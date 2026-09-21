"""
Multi-Timeframe Orchestration Layer.
Executes the SMC strategy across HTF (4h), MTF (1h/15m), and LTF (5min) to generate signals.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from config.settings import settings
from data.candle_store import candle_store
from db.repositories import liquidity_pools, order_blocks, signals, structure_events
from strategies.models import TradeSignal
from strategies.sessions import SessionManager
from strategies.smc import (
    calculate_ote_zone,
    detect_fair_value_gaps,
    detect_inducement,
    detect_liquidity_pools,
    detect_liquidity_sweep,
    find_order_blocks,
    generate_trade_signal,
    get_premium_discount_zone,
    is_ob_mitigated,
)
from strategies.structure import classify_structure, detect_bos, detect_choch, detect_mss, detect_swings
from utils.math_helpers import calculate_atr, is_within_range
from utils.logger import logger


class ConfluenceEngine:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.htf = settings.HTF_TIMEFRAMES[0]  # "4h"
        self.mtf = settings.HTF_TIMEFRAMES[1]  # "1h"
        self.mtf_conf = "15min"  # Intermediate confirmation
        self.ltf = settings.ENTRY_TIMEFRAME  # "5min"

    async def _update_structure_state(
        self, df: pd.DataFrame, tf: str, current_bias: Optional[str] = None
    ) -> Optional[str]:
        """Manages running bias sequentially."""
        swings = detect_swings(df)

        if current_bias is None:
            current_bias = await structure_events.get_latest_bias(self.symbol, tf)
        if current_bias is None:
            current_bias = classify_structure(df, swings)  # Bootstrap fallback

        # 1. Check for Continuation (BOS)
        bos = detect_bos(df, swings, self.symbol, tf)
        if bos:
            await structure_events.save_event(bos)
            logger.info("structure_break", event_type="BOS", direction=bos.direction, tf=tf)
            return bos.direction

        # 2. Check for Market Structure Shift
        mss = detect_mss(df, swings, current_bias or "ranging", self.symbol, tf)
        if mss:
            await structure_events.save_event(mss)
            logger.info("structure_break", event_type="MSS", direction=mss.direction, tf=tf)
            return mss.direction

        # 3. Check for standard Reversal
        choch = detect_choch(df, swings, current_bias or "ranging", self.symbol, tf)
        if choch:
            await structure_events.save_event(choch)
            logger.info("structure_break", event_type="CHoCH", direction=choch.direction, tf=tf)
            return choch.direction

        return current_bias

    async def scan_htf(self) -> None:
        """Run every 4 hours. Updates overarching bias."""
        df = await candle_store.get_candles(self.symbol, self.htf)
        if len(df) < 20:
            return

        # Pass None initially
        bias = await self._update_structure_state(df, self.htf)
        logger.info("htf_scan_complete", bias=bias)

    async def scan_mtf(self) -> None:
        """Run every 1 hour. Finds new Order Blocks."""
        df = await candle_store.get_candles(self.symbol, self.mtf)
        if len(df) < 20:
            return

        swings = detect_swings(df)
        bias = await self._update_structure_state(df, self.mtf)

        if bias in ["bullish", "bearish"]:
            obs = find_order_blocks(df, swings, bias, self.symbol, self.mtf)
            for ob in obs:
                await order_blocks.save_order_block(ob)
                logger.info("order_block_detected", tf=self.mtf, direction=bias, price=ob.ob_50)

    async def scan_mtf_confirmation(self) -> None:
        """Run every 15 minutes to evaluate MTF confirmation layer."""
        df = await candle_store.get_candles(self.symbol, self.mtf_conf)
        if len(df) < 20:
            return

        bias = await self._update_structure_state(df, self.mtf_conf)
        logger.info("mtf_15m_scan_complete", bias=bias)

    async def scan_ltf_entry(self) -> Optional[TradeSignal]:
        """Run on every 5min candle close. Evaluates deep SMC confluence for execution."""
        df = await candle_store.get_candles(self.symbol, self.ltf)
        if len(df) < 20:
            return None

        # 1. Check Baseline HTF Alignment
        htf_bias = await structure_events.get_latest_bias(self.symbol, self.htf)
        if not htf_bias:
            return None

        current_price = df.iloc[-1]["close"]
        swings_ltf = detect_swings(df)

        factors = [f"HTF Bias: {htf_bias.upper()}"]
        confluence_score = 20

        # Premium / Discount Validation
        highs, lows = [s for s in swings_ltf if s.type == "high"], [s for s in swings_ltf if s.type == "low"]
        pd_zone = (
            get_premium_discount_zone(lows[-1].price, highs[-1].price, current_price)
            if highs and lows
            else "Equilibrium"
        )

        # Gate Signal Logic: Reject sub-optimal positional setups
        if (htf_bias == "bullish" and pd_zone == "Premium") or (htf_bias == "bearish" and pd_zone == "Discount"):
            logger.info("signal_rejected_pd_array", direction=htf_bias, pd_zone=pd_zone)
            return None

        # Session Killzone Integration
        active_session = SessionManager.get_active_killzone(datetime.now(timezone.utc))
        if active_session != "Out of Session":
            factors.append(f"Killzone Active ({active_session})")
            confluence_score += 5

        # Check Active OBs and Breaker Blocks
        valid_ob = None
        entry_model = "Unknown Setup"

        active_obs = await order_blocks.get_active_order_blocks(self.symbol, self.mtf)
        for ob in active_obs:
            if ob["direction"] != htf_bias:
                continue

            if is_ob_mitigated(df, ob):
                await order_blocks.mark_mitigated(ob["id"])
                continue

            # Price action inside the unmitigated OB zone
            if is_within_range(current_price, ob["ob_low"], ob["ob_high"]):
                valid_ob = ob
                entry_model = "MTF OB Entry"
                factors.append(f"MTF {ob['direction'].capitalize()} OB Tap")
                confluence_score += 25
                break

        # Fallback to Breaker Block if no standard OB is tapped
        if not valid_ob:
            breakers = await order_blocks.get_active_breaker_blocks(self.symbol, self.mtf)
            for brk in breakers:
                # A bullish setup requires tapping a bearish order block that was broken upwards
                if brk["direction"] != htf_bias:
                    if is_within_range(current_price, brk["ob_low"], brk["ob_high"]):
                        valid_ob = brk
                        entry_model = "Breaker Block Retest"
                        factors.append("MTF Breaker Block Tap")
                        confluence_score += 25
                        break

        if not valid_ob:
            return None

        # Check FVG Combo
        fvgs = detect_fair_value_gaps(df.tail(10), self.symbol, self.ltf)
        if any(f.direction == htf_bias and is_within_range(current_price, f.bottom, f.top) for f in fvgs):
            entry_model = "OB + FVG Combo"
            factors.append("FVG Tap Confirmed")
            confluence_score += 10

        # Liquidity Sweeps
        pools = detect_liquidity_pools(swings_ltf, self.symbol, self.ltf, settings.XAUUSD_PIP_TOLERANCE)
        swept_pool = detect_liquidity_sweep(df, pools)

        if swept_pool:
            # Confirm sweep aligns with institutional intent (e.g. sweep EQL to go long)
            if (htf_bias == "bullish" and swept_pool.pool_type == "EQL") or (
                htf_bias == "bearish" and swept_pool.pool_type == "EQH"
            ):
                factors.append(f"Liquidity Sweep ({swept_pool.pool_type})")
                confluence_score += 20
                await liquidity_pools.save_pool(swept_pool)

        # Inducement Tracking
        idm = detect_inducement(swings_ltf, htf_bias)
        if idm and (
            (htf_bias == "bullish" and current_price < idm.price)
            or (htf_bias == "bearish" and current_price > idm.price)
        ):
            factors.append("Inducement (IDM) Swept")
            confluence_score += 15

        # OTE Mapping
        if highs and lows:
            ote_zone = calculate_ote_zone(lows[-1].price, highs[-1].price, valid_ob["direction"], self.symbol, self.ltf)

            if is_within_range(current_price, ote_zone.ote_entry, ote_zone.ote_top):
                entry_model = "OTE Retracement Tap"
                factors.append(f"OTE Tap ({ote_zone.ote_entry:.2f}–{ote_zone.ote_top:.2f})")
                confluence_score += 20

        ltf_choch = detect_choch(df, swings_ltf, "ranging", self.symbol, self.ltf)
        if ltf_choch and ltf_choch.direction == htf_bias:
            factors.append("LTF CHoCH Confirmed")
            confluence_score += 10

        min_required_score = getattr(settings, "MIN_CONFLUENCE_SCORE", 70)
        if confluence_score < min_required_score:
            logger.info("signal_rejected_low_confluence", score=confluence_score, min_req=min_required_score)
            return None

        # Generate Validated Signal
        atr_val = calculate_atr(df)
        sl_buffer = atr_val * 0.5
        trade_dir = "buy" if valid_ob["direction"] == "bullish" else "sell"
        stop_loss = valid_ob["ob_low"] - sl_buffer if trade_dir == "buy" else valid_ob["ob_high"] + sl_buffer

        # Check Deduplication
        is_dup = await signals.is_duplicate(
            self.symbol, trade_dir, current_price, settings.XAUUSD_PIP_TOLERANCE, settings.SIGNAL_COOLDOWN_HOURS
        )
        if is_dup:
            logger.info("signal_suppressed_duplicate", symbol=self.symbol)
            return None

        signal = generate_trade_signal(
            symbol=self.symbol,
            tf=self.ltf,
            direction=trade_dir,
            entry_price=current_price,
            stop_loss=stop_loss,
            factors=factors,
            confluence_score=confluence_score,
            timestamp=datetime.now(timezone.utc),
            entry_model=entry_model,
            session=active_session,
            pd_zone=pd_zone,
        )

        await signals.save_signal(signal)
        logger.info("trade_signal_generated", signal_id=signal.signal_id, score=confluence_score)

        return signal
