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
from strategies.models import StructureEvent, TradeSignal
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
from strategies.structure import classify_structure, detect_bos, detect_choch, detect_cisd, detect_mss, detect_swings
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

        # 1. HTF/MTF Path: Extract bias cleanly through full continuous replay to catch structural bias on cold start
        if tf != self.ltf:
            new_bias = classify_structure(df, swings)
            if new_bias != current_bias and new_bias != "ranging":
                synthetic_event = StructureEvent(
                    symbol=self.symbol,
                    timeframe=tf,
                    event_type="BOS",
                    direction=new_bias,
                    price_level=swings[-1].price if swings else 0.0,
                    timestamp=df.iloc[-1]["timestamp"],
                    confirmed=True,
                )
                await structure_events.save_event(synthetic_event)
            return new_bias

        # 2. LTF Path: Relies on localized candle breaks for precision event tracking
        if current_bias is None:
            current_bias = classify_structure(df, swings)

        bos = detect_bos(df, swings, current_bias or "ranging", self.symbol, tf)
        if bos:
            await structure_events.save_event(bos)
            logger.info("structure_break", event_type="BOS", direction=bos.direction, tf=tf)
            return bos.direction

        mss = detect_mss(df, swings, current_bias or "ranging", self.symbol, tf)
        if mss:
            await structure_events.save_event(mss)
            logger.info("structure_break", event_type="MSS", direction=mss.direction, tf=tf)
            return mss.direction

        choch = detect_choch(df, swings, current_bias or "ranging", self.symbol, tf)
        if choch:
            await structure_events.save_event(choch)
            logger.info("structure_break", event_type="CHoCH", direction=choch.direction, tf=tf)
            return choch.direction

        # Optional: Secondary validation to trace structural reversals natively against existing OB zones
        active_obs = await order_blocks.get_active_order_blocks(self.symbol, tf)
        for ob in active_obs:
            cisd = detect_cisd(df, ob, self.symbol, tf)
            if cisd:
                await structure_events.save_event(cisd)
                logger.info("structure_break", event_type="CISD", direction=cisd.direction, tf=tf)

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
        """Run every 1 hour. Finds new Order Blocks and converts validated structures to Breakers."""
        df = await candle_store.get_candles(self.symbol, self.mtf)
        if len(df) < 20:
            return

        swings = detect_swings(df)
        current_bias = await structure_events.get_latest_bias(self.symbol, self.mtf)
        bias = await self._update_structure_state(df, self.mtf, current_bias)

        pip_tol = getattr(settings, "XAUUSD_PIP_TOLERANCE", 40.0)
        pools = detect_liquidity_pools(swings, self.symbol, self.mtf, pip_tol)
        for pool in pools:
            await liquidity_pools.save_pool(pool)

        if current_bias and bias and bias != current_bias and bias in ["bullish", "bearish"]:
            opposing = "bullish" if bias == "bearish" else "bearish"
            await order_blocks.mark_as_breaker(self.symbol, self.mtf, opposing)

        if bias in ["bullish", "bearish"]:
            obs = find_order_blocks(df, swings, bias, self.symbol, self.mtf)
            for ob in obs:
                await order_blocks.save_order_block(ob)
                logger.info("order_block_detected", tf=self.mtf, direction=bias, price=ob.ob_50)

    async def scan_mtf_confirmation(self) -> None:
        """Run every 15 minutes to evaluate MTF confirmation layer and detect local Order Blocks."""
        df = await candle_store.get_candles(self.symbol, self.mtf_conf)
        if len(df) < 20:
            return

        swings = detect_swings(df)
        current_bias = await structure_events.get_latest_bias(self.symbol, self.mtf_conf)
        bias = await self._update_structure_state(df, self.mtf_conf, current_bias)

        pip_tol = getattr(settings, "XAUUSD_PIP_TOLERANCE", 40.0)
        pools = detect_liquidity_pools(swings, self.symbol, self.mtf_conf, pip_tol)
        for pool in pools:
            await liquidity_pools.save_pool(pool)

        # Save 15m order blocks to give 5m entry trigger more localized targets
        if bias in ["bullish", "bearish"]:
            obs = find_order_blocks(df, swings, bias, self.symbol, self.mtf_conf)
            for ob in obs:
                await order_blocks.save_order_block(ob)

        logger.info("mtf_15m_scan_complete", bias=bias)

    async def scan_ltf_entry(self) -> Optional[TradeSignal]:
        """Run on every 5min candle close. Evaluates deep SMC confluence for execution."""
        df = await candle_store.get_candles(self.symbol, self.ltf)
        if len(df) < 20:
            return None

        htf_bias = await structure_events.get_latest_bias(self.symbol, self.htf)
        if not htf_bias:
            return None

        current_price = df.iloc[-1]["close"]
        swings_ltf = detect_swings(df)
        ltf_highs = [s for s in swings_ltf if s.type == "high"]
        ltf_lows = [s for s in swings_ltf if s.type == "low"]

        factors = [f"HTF Bias: {htf_bias.upper()}"]
        confluence_score = 20

        # MTF Context Gate Checks
        mtf_bias = await structure_events.get_latest_bias(self.symbol, self.mtf)
        if mtf_bias == htf_bias:
            factors.append(f"MTF Alignment ({self.mtf})")
            confluence_score += 15

        mtf_15m_bias = await structure_events.get_latest_bias(self.symbol, self.mtf_conf)
        if mtf_15m_bias == htf_bias:
            factors.append("15M Structure Aligned")
            confluence_score += 10

        # Localized PD Zone (Anchored to 1H MTF Dealing Range)
        mtf_df = await candle_store.get_candles(self.symbol, self.mtf)
        mtf_swings = detect_swings(mtf_df)
        mtf_highs = [s for s in mtf_swings if s.type == "high"]
        mtf_lows = [s for s in mtf_swings if s.type == "low"]

        if mtf_highs and mtf_lows:
            pd_zone = get_premium_discount_zone(mtf_lows[-1].price, mtf_highs[-1].price, current_price)
        else:
            pd_zone = "Equilibrium"

        # Early Sweep Detection to validate potential PD Array overrides
        pip_tol = getattr(settings, "XAUUSD_PIP_TOLERANCE", 40.0)
        pools = detect_liquidity_pools(swings_ltf, self.symbol, self.ltf, pip_tol)
        swept_pool = detect_liquidity_sweep(df, pools)

        valid_sweep = swept_pool and (
            (htf_bias == "bullish" and swept_pool.pool_type == "EQL")
            or (htf_bias == "bearish" and swept_pool.pool_type == "EQH")
        )

        valid_ote = False
        ote_entry_model_text = ""
        if ltf_highs and ltf_lows:
            ote_zone = calculate_ote_zone(ltf_lows[-1].price, ltf_highs[-1].price, htf_bias, self.symbol, self.ltf)
            if is_within_range(current_price, ote_zone.ote_entry, ote_zone.ote_top):
                valid_ote = True
                ote_entry_model_text = f"OTE Tap ({ote_zone.ote_entry:.2f}–{ote_zone.ote_top:.2f})"

        # Strict SMC Rule: Never buy in Premium, never sell in Discount
        # (unless swept liquidity OR an OTE retracement setup overrides)
        if (htf_bias == "bullish" and pd_zone == "Premium") or (htf_bias == "bearish" and pd_zone == "Discount"):
            if not valid_sweep and not valid_ote:
                logger.info("signal_rejected_pd_array", direction=htf_bias, pd_zone=pd_zone)
                return None
            else:
                factors.append("PD Array Override (Liquidity Sweep / OTE)")
        else:
            factors.append(f"Optimal PD Zone ({pd_zone})")
            confluence_score += 10

        active_session = SessionManager.get_active_killzone(datetime.now(timezone.utc))
        if active_session != "Out of Session":
            factors.append(f"Killzone Active ({active_session})")
            confluence_score += 5

        valid_ob = None
        entry_model = "Unknown Setup"

        # Check BOTH 1H and 15M active order blocks
        active_obs = await order_blocks.get_active_order_blocks(self.symbol, self.mtf)
        active_obs.extend(await order_blocks.get_active_order_blocks(self.symbol, self.mtf_conf))

        for ob in active_obs:
            if ob["direction"] != htf_bias:
                continue

            if is_ob_mitigated(df, ob):
                await order_blocks.mark_mitigated(ob["id"])
                continue

            if is_within_range(current_price, float(ob["ob_low"]), float(ob["ob_high"])):
                valid_ob = ob
                entry_model = f"{ob['timeframe']} OB Entry"
                factors.append(f"{ob['timeframe']} {ob['direction'].capitalize()} OB Tap")
                confluence_score += 25
                break

        if not valid_ob:
            # Check BOTH 1H and 15M active breaker blocks
            breakers = await order_blocks.get_active_breaker_blocks(self.symbol, self.mtf)
            breakers.extend(await order_blocks.get_active_breaker_blocks(self.symbol, self.mtf_conf))
            for brk in breakers:
                if brk["direction"] != htf_bias:
                    if is_within_range(current_price, float(brk["ob_low"]), float(brk["ob_high"])):
                        valid_ob = brk
                        entry_model = f"{brk['timeframe']} Breaker Block Retest"
                        factors.append(f"{brk['timeframe']} Breaker Block Tap")
                        confluence_score += 25
                        break

        if not valid_ob:
            return None

        cisd_event = detect_cisd(df, valid_ob, self.symbol, self.ltf)
        if cisd_event:
            factors.append("CISD Confirmed")
            confluence_score += 15
            await structure_events.save_event(cisd_event)

        fvgs = detect_fair_value_gaps(df.tail(10), self.symbol, self.ltf)
        if any(f.direction == htf_bias and is_within_range(current_price, f.bottom, f.top) for f in fvgs):
            entry_model = "OB + FVG Combo"
            factors.append("FVG Tap Confirmed")
            confluence_score += 10

        if valid_sweep:
            factors.append(f"Liquidity Sweep ({swept_pool.pool_type})")
            confluence_score += 20
            await liquidity_pools.save_pool(swept_pool)

        idm = detect_inducement(swings_ltf, htf_bias)
        if idm and (
            (htf_bias == "bullish" and current_price < idm.price)
            or (htf_bias == "bearish" and current_price > idm.price)
        ):
            factors.append("Inducement (IDM) Swept")
            confluence_score += 15

        if valid_ote:
            entry_model = "OTE Retracement Tap"
            factors.append(ote_entry_model_text)
            confluence_score += 20

        ltf_bias = await structure_events.get_latest_bias(self.symbol, self.ltf) or htf_bias or "ranging"
        ltf_choch = detect_choch(df, swings_ltf, ltf_bias, self.symbol, self.ltf)
        if ltf_choch and ltf_choch.direction == htf_bias:
            factors.append("LTF CHoCH Confirmed")
            confluence_score += 10

        # Adjust score thresholds slightly to account for the PD array logic shift
        if active_session == "Out of Session":
            min_required_score = getattr(settings, "MIN_CONFLUENCE_SCORE_OOS", 75)
        else:
            min_required_score = getattr(settings, "MIN_CONFLUENCE_SCORE", 65)

        if confluence_score < min_required_score:
            logger.info("signal_rejected_low_confluence", score=confluence_score, min_req=min_required_score)
            return None

        # Generate Validated Signal
        atr_val = calculate_atr(df)
        sl_buffer = atr_val * 0.5
        trade_dir = "buy" if valid_ob["direction"] == "bullish" else "sell"
        stop_loss = (
            float(valid_ob["ob_low"]) - sl_buffer if trade_dir == "buy" else float(valid_ob["ob_high"]) + sl_buffer
        )

        # Check Deduplication
        is_dup = await signals.is_duplicate(self.symbol, self.ltf, trade_dir, settings.SIGNAL_COOLDOWN_MINUTES)
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
