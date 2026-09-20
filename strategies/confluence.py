"""
Multi-Timeframe Orchestration Layer.
Executes the SMC strategy across HTF (4h), MTF (1h), and LTF (5min) to generate signals.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from config.settings import settings
from data.candle_store import candle_store
from db.repositories import order_blocks, structure_events, signals
from strategies.models import TradeSignal
from strategies.smc import find_order_blocks, is_ob_mitigated, generate_trade_signal
from strategies.structure import detect_swings, classify_structure, detect_bos
from utils.logger import logger


class ConfluenceEngine:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.htf = settings.HTF_TIMEFRAMES[0]  # "4h"
        self.mtf = settings.HTF_TIMEFRAMES[1]  # "1h"
        self.ltf = settings.ENTRY_TIMEFRAME  # "5min"

    async def _update_structure_state(self, df: pd.DataFrame, tf: str) -> Optional[str]:
        """Detects and saves structural breaks, returning the current bias."""
        swings = detect_swings(df)
        bias = classify_structure(swings)

        # Check for immediate structural breaks on the latest candle
        bos = detect_bos(df, swings, self.symbol, tf)
        if bos:
            await structure_events.save_event(bos)
            logger.info("structure_break", event="BOS", direction=bos.direction, tf=tf)
            return bos.direction

        return bias

    async def scan_htf(self) -> None:
        """Run every 4 hours. Updates overarching bias."""
        df = await candle_store.get_candles(self.symbol, self.htf)
        if len(df) < 20:
            return

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

    async def scan_ltf_entry(self) -> Optional[TradeSignal]:
        """Run on every 5min candle close. Looks for entry triggers within active MTF zones."""
        df = await candle_store.get_candles(self.symbol, self.ltf)
        if len(df) < 20:
            return None

        # 1. Get overarching HTF Bias from DB
        htf_bias = await structure_events.get_latest_bias(self.symbol, self.htf)
        if not htf_bias:
            return None

        # 2. Get active MTF Order Blocks
        active_obs = await order_blocks.get_active_order_blocks(self.symbol, self.mtf)
        if not active_obs:
            return None

        current_price = df.iloc[-1]["close"]
        factors = [f"HTF Bias: {htf_bias.upper()}"]

        # 3. Check for OB Entry Tap
        valid_ob = None
        for ob in active_obs:
            if ob.direction != htf_bias:
                continue

            # Check if OB is mitigated by the LTF price action
            if is_ob_mitigated(df, ob):
                await order_blocks.mark_mitigated(ob.id, datetime.now(timezone.utc))
                continue

            # If price is inside the OB zone, this is an entry condition
            if min(ob.ob_low, ob.ob_high) <= current_price <= max(ob.ob_low, ob.ob_high):
                valid_ob = ob
                factors.append(f"MTF {ob.direction.capitalize()} OB Tap")
                break

        if not valid_ob:
            return None

        # 4. Generate Signal
        trade_dir = "buy" if valid_ob.direction == "bullish" else "sell"
        stop_loss = valid_ob.ob_low - 2.0 if trade_dir == "buy" else valid_ob.ob_high + 2.0

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
            timestamp=datetime.now(timezone.utc),
        )

        await signals.save_signal(signal)
        logger.info("trade_signal_generated", signal_id=signal.signal_id)

        return signal
