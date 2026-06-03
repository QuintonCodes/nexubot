import asyncio
import logging
import math
import time
import uuid
from typing import Tuple

from src.config import CANDLE_LIMIT, TIMEFRAME

logger = logging.getLogger(__name__)


class OfflineManager:
    """Handles trades that were active during a shutdown or restart, calculates their outcomes based on historical price data, and manages any necessary post-trade processing or notifications."""

    def __init__(self, engine):
        self.engine = engine

    def _calculate_offline_result(
        self, signal: dict, start_time: float, klines: list, currency: str
    ) -> Tuple[str, float, bool]:
        """Calculates the outcome of an offline trade based on historical price data and the original trade signal parameters."""
        if not klines:
            return None, 0.0, False

        sl = signal["sl"]
        tp = signal["tp"]
        entry = signal["price"]
        is_long = signal["direction"] == "LONG"
        order_type = signal.get("order_type", "MARKET")
        trade_duration = 14400  # Max tracking 4 hours

        # Pull correct currency field priorities
        risk_val = signal.get("risk_zar", 0) if currency == "ZAR" else signal.get("risk_account", 0)
        profit_val = signal.get("profit_zar", 0) if currency == "ZAR" else signal.get("profit_account", 0)

        outcome = None
        pnl = 0.0
        filled_offline = order_type == "MARKET"

        tp1 = signal.get("tp1")
        tp2 = signal.get("tp2")
        atr_buf = signal.get("atr", 0) * 0.15

        tp1_hit, tp2_hit, tp3_hit = False, False, False

        for k in klines:
            if k["time"] < start_time:
                continue

            if (k["time"] - start_time) > trade_duration:
                outcome = "TIMEOUT (Offline)"
                break

            if not filled_offline:
                if is_long:
                    if k["low"] <= entry:
                        filled_offline = True
                else:
                    if k["high"] >= entry:
                        filled_offline = True
                if filled_offline:
                    continue

            # Multi-TP Check Sequence
            if is_long:
                if k["low"] <= sl:
                    break

                if tp1 and k["high"] >= tp1 and not tp1_hit:
                    tp1_hit = True
                    sl = entry + atr_buf
                if tp2 and k["high"] >= tp2 and not tp2_hit:
                    tp2_hit = True
                    sl = tp1
                if k["high"] >= tp:
                    tp3_hit = True
                    break
            else:
                if k["high"] >= sl:
                    break

                if tp1 and k["low"] <= tp1 and not tp1_hit:
                    tp1_hit = True
                    sl = entry - atr_buf
                if tp2 and k["low"] <= tp2 and not tp2_hit:
                    tp2_hit = True
                    sl = tp1
                if k["low"] <= tp:
                    tp3_hit = True
                    break

        if outcome == "TIMEOUT (Offline)":
            return outcome, 0.0, filled_offline

        # Apply identical win/loss structure mirroring backfill
        if tp3_hit:
            outcome = "WIN (TP3 Offline)"
            pnl = profit_val
        elif tp2_hit:
            outcome = "WIN (TP2 Offline)"
            pnl = profit_val * 0.66
        elif tp1_hit:
            outcome = "WIN (TP1 Offline)"
            pnl = profit_val * 0.33
        else:
            outcome = "LOSS (SL Offline)"
            pnl = -risk_val

        return outcome, pnl, filled_offline

    async def check_offline_trades(self) -> None:
        """Checks for any active trades that were not monitored in real-time (e.g., due to shutdown) and calculates their outcomes based on historical price data."""
        logger.info("🔄 Checking for interrupted trades...")
        active_trades = await self.engine.db.get_active_trades()

        if not active_trades:
            logger.info("✅ No interrupted trades found.")
            return

        for symbol, signal, start_time in active_trades:
            self.engine.ai_engine.register_active_trade(symbol, signal.get("strategy", "Unknown"))

            elapsed = time.time() - start_time
            candles_needed = min(math.ceil(elapsed / 300) + 5, 1440)
            klines = await self.engine.provider.fetch_klines(symbol, TIMEFRAME, min(candles_needed, CANDLE_LIMIT))

            currency = self.engine.session_stats.get("currency", "USD")
            outcome, pnl, filled = self._calculate_offline_result(signal, start_time, klines, currency)

            curr_sym = "R" if currency == "ZAR" else "$"

            # Case 1: Trade finished (TP or SL hit)
            if outcome and "TIMEOUT" not in outcome:
                won = pnl > 0
                logger.info(f"🔔 Offline Result ({symbol}): {outcome} | PnL: {curr_sym}{pnl:.2f}")
                unique_id = f"{symbol}_OFFLINE_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                await self.engine.db.log_trade(
                    {
                        "id": unique_id,
                        "symbol": symbol,
                        "signal": signal["signal"],
                        "confidence": signal["confidence"],
                        "entry": signal["price"],
                        "exit": signal["tp"] if won else signal["sl"],
                        "won": won,
                        "pnl": pnl,
                        "strategy": signal["strategy"] + " (Offline)",
                        "lot_size": signal["lot_size"],
                    }
                )
                await self.engine.db.delete_active_trade(symbol)
                self.engine.ai_engine.active_features.pop(symbol, None)

            # Case 2: Trade Timed Out
            elif outcome == "TIMEOUT (Offline)" or elapsed > 14400:
                if not filled:
                    logger.info(f"🚫 Offline Result ({symbol}): CANCELLED (Never Filled)")
                else:
                    # Calculate Floating PnL
                    last_close = klines[-1]["close"] if klines else signal["price"]
                    tick_val = signal.get("tick_value", 0.0)
                    point = signal.get("point", 0.00001)
                    lot = signal.get("lot_size", 0.1)

                    diff = (
                        (last_close - signal["price"])
                        if signal["direction"] == "LONG"
                        else (signal["price"] - last_close)
                    )
                    pnl = (diff / point) * tick_val * lot
                    won = pnl > 0

                    outcome_str = "WIN (Timeout)" if won else "LOSS (Timeout)"
                    logger.info(f"🔔 Offline Result ({symbol}): {outcome_str} | PnL: {curr_sym}{pnl:.2f}")

                    unique_id = f"{symbol}_TIMEOUT_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                    await self.engine.db.log_trade(
                        {
                            "id": unique_id,
                            "symbol": symbol,
                            "signal": signal["signal"],
                            "confidence": signal["confidence"],
                            "entry": signal["price"],
                            "exit": last_close,
                            "won": won,
                            "pnl": pnl,
                            "strategy": signal["strategy"] + " (Timeout)",
                            "lot_size": lot,
                        }
                    )
                await self.engine.db.delete_active_trade(symbol)
                self.engine.ai_engine.active_features.pop(symbol, None)

            # Case 3: Still Active
            else:
                logger.info(f"♻️ Resuming Active Trade: {symbol} (Strategy: {signal.get('strategy', 'Unknown')})")
                self.engine.active_signals.append(signal)
                self.engine.monitored_tasks[symbol] = asyncio.create_task(
                    self.engine.monitor.verify_trade_realtime(symbol, signal, start_time)
                )
