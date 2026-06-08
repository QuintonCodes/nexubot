import asyncio
import logging
import time
import uuid

logger = logging.getLogger(__name__)


class TradeMonitor:
    """Monitors active trades in real-time for TP/SL hits, manages trailing logic, and handles post-trade processing."""

    def __init__(self, engine):
        self.engine = engine

    async def _ghost_track_tp3(self, symbol: str, entry: float, tp3: float, is_long: bool, point: float) -> None:
        """Silently tracks if TP3 is eventually hit after an early Breakeven/TP1 exit."""
        logger.info(f"👻 Ghost tracking {symbol} for TP3 hit ...")
        start = time.time()
        try:
            while time.time() - start < 14400 and self.engine.is_running:  # Track up to 4 hrs
                tick = await self.engine.provider.get_current_tick(symbol)
                if not tick:
                    await asyncio.sleep(2)
                    continue

                curr_price = tick.bid if is_long else tick.ask
                hit_tp3 = curr_price >= tp3 if is_long else curr_price <= tp3

                if hit_tp3:
                    pips = (tp3 - entry) / (point * 10) if is_long else (entry - tp3) / (point * 10)
                    msg = f"👻 *Ghost Tracker Update:* {symbol} eventually hit TP3 for a theoretical {pips:.1f} Pips!"
                    await self.engine.notifier.send_message(msg)
                    break

                await asyncio.sleep(2)
        except asyncio.CancelledError:
            return
        finally:
            self.engine.monitored_tasks.pop(f"ghost_{symbol}", None)

    async def verify_trade_realtime(self, symbol: str, signal: dict, resume_start_time=None) -> None:
        """Monitors an active trade in real-time for TP/SL hits, manages trailing logic, and handles post-trade processing."""
        logger.info(f"👀 Monitoring trade {symbol} for outcome...")
        await self.engine.db.save_active_trade(symbol, signal)

        entry = signal["price"]
        sl = signal["sl"]
        tp1 = signal.get("tp1")
        tp2 = signal.get("tp2")
        tp3 = signal.get("tp3", signal["tp"])

        is_long = signal["direction"] == "LONG"
        point = signal.get("point", 0.00001)

        lot_size = signal["lot_size"]
        tick_value = signal.get("tick_value", 0.0)
        atr = signal["atr"]

        # Order Management
        is_filled = signal.get("order_type", "MARKET") == "MARKET"

        # Trailer State
        duration = 14400  # 4 hours
        start_time = resume_start_time if resume_start_time else time.time()
        interval = 1  # Check every second

        be_stage = 0  # 0=None, 1=BE, 2=Lock 1R, 3=Lock 2R
        tp1_hit = False
        tp2_hit = False
        outcome = "TIMEOUT"
        final_pips = 0.0
        final_pnl = 0.0
        won = False
        max_favorable_dist = 0.0

        try:
            while (time.time() - start_time) < duration and self.engine.is_running:
                tick = await self.engine.provider.get_current_tick(symbol)
                if not tick:
                    await asyncio.sleep(interval)
                    continue

                    #   --- 1. TRADE MONITORING (FILLED)   ---
                curr_price = tick.bid if is_long else tick.ask
                curr_dist = (tick.bid - entry) if is_long else (entry - tick.ask)

                max_favorable_dist = max(max_favorable_dist, curr_dist)

                # SL Evaluation
                hit_sl = curr_price <= sl if is_long else curr_price >= sl
                if hit_sl:
                    if be_stage > 0:
                        outcome = "WIN (Stopped in Profit/BE)"
                        won = True
                    else:
                        outcome = "LOSS (SL Hit)"
                        won = False

                    points_diff = (curr_price - entry) / point if is_long else (entry - curr_price) / point
                    final_pnl = points_diff * tick_value * lot_size
                    final_pips = abs(curr_price - entry) / (point * 10)
                    break

                # TP1 Milestone Tracking
                if not tp1_hit and tp1 is not None:
                    hit_tp1 = curr_price >= tp1 if is_long else curr_price <= tp1
                    if hit_tp1:
                        tp1_hit = True
                        be_stage = max(be_stage, 1)
                        # Move SL to Breakeven (+ dynamic ATR buffer)
                        buffer = atr * 0.15
                        sl = entry + buffer if is_long else entry - buffer
                        await self.engine.notifier.send_message(f"🎯 *{symbol} Hit TP1!* Moving SL to Breakeven.")

                # TP2 Milestone Tracking
                if tp1_hit and not tp2_hit and tp2 is not None:
                    hit_tp2 = curr_price >= tp2 if is_long else curr_price <= tp2
                    if hit_tp2:
                        tp2_hit = True
                        be_stage = max(be_stage, 2)
                        # Move SL to TP1
                        sl = tp1
                        await self.engine.notifier.send_message(f"🎯 *{symbol} Hit TP2!* Trailing SL to TP1.")

                # TP3 Final Target
                if tp3 is not None:
                    hit_tp3 = curr_price >= tp3 if is_long else curr_price <= tp3
                    if hit_tp3:
                        outcome = "WIN (Full TP3 Hit)"
                        points_diff = (tp3 - entry) if is_long else (entry - tp3)
                        points_won = (tp3 - entry) / point if is_long else (entry - tp3) / point
                        final_pnl = points_won * tick_value * lot_size
                        final_pips = points_diff / (point * 10)
                        won = True
                        break

                await asyncio.sleep(interval)

            #   --- POST TRADE PROCESSING   ---
            self.engine.active_signals = [s for s in self.engine.active_signals if s["symbol"] != symbol]

            if outcome == "TIMEOUT":
                tick = await self.engine.provider.get_current_tick(symbol)
                if tick:
                    close_p = tick.bid if is_long else tick.ask
                    points_diff = (close_p - entry) / point if is_long else (entry - close_p) / point
                    points_timeout = (close_p - entry) if is_long else (entry - close_p)
                    final_pnl = points_diff * tick_value * lot_size
                    final_pips = points_timeout / (point * 10)
                    won = final_pnl > 0 or final_pips > 0
                    outcome = "WIN (Floating)" if won else "LOSS (Floating)"

            final_pnl = round(final_pnl, 2)
            currency = self.engine.session_stats.get("currency", "USD")
            curr_sym = "R" if currency == "ZAR" else "$"
            logger.info(f"🏁 Result ({symbol}): {outcome} | PnL: {curr_sym}{final_pnl:.2f} | Pips: {final_pips:.1f}")

            if is_filled:
                await self.engine.notifier.send_trade_result(symbol, outcome, final_pips, won, final_pnl, currency)

                # Initiate Ghost Tracking for Early Exits
                if "Stopped in Profit/BE" in outcome:
                    ghost_task = asyncio.create_task(self._ghost_track_tp3(symbol, entry, tp3, is_long, point))
                    self.engine.monitored_tasks[f"ghost_{symbol}"] = ghost_task

                if won:
                    self.engine.session_stats["wins"] += 1
                else:
                    self.engine.session_stats["losses"] += 1

                self.engine.session_stats["total"] += 1
                self.engine.session_stats["pnl"] += final_pnl

                await self.engine.db.log_trade(
                    {
                        "id": f"{symbol}_{int(time.time())}_{uuid.uuid4().hex[:6]}",
                        "symbol": symbol,
                        "signal": signal["signal"],
                        "confidence": signal["confidence"],
                        "entry": entry,
                        "exit": tp3 if "TP" in outcome else sl if "SL" in outcome else entry,
                        "won": won,
                        "pnl": final_pnl,
                        "currency": currency,
                        "strategy": signal["strategy"],
                        "lot_size": lot_size,
                    }
                )

                target_excursion = max_favorable_dist / atr if atr > 0 else 0.0
                self.engine.ai_engine.record_trade_outcome(symbol, won, final_pnl, target_excursion)

        except asyncio.CancelledError:
            logger.info(f"Monitor for {symbol} cancelled.")
        except Exception as e:
            logger.error(f"Error verifying {symbol}: {e}")
        finally:
            try:
                await self.engine.db.delete_active_trade(symbol)
                self.engine.monitored_tasks.pop(symbol, None)
            except Exception as e:
                logger.error(f"Failed to delete active trade during cleanup: {e}")
