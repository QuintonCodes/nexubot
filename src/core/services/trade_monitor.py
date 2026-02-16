import asyncio
import logging
import time
import uuid

logger = logging.getLogger(__name__)


class TradeMonitor:
    def __init__(self, engine):
        self.engine = engine

    def _update_signal_status(self, symbol: str, status: str):
        """Helper to update status text in UI list"""
        for s in self.engine.active_signals:
            if s["symbol"] == symbol:
                s["status"] = status
                break

    async def force_close_trade(self, symbol: str) -> bool:
        """Manually closes an active signal/trade."""
        self.engine.active_signals = [s for s in self.engine.active_signals if s["symbol"] != symbol]
        logger.info(f"🛑 Trade {symbol} Force Closed by User.")
        return True

    async def verify_trade_realtime(self, symbol: str, signal: dict, resume_start_time=None):
        """
        Monitors price and Logs Data for ML.
        Updates self.active_signals state for UI.
        """
        is_shadow = signal.get("is_shadow", False)
        if not is_shadow:
            logger.info(f"👀 Monitoring trade {symbol} for outcome...")
            await self.engine.db.save_active_trade(symbol, signal)

        entry = signal["price"]
        sl = signal["sl"]
        tp = signal["tp"]
        is_long = signal["direction"] == "LONG"
        atr = signal.get("atr", 1.0)
        lot_size = signal["lot_size"]
        tick_value = signal.get("tick_value", 0.0)
        point = signal.get("point", 0.00001)

        # Order Management
        is_filled = signal.get("order_type", "MARKET") == "MARKET"
        if is_filled and not is_shadow:
            self._update_signal_status(symbol, "OPEN")

        # Trailer State
        be_stage = 0  # 0=None, 1=BE, 2=Lock 1R, 3=Lock 2R
        duration = 14400  # 4 hours
        start_time = resume_start_time if resume_start_time else time.time()
        interval = 1  # Check every second

        outcome = "TIMEOUT"
        final_pnl = 0.0
        max_favorable_dist = 0.0
        won = False

        try:
            while (time.time() - start_time) < duration and self.engine.is_running:
                tick = await self.engine.provider.get_current_tick(symbol)
                if not tick:
                    await asyncio.sleep(interval)
                    continue

                current_bid = tick.bid
                current_ask = tick.ask
                spread = current_ask - current_bid

                # --- 1. GHOST ORDER LOGIC (WAIT FOR FILL) ---
                if not is_filled:
                    filled_cond = (current_ask <= (entry - spread)) if is_long else (current_bid >= (entry + spread))

                    if filled_cond:
                        is_filled = True
                        if not is_shadow:
                            self._update_signal_status(symbol, "FILLED")
                            logger.info(f"⚡ {symbol} Ghost Order Filled at {entry}")
                    else:
                        # Runaway cancellation
                        dist_away = (current_ask - entry) if is_long else (entry - current_bid)
                        if dist_away > (atr * 2):
                            outcome = "CANCELLED (Runaway)"
                            break

                        await asyncio.sleep(interval)
                        continue

                # --- 2. TRADE MONITORING (FILLED) ---
                curr_price = current_bid if is_long else current_ask

                hit_sl = curr_price <= sl if is_long else curr_price >= sl
                hit_tp = curr_price >= tp if is_long else curr_price <= tp

                if hit_sl:
                    outcome = "LOSS (SL Hit)"
                    points_lost = (sl - entry) / point if is_long else (entry - sl) / point
                    final_pnl = points_lost * tick_value * lot_size
                    won = False
                    break
                elif hit_tp:
                    outcome = "WIN (TP Hit)"
                    points_won = (tp - entry) / point if is_long else (entry - tp) / point
                    final_pnl = points_won * tick_value * lot_size
                    won = True
                    break

                # Update Max Excursion
                curr_dist = (current_bid - entry) if is_long else (entry - current_ask)
                max_favorable_dist = max(max_favorable_dist, curr_dist)

                # --- 3. MULTI-STAGE TRAILING ---
                if not is_shadow:
                    # Stage 3: Lock 2R
                    if be_stage < 3 and max_favorable_dist > (atr * 3.0):
                        new_sl = entry + (atr * 2.0) if is_long else entry - (atr * 2.0)
                        # Ensure we are moving SL in favorable direction
                        if (is_long and new_sl > sl) or (not is_long and new_sl < sl):
                            sl = new_sl
                            be_stage = 3
                            logger.info(f"🛡️ {symbol} Locked 2R Profit")

                    # Stage 2: Lock 1R
                    elif be_stage < 2 and max_favorable_dist > (atr * 2.0):
                        new_sl = entry + (atr * 1.0) if is_long else entry - (atr * 1.0)
                        if (is_long and new_sl > sl) or (not is_long and new_sl < sl):
                            sl = new_sl
                            be_stage = 2
                            logger.info(f"🛡️ {symbol} Locked 1R Profit")

                    # Stage 1: Breakeven
                    elif be_stage < 1 and max_favorable_dist > (atr * 1.0):
                        # Small buffer to cover commissions/spread
                        buffer = 20 * point
                        new_sl = entry + buffer if is_long else entry - buffer
                        if (is_long and new_sl > sl) or (not is_long and new_sl < sl):
                            sl = new_sl
                            be_stage = 1
                            logger.info(f"🛡️ {symbol} SL Moved to Breakeven")

                await asyncio.sleep(interval)

            # --- POST TRADE PROCESSING ---
            if not is_shadow:
                await self.engine.db.delete_active_trade(symbol)
                self.engine.active_signals = [s for s in self.engine.active_signals if s["symbol"] != symbol]

                if outcome == "TIMEOUT":
                    tick = await self.engine.provider.get_current_tick(symbol)
                    if tick:
                        close_p = tick.bid if is_long else tick.ask
                        points_diff = (close_p - entry) / point if is_long else (entry - close_p) / point
                        final_pnl = points_diff * tick_value * lot_size
                        won = final_pnl > 0
                        outcome = "WIN (Floating)" if won else "LOSS (Floating)"

                if outcome == "CANCELLED":
                    logger.info(f"🚫 {symbol} Order Cancelled")
                    return

                logger.info(f"🔔 Result ({symbol}): {outcome} | PnL: R{final_pnl:.2f}")

                if is_filled:
                    if won:
                        self.engine.session_stats["wins"] += 1
                    else:
                        self.engine.session_stats["losses"] += 1
                    self.engine.session_stats["total"] += 1
                    self.engine.session_stats["pnl"] += final_pnl

                    unique_id = f"{symbol}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                    await self.engine.db.log_trade(
                        {
                            "id": unique_id,
                            "symbol": symbol,
                            "signal": signal["signal"],
                            "confidence": signal["confidence"],
                            "entry": entry,
                            "exit": tp if "TP" in outcome else sl if "SL" in outcome else entry,
                            "won": won,
                            "pnl": final_pnl,
                            "strategy": signal["strategy"],
                            "lot_size": lot_size,
                        }
                    )
            else:
                logger.info(f"👻 Shadow Result ({symbol}): {outcome} (Virtual)")

            if is_filled:
                self.engine.ai_engine.record_trade_outcome(
                    symbol, won, final_pnl, max_favorable_dist / atr if atr else 0, is_shadow
                )

        except Exception as e:
            logger.error(f"Error verifying {symbol}: {e}")
        finally:
            if not is_shadow:
                asyncio.create_task(self.engine.db.delete_active_trade(symbol))
