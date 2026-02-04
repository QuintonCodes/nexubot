import asyncio
import eel
import math
import os
import sys
import time
import threading
import uuid
from datetime import datetime, timedelta

from src.data.provider import DataProvider
from src.database.manager import DatabaseManager
from src.engine.ai_engine import AITradingEngine
from src.utils.backfill import backfill_data
from src.utils.logger import setup_logging
from src.utils.trainer import ModelTrainer
from src.config import (
    CANDLE_LIMIT,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_RISK_PCT,
    FALLBACK_CRYPTO,
    FALLBACK_FOREX,
    MAX_SIGNALS_PER_SCAN,
    SCAN_INTERVAL_CRYPTO,
    SCAN_INTERVAL_FOREX,
    TIMEFRAME,
    VERSION,
)

# Initialize Logging immediately
logger = setup_logging()

# Global instance to hold the bot state
bot_instance = None
_background_loop = None
_loop_thread = None


def _get_persistent_loop():
    """Ensures a background loop is running and returns it."""
    global _background_loop, _loop_thread
    if _background_loop is None:
        _background_loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(target=_start_background_loop, args=(_background_loop,), daemon=True)
        _loop_thread.start()
        logger.info("⚡ Background AsyncIO Loop Started")
    return _background_loop


def _read_recent_logs(limit=20):
    """Reads the last N lines from the log file."""
    log_file = "nexubot.log"
    if not os.path.exists(log_file):
        return []

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # simple filter to remove clutter
            filtered = [l.strip() for l in lines if "geventwebsocket" not in l and "aiohttp" not in l]
            return filtered[-limit:]
    except Exception:
        return []


def _safe_get_result(future, timeout=3.0):
    try:
        return future.result(timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError):
        return None
    except Exception as e:
        logger.error(f"Async Task Error: {e}")
        return None


def _start_background_loop(loop):
    """Runs the asyncio loop forever in a separate thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


class NexubotGUI:
    def __init__(self):
        self.provider = DataProvider()
        self.db = DatabaseManager()
        self.ai_engine = AITradingEngine()

        self.is_running = False
        self.execution_mode = "SIGNAL_ONLY"
        self.system_status = "IDLE"  # IDLE, BACKFILLING, TRAINING

        self.session_start = time.time()
        self.session_id = f"SESSION_{int(time.time())}"
        self.session_stats = {
            "wins": 0,
            "losses": 0,
            "total": 0,
            "pnl": 0.0,
            "start": datetime.now(),
        }
        self.active_signals = []
        self.active_crypto_list = []
        self.active_forex_list = []
        self.monitored_tasks = {}
        self.settings = {}

        self._scanner_task = None

    def _apply_settings_to_engine(self):
        """Pushes settings to the AI Engine instance."""
        self.ai_engine.update_config(self.settings)

    def _calculate_offline_result(self, symbol: str, signal: dict, start_time: float, klines: list):
        """Synchronous CPU-bound calculation logic for offline verification."""
        if not klines:
            return None, 0.0, False

        sl = signal["sl"]
        tp = signal["tp"]
        entry = signal["price"]
        is_long = signal["direction"] == "LONG"
        order_type = signal.get("order_type", "MARKET")
        trade_duration = 14400

        outcome = None
        pnl = 0.0
        filled_offline = order_type == "MARKET"

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

            if filled_offline:
                if is_long:
                    if k["low"] <= sl:
                        outcome = "LOSS (SL Offline)"
                        pnl = -signal["risk_zar"]
                        break
                    if k["high"] >= tp:
                        outcome = "WIN (TP Offline)"
                        pnl = signal["profit_zar"]
                        break
                else:
                    if k["high"] >= sl:
                        outcome = "LOSS (SL Offline)"
                        pnl = -signal["risk_zar"]
                        break
                    if k["low"] <= tp:
                        outcome = "WIN (TP Offline)"
                        pnl = signal["profit_zar"]
                        break

        return outcome, pnl, filled_offline

    async def _check_offline_trades(self):
        """Resumes or closes trades that were active before shutdown."""
        logger.info("🔄 Checking for interrupted trades...")
        active_trades = await self.db.get_active_trades()

        for symbol, signal, start_time in active_trades:
            self.ai_engine.register_active_trade(symbol)

            elapsed = time.time() - start_time
            candles_needed = math.ceil(elapsed / 900) + 5
            klines = await self.provider.fetch_klines(symbol, TIMEFRAME, min(candles_needed, 1000))

            outcome, pnl, filled = self._calculate_offline_result(symbol, signal, start_time, klines)

            # Case 1: Trade finished (TP or SL hit)
            if outcome and "TIMEOUT" not in outcome:
                won = pnl > 0
                logger.info(f"🔔 Offline Result ({symbol}): {outcome} | PnL: R{pnl:.2f}")
                unique_id = f"{symbol}_OFFLINE_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                await self.db.log_trade(
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
                await self.db.delete_active_trade(symbol)

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
                    logger.info(f"🔔 Offline Result ({symbol}): {outcome_str} | PnL: R{pnl:.2f}")

                    unique_id = f"{symbol}_TIMEOUT_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                    await self.db.log_trade(
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
                await self.db.delete_active_trade(symbol)

            # Case 3: Still Active
            else:
                logger.info(f"Resuming {symbol}...")
                self.active_signals.append(signal)
                self.monitored_tasks[symbol] = asyncio.create_task(
                    self.verify_trade_realtime(symbol, signal, resume_start_time=start_time)
                )

    async def _refresh_market_watch_symbols(self):
        """Fetches current Market Watch from Provider."""
        data = await self.provider.get_dynamic_symbols()
        self.active_crypto_list = data.get("crypto", [])
        self.active_forex_list = data.get("forex", [])

        if not self.active_crypto_list and not self.active_forex_list:
            # Fallback if MT5 returns nothing
            self.active_crypto_list = list(FALLBACK_CRYPTO)
            self.active_forex_list = list(FALLBACK_FOREX)

    def _update_signal_status(self, symbol, status):
        """Helper to update status text in UI list"""
        for s in self.active_signals:
            if s["symbol"] == symbol:
                s["status"] = status
                break

    async def force_close_trade(self, symbol):
        """Manually closes an active signal/trade."""
        # Remove from active list
        self.active_signals = [s for s in self.active_signals if s["symbol"] != symbol]
        logger.info(f"🛑 Trade {symbol} Force Closed by User.")
        return True

    async def get_dashboard_data(self):
        """
        Aggregates data for the dashboard:
        1. Live MT5 Balance/Equity
        2. Database Stats (Win Rate, Total PnL)
        3. Chart Data (Equity Curve simulation)
        """
        safe_response = {
            "balance": 0.0,
            "equity": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "wins": 0,
            "losses": 0,
            "chart_labels": [],
            "chart_data": [],
            "mode": self.execution_mode,
            "system_status": self.system_status,
            "recent_trades": [],
            "timestamp": time.time(),
        }

        try:
            # 1. Live Account Info
            acct = await self.provider.get_account_summary()
            if acct:
                safe_response["balance"] = acct.get("balance", 0.0)
                safe_response["equity"] = acct.get("equity", 0.0)

            # 2. Database Stats
            total_pnl = 0.0
            wins = 0
            losses = 0
            chart_labels = []
            chart_balance = []
            recent_trades_data = []

            running_pnl = 0.0

            all_trades = await self.db.get_dashboard_chart_data()

            for t in all_trades:
                total_pnl += t.pnl_zar
                if t.result == 1:
                    wins += 1
                else:
                    losses += 1

                # Chart Data Construction
                running_pnl += t.pnl_zar
                dt_obj = datetime.fromtimestamp(t.timestamp)
                chart_labels.append(dt_obj.strftime("%d %b"))
                chart_balance.append(round(running_pnl, 2))

            for t in reversed(all_trades[-5:]):
                recent_trades_data.append(
                    {
                        "time": datetime.fromtimestamp(t.timestamp).strftime("%H:%M:%S"),
                        "symbol": t.symbol,
                        "signal_type": t.signal_type,
                        "entry": float(t.entry_price),
                        "exit": float(t.exit_price),
                        "size": getattr(t, "size", 0.01),
                        "pnl": float(t.pnl_zar),
                        "result": int(t.result),
                    }
                )

            total_trades = wins + losses
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

            safe_response.update(
                {
                    "total_pnl": total_pnl,
                    "win_rate": win_rate,
                    "wins": wins,
                    "losses": losses,
                    "chart_labels": chart_labels[-20:],
                    "chart_data": chart_balance[-20:],
                    "recent_trades": recent_trades_data,
                }
            )

            return safe_response
        except Exception as e:
            logger.error(f"Dashboard Data Error: {e}")
            return safe_response

    async def get_latency(self):
        """Returns the connection latency (ping) to the broker server."""
        return self.provider.get_ping()

    async def get_signal_page_data(self):
        """Aggregates all data needed for the Signal Page."""
        safe_response = {
            "account": {"balance": 0.0, "equity": 0.0},
            "stats": {
                "lifetime_wr": 0.0,
                "active_count": 0,
                "session_pnl": 0.0,
                "session_wins": 0,
                "session_losses": 0,
                "session_total": 0,
                "time_running": "0:00:00",
            },
            "signals": [],
            "logs": [],
            "mode": self.execution_mode,
        }

        try:
            acct = await self.provider.get_account_summary()
            if acct:
                safe_response["account"] = acct

            lifetime_wr = await self.db.get_total_historical_win_rate()
            elapsed = timedelta(seconds=int(time.time() - self.session_start))

            safe_response["stats"] = {
                "lifetime_wr": lifetime_wr,
                "active_count": len(self.active_signals),
                "session_pnl": self.session_stats["pnl"],
                "session_wins": self.session_stats["wins"],
                "session_losses": self.session_stats["losses"],
                "session_total": self.session_stats["total"],
                "time_running": str(elapsed),
            }
            safe_response["signals"] = self.active_signals
            safe_response["logs"] = _read_recent_logs(30)

            return safe_response
        except Exception as e:
            logger.error(f"Signal Data Error: {e}")
            return safe_response

    async def get_trade_history(self, filters=None):
        """Fetches filtered trade history and global lifetime stats."""
        alltime_trades = await self.db.get_alltime_trade_history(self.provider, filters)
        if not alltime_trades:
            alltime_trades = {}

        stats = {
            "balance": self.ai_engine.user_balance_zar or 0.0,
            "lifetime_wr": 0.0,
            "total_trades": 0,
            "lifetime_pnl": 0.0,
        }

        total_trades = alltime_trades.get("total_trades", 0)
        lifetime_pnl = alltime_trades.get("lifetime_pnl", 0.0)
        total_wins = alltime_trades.get("total_wins", 0)
        lifetime_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

        stats["total_trades"] = total_trades or 0
        stats["lifetime_pnl"] = lifetime_pnl or 0.0
        stats["lifetime_wr"] = lifetime_wr

        table_data = []
        raw_trades = alltime_trades.get("trades", [])
        if raw_trades:
            for t in raw_trades:
                table_data.append(
                    {
                        "time": datetime.fromtimestamp(t.timestamp).strftime("%Y-%m-%d %H:%M"),
                        "symbol": t.symbol,
                        "signal_type": t.signal_type,
                        "entry": float(t.entry_price),
                        "exit": float(t.exit_price),
                        "pnl": float(t.pnl_zar),
                        "result": int(t.result),
                        "confidence": float(t.confidence),
                        "size": getattr(t, "size", 0.01),
                    }
                )

        return {
            "stats": stats,
            "history": table_data,
            "pagination": {
                "current": alltime_trades.get("page", 1),
                "total_pages": (
                    math.ceil(total_trades / alltime_trades.get("limit", 10)) if alltime_trades.get("limit") else 1
                ),
                "total_records": total_trades,
            },
        }

    async def initialize_connection(self, login_id, server, password):
        """
        Attempt to connect to MT5 using credentials from the GUI.
        """
        # --- Prevent Duplicate Connection ---
        if self.provider.connected:
            logger.info("✅ Already connected to MT5.")
            self.is_running = True
            if not self._scanner_task or self._scanner_task.done():
                self._scanner_task = asyncio.create_task(self.scanner_loop())
            return {"success": True, "message": "Already Connected"}

        try:
            mt5_login = int(login_id)
        except ValueError:
            return {"success": False, "message": "Login ID must be a number."}

        # 1. Update Provider Credentials dynamically
        self.provider._login = mt5_login
        self.provider._password = password
        self.provider._server = server

        logger.info(f"🖥️ Connecting to {server}...")

        for attempt in range(2):
            try:
                connected = await asyncio.wait_for(self.provider.initialize(), timeout=30)
                if connected:
                    break
                await asyncio.sleep(1)  # Wait 1s before retry
            except asyncio.TimeoutError:
                if attempt == 1:
                    return {"success": False, "message": "MT5 Launch Timeout"}
            except Exception as e:
                logger.error(f"Connection Error: {e}")

        if self.provider.connected:
            self.is_running = True
            await self.initialize_settings()

            # Save valid credentials to DB
            new_settings = self.settings.copy()
            new_settings.update({"login": login_id, "server": server, "password": password})
            await self.db.save_settings(new_settings)

            self.ai_engine.set_context(500.0, self.db)
            await self._check_offline_trades()

            # 3. Double-Loop Prevention
            if not self._scanner_task or self._scanner_task.done():
                self._scanner_task = asyncio.create_task(self.scanner_loop())

            return {"success": True, "message": "Connection Established"}
        else:
            return {"success": False, "message": "MT5 Connection Failed. Check Credentials."}

    async def initialize_settings(self):
        await self.db.init_database()
        await self.db.cleanup_db()
        db_settings = await self.db.get_settings()

        if not db_settings or not db_settings["login"]:
            logger.warning("⚠️ No Login Details Found. Waiting for user input via GUI...")
            return

        default_settings = {
            "login": "",
            "server": "MetaQuotes-Demo",
            "password": "",
            "lot_size": 0.10,
            "risk": DEFAULT_RISK_PCT,
            "confidence": DEFAULT_MIN_CONFIDENCE,
            "high_vol": False,
        }

        if db_settings:
            default_settings.update({k: v for k, v in db_settings.items() if v is not None})
        self.settings = default_settings
        self._apply_settings_to_engine()

    async def process_batch(self, symbols: list):
        """Processes a list of symbols concurrently."""
        if self.system_status != "IDLE":
            return

        signals_found = 0

        for sym in symbols:
            if not self.is_running:
                break

            # Limit signals per batch
            if signals_found >= MAX_SIGNALS_PER_SCAN:
                break

            # Skip if already active in UI
            if any(s.get("symbol") == sym for s in self.active_signals):
                continue

            klines = await self.provider.fetch_klines(sym, TIMEFRAME, CANDLE_LIMIT)
            if klines:
                signal = await self.ai_engine.analyze_market(sym, klines, self.provider)
                if signal:
                    signal.setdefault("symbol", sym)
                    signal["detected_at"] = time.time()
                    signal["status"] = "PENDING"  # Initial status

                    is_shadow = signal.get("is_shadow", False)

                    if not is_shadow:
                        signals_found += 1
                        self.active_signals.insert(0, signal)

                        if self.execution_mode == "FULL_AUTO":
                            placed = await self.provider.execute_trade_on_mt5(signal)
                            if placed:
                                signal["status"] = "PLACED"
                        self.monitored_tasks[sym] = asyncio.create_task(self.verify_trade_realtime(sym, signal))
                    else:
                        asyncio.create_task(self.verify_trade_realtime(sym, signal))

    async def restart_system(self):
        """Stops scanner, re-inits DB/Provider with new settings."""
        logger.info("♻️ Restarting System...")
        self.is_running = False
        if self._scanner_task:
            self._scanner_task.cancel()

        await asyncio.sleep(1)
        self._apply_settings_to_engine()
        self.ai_engine.nn_brain = asyncio.to_thread(lambda: self.ai_engine.nn_brain.__init__(auto_load=True))

        await self.initialize_connection(self.settings["login"], self.settings["server"], self.settings["password"])
        return True

    async def scanner_loop(self):
        """Background loop to scan markets."""
        logger.info("🚀 Scanner Loop Initiated...")

        active_crypto = []
        active_forex = []
        last_crypto = 0
        last_forex = 0
        last_sort_time = 0

        await self._refresh_market_watch_symbols()

        while self.is_running:
            try:
                # If training, pause scanning
                if self.system_status != "IDLE":
                    await asyncio.sleep(2)
                    continue

                # Fail-safe: Check if MT5 is actually alive
                if not self.provider.connected:
                    logger.warning("⚠️ MT5 Disconnected. Pausing scanner...")
                    await asyncio.sleep(5)
                    continue

                # 1. Update Balance context
                acct = await self.provider.get_account_summary()
                if acct:
                    self.ai_engine.set_context(acct["balance"], self.db)

                now = time.time()

                # 2. Sort Pairs every 15 mins (Logic from console.py)
                if now - last_sort_time > 900:
                    logger.info("📊 Re-ranking pairs by volatility...")
                    await self._refresh_market_watch_symbols()

                    active_crypto = await self.sort_pairs(active_crypto)
                    active_forex = await self.sort_pairs(active_forex)
                    last_sort_time = now

                # 3. Batch Process (Parallel Scanning)
                tasks = []

                if now - last_crypto > SCAN_INTERVAL_CRYPTO:
                    tasks.append(self.process_batch(self.active_crypto_list[:5]))
                    last_crypto = now

                if now - last_forex > SCAN_INTERVAL_FOREX:
                    tasks.append(self.process_batch(self.active_forex_list[:10]))
                    last_forex = now

                if tasks:
                    await asyncio.gather(*tasks)

                await asyncio.sleep(1)

            except RuntimeError as e:
                if "shutdown" in str(e) or "closed" in str(e):
                    logger.info("🛑 Scanner loop stopping due to shutdown.")
                    break
                logger.error(f"Scanner Runtime Error: {e}")
            except Exception as e:
                logger.error(f"Scanner Loop Error: {e}")
                await asyncio.sleep(5)

    def set_execution_mode(self, mode_str):
        """Sets the bot state (SIGNAL_ONLY vs FULL_AUTO)"""
        self.execution_mode = mode_str
        print(f"⚙️ Execution Mode Changed: {self.execution_mode}")
        return True

    async def sort_pairs(self, symbols: list) -> list:
        """Fetches 100 candles for all pairs to rank them by volatility (Ported from console.py)."""
        data_map = {}
        for sym in symbols:
            # Quick fetch
            k = await self.provider.fetch_klines(sym, TIMEFRAME, 100)
            if k:
                df = self.ai_engine.prepare_data(k, heavy=False)
                if df is not None:
                    data_map[sym] = df

        ranked = self.ai_engine.rank_symbols_by_volatility(symbols, data_map)
        result = ranked + [s for s in symbols if s not in ranked]
        return result

    async def stop_session(self):
        """Stops the bot operations & closes resources."""
        logger.info("🛑 Stopping Engine & Saving Session...")
        self.is_running = False

        if self._scanner_task:
            self._scanner_task.cancel()

        # Cancel monitoring tasks
        for task in self.monitored_tasks.values():
            task.cancel()
        self.monitored_tasks.clear()
        self.active_signals.clear()

        await self.db.log_session(self.session_id, self.session_stats["start"].timestamp(), self.session_stats)
        await self.db.close()
        await self.provider.shutdown()
        return True

    async def trigger_manual_training(self, symbol=None):
        """
        Orchestrates the entire training lifecycle:
        1. Backfill Data (Full or Partial)
        2. Train Model
        3. Reload Engine
        """
        if self.system_status != "IDLE":
            logger.warning("⚠️ Training already in progress.")
            return False

        try:
            self.system_status = "BACKFILLING"
            logger.info(f"🔄 Starting Manual Training Cycle. Target: {symbol if symbol else 'ALL'}")

            # 1. Backfill
            target = [symbol] if symbol else None
            await backfill_data(target_symbols=target)

            # 2. Train
            self.system_status = "TRAINING"
            logger.info("🧠 Backfill Complete. Starting Neural Training...")

            # Run training in thread to avoid blocking main loop
            await asyncio.to_thread(ModelTrainer.train_if_needed, force=True)

            # 3. Restart to apply
            logger.info("✅ Training Complete. Reloading Systems...")
            await self.restart_system()

            self.system_status = "IDLE"
            return True
        except Exception as e:
            logger.error(f"Training Cycle Failed: {e}")
            self.system_status = "IDLE"
            return False

    async def verify_trade_realtime(self, symbol: str, signal: dict, resume_start_time=None):
        """
        Monitors price and Logs Data for ML.
        Updates self.active_signals state for UI.
        """
        is_shadow = signal.get("is_shadow", False)
        if not is_shadow:
            logger.info(f"👀 Monitoring trade {symbol} for outcome...")
            await self.db.save_active_trade(symbol, signal)

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
            while (time.time() - start_time) < duration and self.is_running:
                tick = await self.provider.get_current_tick(symbol)
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
                    break
                elif hit_tp:
                    outcome = "WIN (TP Hit)"
                    points_won = (tp - entry) / point if is_long else (entry - tp) / point
                    final_pnl = points_won * tick_value * lot_size
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
                await self.db.delete_active_trade(symbol)
                self.active_signals = [s for s in self.active_signals if s["symbol"] != symbol]

                if outcome == "TIMEOUT":
                    tick = await self.provider.get_current_tick(symbol)
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
                        self.session_stats["wins"] += 1
                    else:
                        self.session_stats["losses"] += 1
                    self.session_stats["total"] += 1
                    self.session_stats["pnl"] += final_pnl

                    unique_id = f"{symbol}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                    await self.db.log_trade(
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
                self.ai_engine.record_trade_outcome(
                    symbol, won, final_pnl, max_favorable_dist / atr if atr else 0, is_shadow
                )

        except Exception as e:
            logger.error(f"Error verifying {symbol}: {e}")
        finally:
            if not is_shadow:
                asyncio.create_task(self.db.delete_active_trade(symbol))


# --- Expose Functions to JavaScript ---
@eel.expose
def attempt_login(login_id, server, password):
    """Called from login.html when user clicks 'Initialize Connection'"""
    global bot_instance
    if not bot_instance:
        bot_instance = NexubotGUI()

    loop = _get_persistent_loop()
    future = asyncio.run_coroutine_threadsafe(bot_instance.initialize_connection(login_id, server, password), loop)
    return _safe_get_result(future, timeout=45.0)


@eel.expose
def close_app():
    """Cleanup when window closes"""
    print("Saving session and closing...")
    sys.exit(0)


@eel.expose
def fetch_dashboard_update():
    """Called by JS interval to get live data"""
    global bot_instance
    default_data = {
        "balance": 0.0,
        "equity": 0.0,
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "wins": 0,
        "losses": 0,
        "recent_trades": [],
        "chart_labels": [],
        "chart_data": [],
        "mode": "SIGNAL_ONLY",
        "system_status": "IDLE",
    }
    if not bot_instance:
        return default_data

    loop = _get_persistent_loop()

    try:
        future = asyncio.run_coroutine_threadsafe(bot_instance.get_dashboard_data(), loop)
        res = _safe_get_result(future, timeout=3.0)

        final_data = res if res else default_data.copy()

        if bot_instance and bot_instance.provider:
            final_data["latency"] = bot_instance.provider.get_ping()

        return final_data
    except Exception as e:
        logger.error(f"Dashboard Update Failed: {e}")
        return default_data


@eel.expose
def fetch_signal_updates():
    """Polled by signal.html"""
    global bot_instance
    default_data = {
        "account": {"balance": 0, "equity": 0},
        "stats": {
            "lifetime_wr": 0,
            "active_count": 0,
            "session_pnl": 0,
            "session_wins": 0,
            "session_losses": 0,
            "session_total": 0,
            "time_running": "--",
        },
        "signals": [],
        "logs": [],
        "mode": "SIGNAL_ONLY",
    }
    if not bot_instance:
        return default_data

    loop = _get_persistent_loop()

    try:
        future = asyncio.run_coroutine_threadsafe(bot_instance.get_signal_page_data(), loop)
        res = _safe_get_result(future, timeout=3.0)

        final_data = res if res else default_data.copy()

        if bot_instance and bot_instance.provider:
            final_data["latency"] = bot_instance.provider.get_ping()

        return final_data
    except Exception as e:
        logger.error(f"Signal Update Failed: {e}")
        return default_data


@eel.expose
def fetch_trade_history(filters=None):
    """
    Fetches filtered trade history and global lifetime stats.
    """
    global bot_instance
    default_res = {
        "stats": {"balance": 0.0, "lifetime_wr": 0.0, "total_trades": 0, "lifetime_pnl": 0.0},
        "history": [],
        "pagination": {"current": 1, "total_pages": 1, "total_records": 0},
    }
    if not bot_instance:
        return default_res

    loop = _get_persistent_loop()

    try:
        future = asyncio.run_coroutine_threadsafe(bot_instance.get_trade_history(filters), loop)
        res = _safe_get_result(future, timeout=3.0)

        final_data = res if res else default_res.copy()

        if bot_instance and bot_instance.provider:
            final_data["latency"] = bot_instance.provider.get_ping()

        return final_data
    except Exception as e:
        logger.error(f"History Fetch Error: {e}")
        return default_res


@eel.expose
def force_close(symbol):
    global bot_instance
    if bot_instance:
        loop = _get_persistent_loop()
        asyncio.run_coroutine_threadsafe(bot_instance.force_close_trade(symbol), loop)


@eel.expose
def get_user_settings():
    """Returns current settings to populate the Settings Page."""
    global bot_instance
    if not bot_instance:
        bot_instance = NexubotGUI()
        asyncio.run(bot_instance.initialize_settings())

    bot_instance.settings["latency"] = bot_instance.provider.get_ping()

    # Inject Neural Logic Data (Dynamic)
    bot_instance.settings["neural_meta"] = {
        "model": f"Transformer-XL {VERSION}",
        "epochs": "50,000",
        "bias": "Conservative" if bot_instance.ai_engine.min_confidence > 80 else "Balanced",
    }

    return bot_instance.settings


@eel.expose
def save_settings(data):
    """Saves settings and restarts the engine."""
    global bot_instance
    if not bot_instance:
        return False

    loop = _get_persistent_loop()

    async def _save_and_restart():
        await bot_instance.db.save_settings(data)
        await bot_instance.restart_system()
        return True

    future = asyncio.run_coroutine_threadsafe(_save_and_restart(), loop)
    return _safe_get_result(future, timeout=10.0)


@eel.expose
def set_mode(is_auto):
    """Called by the toggle switch"""
    global bot_instance
    if bot_instance:
        mode = "FULL_AUTO" if is_auto else "SIGNAL_ONLY"
        bot_instance.set_execution_mode(mode)


@eel.expose
def shutdown_bot():
    """Stops the running bot instance gracefully."""
    global bot_instance
    if bot_instance:
        loop = _get_persistent_loop()
        # Clean shutdown of DB and MT5
        future = asyncio.run_coroutine_threadsafe(bot_instance.stop_session(), loop)
        _safe_get_result(future, timeout=5.0)

    # Kill the process
    os._exit(0)


@eel.expose
def stop_and_reset():
    """Stops the bot logic and prepares for redirect to login."""
    global bot_instance
    if bot_instance:
        loop = _get_persistent_loop()
        future = asyncio.run_coroutine_threadsafe(bot_instance.stop_session(), loop)
        return _safe_get_result(future, timeout=3.0)
    return True


@eel.expose
def trigger_training(symbol=None):
    """Triggered from Frontend to start backfill/training."""
    global bot_instance
    if not bot_instance:
        return False
    loop = _get_persistent_loop()
    future = asyncio.run_coroutine_threadsafe(bot_instance.trigger_manual_training(symbol), loop)
    return True
