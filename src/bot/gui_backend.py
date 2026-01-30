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
from src.utils.logger import setup_logging
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
        # Log warning but don't crash
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
        # 1. Live Account Info
        acct = await self.provider.get_account_summary()

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

        return {
            "balance": acct["balance"],
            "equity": acct["equity"],
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "chart_labels": chart_labels[-20:],
            "chart_data": chart_balance[-20:],
            "mode": self.execution_mode,
            "recent_trades": recent_trades_data,
            "timestamp": time.time(),
        }

    async def get_latency(self):
        """Returns the connection latency (ping) to the broker server."""
        return self.provider.get_ping()

    async def get_signal_page_data(self):
        """Aggregates all data needed for the Signal Page."""
        acct = await self.provider.get_account_summary()
        # Calculate Lifetime WR
        lifetime_wr = await self.db.get_total_historical_win_rate()
        # Session Time
        elapsed = timedelta(seconds=int(time.time() - self.session_start))

        return {
            "account": {"balance": acct.get("balance", 0.0), "equity": acct.get("equity", 0.0)},
            "stats": {
                "lifetime_wr": lifetime_wr,
                "active_count": len(self.active_signals),
                "session_pnl": self.session_stats["pnl"],
                "session_wins": self.session_stats["wins"],
                "session_losses": self.session_stats["losses"],
                "session_total": self.session_stats["total"],
                "time_running": str(elapsed),
            },
            "signals": self.active_signals,
            "logs": _read_recent_logs(30),
            "mode": self.execution_mode,
        }

    async def get_trade_history(self, filters=None):
        """Fetches filtered trade history and global lifetime stats."""
        alltime_trades = self.db.get_alltime_trade_history(self.provider, filters)

        total_trades = alltime_trades["total_trades"] or 0
        lifetime_pnl = alltime_trades["lifetime_pnl"] or 0.0
        total_wins = alltime_trades["total_wins"] or 0
        lifetime_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

        table_data = []
        for t in alltime_trades["trades"]:
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
            "stats": {
                "balance": self.ai_engine.user_balance_zar,
                "lifetime_wr": lifetime_wr,
                "total_trades": total_trades or 0,
                "lifetime_pnl": lifetime_pnl or 0,
            },
            "history": table_data,
            "pagination": {
                "current": alltime_trades["page"],
                "total_pages": math.ceil(alltime_trades["total_records"] / alltime_trades["limit"]),
                "total_records": alltime_trades["total_records"],
            },
        }

    async def initialize_connection(self, login_id, server, password):
        """
        Attempt to connect to MT5 using credentials from the GUI.
        """
        # --- Prevent Duplicate Connection ---
        if self.is_running and self.provider.connected:
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
        try:
            connected = await asyncio.wait_for(self.provider.initialize(), timeout=30)
        except asyncio.TimeoutError:
            return {"success": False, "message": "MT5 Launch Timeout"}
        except Exception as e:
            return {"success": False, "message": f"Driver Error: {str(e)}"}

        if connected:
            self.is_running = True
            await self.initialize_settings()

            # Save valid credentials to DB
            new_settings = self.settings.copy()
            new_settings.update({"login": login_id, "server": server, "password": password})
            await self.db.save_settings(new_settings)

            self.ai_engine.set_context(500.0, self.db)

            # Check Offline/Interrupted Trades
            await self._check_offline_trades()

            # START THE SCANNER LOOP
            asyncio.create_task(self.scanner_loop())
            return {"success": True, "message": "Connection Established"}
        else:
            return {"success": False, "message": "MT5 Connection Failed. Check Credentials."}

    async def initialize_settings(self):
        """Load settings from DB or create defaults."""
        await self.db.init_database()
        # Cleanup database
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

        # Merge DB settings into defaults
        default_settings.update({k: v for k, v in db_settings.items() if v is not None})
        self.settings = default_settings
        self._apply_settings_to_engine()

    async def monitor_trade(self, symbol, signal):
        """
        Simplified monitoring to track outcome active signals.
        Removes signal from UI when closed.
        """
        # Duration 4 hours
        end_time = time.time() + 14400
        entry = signal["price"]
        sl = signal["sl"]
        tp = signal["tp"]
        is_long = signal["direction"] == "LONG"

        while time.time() < end_time and self.is_running:
            tick = await self.provider.get_current_tick(symbol)
            if tick:
                price = tick.bid if is_long else tick.ask

                # Check Outcome
                hit_sl = price <= sl if is_long else price >= sl
                hit_tp = price >= tp if is_long else price <= tp

                if hit_sl or hit_tp:
                    pnl = signal["risk_zar"] * -1 if hit_sl else signal["profit_zar"]

                    # Update Session Stats
                    self.session_stats["total"] += 1
                    if hit_tp:
                        self.session_stats["wins"] += 1
                    else:
                        self.session_stats["losses"] += 1
                    self.session_stats["pnl"] += pnl

                    # Log to DB
                    await self.db.log_trade(
                        {
                            "id": f"{symbol}_{int(time.time())}",
                            "symbol": symbol,
                            "signal": signal["signal"],
                            "confidence": signal["confidence"],
                            "entry": entry,
                            "exit": sl if hit_sl else tp,
                            "won": hit_tp,
                            "pnl": pnl,
                            "strategy": signal["strategy"],
                        }
                    )

                    # Remove from UI list
                    self.active_signals = [s for s in self.active_signals if s["symbol"] != symbol]
                    break

            await asyncio.sleep(1)

    async def process_batch(self, symbols: list):
        """Processes a list of symbols concurrently."""
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
                        # Add to UI
                        signals_found += 1
                        self.active_signals.insert(0, signal)

                        if self.execution_mode == "FULL_AUTO":
                            placed = await self.provider.execute_trade_on_mt5(signal)
                            if placed:
                                signal["status"] = "PLACED"
                        self.monitored_tasks[sym] = asyncio.create_task(self.verify_trade_realtime(sym, signal))
                    else:
                        # Shadow tracking (no UI)
                        asyncio.create_task(self.verify_trade_realtime(sym, signal))

    async def restart_system(self):
        """Stops scanner, re-inits DB/Provider with new settings."""
        logger.info("♻️ Restarting System...")
        self.is_running = False
        await asyncio.sleep(1)  # Let loops die

        # Apply settings
        self._apply_settings_to_engine()

        # Re-login (using saved settings)
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

        if not self.active_crypto_list and not self.active_forex_list:
            logger.warning("⚠️ Market Watch empty. Using Fallback Symbols.")
            self.active_crypto_list = list(FALLBACK_CRYPTO)
            self.active_forex_list = list(FALLBACK_FOREX)

        while self.is_running:
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

            # 3. Batch Process (Paralle Scanning)
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
        """Stops the bot operations but keeps the app open (for logout)."""
        logger.info("🛑 Stopping Engine & Logging Out...")
        self.is_running = False

        # Cancel monitoring tasks
        for task in self.monitored_tasks.values():
            task.cancel()
        self.monitored_tasks.clear()
        self.active_signals.clear()

        await self.db.log_session(self.session_id, self.session_stats["start"].timestamp(), self.session_stats)
        await self.db.close()

        # Close MT5 connection
        await self.provider.shutdown()
        return True

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
        won = final_pnl > 0

        try:
            while (time.time() - start_time) < duration and self.is_running:
                # Get Tick for Spread Logic
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
                # Determine Exit Prices
                curr_price = current_bid if is_long else current_ask

                # Check SL/TP
                hit_sl = curr_price <= sl if is_long else curr_price >= sl
                hit_tp = curr_price >= tp if is_long else curr_price <= tp

                if hit_sl:
                    outcome = "LOSS (SL Hit)"
                    # Accurate PnL Calc based on points moved
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
                        outcome = "WIN (Floating)" if final_pnl > 0 else "LOSS (Floating)"

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
                    self.session_stats["pnl_zar"] += final_pnl

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


# --- Expose Functions to JavaScript ---
@eel.expose
def attempt_login(login_id, server, password):
    """Called from login.html when user clicks 'Initialize Connection'"""
    global bot_instance

    if not bot_instance:
        bot_instance = NexubotGUI()

    loop = _get_persistent_loop()

    future = asyncio.run_coroutine_threadsafe(bot_instance.initialize_connection(login_id, server, password), loop)
    return _safe_get_result(future, timeout=10.0)


@eel.expose
def close_app():
    """Cleanup when window closes"""
    print("Saving session and closing...")
    sys.exit(0)


@eel.expose
def fetch_dashboard_update():
    """Called by JS interval to get live data"""
    global bot_instance
    if not bot_instance:
        return {}

    loop = _get_persistent_loop()

    try:
        future = asyncio.run_coroutine_threadsafe(bot_instance.get_dashboard_data(), loop)
        res = _safe_get_result(future, timeout=3.0) or {}

        if bot_instance and bot_instance.provider:
            res["latency"] = bot_instance.provider.get_ping()
        return res
    except Exception as e:
        logger.error(f"Dashboard Update Failed: {e}")
        return {}


@eel.expose
def fetch_signal_updates():
    """Polled by signal.html"""
    global bot_instance
    if not bot_instance:
        return {}

    loop = _get_persistent_loop()

    try:
        future = asyncio.run_coroutine_threadsafe(bot_instance.get_signal_page_data(), loop)
        res = _safe_get_result(future, timeout=3.0) or {}

        if bot_instance and bot_instance.provider:
            res["latency"] = bot_instance.provider.get_ping()
        return res
    except Exception as e:
        logger.error(f"Signal Update Failed: {e}")
        return {}


@eel.expose
def fetch_trade_history(filters=None):
    """
    Fetches filtered trade history and global lifetime stats.
    """
    global bot_instance
    if not bot_instance:
        return {}

    loop = _get_persistent_loop()

    try:
        future = asyncio.run_coroutine_threadsafe(bot_instance.get_trade_history(filters), loop)
        res = _safe_get_result(future, timeout=3.0) or {}

        if bot_instance and bot_instance.provider:
            res["latency"] = bot_instance.provider.get_ping()
        return res
    except Exception as e:
        logger.error(f"History Fetch Error: {e}")
        return {}


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

    asyncio.run_coroutine_threadsafe(_save_and_restart(), loop)
    return True


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
        bot_instance.is_running = False
        if bot_instance.provider:
            pass


@eel.expose
def stop_and_reset():
    """Stops the bot logic and prepares for redirect to login."""
    global bot_instance
    if bot_instance:
        loop = _get_persistent_loop()
        future = asyncio.run_coroutine_threadsafe(bot_instance.stop_session(), loop)
        return _safe_get_result(future, timeout=3.0)
    return True
