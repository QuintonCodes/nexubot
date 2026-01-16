import asyncio
import json
import eel
import logging
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from sqlalchemy import and_, select
from src.data.provider import DataProvider
from src.database.manager import DatabaseManager, TradeResult
from src.engine.ai_engine import AITradingEngine
from src.config import ALL_SYMBOLS, TIMEFRAME, CANDLE_LIMIT, SCAN_INTERVAL_CRYPTO, CRYPTO_SYMBOLS, FOREX_SYMBOLS

# Global instance to hold the bot state
bot_instance = None
logger = logging.getLogger(__name__)

SETTINGS_FILE = "user_preferences.json"

# --- BACKGROUND LOOP MANAGEMENT ---
_background_loop = None
_loop_thread = None


def _start_background_loop(loop):
    """Runs the asyncio loop forever in a separate thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


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
            filtered = [l.strip() for l in lines if "aiohttp" not in l and "asyncio" not in l]
            return filtered[-limit:]
    except Exception:
        return []


class NexubotGUI:
    def __init__(self):
        self.provider = DataProvider()
        self.db = DatabaseManager()
        self.ai_engine = AITradingEngine()
        self.is_running = False
        self.execution_mode = "SIGNAL_ONLY"  # Default state

        # Session Tracking
        self.session_start = time.time()
        self.session_stats = {"wins": 0, "losses": 0, "total": 0, "pnl": 0.0}
        self.active_signals = []
        self.monitored_tasks = {}

        # Load Settings on Init
        self.settings = self._load_settings()
        self._apply_settings_to_engine()

    def _load_settings(self):
        """Loads user preferences from JSON."""
        defaults = {
            "login": "",
            "server": "MetaQuotes-Demo",
            "password": "",  # Stored (ideally encrypted, but plain for now per request)
            "lot_size": 0.10,
            "risk": 2.0,
            "confidence": 75,
            "high_vol": False,
        }
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                    defaults.update(data)
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")
        return defaults

    def _save_settings_to_file(self, data):
        """Writes settings to JSON."""
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=4)
            self.settings = data
            return True
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False

    def _apply_settings_to_engine(self):
        """Pushes settings to the AI Engine instance."""
        self.ai_engine.update_config(self.settings)

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

        running_pnl = 0.0

        if self.db.engine:
            async with self.db.async_session() as session:
                # Get all trades ordered by time
                stmt = select(TradeResult).order_by(TradeResult.timestamp.asc())
                result = await session.execute(stmt)
                all_trades = result.scalars().all()

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

                # Fetch recent 5 for table (descending order)
                recent_trades_data = []
                # We can just slice the all_trades list reversed since we already fetched it
                # or do a separate query. Slicing is faster for small datasets.
                for t in reversed(all_trades[-5:]):
                    recent_trades_data.append(
                        {
                            "time": datetime.fromtimestamp(t.timestamp).strftime("%H:%M:%S"),
                            "symbol": t.symbol,
                            "signal_type": t.signal_type,  # BUY/SELL
                            "entry": float(t.entry_price),
                            "exit": float(t.exit_price),
                            "size": 0.01,  # Placeholder if lot size isn't in TradeResult model, or fetch if available
                            "pnl": float(t.pnl_zar),
                            "result": int(t.result),  # 1 or 0
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
            "chart_labels": chart_labels[-20:],  # Last 20 trades for clean chart
            "chart_data": chart_balance[-20:],
            "mode": self.execution_mode,
            "recent_trades": recent_trades_data,
            "timestamp": time.time(),
        }

    async def initialize_connection(self, login_id, server, password):
        """
        Attempt to connect to MT5 using credentials from the GUI.
        """
        # Update settings with login info on successful attempt
        self.settings.update({"login": login_id, "server": server, "password": password})
        self._save_settings_to_file(self.settings)

        try:
            mt5_login = int(login_id)
        except ValueError:
            return {"success": False, "message": "Login ID must be a number."}

        # 1. Update Provider Credentials dynamically
        self.provider._login = mt5_login
        self.provider._password = password
        self.provider._server = server

        # 2. Attempt Connection
        print(f"🖥️ GUI Request: Connecting to {server}...")
        try:
            connected = await self.provider.initialize()
        except Exception as e:
            logger.error(f"MT5 Init Error: {e}")
            return {"success": False, "message": f"Driver Error: {str(e)}"}

        if connected:
            self.is_running = True
            await self.db.init_database()
            self.ai_engine.set_context(500.0, self.db)
            self._apply_settings_to_engine()  # Ensure engine has latest config

            # START THE SCANNER LOOP
            asyncio.create_task(self.scanner_loop())
            return {"success": True, "message": "Connection Established"}
        else:
            return {"success": False, "message": "MT5 Connection Failed. Check Credentials."}

    async def scanner_loop(self):
        """Background loop to scan markets."""
        print("🚀 Scanner Loop Initiated...")
        while self.is_running:
            # 1. Update Balance context
            acct = await self.provider.get_account_summary()
            if acct:
                self.ai_engine.set_context(acct["balance"], self.db)

            # 2. Scan Logic (Simplified for GUI responsiveness)
            for sym in ALL_SYMBOLS:
                if not self.is_running:
                    break

                # Skip if already active
                if any((isinstance(s, dict) and s.get("symbol") == sym) for s in self.active_signals):
                    continue

                klines = await self.provider.fetch_klines(sym, TIMEFRAME, CANDLE_LIMIT)
                if klines:
                    signal = await self.ai_engine.analyze_market(sym, klines, self.provider)
                    if signal:
                        # Ensure the signal dict always has the symbol key to avoid later KeyErrors
                        if not isinstance(signal, dict):
                            signal = {"symbol": sym, "price": None}
                        else:
                            signal.setdefault("symbol", sym)

                        # Add metadata for UI
                        signal["detected_at"] = time.time()
                        signal["formatted_time"] = datetime.now().strftime("%H:%M:%S")

                        if "neural_info" not in signal:
                            signal["neural_info"] = {
                                "prediction": "CONTINUATION" if signal["confidence"] > 75 else "REVERSION",
                                "sentiment": "GREED" if signal["direction"] == "LONG" else "FEAR",
                                "volatility": "EXPANSION",
                            }

                        self.active_signals.insert(0, signal)  # Prepend

                        # Auto-Monitor Task
                        self.monitored_tasks[sym] = asyncio.create_task(self.monitor_trade(sym, signal))

            await asyncio.sleep(SCAN_INTERVAL_CRYPTO)

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

    def set_execution_mode(self, mode_str):
        """Sets the bot state (SIGNAL_ONLY vs FULL_AUTO)"""
        self.execution_mode = mode_str
        print(f"⚙️ Execution Mode Changed: {self.execution_mode}")
        return True

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

    async def force_close_trade(self, symbol):
        """Manually closes an active signal/trade."""
        # Remove from active list
        self.active_signals = [s for s in self.active_signals if s["symbol"] != symbol]
        logger.info(f"🛑 Trade {symbol} Force Closed by User.")
        return True

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


# --- Expose Functions to JavaScript ---


@eel.expose
def attempt_login(login_id, server, password):
    """Called from login.html when user clicks 'Initialize Connection'"""
    global bot_instance

    if not bot_instance:
        bot_instance = NexubotGUI()

    # 2. Get Persistent Loop
    loop = _get_persistent_loop()

    # 3. Dispatch to background thread and wait for result safely
    future = asyncio.run_coroutine_threadsafe(bot_instance.initialize_connection(login_id, server, password), loop)
    return future.result()


@eel.expose
def close_app():
    """Cleanup when window closes"""
    # Add any specific cleanup here if needed
    print("Saving session and closing...")
    sys.exit(0)


@eel.expose
def fetch_dashboard_update():
    """Called by JS interval to get live data"""
    global bot_instance
    if not bot_instance:
        return {}

    loop = _get_persistent_loop()

    # Dispatch safely without closing the loop
    try:
        future = asyncio.run_coroutine_threadsafe(bot_instance.get_dashboard_data(), loop)
        return future.result()
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
        return future.result()
    except Exception:
        return {}


@eel.expose
def fetch_trade_history(filters=None):
    """
    Fetches filtered trade history and global lifetime stats.
    """
    global bot_instance
    if not bot_instance:
        return {}

    db = bot_instance.db
    if not db.engine:
        return {}

    loop = _get_persistent_loop()

    async def _query_db():
        async with db.async_session() as session:
            # 1. Fetch Global Stats (Lifetime)
            stmt_all = select(TradeResult)
            result_all = await session.execute(stmt_all)
            all_trades = result_all.scalars().all()

            total_count = len(all_trades)
            lifetime_pnl = sum(t.pnl_zar for t in all_trades)
            wins = sum(1 for t in all_trades if t.result == 1)
            lifetime_wr = (wins / total_count * 100) if total_count > 0 else 0.0

            # 2. Build Filtered Query
            query = select(TradeResult).order_by(TradeResult.timestamp.desc())

            conditions = []

            if filters:
                # Date Filter
                if filters.get("startDate"):
                    try:
                        start_ts = datetime.strptime(filters["startDate"], "%Y-%m-%d").timestamp()
                        conditions.append(TradeResult.timestamp >= start_ts)
                    except:
                        pass

                if filters.get("endDate"):
                    try:
                        end_ts = datetime.strptime(filters["endDate"], "%Y-%m-%d").timestamp() + 86400  # End of day
                        conditions.append(TradeResult.timestamp <= end_ts)
                    except:
                        pass

                # Outcome Filter
                outcome = filters.get("outcome", "ALL")
                if outcome == "WINS":
                    conditions.append(TradeResult.result == 1)
                elif outcome == "LOSSES":
                    conditions.append(TradeResult.result == 0)

                # Asset Class Filter
                assets = filters.get("assets", [])
                if assets and "ALL" not in assets:
                    symbol_conditions = []
                    if "CRYPTO" in assets:
                        symbol_conditions.extend(CRYPTO_SYMBOLS)
                    if "FOREX" in assets:
                        symbol_conditions.extend(FOREX_SYMBOLS)
                    # For INDICES or COMMODITIES, you would add logic here based on your symbol naming convention
                    # e.g. "NAS100", "US30"

                    if symbol_conditions:
                        conditions.append(TradeResult.symbol.in_(symbol_conditions))

            if conditions:
                query = query.where(and_(*conditions))

            # Execute Filtered Query
            result_filtered = await session.execute(query)
            filtered_trades = result_filtered.scalars().all()

            # Format for UI
            table_data = []
            for t in filtered_trades:
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
                        # Size is mock for now if not in DB, or assume fixed 0.01 if not stored
                        "size": getattr(t, "size", 0.01),
                    }
                )

            return {
                "stats": {
                    "balance": bot_instance.ai_engine.user_balance_zar,  # Live balance
                    "lifetime_wr": lifetime_wr,
                    "total_trades": total_count,
                    "lifetime_pnl": lifetime_pnl,
                },
                "history": table_data,
            }

    future = asyncio.run_coroutine_threadsafe(_query_db(), loop)
    return future.result()


@eel.expose
def force_close(symbol):
    global bot_instance
    if bot_instance:
        loop = _get_persistent_loop()
        asyncio.run_coroutine_threadsafe(bot_instance.force_close_trade(symbol), loop)


@eel.expose
def set_mode(is_auto):
    """Called by the toggle switch"""
    global bot_instance
    if bot_instance:
        mode = "FULL_AUTO" if is_auto else "SIGNAL_ONLY"
        bot_instance.set_execution_mode(mode)


@eel.expose
def get_user_settings():
    """Returns current settings to populate the Settings Page."""
    global bot_instance
    if not bot_instance:
        bot_instance = NexubotGUI()
    return bot_instance.settings


@eel.expose
def save_settings(data):
    """Saves settings and restarts the engine."""
    global bot_instance
    if not bot_instance:
        return False

    # Save to file
    if bot_instance._save_settings_to_file(data):
        # Trigger Restart in background
        loop = _get_persistent_loop()
        asyncio.run_coroutine_threadsafe(bot_instance.restart_system(), loop)
        return True
    return False
