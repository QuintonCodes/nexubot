import asyncio
import time
from datetime import datetime

from src.data.provider import DataProvider
from src.database.manager import DatabaseManager
from src.engine.ai_engine import AITradingEngine
from src.utils.backfill import backfill_data
from src.utils.logger import setup_logging
from src.utils.concurrency import get_persistent_loop
from src.core.services.application_data import ApplicationData
from src.core.services.market_scanner import MarketScanner
from src.core.services.offline_manager import OfflineManager
from src.core.services.trade_monitor import TradeMonitor
from src.config import DEFAULT_MIN_CONFIDENCE, DEFAULT_RISK_PCT, ConfigManager

# Initialize Logging immediately
logger = setup_logging()
bot_instance = None


class NexubotEngine:
    def __init__(self):
        self.provider = DataProvider()
        self.db = DatabaseManager()
        self.ai_engine = AITradingEngine()
        self.scanner = MarketScanner(self)
        self.monitor = TradeMonitor(self)
        self.offline = OfflineManager(self)
        self.data_service = ApplicationData(self)

        self._db_initialized = False
        self._db_lock = None
        self._scanner_task = None

        self.is_running = False
        self.execution_mode = "SIGNAL_ONLY"
        self.system_status = "IDLE"  # IDLE, BACKFILLING, TRAINING
        self.session_id = f"SESSION_{int(time.time())}"
        self.session_stats = {
            "wins": 0,
            "losses": 0,
            "total": 0,
            "pnl": 0.0,
            "currency": "USD",
            "start": datetime.now(),
        }
        self.active_signals = []
        self.monitored_tasks = {}
        self.settings = {}
        self.active_crypto_list = []
        self.active_forex_list = []
        self.session_start = time.time()

    def _apply_settings_to_engine(self):
        """Pushes settings to the AI Engine instance."""
        settings_with_mode = self.settings.copy()
        settings_with_mode["execution_mode"] = self.execution_mode
        self.ai_engine.update_config(settings_with_mode)

    def get_latency(self):
        """Returns the connection latency (ping) to the broker server."""
        return self.provider.get_ping()

    async def initialize_connection(self, login_id, server, password, mt5_path=None):
        """
        Attempt to connect to MT5 using credentials from the GUI.
        """
        # --- Prevent Duplicate Connection ---
        if self.provider.connected:
            logger.info("✅ Already connected to MT5.")
            self.is_running = True
            if not self._scanner_task or self._scanner_task.done():
                self._scanner_task = asyncio.create_task(self.scanner.scanner_loop())
            return {"success": True, "message": "Already Connected"}

        try:
            mt5_login = int(login_id)
        except ValueError:
            return {"success": False, "message": "Login ID must be a number."}

        # 1. Update Provider Credentials dynamically
        self.provider._login = mt5_login
        self.provider._password = password
        self.provider._server = server
        if mt5_path:
            self.provider._path = mt5_path

        logger.info(f"🖥️ Connecting to {server}...")

        connected = await self.provider.initialize()

        if connected:
            self.is_running = True
            await self.initialize_settings()

            asyncio.create_task(self.ai_engine.initialize())

            # Get actual account currency for session stats
            acct = await self.provider.get_account_summary()
            if acct:
                self.session_stats["currency"] = acct.get("currency", "USD")

            # Save valid credentials to DB
            new_settings = self.settings.copy()
            new_settings.update(
                {"login": login_id, "server": server, "password": password, "execution_mode": self.execution_mode}
            )
            await self.db.save_settings(new_settings)

            self.ai_engine.set_context(500.0, self.db)
            await self.offline.check_offline_trades()

            # 3. Double-Loop Prevention
            if not self._scanner_task or self._scanner_task.done():
                self._scanner_task = asyncio.create_task(self.scanner.scanner_loop())

            return {"success": True, "message": "Connection Established"}
        else:
            return {"success": False, "message": "MT5 Connection Failed. Check Credentials."}

    async def initialize_settings(self):
        """Loads settings. Safe to call multiple times."""
        if self._db_lock is None:
            self._db_lock = asyncio.Lock()

        async with self._db_lock:
            if not self._db_initialized:
                await self.db.init_database()
                await self.db.cleanup_db()
                self._db_initialized = True

            db_settings = await self.db.get_settings()

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
            self.execution_mode = self.settings.get("execution_mode", "SIGNAL_ONLY")
            self._apply_settings_to_engine()

    async def restart_system(self):
        """Stops scanner, re-inits DB/Provider with new settings."""
        logger.info("♻️ Restarting System...")
        self.is_running = False
        if self._scanner_task:
            self._scanner_task.cancel()

        if self.provider.connected:
            await self.provider.shutdown()
            self.provider.connected = False
            await asyncio.sleep(2)

        await self.initialize_settings()
        await self.ai_engine.initialize()

        await self.initialize_connection(self.settings["login"], self.settings["server"], self.settings["password"])
        return True

    def set_execution_mode(self, mode_str):
        """Sets the bot state (SIGNAL_ONLY vs FULL_AUTO)"""
        self.execution_mode = mode_str
        logger.info(f"⚙️ Execution Mode Changed: {self.execution_mode}")

        loop = get_persistent_loop()
        asyncio.run_coroutine_threadsafe(self.db.save_settings({"execution_mode": mode_str}), loop)
        return True

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
        Developer Orchestration:
        1. Backfill Data (Full or Partial)
        2. Train Model
        3. Reload Engine
        """
        if self.system_status != "IDLE":
            logger.warning("⚠️ System must be IDLE to run training.")
            return False

        try:
            self.system_status = "BACKFILLING"
            logger.info(f"🔄 Starting Manual SMC Training Cycle.")

            # 1. Backfill - Pass the existing provider and engine to avoid MT5 connection clash!
            target = [symbol] if symbol else None
            await backfill_data(self.provider, self.ai_engine, target_symbols=target)

            # 2. Train
            self.system_status = "TRAINING"
            logger.info("🧠 Backfill Complete. Starting Neural Training...")

            await asyncio.to_thread(self.ai_engine.nn_brain.train_network)

            # 3. Restart to apply
            logger.info("✅ Training Complete. Reloading Systems...")
            await self.restart_system()

            self.system_status = "IDLE"
            return True
        except Exception as e:
            logger.error(f"Training Cycle Failed: {e}")
            self.system_status = "IDLE"
            return False
