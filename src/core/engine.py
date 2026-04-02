import asyncio
import time
from datetime import datetime

from src.data.provider import DataProvider
from src.database.manager import DatabaseManager
from src.engine.ai_engine import AITradingEngine
from src.utils.logger import setup_logging
from src.core.services.market_scanner import MarketScanner
from src.core.services.offline_manager import OfflineManager
from src.core.services.trade_monitor import TradeMonitor

logger = setup_logging()


class NexubotEngine:
    """The central control of the bot."""

    def __init__(self, notifier):
        self.notifier = notifier
        self.provider = DataProvider()
        self.db = DatabaseManager()
        self.ai_engine = AITradingEngine()
        self.scanner = MarketScanner(self)
        self.monitor = TradeMonitor(self)
        self.offline = OfflineManager(self)

        self._scanner_task = None
        self.is_running = False
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
        self.active_crypto_list = []
        self.active_forex_list = []
        self.active_indices_list = []
        self.session_start = time.time()

    async def initialize_connection(self, login_id: int, server: str, password: str, mt5_path=None) -> bool:
        """
        Headless connection to MT5 using environment credentials.
        """
        # --- Prevent Duplicate Connection ---
        if self.provider.connected:
            logger.info("✅ Already connected to MT5.")
            self.is_running = True
            if not self._scanner_task or self._scanner_task.done():
                self._scanner_task = asyncio.create_task(self.scanner.scanner_loop())
            return True

        try:
            mt5_login = int(login_id)
        except ValueError:
            logger.error("Login ID must be a number.")
            return False

        # Update Provider Credentials dynamically
        self.provider._login = mt5_login
        self.provider._password = password
        self.provider._server = server
        if mt5_path:
            self.provider._path = mt5_path

        logger.info(f"🖥️ Connecting to {server}...")

        connected = await self.provider.initialize()

        if connected:
            self.is_running = True

            # Get actual account currency for session stats
            acct = await self.provider.get_account_summary()
            if acct:
                self.session_stats["currency"] = acct.get("currency", "USD")

            self.ai_engine.set_context(acct["balance"], self.db, acct.get("currency", "USD"))

            await self.offline.check_offline_trades()

            if not self._scanner_task or self._scanner_task.done():
                self._scanner_task = asyncio.create_task(self.scanner.scanner_loop())

            return True
        else:
            logger.error("MT5 Connection Failed. Check Credentials.")
            return False

    async def stop_session(self):
        """Stops the bot operations & closes resources."""
        logger.info("🛑 Stopping Engine & Saving Session...")
        self.is_running = False

        if self._scanner_task and not self._scanner_task.done():
            self._scanner_task.cancel()
            try:
                await self._scanner_task  # Wait for it to properly close
            except asyncio.CancelledError:
                pass

        # Cancel monitoring tasks
        for symbol, task in self.monitored_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.monitored_tasks.clear()
        self.active_signals.clear()

        await self.db.log_session(self.session_id, self.session_stats["start"].timestamp(), self.session_stats)
        await self.db.close()
        await self.provider.shutdown()
        return True
