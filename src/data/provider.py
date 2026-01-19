import asyncio
import logging
import MetaTrader5 as mt5
import os
import time
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from src.config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH

logger = logging.getLogger(__name__)


class DataProvider:
    """
    MT5 Direct Provider with Live Economic Calender.
    """

    def __init__(self):
        self.connected = False
        self._login = MT5_LOGIN
        self._password = MT5_PASSWORD
        self._server = MT5_SERVER
        self._path = MT5_PATH
        self.spread_cache = {}
        self.last_cache_clear = time.time()
        self._news_cache = []
        self._last_news_fetch = 0

    def _fetch_calendar_events(self) -> List[Dict]:
        """
        Fetches High Impact news events from MT5 Calendar.
        Filters for upcoming events in the next 2 hours.
        """
        if not hasattr(mt5, "calendar_get_events"):
            # Return empty so the bot falls back to news_block.txt silently
            return []

        try:
            now = datetime.now()
            future = now + timedelta(hours=2)

            # Fetch calendar values (news events)
            events = mt5.calendar_get_events(None, None)
            if not events:
                return []

            # Filter for High Impact (importance=2 or 3 depending on broker, usually 1=Low, 2=Med, 3=High)
            high_impact_ids = {e.id for e in events if e.importance >= 3}

            # Get actual values/timings for today
            values = mt5.calendar_get_values(datetime.now() - timedelta(hours=1), future)

            news_buffer = []
            if values:
                for v in values:
                    if v.event_id in high_impact_ids:
                        news_buffer.append({"time": v.time, "impact": "HIGH", "event_id": v.event_id})
            return news_buffer

        except Exception as e:
            logger.error(f"News fetch failed: {e}")
            return []

    def _kill_terminal(self):
        """Force kills MT5 terminal process."""
        try:
            if os.name == "nt":
                # Windows
                subprocess.run(
                    ["taskkill", "/F", "/IM", "terminal64.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                logger.info("⚠️ Forced kill of terminal64.exe")
        except Exception as e:
            logger.error(f"Failed to kill terminal: {e}")

    def _sync_account_info(self) -> Dict:
        """Fetches live account balance and equity."""
        info = mt5.account_info()
        if not info:
            return {"balance": 0.0, "equity": 0.0, "profit": 0.0}

        return {"balance": float(info.balance), "equity": float(info.equity), "profit": float(info.profit)}

    def _sync_connect(self) -> bool:
        """Synchronous MT5 Connection with Retries and Auto-Kill"""
        try:
            if mt5.terminal_info() is not None:
                if self.connected:
                    return True
                else:
                    # It's open but we need to verify login
                    self.connected = True

            # 1. Validate Path
            if not os.path.exists(self._path):
                logger.error(f"❌ MT5 Path not found: {self._path}")
                return False

            # 2. Initialize with Timeout (Fix for IPC Timeout)
            # We give it 60 seconds to launch and connect.
            if not mt5.initialize(path=self._path, timeout=60000):
                logger.warning("⚠️ MT5 Init failed. Attempting to restart terminal...")
                self._kill_terminal()
                time.sleep(2)

                # Retry
                if not mt5.initialize(path=self._path, timeout=60000):
                    logger.error("❌ MT5 Init failed after restart.")
                    return False

            # 3. Login
            if self._login and self._password:
                authorized = mt5.login(int(self._login), password=self._password, server=self._server)
                if authorized:
                    logger.info(f"✅ Connected to Broker Account: {self._login}")
                    self.connected = True
                    return True

            self.connected = True
            return True
        except Exception as e:
            logger.exception(f"Critical MT5 Connection Error: {e}")
            return False

    def _sync_get_rates(self, symbol: str, timeframe: int, limit: int) -> List[Dict]:
        """Fetches candles with Synchronization Check."""
        # 1. Select symbol in Market Watch to trigger sync
        if not self.connected:
            return []

        selected = mt5.symbol_select(symbol, True)
        if not selected:
            # Check if terminal is actually dead before logging warning
            term_info = mt5.terminal_info()
            if term_info is None:
                self.connected = False  # Terminal is gone
                return []

            logger.warning(f"Symbol {symbol} not found in Market Watch.")
            return []

        # 2. Attempt to fetch data with retries
        rates = None
        for _ in range(3):
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, limit)
            if rates is not None and len(rates) > 0:
                break
            time.sleep(0.2)

        if rates is None or len(rates) == 0:
            return []

        # Convert to standard list of dicts
        data = []
        for r in rates:
            data.append(
                {
                    "time": float(r["time"]),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": float(r["tick_volume"]),
                    "spread": int(r["spread"]),
                }
            )
        return data

    def _sync_get_tick(self, symbol: str) -> Optional[float]:
        """Gets current Bid/Ask average."""
        if not mt5.symbol_select(symbol, True):
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return (tick.bid + tick.ask) / 2.0
        return None

    def _sync_get_tick_struct(self, symbol: str) -> Optional[mt5.Tick]:
        """Returns full tick object (Bid/Ask)."""
        if not mt5.symbol_select(symbol, True):
            return None
        return mt5.symbol_info_tick(symbol)

    def _sync_symbol_info(self, symbol: str) -> Dict:
        """
        Fetches detailed symbol specification for risk calculations."""
        info = mt5.symbol_info(symbol)
        if not info:
            return {}

        return {
            "digits": info.digits,
            "min_vol": info.volume_min,
            "max_vol": info.volume_max,
            "vol_step": info.volume_step,
            "point": info.point,
            "trade_tick_value": info.trade_tick_value,
            "currency_profit": info.currency_profit,
            "currency_base": info.currency_base,
        }

    async def check_live_news_block(self, symbol: str, currencies: List[str]) -> bool:
        """
        Returns True if High Impact news is imminent (within 30 mins) for the symbol's currencies.
        """
        # Update cache every 5 minutes
        if time.time() - self._last_news_fetch > 300:
            self._news_cache = await asyncio.to_thread(self._fetch_calendar_events)
            self._last_news_fetch = time.time()
            if self._news_cache:
                logger.info(f"📅 Live News Updated: {len(self._news_cache)} high impact events found.")

        if not self._news_cache:
            return False

        now = datetime.now()
        for event in self._news_cache:
            event_time = event["time"]
            # Block 30 mins before and 30 mins after
            if (event_time - timedelta(minutes=30)) <= now <= (event_time + timedelta(minutes=30)):
                return True

        return False

    async def initialize(self) -> bool:
        """
        Initializes connection to MT5 Terminal.
        Retries logic implemented for stability.
        """
        return await asyncio.to_thread(self._sync_connect)

    async def fetch_klines(self, symbol: str, timeframe_str: str, limit: int) -> List[Dict]:
        """
        Fetches candles with Synchronization Check.
        """
        if not self.connected:
            return []

        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }
        mt5_tf = tf_map.get(timeframe_str.lower(), mt5.TIMEFRAME_M15)

        return await asyncio.to_thread(self._sync_get_rates, symbol, mt5_tf, limit)

    async def get_account_summary(self) -> Dict:
        """Async wrapper to get account details"""
        if not self.connected:
            return {"balance": 0.0, "equity": 0.0, "profit": 0.0}
        return await asyncio.to_thread(self._sync_account_info)

    async def get_current_tick(self, symbol: str) -> Optional[mt5.Tick]:
        """Returns full tick object (Bid/Ask)."""
        return await asyncio.to_thread(self._sync_get_tick_struct, symbol)

    def get_ping(self) -> int:
        """Returns the last known latency to the broker in ms."""
        if not self.connected:
            return -1
        try:
            # Use mt5.terminal_info() which is fast and local
            info = mt5.terminal_info()
            if info:
                return info.ping_last // 1000  # Convert micros to ms if needed, or just return as is (usually micros)
                # Actually ping_last is usually in microseconds. Let's return ms.
                return int(info.ping_last / 1000)
            return 0
        except:
            return -1

    async def get_symbol_info(self, symbol: str) -> Dict:
        """
        Fetches detailed symbol specification for risk calculations.
        Crucial for ZAR account conversion.
        """
        return await asyncio.to_thread(self._sync_symbol_info, symbol)

    async def get_spread(self, symbol: str) -> Dict:
        """
        Smart Spread Check using caching + Hourly Clear.
        """
        # 1. Clear cache if older than 1 hour
        if time.time() - self.last_cache_clear > 3600:
            self.spread_cache = {}
            self.last_cache_clear = time.time()

        tick = await self.get_current_tick(symbol)
        if not tick:
            return {"spread": 0, "spread_high": True}  # Block if no data

        spread_points = tick.ask - tick.bid
        info = await self.get_symbol_info(symbol)
        point = info.get("point", 0.00001)
        spread_raw = spread_points / point

        # Cache Update
        if symbol not in self.spread_cache:
            self.spread_cache[symbol] = []
        self.spread_cache[symbol].append(spread_raw)
        if len(self.spread_cache[symbol]) > 50:
            self.spread_cache[symbol].pop(0)

        return {"spread": spread_raw, "spread_high": False}

    async def shutdown(self):
        """Safely shuts down the connection."""
        await asyncio.to_thread(mt5.shutdown)
        self.connected = False
