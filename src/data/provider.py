import asyncio
import logging
import MetaTrader5 as mt5
import os
import subprocess
import time
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
        self._symbol_type_cache = {}

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

    def _sync_get_market_watch_symbols(self) -> Dict:
        """
        Fetches all symbols currently visible in MT5 Market Watch.
        Categorizes them into Crypto and Forex based on simple heuristics.
        """
        if not self.connected:
            return {}

        # Get only selected symbols (Market Watch)
        symbols = mt5.symbols_get(selected=True)
        if not symbols:
            return {}

        categorized = {"crypto": [], "forex": []}

        for s in symbols:
            category = self.get_symbol_type(s.name)

            if category == "CRYPTO":
                categorized["crypto"].append(s.name)
            else:
                # Treat Indices/Metals/Forex as 'Forex' for strategy purposes
                categorized["forex"].append(s.name)

        return categorized

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

    async def execute_trade_on_mt5(self, signal: dict) -> bool:
        """Places the trade on MT5 if in FULL_AUTO mode."""
        if not self.connected:
            return False

        symbol = signal["symbol"]
        action = mt5.TRADE_ACTION_DEAL
        type_order = mt5.ORDER_TYPE_BUY if signal["signal"] == "BUY" else mt5.ORDER_TYPE_SELL

        # Handle Limit Orders
        if signal.get("order_type") == "LIMIT":
            action = mt5.TRADE_ACTION_PENDING
            type_order = mt5.ORDER_TYPE_BUY_LIMIT if signal["signal"] == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT

        request = {
            "action": action,
            "symbol": symbol,
            "volume": signal["lot_size"],
            "type": type_order,
            "price": signal["price"],
            "sl": signal["sl"],
            "tp": signal["tp"],
            "deviation": 10,
            "magic": 123456,
            "comment": f"Nexubot {signal['strategy']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = await asyncio.to_thread(mt5.order_send, request)

        if result and result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ AUTOMATION: Trade Placed on MT5 for {symbol} ({signal['lot_size']} lots)")
            return True
        else:
            err = result.comment if result else "Unknown Error"
            logger.error(f"❌ AUTOMATION FAILED {symbol}: {err}")
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

    async def get_dynamic_symbols(self) -> Dict:
        """Async wrapper to get market watch symbols."""
        return await asyncio.to_thread(self._sync_get_market_watch_symbols)

    def get_ping(self) -> int:
        """Returns the last known latency to the broker in ms."""
        if not self.connected:
            return -1
        try:
            info = mt5.terminal_info()
            if info:
                return int(info.ping_last / 1000)
            return 0
        except:
            return -1

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

        # Keep last 10 ticks
        if len(self.spread_cache[symbol]) > 10:
            self.spread_cache[symbol].pop(0)

        avg_spread = sum(self.spread_cache[symbol]) / len(self.spread_cache[symbol])

        return {"spread": spread_raw, "avg_spread": avg_spread, "spread_high": False}

    async def get_symbol_info(self, symbol: str) -> Dict:
        """
        Fetches detailed symbol specification for risk calculations.
        Crucial for ZAR account conversion.
        """
        return await asyncio.to_thread(self._sync_symbol_info, symbol)

    def get_symbol_type(self, symbol: str) -> str:
        """
        Robustly determines if a symbol is 'CRYPTO' or 'FOREX'
        based on MT5 internal classification paths.
        """
        # Return cached result if available
        if symbol in self._symbol_type_cache:
            return self._symbol_type_cache[symbol]

        # 1. Ask MT5 for the symbol path
        info = mt5.symbol_info(symbol)

        result = "FOREX"  # Default safety

        if info:
            path = info.path.lower()
            # MT5 Path Check (Most Accurate)
            if "crypto" in path or "bitcoin" in path or "digital" in path:
                result = "CRYPTO"
            if "indices" in path or "stock" in path or "nas" in path:
                result = "INDICES"
            if "forex" in path or "majors" in path or "minors" in path or "exotics" in path:
                result = "FOREX"
        else:
            # 2. Fallback: Name-based Heuristic (if MT5 info fails or is vague)
            s = symbol.upper()

            # Common Crypto Bases
            crypto_bases = ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "LTC"]
            if any(base in s for base in crypto_bases):
                result = "CRYPTO"

            # Common Forex/Metals
            forex_bases = ["EUR", "USD", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF", "XAU", "XAG"]
            if any(base in s for base in forex_bases):
                result = "FOREX"

        self._symbol_type_cache[symbol] = result
        return result

    async def shutdown(self):
        """Safely shuts down the connection."""
        await asyncio.to_thread(mt5.shutdown)
        self.connected = False
