import asyncio
import logging
import MetaTrader5 as mt5
import os
import subprocess
import time
from collections import deque
from typing import Dict, List, Optional

from src.config import (
    FALLBACK_CRYPTO,
    FALLBACK_FOREX,
    FALLBACK_INDICES,
    FALLBACK_METALS,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_PATH,
    MT5_SERVER,
)

logger = logging.getLogger(__name__)

# Markets to ignore
IGNORED_CURRENCIES = ["RUB", "TRY", "ZAR", "MXN", "CNH", "HKD", "SGD", "NOK", "SEK", "PLN", "DKK", "HUF"]


class DataProvider:
    """Centralized Data Provider for MT5 Terminal."""

    def __init__(self):
        self.connected = False
        self._login = MT5_LOGIN
        self._password = MT5_PASSWORD
        self._server = MT5_SERVER
        self._path = MT5_PATH
        self.spread_cache = {}
        self.last_cache_clear = time.time()
        self._symbol_type_cache = {}
        self._cached_usdzar = None
        self._cached_usdzar_time = 0

    def _kill_terminal(self) -> None:
        """Force kills MT5 terminal process."""
        try:
            if os.name == "nt":
                # Windows
                subprocess.run(
                    ["taskkill", "/F", "/IM", "terminal64.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                logger.info("⚠️ Forced kill of terminal64.exe executed to recover connection.")
        except Exception as e:
            logger.error(f"Failed to kill terminal: {e}")

    def _sync_account_info(self) -> Dict:
        """Fetches live account summary (balance, equity, profit). Returns zeros if not connected or error occurs."""
        info = mt5.account_info()
        if not info:
            return {"balance": 0.0, "equity": 0.0, "profit": 0.0, "currency": "USD"}

        return {
            "balance": float(info.balance),
            "equity": float(info.equity),
            "profit": float(info.profit),
            "currency": getattr(info, "currency", "USD"),
        }

    def _sync_connect(self) -> bool:
        """Initializes connection to MT5 Terminal."""
        try:
            if mt5.terminal_info() is not None:
                acc_info = mt5.account_info()
                if acc_info is not None and str(acc_info.login) == str(self._login):
                    self.connected = True
                    return True
                else:
                    self.connected = False

            # 1. Validate Path
            if not os.path.exists(self._path):
                logger.error(f"❌ MT5 Path not found: {self._path}")
                return False

            # 2. Initialize with Timeout
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
        Strictly filters to only include the highly targeted assets defined in config.
        """
        if not self.connected:
            return {}

        # Combine all strictly allowed targets from config
        allowed_symbols = set(FALLBACK_CRYPTO + FALLBACK_FOREX + FALLBACK_INDICES + FALLBACK_METALS)

        # Get only selected symbols (Market Watch)
        symbols = mt5.symbols_get(selected=True)
        if not symbols:
            return {}

        categorized = {"crypto": [], "forex": [], "indices": [], "metals": []}

        for s in symbols:
            name = s.name.upper()

            # 1. Ignore Trash
            if s.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED or not s.visible:
                continue
            if any(ign in name for ign in IGNORED_CURRENCIES):
                continue

            # Strict Targeting: Only process symbols explicitly defined in config
            is_allowed = name in allowed_symbols or any(a in name for a in allowed_symbols)
            if not is_allowed:
                continue

            category = self.get_symbol_type(s.name)
            if category == "CRYPTO":
                categorized["crypto"].append(s.name)
            elif category == "INDICES":
                categorized["indices"].append(s.name)
            elif category == "METALS":
                categorized["metals"].append(s.name)
            else:
                categorized["forex"].append(s.name)

        # Fallback if Market Watch is empty or missing them
        if not categorized["crypto"] and not categorized["forex"] and not categorized["metals"]:
            categorized["crypto"] = list(FALLBACK_CRYPTO)
            categorized["forex"] = list(FALLBACK_FOREX)
            categorized["indices"] = list(FALLBACK_INDICES)
            categorized["metals"] = list(FALLBACK_METALS)

        return categorized

    def _sync_get_rates(self, symbol: str, timeframe: int, limit: int) -> List[Dict]:
        """Fetches historical OHLCV data for a symbol and timeframe. Returns empty list if not connected or symbol issues."""
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

        # Attempt to fetch data with retries
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

    def _sync_get_tick_struct(self, symbol: str) -> Optional[mt5.Tick]:
        """Returns full tick object (Bid/Ask)."""
        if not mt5.symbol_select(symbol, True):
            return None
        return mt5.symbol_info_tick(symbol)

    def _sync_symbol_info(self, symbol: str) -> Dict:
        """Fetches detailed symbol specification for risk calculations."""
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
            "filling_mode": info.filling_mode,
        }

    async def fetch_klines(self, symbol: str, timeframe_str: str, limit: int) -> List[Dict]:
        """Async wrapper to fetch klines with MT5 timeframe mapping and synchronization check."""
        if not self.connected:
            return []

        tf_map = {
            "m1": mt5.TIMEFRAME_M1,
            "1m": mt5.TIMEFRAME_M1,
            "m5": mt5.TIMEFRAME_M5,
            "5m": mt5.TIMEFRAME_M5,
            "m15": mt5.TIMEFRAME_M15,
            "15m": mt5.TIMEFRAME_M15,
            "m30": mt5.TIMEFRAME_M30,
            "30m": mt5.TIMEFRAME_M30,
            "h1": mt5.TIMEFRAME_H1,
            "1h": mt5.TIMEFRAME_H1,
            "h4": mt5.TIMEFRAME_H4,
            "4h": mt5.TIMEFRAME_H4,
            "d1": mt5.TIMEFRAME_D1,
            "1d": mt5.TIMEFRAME_D1,
        }
        mt5_tf = tf_map.get(timeframe_str.lower(), mt5.TIMEFRAME_M5)

        return await asyncio.to_thread(self._sync_get_rates, symbol, mt5_tf, limit)

    async def get_account_summary(self) -> Dict:
        """Async wrapper to fetch live account summary (balance, equity, profit). Returns zeros if not connected."""
        if not self.connected:
            return {"balance": 0.0, "equity": 0.0, "profit": 0.0, "currency": "USD"}
        return await asyncio.to_thread(self._sync_account_info)

    async def get_current_tick(self, symbol: str) -> Optional[mt5.Tick]:
        """Async wrapper to get current tick (Bid/Ask) for spread calculations."""
        return await asyncio.to_thread(self._sync_get_tick_struct, symbol)

    async def get_dynamic_symbols(self) -> Dict:
        """Async wrapper to fetch dynamic list of symbols from Market Watch with strict filtering."""
        return await asyncio.to_thread(self._sync_get_market_watch_symbols)

    async def get_spread(self, symbol: str) -> Dict:
        """Returns current spread and average spread for the symbol."""
        # 1. Clear cache if older than 1 hour
        if time.time() - self.last_cache_clear > 3600:
            self.spread_cache = {}
            self.last_cache_clear = time.time()

        tick = await self.get_current_tick(symbol)
        if not tick:
            return {"spread": 0, "spread_high": True}

        spread_points = tick.ask - tick.bid
        info = await self.get_symbol_info(symbol)
        point = info.get("point", 0.00001)
        spread_raw = spread_points / point

        # Cache Update
        if symbol not in self.spread_cache:
            self.spread_cache[symbol] = deque(maxlen=10)

        self.spread_cache[symbol].append(spread_raw)
        avg_spread = sum(self.spread_cache[symbol]) / len(self.spread_cache[symbol])

        return {"spread": spread_raw, "avg_spread": avg_spread, "spread_high": False}

    async def get_symbol_info(self, symbol: str) -> Dict:
        """Async wrapper to get symbol info for risk calculations."""
        return await asyncio.to_thread(self._sync_symbol_info, symbol)

    def get_symbol_type(self, symbol: str) -> str:
        """Determines if a symbol is Forex, Crypto, Indices, or Metals based on MT5 info and heuristics."""
        # Return cached result if available
        if symbol in self._symbol_type_cache:
            return self._symbol_type_cache[symbol]

        # 1. Ask MT5 for the symbol path
        info = mt5.symbol_info(symbol)
        result = "FOREX"

        if info:
            path = info.path.lower()
            if "crypto" in path or "bitcoin" in path:
                result = "CRYPTO"
            if "indices" in path:
                result = "INDICES"
            elif "metals" in path or "gold" in path or "silver" in path:
                result = "METALS"
            if "forex" in path or "majors" in path or "minors" in path or "exotics" in path:
                result = "FOREX"
        else:
            # 2. Fallback: Name-based Heuristic (if MT5 info fails or is vague)
            s = symbol.upper()

            if any(base in s for base in ["XAU", "XAG"]):
                result = "METALS"
            elif any(base in s for base in ["BTC", "ETH", "SOL", "XRP"]):
                result = "CRYPTO"
            elif any(base in s for base in ["US30", "GER30"]):
                result = "INDICES"
            else:
                result = "FOREX"

        self._symbol_type_cache[symbol] = result
        return result

    async def get_usdzar_rate(self) -> float:
        """Fetches the current USDZAR exchange rate. Uses caching to minimize MT5 calls. Returns a fallback rate if not available."""
        # Return cached if fresh (1 minute)
        if self._cached_usdzar and (time.time() - self._cached_usdzar_time) < 60:
            return self._cached_usdzar

        possible_pairs = ["USDZAR", "USDZARm", "USDZAR.", "USDZAR_OT"]

        for pair in possible_pairs:
            if mt5.symbol_select(pair, True):
                tick = await self.get_current_tick(pair)
                if tick:
                    rate = (tick.bid + tick.ask) / 2.0
                    self._cached_usdzar = rate
                    self._cached_usdzar_time = time.time()
                    return rate

        # Fallback if the broker entirely lacks USDZAR
        logger.critical("⚠️ MT5 Broker lacks USDZAR! ZAR Lot sizing falling back to rough 16.5 estimate.")
        return 16.5

    async def initialize(self) -> bool:
        """Initializes connection to MT5 Terminal. Returns True if successful, False otherwise."""
        return await asyncio.to_thread(self._sync_connect)

    async def shutdown(self):
        """Safely shuts down the connection."""
        await asyncio.to_thread(mt5.shutdown)
        self.connected = False
