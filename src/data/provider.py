import asyncio
import logging
import MetaTrader5 as mt5
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from src.config import FALLBACK_CRYPTO, FALLBACK_FOREX, FALLBACK_INDICES, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH

logger = logging.getLogger(__name__)

# Markets to ignore
IGNORED_CURRENCIES = ["RUB", "TRY", "ZAR", "MXN", "CNH", "HKD", "SGD", "NOK", "SEK", "PLN", "DKK", "HUF"]


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
        self._symbol_type_cache = {}
        self._cached_usdzar = None
        self._cached_usdzar_time = 0

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
            return {"balance": 0.0, "equity": 0.0, "profit": 0.0, "currency": "USD"}

        return {
            "balance": float(info.balance),
            "equity": float(info.equity),
            "profit": float(info.profit),
            "currency": getattr(info, "currency", "USD"),
        }

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

        # Fallbacks from Config (Prioritized)
        priority_crypto = set(FALLBACK_CRYPTO)
        priority_forex = set(FALLBACK_FOREX)
        priority_indices = set(FALLBACK_INDICES)

        # Also allow major pairs explicitly
        major_forex_bases = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCAD",
            "AUDUSD",
            "XAUUSD",
            "XAGUSD",
            "GBPJPY",
            "NZDUSD",
            "BTCJPY",
            "CHFJPY",
            "EURJPY",
            "AUDJPY",
            "CADJPY",
            "EURAUD",
        ]
        major_crypto_bases = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "BTCUSDT", "ETHUSDT"]
        major_indices_bases = ["US30", "NAS100", "GER30", "SPX500", "UK100", "JPN225"]

        # Get only selected symbols (Market Watch)
        symbols = mt5.symbols_get(selected=True)
        if not symbols:
            return {}

        categorized = {"crypto": [], "forex": [], "indices": []}

        for s in symbols:
            name = s.name.upper()
            # 1. Ignore Trash
            if s.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED or not s.visible:
                continue
            if any(ign in name for ign in IGNORED_CURRENCIES):
                continue

            # 2. Strict Filtering Logic
            is_priority = (name in priority_crypto) or (name in priority_forex)
            is_major = any(m in name for m in major_forex_bases) or any(m in name for m in major_crypto_bases)
            is_index = any(m in name for m in major_indices_bases)

            # Only add if it's in our Priority List or is a Major Pair
            if not (is_priority or is_major or is_index):
                continue

            category = self.get_symbol_type(s.name)
            if category == "CRYPTO":
                categorized["crypto"].append(s.name)
            elif category == "INDICES":
                categorized["indices"].append(s.name)
            else:
                categorized["forex"].append(s.name)

        # Fallback if empty (Force default list)
        if not categorized["crypto"] and not categorized["forex"]:
            categorized["crypto"] = list(priority_crypto)
            categorized["forex"] = list(priority_forex)
            categorized["indices"] = list(priority_indices)

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
            "filling_mode": info.filling_mode,
        }

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
            return {"balance": 0.0, "equity": 0.0, "profit": 0.0, "currency": "USD"}
        return await asyncio.to_thread(self._sync_account_info)

    async def get_current_tick(self, symbol: str) -> Optional[mt5.Tick]:
        """Returns full tick object (Bid/Ask)."""
        return await asyncio.to_thread(self._sync_get_tick_struct, symbol)

    async def get_dynamic_symbols(self) -> Dict:
        """Async wrapper to get market watch symbols."""
        return await asyncio.to_thread(self._sync_get_market_watch_symbols)

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
        result = "FOREX"

        if info:
            path = info.path.lower()
            if "crypto" in path or "bitcoin" in path:
                result = "CRYPTO"
            if "indices" in path or "nas" in path:
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
            elif any(base in s for base in ["US30", "NAS", "GER30"]):
                result = "INDICES"
            else:
                result = "FOREX"

        self._symbol_type_cache[symbol] = result
        return result

    async def get_usdzar_rate(self) -> float:
        """
        Fetches the current USDZAR exchange rate for currency conversion.
        Tries standard permutations like USDZAR, USDZARm, USDZAR.
        """
        # Return cached if fresh (1 minute)
        if self._cached_usdzar and (time.time() - self._cached_usdzar_time) < 60:
            return self._cached_usdzar

        possible_pairs = ["USDZAR", "USDZARm", "USDZAR.", "USDZAR_OT"]

        for pair in possible_pairs:
            tick = await self.get_current_tick(pair)
            if tick:
                rate = (tick.bid + tick.ask) / 2.0
                self._cached_usdzar = rate
                self._cached_usdzar_time = time.time()
                return rate

        # Fallback if pair not found in market watch (Rough Estimate)
        return 18.5

    async def initialize(self) -> bool:
        """
        Initializes connection to MT5 Terminal.
        Retries logic implemented for stability.
        """
        return await asyncio.to_thread(self._sync_connect)

    async def shutdown(self):
        """Safely shuts down the connection."""
        await asyncio.to_thread(mt5.shutdown)
        self.connected = False
