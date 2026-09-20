"""
In-Memory Rolling Candle Buffer.
Acts as the central repository for OHLCV data. Replaces legacy MT5 terminal polling.
Uses fast O(1) deques and asyncio task locks to prevent race conditions between
incoming WebSocket ticks and outgoing Scheduler strategy scans.
"""

import asyncio
import pandas as pd
from collections import deque
from typing import Dict, Optional

from config.settings import settings


class CandleStore:
    def __init__(self):
        # Internal structure: self._data["XAUUSD"]["5min"] = deque(...)
        self._data: Dict[str, Dict[str, deque]] = {}
        # Task safety locks: self._locks["XAUUSD"]["5min"] = asyncio.Lock()
        self._locks: Dict[str, Dict[str, asyncio.Lock]] = {}

        # Buffer limit pulled directly from validated settings
        self.max_len = settings.CANDLE_BUFFER_SIZE

    def _ensure_initialized(self, symbol: str, timeframe: str) -> None:
        """Internal helper to bootstrap nested dicts and locks for new assets/TFs."""
        if symbol not in self._data:
            self._data[symbol] = {}
            self._locks[symbol] = {}
        if timeframe not in self._data[symbol]:
            self._data[symbol][timeframe] = deque(maxlen=self.max_len)
            self._locks[symbol][timeframe] = asyncio.Lock()

    async def add_candle(self, symbol: str, timeframe: str, candle: pd.Series) -> None:
        """
        Appends a newly closed candle to the rolling buffer.
        If a candle with the exact same timestamp arrives, it replaces the existing one.
        """
        self._ensure_initialized(symbol, timeframe)

        async with self._locks[symbol][timeframe]:
            store = self._data[symbol][timeframe]

            # Handle potential updates to the current active candle
            if store and store[-1]["timestamp"] == candle["timestamp"]:
                store[-1] = candle
            else:
                store.append(candle)

    async def get_candles(self, symbol: str, timeframe: str, count: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieves a DataFrame of the requested candle history.

        Args:
            symbol (str): Target asset (e.g., 'XAUUSD').
            timeframe (str): Target timeframe (e.g., '5min').
            count (int, optional): Number of recent candles to retrieve. Returns all if None.

        Returns:
            pd.DataFrame: A standardized DataFrame ready for SMC computations.
        """
        self._ensure_initialized(symbol, timeframe)

        async with self._locks[symbol][timeframe]:
            store = self._data[symbol][timeframe]
            if count is None or count >= len(store):
                data = list(store)
            else:
                data = list(store)[-count:]

        # Return an empty DataFrame with the correct schema if no data exists
        if not data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        return pd.DataFrame(data)

    async def is_ready(self, symbol: str, timeframe: str, min_candles: int = 100) -> bool:
        """
        Checks if the buffer has enough historical data for reliable structure calculation.
        """
        self._ensure_initialized(symbol, timeframe)

        async with self._locks[symbol][timeframe]:
            return len(self._data[symbol][timeframe]) >= min_candles

    async def initialize(self, symbol: str, timeframe: str, historical_df: pd.DataFrame) -> None:
        """
        Bulk-loads a REST API historical DataFrame into the deque during startup.
        Clears any existing data for that timeframe.
        """
        self._ensure_initialized(symbol, timeframe)

        async with self._locks[symbol][timeframe]:
            store = self._data[symbol][timeframe]
            store.clear()

            # Iterate and append rows as pandas Series
            for _, row in historical_df.iterrows():
                store.append(row)


# Export the singleton instance to be used globally across the engine
candle_store = CandleStore()
