"""
Twelve Data REST & WebSocket Client.
Handles historical data fetching, real-time tick streaming, candle aggregation,
API rate limit tracking, and exponential backoff for disconnects.
"""

import aiohttp
import asyncio
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional

from config.settings import settings
from utils.logger import logger
from utils.rate_limiter import rate_limiter


class TwelveDataClient:
    def __init__(self):
        self.api_key = settings.TWELVE_DATA_API_KEY
        self.rest_base_url = "https://api.twelvedata.com"
        self.ws_url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.api_key}"

        self.ws_connection: Optional[aiohttp.ClientWebSocketResponse] = None
        self.is_streaming = False

        # State for tick-to-candle aggregation
        self.active_candles: Dict[str, dict] = {}
        self.interval_minutes = self._parse_interval_minutes(settings.ENTRY_TIMEFRAME)

    def _parse_interval_minutes(self, tf: str) -> int:
        """Parses string timeframes into integer minutes."""
        if tf.endswith("min"):
            return int(tf.replace("min", ""))
        if tf.endswith("h"):
            return int(tf.replace("h", "")) * 60
        raise ValueError(f"Unsupported timeframe format: {tf}")

    async def get_historical_ohlcv(self, symbol: str, interval: str, outputsize: int = 500) -> pd.DataFrame:
        """Fetches historical OHLCV data via REST API safely mapped through rate limiter."""
        # Enforce unified Rate Limiter verification before dispatching network request
        await rate_limiter.acquire()

        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON",
        }

        url = f"{self.rest_base_url}/time_series"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if "values" not in data:
                    logger.error("twelvedata_rest_error", response=data)
                    raise ValueError(f"Failed to fetch data for {symbol}: {data.get('message', 'Unknown error')}")

                return pd.DataFrame(data["values"])

    def _get_candle_boundary(self, dt: datetime) -> datetime:
        """Rounds a datetime down to the nearest multiple of the interval (e.g., 5min)."""
        minute = dt.minute - (dt.minute % self.interval_minutes)
        return dt.replace(minute=minute, second=0, microsecond=0)

    async def _process_tick(self, tick: dict, on_candle_close: Callable[[str, str, pd.Series], Awaitable[None]]):
        """Aggregates real-time price ticks into M5 candles and triggers callbacks."""
        symbol = tick.get("symbol")
        price = float(tick.get("price"))
        tick_time = datetime.fromtimestamp(tick.get("timestamp"), tz=timezone.utc)

        # Lazy runtime import inside the execution scope to avoid circular initialization loops
        from strategies.pipeline import monitor_active_signals

        asyncio.create_task(monitor_active_signals(symbol, price))

        candle_start = self._get_candle_boundary(tick_time)

        # Initialize candle state for the symbol if it doesn't exist
        if symbol not in self.active_candles:
            self.active_candles[symbol] = {
                "timestamp": candle_start,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1.0,  # Representing tick count
            }
            return

        current = self.active_candles[symbol]

        # Check if the tick belongs to a new candle
        if candle_start > current["timestamp"]:
            # 1. Previous candle is officially closed. Push it.
            closed_candle = pd.Series(current)
            logger.info("candle_closed", symbol=symbol, timestamp=str(current["timestamp"]))

            # Fire the callback to the strategy engine asynchronously
            asyncio.create_task(on_candle_close(symbol, settings.ENTRY_TIMEFRAME, closed_candle))

            # 2. Reset the active candle for the new boundary
            self.active_candles[symbol] = {
                "timestamp": candle_start,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1.0,
            }
        else:
            # Update the currently forming candle
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price
            current["volume"] += 1.0

    async def start_websocket_stream(
        self, symbols: List[str], on_candle_close: Callable[[str, str, pd.Series], Awaitable[None]]
    ) -> None:
        """
        Connects to the WebSocket, streams ticks, and handles auto-reconnection
        with exponential backoff.
        """
        self.is_streaming = True
        backoff = 1

        while self.is_streaming:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.ws_url) as ws:
                        self.ws_connection = ws
                        logger.info("websocket_connected", url="wss://ws.twelvedata.com")

                        # Subscribe to symbols
                        subscribe_msg = {"action": "subscribe", "params": {"symbols": ",".join(symbols)}}
                        await ws.send_json(subscribe_msg)
                        backoff = 1  # Reset backoff on successful connection

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                if data.get("event") == "price":
                                    await self._process_tick(data, on_candle_close)
                                elif data.get("event") == "heartbeat":
                                    pass  # Keepalive ping
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break

            except Exception as e:
                logger.error("websocket_error", error=str(e))

            if self.is_streaming:
                logger.warning("websocket_reconnecting", backoff_seconds=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)  # Max 60s delay

    async def stop_websocket_stream(self) -> None:
        """Cleanly terminates the WebSocket connection."""
        self.is_streaming = False
        if self.ws_connection and not self.ws_connection.closed:
            await self.ws_connection.close()
            logger.info("websocket_disconnected")


# Singleton instance used by the rest of the application
client = TwelveDataClient()
