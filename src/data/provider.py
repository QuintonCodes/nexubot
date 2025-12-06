import aiohttp
import logging
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Optional
from src.config import (
  TWELVEDATA_API_KEY, TWELVEDATA_BASE_URL,
  BINANCE_API_URL, FALLBACK_USD_ZAR, FOREX_SYMBOLS
)

logger = logging.getLogger(__name__)

class DataProvider:
  """
  Hybrid Provider: Routes requests to Binance (Crypto) or TwelveData (Forex).
  """
  def __init__(self, session: aiohttp.ClientSession):
    self.session = session
    self.api_key = TWELVEDATA_API_KEY
    self.forex_lock = asyncio.Lock()
    self._cached_usd_zar = FALLBACK_USD_ZAR
    self._last_zar_update = 0

  async def get_usd_zar(self) -> float:
    """Updates ZAR rate hourly."""
    if time.time() - self._last_zar_update > 3600:
      rate = await self._fetch_twelve_price("USD/ZAR")
      if rate:
          self._cached_usd_zar = rate
          self._last_zar_update = time.time()
    return self._cached_usd_zar

  async def fetch_klines(self, symbol: str, interval: str, limit: int) -> List[Dict]:
    """Router for Klines."""
    if symbol in FOREX_SYMBOLS:
      return await self._fetch_forex_klines(symbol, interval, limit)
    return await self._fetch_crypto_klines(symbol, interval, limit)

  async def get_latest_price(self, symbol: str) -> Optional[float]:
    """Unified price fetcher for trade verification."""
    if symbol in FOREX_SYMBOLS:
      return await self._fetch_twelve_price(symbol)
    return await self._fetch_binance_price(symbol)

    # --- BINANCE (CRYPTO) ---
  async def _fetch_crypto_klines(self, symbol: str, interval: str, limit: int) -> List[Dict]:
    url = f"{BINANCE_API_URL}/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}

    try:
      async with self.session.get(url, params=params) as resp:
        if resp.status != 200:
          logger.warning(f"Binance error {resp.status} for {symbol}")
          return []
        data = await resp.json()

        # Binance returns Oldest -> Newest
        return [{
            'time': float(k[0]) / 1000,
            'open': float(k[1]), 'high': float(k[2]),
            'low': float(k[3]), 'close': float(k[4]),
            'volume': float(k[5])
        } for k in data]
    except Exception:
      return []

  async def _fetch_binance_price(self, symbol: str) -> Optional[float]:
    url = f"{BINANCE_API_URL}/ticker/price"
    try:
      async with self.session.get(url, params={'symbol': symbol}) as resp:
        if resp.status == 200:
          data = await resp.json()
          return float(data['price'])
    except Exception:
      return None

    # --- TWELVEDATA (FOREX) ---
  async def _fetch_forex_klines(self, symbol: str, interval: str, limit: int) -> List[Dict]:
    if not self.api_key: return []

    async with self.forex_lock:
      await asyncio.sleep(1.5) # Rate limit protection
      params = {
        'symbol': symbol, 'interval': interval,
        'outputsize': limit, 'apikey': self.api_key,
        'order': 'ASC'
      }
      try:
        async with self.session.get(f"{TWELVEDATA_BASE_URL}/time_series", params=params) as resp:
          if resp.status != 200:
            logger.warning(f"TwelveData error {resp.status} for {symbol}")
            return []
          data = await resp.json()
          if 'values' not in data: return []

          normalized = []
          for k in data['values']:
            try:
              dt = datetime.strptime(k['datetime'], "%Y-%m-%d %H:%M:%S")
              ts = dt.timestamp()
              normalized.append({
                'time': ts,
                'open': float(k['open']), 'high': float(k['high']),
                'low': float(k['low']), 'close': float(k['close']),
                'volume': float(k.get('volume', 0))
              })
            except ValueError:
                 continue
          return normalized
      except Exception:
        return []

  async def _fetch_twelve_price(self, symbol: str) -> Optional[float]:
    try:
      async with self.session.get(f"{TWELVEDATA_BASE_URL}/price", params={'symbol': symbol, 'apikey': self.api_key}) as resp:
        if resp.status == 200:
          data = await resp.json()
          return float(data['price'])
    except Exception:
      return None