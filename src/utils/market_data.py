import aiohttp
import logging
from typing import List
from src.config import BINANCE_API_URL, MIN_VOLUME_USDT, TOP_SYMBOLS_COUNT

logger = logging.getLogger(__name__)

async def fetch_top_volatile_symbols(session: aiohttp.ClientSession) -> List[str]:
    """
    Fetches the top USDT trading pairs by 24h volume and volatility.
    Parses specific Binance fields: symbol, quoteVolume, highPrice, lowPrice
    """
    try:
        url = f"{BINANCE_API_URL}/ticker/24hr"
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Binance API Error: {response.status}")
                return []

            data = await response.json()

            # Filter for USDT pairs, exclude stablecoins/leverage tokens
            usdt_pairs = [
                ticker for ticker in data
                if ticker['symbol'].endswith('USDT')
                and 'UP' not in ticker['symbol']
                and 'DOWN' not in ticker['symbol']
                and ticker['symbol'] not in ['USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'USDPUSDT']
            ]

            # Filter by minimum volume (Liquidity check)
            valid_pairs = [
                p for p in usdt_pairs
                if float(p['quoteVolume']) > MIN_VOLUME_USDT
            ]

            if not valid_pairs:
                logger.warning("No pairs met volume criteria.")
                return []

            # Sort by High-Low percentage (Volatility) - Best for Binary Options
            # Formula: (High - Low) / Low
            valid_pairs.sort(
                key=lambda x: (float(x['highPrice']) - float(x['lowPrice'])) / float(x['lowPrice']),
                reverse=True
            )

            # Get top N symbols
            top_symbols = [p['symbol'] for p in valid_pairs[:TOP_SYMBOLS_COUNT]]

            logger.info(f"🌊 Market Scan: Found {len(top_symbols)} high-volatility pairs")
            return top_symbols

    except Exception as e:
        logger.error(f"Error fetching top symbols: {str(e)}")
        return []