import aiohttp
import logging
from typing import List
from src.config import BINANCE_API_URL, MIN_VOLUME_USDT, TOP_SYMBOLS_COUNT, USD_ZAR_RATE

logger = logging.getLogger(__name__)

async def fetch_top_volatile_symbols(session: aiohttp.ClientSession) -> List[str]:
    """
    Fetches the top USDT trading pairs by 24h volume and volatility.

    Args:
        session: The active aiohttp session.

    Returns:
        List[str]: A list of symbol strings (e.g. ['BTCUSDT', 'ETHUSDT'])
    """
    try:
        url = f"{BINANCE_API_URL}/ticker/24hr"
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Binance API Error: {response.status}")
                return []

            data = await response.json()

            # Filter for Liquid USDT Pairs (Excluding Stablecoins)
            valid_pairs = []
            for ticker in data:
                symbol = ticker['symbol']
                # Check format and exclusion list
                if not symbol.endswith('USDT'): continue
                if any(ex in symbol for ex in ['UP', 'DOWN', 'USDC', 'FDUSD', 'TUSD']): continue

                # Check Volume (Converted to Float)
                quote_vol = float(ticker['quoteVolume'])
                if quote_vol < MIN_VOLUME_USDT: continue

                valid_pairs.append(ticker)

            if not valid_pairs:
                logger.warning("No pairs met liquidity criteria.")
                return []

            # Sort by Volatility (High - Low % Diff)
            # We want pairs that are moving, not flat-lining.
            valid_pairs.sort(
                key=lambda x: (float(x['highPrice']) - float(x['lowPrice'])) / float(x['lowPrice']),
                reverse=True
            )

            # Select Top N
            top_tickers = valid_pairs[:TOP_SYMBOLS_COUNT]
            top_symbols = [p['symbol'] for p in top_tickers]

            # Log the stats for the #1 pair
            top_vol_zar = float(top_tickers[0]['quoteVolume']) * USD_ZAR_RATE
            logger.info(f"🌊 Market Scan: Found {len(top_symbols)} pairs. Top Vol: R{top_vol_zar/1000000:.0f}M ({top_symbols[0]})")

            return top_symbols

    except Exception as e:
        logger.error(f"Error fetching top symbols: {str(e)}")
        return []