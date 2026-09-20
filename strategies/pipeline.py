import pandas as pd

from bot.dispatcher import broadcast_signal
from bot.formatters.signal_formatter import format_trade_signal
from config.settings import settings
from data.candle_store import candle_store
from data.normalizer import normalize_ohlcv
from data.twelve_data_client import TwelveDataClient
from strategies.confluence import ConfluenceEngine
from utils.logger import logger


async def bootstrap_historical_data(client: TwelveDataClient, symbol: str):
    """Fetches initial data for all timeframes on startup."""
    timeframes = settings.HTF_TIMEFRAMES + [settings.ENTRY_TIMEFRAME]

    for tf in timeframes:
        logger.info("bootstrapping_data", symbol=symbol, timeframe=tf)
        raw_df = await client.get_historical_ohlcv(symbol, tf, outputsize=500)
        norm_df = normalize_ohlcv(raw_df)
        await candle_store.initialize(symbol, tf, norm_df)

    # Run initial HTF and MTF scans immediately to populate DB with existing state
    engine = ConfluenceEngine(symbol)
    await engine.scan_htf()
    await engine.scan_mtf()


async def on_candle_close(symbol: str, timeframe: str, candle: pd.Series) -> None:
    """Callback fired by WebSocket tick aggregator precisely on 5-minute rollovers."""
    # 1. Add the new LTF candle to the store
    await candle_store.add_candle(symbol, timeframe, candle)

    # 2. Trigger the Confluence Engine to check for setups
    engine = ConfluenceEngine(symbol)
    signal = await engine.scan_ltf_entry()

    # 3. If a signal is generated, format and broadcast it to Telegram
    if signal:
        logger.info("signal_confirmed_broadcasting", signal_id=signal.signal_id)
        msg_html = format_trade_signal(signal)
        await broadcast_signal(msg_html)
