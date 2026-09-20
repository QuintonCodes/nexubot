"""
Nexubot - Full Cloud Production Entry Point.
Initializes Database, loads initial data, starts background schedulers,
connects to WebSocket, and begins Telegram polling
"""

import asyncio
import sys
import pandas as pd

from config.settings import settings
from utils.logger import logger
from data.twelve_data_client import TwelveDataClient
from data.candle_store import candle_store
from data.normalizer import normalize_ohlcv
from db.database import init_pool, close_pool
from strategies.confluence import ConfluenceEngine
from scheduler.jobs import setup_scheduler
from bot.dispatcher import start_bot, broadcast_signal
from bot.formatters.signal_formatter import format_trade_signal


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


async def main() -> None:
    logger.info("starting_nexubot_production", symbols=settings.SYMBOLS)
    client = TwelveDataClient()
    symbol = settings.SYMBOLS[0]

    try:
        # 1. Initialize PostgreSQL Connection Pool
        await init_pool()

        # 2. Bootstrap Market Data
        await bootstrap_historical_data(client, symbol)

        # 3. Start APScheduler (Background HTF/MTF Refreshes)
        scheduler = setup_scheduler(client)
        scheduler.start()
        logger.info("scheduler_started")

        # 4. Start concurrent runtime tasks (Telegram + WebSocket)
        ws_task = asyncio.create_task(client.start_websocket_stream(settings.SYMBOLS, on_candle_close))
        bot_task = asyncio.create_task(start_bot())

        # Keep the event loop running
        await asyncio.gather(ws_task, bot_task)

    except Exception as e:
        logger.error("fatal_runtime_error", error=str(e))
    finally:
        await close_pool()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("shutdown_requested_by_user")
        # Python's asyncio handles the graceful cleanup of tasks on interrupt
