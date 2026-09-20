"""
APScheduler Jobs for Nexubot.
Handles background REST API refreshes for HTF and MTF analysis.
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import settings
from data.twelve_data_client import TwelveDataClient
from data.candle_store import candle_store
from data.normalizer import normalize_ohlcv
from strategies.confluence import ConfluenceEngine
from utils.logger import logger


async def refresh_htf_data(client: TwelveDataClient, symbol: str):
    """Refreshes the 4H Data and recalculates Macro Bias."""
    logger.info("scheduler_running_htf_refresh")
    try:
        raw_df = await client.get_historical_ohlcv(symbol, settings.HTF_TIMEFRAMES[0], outputsize=200)
        norm_df = normalize_ohlcv(raw_df)
        await candle_store.initialize(symbol, settings.HTF_TIMEFRAMES[0], norm_df)

        engine = ConfluenceEngine(symbol)
        await engine.scan_htf()
    except Exception as e:
        logger.error("htf_refresh_failed", error=str(e))


async def refresh_mtf_data(client: TwelveDataClient, symbol: str):
    """Refreshes the 1H Data and scans for new Order Blocks."""
    logger.info("scheduler_running_mtf_refresh")
    try:
        raw_df = await client.get_historical_ohlcv(symbol, settings.HTF_TIMEFRAMES[1], outputsize=200)
        norm_df = normalize_ohlcv(raw_df)
        await candle_store.initialize(symbol, settings.HTF_TIMEFRAMES[1], norm_df)

        engine = ConfluenceEngine(symbol)
        await engine.scan_mtf()
    except Exception as e:
        logger.error("mtf_refresh_failed", error=str(e))


async def refresh_mtf_15m_data(client: TwelveDataClient, symbol: str):
    """Refreshes the 15M Data to confirm intermediate structure sweeps."""
    logger.info("scheduler_running_15m_refresh")
    try:
        tf = "15min"
        raw_df = await client.get_historical_ohlcv(symbol, tf, outputsize=200)
        norm_df = normalize_ohlcv(raw_df)
        await candle_store.initialize(symbol, tf, norm_df)

        engine = ConfluenceEngine(symbol)
        await engine.scan_mtf_confirmation()
    except Exception as e:
        logger.error("mtf_15m_refresh_failed", error=str(e))


def setup_scheduler(client: TwelveDataClient) -> AsyncIOScheduler:
    """Configures and returns the AsyncIOScheduler."""
    scheduler = AsyncIOScheduler()
    symbol = settings.SYMBOLS[0]

    # HTF Refresh: Every 4 hours
    scheduler.add_job(
        refresh_htf_data,
        trigger=IntervalTrigger(hours=4),
        args=[client, symbol],
        id="htf_refresh",
        replace_existing=True,
    )

    # MTF Refresh: Every 1 hour at the top of the hour
    scheduler.add_job(
        refresh_mtf_data,
        trigger=CronTrigger(minute=1),  # 1 minute past the hour to allow TwelveData to finalize the candle
        args=[client, symbol],
        id="mtf_refresh",
        replace_existing=True,
    )

    # 15M Refresh: Every 15 minutes at minutes 1, 16, 31, 46
    scheduler.add_job(
        refresh_mtf_15m_data,
        trigger=CronTrigger(minute="1,16,31,46"),
        args=[client, symbol],
        id="mtf_15m_refresh",
        replace_existing=True,
    )

    return scheduler
