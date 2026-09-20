import asyncio
import signal as os_signal
import sys
from typing import List, Optional

from bot.dispatcher import start_bot
from config.settings import settings
from data.twelve_data_client import client, TwelveDataClient
from db.database import close_pool, init_pool
from scheduler.jobs import setup_scheduler
from strategies.pipeline import bootstrap_historical_data, on_candle_close
from utils.logger import logger


async def shutdown(
    sig_name: str,
    client: Optional[TwelveDataClient],
    scheduler,
    running_tasks: List[asyncio.Task],
) -> None:
    """
    Coordinates graceful termination across network streams, scheduled jobs,
    and database connections upon receiving OS termination signals.
    """
    logger.info("shutdown_signal_received", signal=sig_name)

    # 1. Sever TwelveData WebSocket connection
    if client:
        client.is_streaming = False
        try:
            await client.stop_websocket_stream()
            logger.info("websocket_stream_stopped")
        except Exception as e:
            logger.error("error_stopping_websocket", error=str(e))

    # 2. Halt APScheduler background jobs
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("scheduler_shutdown_complete")

    # 3. Cancel active concurrent background tasks (WebSocket + Telegram Bot)
    for task in running_tasks:
        if task and not task.done():
            task.cancel()

    if running_tasks:
        await asyncio.gather(*running_tasks, return_exceptions=True)
        logger.info("background_tasks_cancelled")

    # 4. Drain PostgreSQL asyncpg connection pool
    try:
        await close_pool()
        logger.info("database_pool_closed")
    except Exception as e:
        logger.error("error_closing_database_pool", error=str(e))

    logger.info("graceful_shutdown_complete")


async def main() -> None:
    logger.info("starting_nexubot_production", symbols=settings.SYMBOLS)
    symbol = settings.SYMBOLS[0]
    scheduler = None
    tasks: List[asyncio.Task] = []
    stop_event = asyncio.Event()

    # Setup cross-platform signal listeners for Railway container lifecycles
    loop = asyncio.get_running_loop()
    for sig in (os_signal.SIGTERM, os_signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: stop_event.set())
        except NotImplementedError:
            # Fallback for development environments without add_signal_handler support
            os_signal.signal(sig, lambda *_: stop_event.set())

    try:
        # 1. Initialize PostgreSQL Connection Pool
        await init_pool()

        # 2. Bootstrap Historical Candles & Strategy State
        await bootstrap_historical_data(client, symbol)

        # 3. Start APScheduler for periodic HTF/MTF scans
        scheduler = setup_scheduler(client)
        scheduler.start()
        logger.info("scheduler_started")

        # 4. Dispatch concurrent runtime tasks
        ws_task = asyncio.create_task(
            client.start_websocket_stream(settings.SYMBOLS, on_candle_close),
            name="TwelveData_WebSocket",
        )
        bot_task = asyncio.create_task(
            start_bot(),
            name="Telegram_Bot_Polling",
        )

        # 5. Monitor tasks and termination signal concurrently
        stop_waiter = asyncio.create_task(stop_event.wait(), name="Termination_Signal_Watcher")
        done, _ = await asyncio.wait(
            [ws_task, bot_task, stop_waiter],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Check if an internal process crashed prematurely before a termination signal
        for completed_task in done:
            if completed_task is not stop_waiter:
                exc = completed_task.exception()
                if exc:
                    logger.error(
                        "fatal_runtime_error",
                        task_name=completed_task.get_name(),
                        error=str(exc),
                    )
                    await shutdown("CRASH", client, scheduler, tasks)
                    sys.exit(1)

        # Clean shutdown initiated by SIGTERM or SIGINT
        await shutdown("SIGTERM/SIGINT", client, scheduler, tasks)
        sys.exit(0)

    except Exception as e:
        logger.error("startup_fatal_error", error=str(e))
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
        await close_pool()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("shutdown_requested_by_user")
