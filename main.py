import asyncio
import os
import sys
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.object.*")
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.api.telegram_notifier import TelegramNotifier
from src.core.engine import NexubotEngine
from src.utils.logger import setup_logging

load_dotenv()
setup_logging()


async def run_daily_reporter(engine, notifier):
    """Background task to send a report every 24 hours."""
    while engine.is_running:
        await asyncio.sleep(86400)  # Wait 24 hours
        stats = engine.session_stats
        notifier.send_daily_report(stats["wins"], stats["losses"], stats["total"], stats["pnl"])


async def main():
    engine = NexubotEngine(None)
    notifier = TelegramNotifier(engine=engine)
    engine.notifier = notifier

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH", r"C:\Program Files\Metatrader 5\terminal64.exe")

    if not all([login, password, server]):
        notifier.send_message("❌ *Boot Error:* Missing MT5 credentials in .env file.")
        return

    # Initialize Headless Connection
    is_connected = await engine.initialize_connection(login, server, password, path)

    if is_connected:
        # Start the Telegram async polling loop
        await notifier.initialize()

        # 2. Auto-Train Neural Network on Boot
        notifier.send_message("⚙️ *Running Pre-Flight ML Optimization...*")
        # Run training in background thread to avoid blocking asyncio loop
        await asyncio.to_thread(engine.ai_engine.nn_brain.train_network)

        # 3. Fetch Initial DB Stats for Startup Message
        win_rate = await engine.db.get_total_historical_win_rate()
        # Rough estimation of total trades from the recent log count if available, otherwise default
        recent_trades = await engine.db.get_recent_trades(limit=1000)
        total_trades = len(recent_trades)

        await notifier.send_startup_message(win_rate, total_trades)

        asyncio.create_task(run_daily_reporter(engine, notifier))

        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            print("Received Cancellation Signal.")
        finally:
            # 🔴 Execute Graceful Shutdown
            print("Initiating Graceful Shutdown...")
            await notifier.send_shutdown_message()
            await engine.stop_session()
            # wait briefly to ensure telegram message dispatches before process dies
            await asyncio.sleep(2)
    else:
        print(f"❌ *MT5 Connection Failed.* Check console logs.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Nexubot Shutdown Complete.")
