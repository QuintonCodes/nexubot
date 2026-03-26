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
    while True:
        await asyncio.sleep(86400)  # Wait 24 hours
        stats = engine.session_stats
        notifier.send_daily_report(stats["wins"], stats["losses"], stats["total"], stats["pnl"])


async def main():
    notifier = TelegramNotifier()
    notifier.send_message("🚀 *Nexubot Boot Sequence Initiated...*")

    # Pass notifier to engine
    engine = NexubotEngine(notifier)

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
        notifier.send_message(f"✅ *Connected to MT5 Server:* {server}")

        # Start the daily reporting loop in the background
        asyncio.create_task(run_daily_reporter(engine, notifier))

        # Keep the main thread alive indefinitely
        while True:
            await asyncio.sleep(3600)
    else:
        notifier.send_message(f"❌ *MT5 Connection Failed.* Check console logs.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Nexubot Shutting Down...")
