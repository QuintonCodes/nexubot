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
from src.config import __version__

load_dotenv()
setup_logging()


async def main() -> None:
    print(f"🚀 Booting Nexubot {__version__} (Pure SMC Engine)...")
    engine = NexubotEngine(None)

    await engine.db.init_database()
    await engine.db.cleanup_db()

    notifier = TelegramNotifier(engine=engine)
    engine.notifier = notifier

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH", r"C:\Program Files\Metatrader 5\terminal64.exe")

    if not all([login, password, server]):
        notifier.send_message("❌ *Boot Error:* Missing MT5 credentials in .env file.")
        print("❌ Boot Error: Missing MT5 credentials in .env file.")
        return

    # 1. Lock the system status BEFORE connecting so the scanner pauses immediately
    engine.system_status = "TRAINING"

    is_connected = await engine.initialize_connection(login, server, password, path)

    if is_connected:
        tg_ready = await notifier.initialize()
        if not tg_ready:
            print("⚠️ Running without Telegram alerts. Check Bot Token or Connection.")

        # 2. Send Startup message FIRST so Telegram gets the notification immediately
        win_rate = await engine.db.get_total_historical_win_rate()
        recent_trades = await engine.db.get_recent_trades(limit=1000)
        total_trades = len(recent_trades)
        await notifier.send_startup_message(win_rate, total_trades)

        # 3. Auto-Train Neural Network while Scanner is paused
        if not os.path.exists("training_data.csv"):
            print("⚠️ WARNING: 'training_data.csv' not found. Run 'python run_backfill.py' first.")
            notifier.send_message("⚠️ *Warning:* No ML training data found. Please run backfill.")
        else:
            print("⚙️ Running Pre-Flight ML Optimization...")
            notifier.send_message("⚙️ *System Note:* Neural Network Training initiated. Scanning will resume shortly.")
            try:
                await asyncio.wait_for(asyncio.to_thread(engine.ai_engine.nn_brain.train_network), timeout=300.0)
            except asyncio.TimeoutError:
                print("⚠️ ML Training timed out. Proceeding with existing model.")
                notifier.send_message("⚠️ ML Training timed out. Running with prior model.")

        # 4. Unlock the system status to IDLE -> Scanner can now fire
        engine.system_status = "IDLE"
        print("✅ Training Complete. Scanner Activated.")

        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            print("Received Cancellation Signal.")
        finally:
            print("Initiating Graceful Shutdown...")
            stats = engine.session_stats

            await notifier.send_daily_report(
                stats["wins"], stats["losses"], stats["total"], stats["pnl"], stats.get("currency", "USD")
            )
            await notifier.send_shutdown_message()

            await engine.stop_session()

            # Catch all pending fire-and-forget tasks
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if pending:
                print(f"Waiting for {len(pending)} background tasks to complete delivery...")
                await asyncio.gather(*pending, return_exceptions=True)
    else:
        print(f"❌ *MT5 Connection Failed.* Check console logs.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Nexubot Shutdown Complete.")
