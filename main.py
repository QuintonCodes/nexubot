import asyncio
import sys
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import the Bot class from the refactored structure
from src.bot.console import NexubotConsole
from src.utils.logger import setup_logging
from src.utils.backfill import backfill_data
from src.config import DATA_FILE

# Initialize logging
logger = setup_logging()


async def check_and_run_backfill():
    """
    Checks if training data exists. If not, runs backfill automatically.
    """
    needs_backfill = False

    if not os.path.exists(DATA_FILE):
        needs_backfill = True
    else:
        # Check if file is empty or too small (just header)
        try:
            with open(DATA_FILE, "r") as f:
                lines = f.readlines()
                if len(lines) < 50:
                    needs_backfill = True
        except Exception:
            needs_backfill = True

    if needs_backfill:
        print("\n⚠️ No training data found (or file is empty).")
        print("⏳ Starting Automatic Backfill to populate ML Data...")
        try:
            await backfill_data()
            print("✅ Backfill Complete! Starting Bot...")
        except Exception as e:
            print(f"❌ Backfill Failed: {e}")
            logger.error(f"Backfill failed: {e}")


async def main():
    """Main function to run the Nexubot application"""
    # 1. Auto-Backfill Check
    await check_and_run_backfill()

    # 2. Start Bot
    bot = None
    try:
        bot = NexubotConsole()
        await bot.start()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        print("\n👋 User requested shutdown.")
    except asyncio.CancelledError:
        logger.info("\nTasks cancelled. Shutting down...")
    except Exception as e:
        logger.exception(f"Fatal error in main loop: {e}")
        print(f"❌ Fatal Error: {e}")

    finally:
        if bot:
            await bot.stop()
            print("✅ Nexubot stopped successfully!")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch the final interrupt if it bubbles up from asyncio.run
        pass
