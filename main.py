import asyncio
import sys
import os
import logging
from dotenv import load_dotenv

# Ensure the src directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the Bot class from the refactored structure
from src.bot.console import NexubotConsole
from src.utils.logger import setup_logging

# Load environment variables from .env file
load_dotenv()

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

async def main():
    """Main function to run the Nexubot application"""
    bot = None

    try:
        # Initialize the bot
        bot = NexubotConsole()

        # Start the bot
        await bot.start()

    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")

    except Exception as e:
        logger.exception(f"Fatal error in main loop: {e}")
        print(f"❌ Fatal Error: {e}")

    finally:
        if bot:
            await bot.stop()
            print("✅ Nexubot stopped successfully!")

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        pass