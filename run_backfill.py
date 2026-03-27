import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data.provider import DataProvider
from src.core.engine import NexubotEngine
from src.utils.backfill import backfill_data

load_dotenv()


async def main():
    print("Initiating Backfill Engine...")
    provider = DataProvider()
    await provider.initialize()

    engine = NexubotEngine(None)
    engine.provider = provider
    engine.strategy_analyzer = engine.ai_engine.strategy_analyzer

    await backfill_data(provider, engine)


if __name__ == "__main__":
    asyncio.run(main())
