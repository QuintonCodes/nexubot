import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data.provider import DataProvider
from src.core.engine import NexubotEngine
from src.utils.backfill import backfill_data
from src.utils.logger import setup_logging
from src.config import CALIBRATOR_FILE, ENTRY_MODEL_FILE, EXIT_MODEL_FILE, EXIT_SCALER_FILE, SCALER_FILE, TRAINING_FILE

load_dotenv()
setup_logging("backfill.log")


def wipe_legacy_ml_data() -> None:
    """
    Wipes old ML artifacts to prevent TensorFlow shape mismatches
    now that the engine has transitioned to Pure SMC features.
    """
    files_to_delete = [CALIBRATOR_FILE, ENTRY_MODEL_FILE, EXIT_MODEL_FILE, EXIT_SCALER_FILE, SCALER_FILE, TRAINING_FILE]
    cleaned = False

    print("🧹 Checking for legacy ML artifacts...")
    for file in files_to_delete:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"   🗑️ Deleted: {file}")
                cleaned = True
            except Exception as e:
                print(f"   ❌ Failed to delete {file}: {e}")

    if cleaned:
        print("✅ Legacy ML data wiped. Ready for Pure SMC Backfill.\n")
    else:
        print("✅ Environment is clean. No legacy data found.\n")


async def main() -> None:
    print("🚀 Initiating Pure SMC Backfill Engine...")
    wipe_legacy_ml_data()

    provider = DataProvider()
    await provider.initialize()

    engine = NexubotEngine(None)
    engine.provider = provider

    await backfill_data(provider)

    print("\n🛑 Shutting down provider...")
    await provider.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBackfill manually interrupted.")
