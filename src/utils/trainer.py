import glob
import logging
import os
import time

from src.engine.ml_engine import NeuralPredictor
from src.config import DATA_FILE, ENTRY_MODEL_FILE

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Manages when and how the Neural Network is trained to prevent
    unnecessary processing on bot startup.
    """

    @staticmethod
    def should_train() -> bool:
        """
        Checks if training is required.
        Returns True if:
        1. Models do not exist.
        2. Training data is newer than the model (by > 1 hour).
        3. Checks ANY file in the data/ directory, not just the main csv.
        """
        if not os.path.exists(ENTRY_MODEL_FILE):
            return True

        # Check main data file
        data_files = [DATA_FILE]
        # Also check for individual symbol logs if they exist (assuming pattern like data/logs/*.csv)
        data_dir = os.path.dirname(DATA_FILE)
        if os.path.exists(data_dir):
            data_files.extend(glob.glob(os.path.join(data_dir, "*.csv")))

        model_mtime = os.path.getmtime(ENTRY_MODEL_FILE)

        needs_training = False

        for f in data_files:
            if os.path.exists(f):
                data_mtime = os.path.getmtime(f)
                # If any data file is significantly newer than the model
                if data_mtime > (model_mtime + 3600):
                    logger.info(f"⚠️ New data detected in {os.path.basename(f)}. Retraining recommended.")
                    needs_training = True
                    break

        if not needs_training and not os.path.exists(DATA_FILE):
            return False

        return needs_training

    @staticmethod
    def train_if_needed(force: bool = False):
        """
        Runs the training process only if necessary or forced.
        """
        if force or ModelTrainer.should_train():
            logger.info("🧠 Starting Neural Network Training...")
            try:
                predictor = NeuralPredictor(auto_load=False)
                predictor.train_network()
                logger.info("✅ Training Complete.")
            except Exception as e:
                logger.error(f"Training Failed: {e}")
        else:
            logger.info("⚡ Models are up to date. Skipping training.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ModelTrainer.train_if_needed(force=True)
