import os
import logging

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
        """
        if not os.path.exists(ENTRY_MODEL_FILE):
            return True

        if not os.path.exists(DATA_FILE):
            return False  # No data to train on

        # Check timestamps
        data_mtime = os.path.getmtime(DATA_FILE)
        model_mtime = os.path.getmtime(ENTRY_MODEL_FILE)

        # If data is significantly newer than the model (more than 1 hour diff)
        if data_mtime > (model_mtime + 3600):
            logger.info("⚠️ Training data is newer than current model. Retraining recommended.")
            return True

        return False

    @staticmethod
    def train_if_needed(force: bool = False):
        """
        Runs the training process only if necessary or forced.
        """
        if force or ModelTrainer.should_train():
            logger.info("🧠 Starting Neural Network Training...")
            predictor = NeuralPredictor(auto_load=False)
            predictor.train_network()
            logger.info("✅ Training Complete.")
        else:
            logger.info("⚡ Models are up to date. Skipping training.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ModelTrainer.train_if_needed(force=True)
