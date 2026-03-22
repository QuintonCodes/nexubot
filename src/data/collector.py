import csv
import logging
import os

from src.config import FEATURE_COLS

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects feature data and trade results for future ML training.
    """

    def __init__(self, filename="training_data.csv"):
        self.filename = filename

    def log_training_data(self, symbol: str, features: dict, won: int, pnl: float, excursion: float = 0.0):
        """
        Logs the strict SMC feature set for ML training.
        """
        file_exists = os.path.isfile(self.filename)

        # We only want the specific features + targets
        headers = ["symbol"] + FEATURE_COLS + ["target_win", "pnl", "target_excursion"]

        row = {"symbol": symbol}
        for col in FEATURE_COLS:
            row[col] = features.get(col, 0.0)

        row["target_win"] = won
        row["pnl"] = pnl
        row["target_excursion"] = excursion

        try:
            with open(self.filename, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to log training data for {symbol}: {e}")
