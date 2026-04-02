import csv
import logging
import os
import pandas as pd

from src.config import FEATURE_COLS

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects feature data and trade results for future ML training.
    """

    def __init__(self, filename="training_data.csv"):
        self.filename = filename
        self.max_rows = 20000

    def log_training_data(self, symbol: str, features: dict, won: int, pnl: float, excursion: float = 0.0):
        """
        Logs the strict SMC feature set for ML training.
        """
        file_exists = os.path.isfile(self.filename)
        headers = ["symbol"] + FEATURE_COLS + ["target_win", "pnl", "target_excursion"]

        row = {"symbol": symbol}
        for col in FEATURE_COLS:
            row[col] = round(float(features.get(col, 0.0)), 4)

        row["target_win"] = won
        row["pnl"] = round(pnl, 2)
        row["target_excursion"] = round(excursion, 4)

        try:
            with open(self.filename, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

            if file_exists:
                df = pd.read_csv(self.filename, on_bad_lines="skip")
                if len(df) > self.max_rows:
                    # Keep the most recent 20,000 trades
                    df.tail(self.max_rows).to_csv(self.filename, index=False)
                    logger.info(f"🧹 Training data pruned to latest {self.max_rows} rows.")
        except Exception as e:
            logger.error(f"Failed to log training data for {symbol}: {e}")
