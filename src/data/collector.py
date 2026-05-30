import csv
import logging
import os
import pandas as pd

from src.config import FEATURE_COLS, MAX_ROWS, TRAINING_FILE

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects and manages training data for the neural network models."""

    def __init__(self, filename=TRAINING_FILE):
        self.filename = filename
        self.max_rows = MAX_ROWS
        self._current_row_count = None

    def _get_row_count(self) -> int:
        """Efficiently counts rows without loading the entire CSV into memory."""
        if not os.path.isfile(self.filename):
            return 0
        with open(self.filename, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # Subtract 1 for the header

    def log_training_data(self, symbol: str, features: dict, won: int, pnl: float, excursion: float = 0.0) -> None:
        """Logs the features and trade outcome to a CSV file for future training."""
        file_exists = os.path.isfile(self.filename)

        # Initialize counter once on boot
        if self._current_row_count is None:
            self._current_row_count = self._get_row_count()

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

            self._current_row_count += 1

            # Only run the heavy Pandas prune operation when over maximum + buffer
            if self._current_row_count > self.max_rows + 500:
                df = pd.read_csv(self.filename, on_bad_lines="skip")
                df.tail(self.max_rows).to_csv(self.filename, index=False)
                self._current_row_count = self.max_rows
                logger.info(f"🧹 Training data pruned to latest {self.max_rows} rows.")
        except Exception as e:
            logger.error(f"Failed to log training data for {symbol}: {e}")
