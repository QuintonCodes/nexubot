import logging
import os
import pandas as pd
from datetime import datetime

from src.config import DATA_FILE

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects feature data and trade results for future ML training.
    """

    @staticmethod
    def log_training_data(symbol: str, features: dict, result: int, pnl: float, excursion: float = 0.0):
        """
        Appends a row of training data to the CSV.

        Args:
            symbol: The pair traded.
            features: Dictionary of indicator values.
            result: 1 (Win) or 0 (Loss).
            pnl: Profit/Loss in ZAR.
            excursion: Max favorable ATR multiple (default 0.0 for live logs).
        """
        try:
            # Prepare the data row
            data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                **features,  # Unpack indicator values
                "target_win": result,
                "target_pnl": pnl,
                "target_excursion": excursion,
            }

            df = pd.DataFrame([data])

            # Append to CSV (create header if file doesn't exist)
            if not os.path.exists(DATA_FILE):
                df.to_csv(DATA_FILE, index=False, mode="w")
            else:
                df.to_csv(DATA_FILE, index=False, mode="a", header=False)

            logger.debug(f"💾 Training data logged for {symbol}")

        except Exception as e:
            logger.error(f"Failed to log training data: {e}")
