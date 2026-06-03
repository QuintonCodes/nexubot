import csv
import logging
import numpy as np
import os
import pandas as pd
import threading

from src.config import FEATURE_COLS, MAX_ROWS, TRAINING_FILE

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects and manages training data for the neural network models."""

    _lock = threading.Lock()

    def __init__(self, filename=TRAINING_FILE):
        self.filename = filename
        self.max_rows = MAX_ROWS
        self._current_row_count = None

    def _get_row_count(self) -> int:
        """Efficiently counts rows without loading the entire CSV into memory."""
        if not os.path.isfile(self.filename):
            return 0
        with open(self.filename, "rb") as f:
            return sum(1 for _ in f) - 1

    @staticmethod
    def engineer_features(raw_features: dict) -> dict:
        """Transforms raw SMC features into the full engineered feature set."""
        f = raw_features

        # Session mappings logic
        active_killzone = int(f.get("active_killzone", 0))
        SESSION_SCORE_MAP = {0: 1.0, 1: 0.9, 2: 0.6, 3: 0.55}
        session_quality_score = SESSION_SCORE_MAP.get(active_killzone, 1.0)
        pd_array_status = f.get("pd_array_status", 0.5)
        raw_distance = max(0.0, float(f.get("distance_to_poi", 0.0)))

        return {
            "log_distance_to_poi": round(float(np.log1p(raw_distance)), 4),
            "is_htf_aligned": float(f.get("is_htf_aligned", 0.0)),
            "is_liquidity_swept_tier": float(f.get("is_liquidity_swept_tier", 0.0)),
            "sweep_depth_atr": float(f.get("sweep_depth_atr", 0.0)),
            "pd_deviation_from_equilibrium": round(abs(pd_array_status - 0.5), 4),
            "zone_overlap_count": float(f.get("zone_overlap_count", 0.0)),
            "body_ratio": round(f.get("body_ratio", 0.0), 4),
            "vol_ratio": round(f.get("vol_ratio", 0.0), 4),
            "momentum_exhaustion_count": float(f.get("momentum_exhaustion_count", 0.0)),
            "dist_to_pdh_atr": round(f.get("dist_to_pdh_atr", 0.0), 4),
            "dist_to_pdl_atr": round(f.get("dist_to_pdl_atr", 0.0), 4),
            "sweep_snapback_vel": round(f.get("sweep_snapback_vel", 0.0), 4),
            "dist_to_asia_extremes_atr": round(f.get("dist_to_asia_extremes_atr", 0.0), 4),
            "hour_sin": round(f.get("hour_sin", 0.0), 4),
            "hour_cos": round(f.get("hour_cos", 0.0), 4),
            "mins_since_kz_open": float(f.get("mins_since_kz_open", 0.0)),
            "sweep_aligned": float(f.get("sweep_aligned", 0.0)),
            "poi_vol_anomaly": round(f.get("poi_vol_anomaly", 0.0), 4),
            "session_quality_score": session_quality_score,
        }

    def log_training_data(
        self, symbol: str, strategy: str, features: dict, won: int, pnl: float, excursion: float = 0.0
    ) -> None:
        """Logs the features and trade outcome to a CSV file for future training."""
        headers = ["symbol", "strategy"] + FEATURE_COLS + ["target_win", "pnl", "target_excursion"]

        row = {"symbol": symbol, "strategy": strategy}
        for col in FEATURE_COLS:
            row[col] = round(float(features.get(col, 0.0)), 4)

        row["target_win"] = won
        row["pnl"] = round(pnl, 2)
        row["target_excursion"] = round(excursion, 4)

        try:
            with self._lock:
                file_exists = os.path.isfile(self.filename)

                if self._current_row_count is None:
                    self._current_row_count = self._get_row_count()

                with open(self.filename, mode="a", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)

                self._current_row_count += 1

                # Only run the heavy Pandas prune operation when over maximum + buffer
                if self._current_row_count > self.max_rows + 500:
                    df = pd.read_csv(self.filename, on_bad_lines="skip")

                    temp_file = self.filename + ".tmp"
                    df.tail(self.max_rows).to_csv(temp_file, index=False)
                    os.replace(temp_file, self.filename)

                    self._current_row_count = self.max_rows
                    logger.info(f"🧹 Training data pruned to latest {self.max_rows} rows.")
        except Exception as e:
            logger.error(f"Failed to log training data for {symbol}: {e}")
