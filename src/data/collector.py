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
        self._last_mtime = 0
        self._is_pruning = False

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

        # Clip severe outliers to prevent scaler corruption
        dist_to_pdh_atr = float(np.clip(f.get("dist_to_pdh_atr", 0.0), 0.0, 30.0))
        dist_to_pdl_atr = float(np.clip(f.get("dist_to_pdl_atr", 0.0), 0.0, 30.0))
        vwap_dist = float(f.get("vwap_distance_atr", 0.0))
        vol_trend = float(f.get("volume_trend_3", 0.0))
        vwap_dist_clipped = float(np.clip(vwap_dist, -5.0, 5.0))
        vol_trend_clipped = float(np.clip(vol_trend, -3.0, 3.0))

        # Base calculations used across multiple terms
        pd_deviation = round(abs(f.get("pd_array_status", 0.5) - 0.5), 4)
        structure_age = float(f.get("structure_age_bars", 0.0))

        return {
            "pd_deviation_from_equilibrium": pd_deviation,
            "dist_to_pdh_atr": round(dist_to_pdh_atr, 4),
            "dist_to_pdl_atr": round(dist_to_pdl_atr, 4),
            "dist_to_asia_extremes_atr": round(f.get("dist_to_asia_extremes_atr", 0.0), 4),
            "vwap_distance_atr": round(vwap_dist_clipped, 4),
            "is_liquidity_swept_tier": float(f.get("is_liquidity_swept_tier", 0.0)),
            "sweep_depth_atr": float(f.get("sweep_depth_atr", 0.0)),
            "body_ratio": round(f.get("body_ratio", 0.0), 4),
            "favor_wick_pct": round(f.get("favor_wick_pct", 0.0), 4),
            "adverse_wick_pct": round(f.get("adverse_wick_pct", 0.0), 4),
            "vol_ratio": round(f.get("vol_ratio", 0.0), 4),
            "volume_trend_3": round(vol_trend_clipped, 4),
            "atr_expansion_ratio": round(f.get("atr_expansion_ratio", 1.0), 4),
            "rr_at_entry": round(f.get("rr_at_entry", 0.0), 4),
            "hour_sin": round(f.get("hour_sin", 0.0), 4),
            "hour_cos": round(f.get("hour_cos", 0.0), 4),
            "structure_age_bars": structure_age,
            "zone_age_bars": round(float(np.log1p(f.get("zone_age_bars", 0.0))), 4),
            "interaction_structure_pd": round(structure_age * pd_deviation, 4),
        }

    def _prune_data_background(self):
        """Background thread function to prune the training data CSV to the latest max_rows entries when it exceeds the threshold."""
        try:
            # Enforce lock on file read operation to circumvent threading vs async race conditions
            with self._lock:
                df = pd.read_csv(self.filename, on_bad_lines="skip")
                temp_file = self.filename + ".tmp"
                df.tail(self.max_rows).to_csv(temp_file, index=False)
                os.replace(temp_file, self.filename)
                self._current_row_count = self.max_rows

            logger.info(f"🧹 Training data pruned to latest {self.max_rows} rows.")
        except Exception as e:
            logger.error(f"Prune background error: {e}")
        finally:
            self._is_pruning = False

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
                current_mtime = os.path.getmtime(self.filename) if file_exists else 0

                if self._current_row_count is None or current_mtime > self._last_mtime + 2:
                    self._current_row_count = self._get_row_count()

                with open(self.filename, mode="a", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)

                self._current_row_count += 1
                self._last_mtime = os.path.getmtime(self.filename)

                if self._current_row_count > self.max_rows + 500 and not self._is_pruning:
                    self._is_pruning = True
                    threading.Thread(target=self._prune_data_background, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to log training data for {symbol}: {e}")
