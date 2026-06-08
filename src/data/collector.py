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

        # Transform structurally-aged parameters to distribute variance dynamically.
        structure_age_bars_raw = float(f.get("structure_age_bars", 0.0))
        structure_age = round(float(np.log1p(structure_age_bars_raw)), 4)
        pd_deviation = round(abs(f.get("pd_array_status", 0.5) - 0.5), 4)

        # Fix right-skew and inverse monotonic signal on RR parameter.
        rr_raw = float(f.get("rr_at_entry", 2.0))
        rr_clipped = np.clip(rr_raw, 2.0, 8.0)
        rr_normalized = round((rr_clipped - 2.0) / 6.0, 4)

        # Clip Expansion Ratio
        atr_exp = round(float(np.clip(f.get("atr_expansion_ratio", 1.0), 0.3, 3.0)), 4)

        # Add Companion binary state for missing logic to distinguish from real 0
        zone_age = round(float(np.log1p(f.get("zone_age_bars", 0.0))), 4) if f.get("has_active_zone", 0.0) else 0.0

        return {
            "is_liquidity_swept_tier": float(f.get("is_liquidity_swept_tier", 0.0)),
            "body_ratio": round(float(f.get("body_ratio", 0.0)), 4),
            "favor_wick_pct": round(float(f.get("favor_wick_pct", 0.0)), 4),
            "adverse_wick_pct": round(float(f.get("adverse_wick_pct", 0.0)), 4),
            "vol_ratio": round(float(f.get("vol_ratio", 1.0)), 4),
            "atr_expansion_ratio": atr_exp,
            "rr_at_entry": rr_normalized,
            "hour_sin": round(float(f.get("hour_sin", 0.0)), 4),
            "hour_cos": round(float(f.get("hour_cos", 0.0)), 4),
            "structure_age_bars": structure_age,
            "zone_age_bars": zone_age,
            "interaction_structure_pd": round(structure_age * pd_deviation, 4),
            "vwap_distance_atr": round(float(np.clip(f.get("vwap_distance_atr", 0.0), -5.0, 5.0)), 4),
            "candle_momentum_score": round(float(f.get("candle_momentum_score", 0.0)), 4),
            "fib_ote_zone_score": round(float(f.get("fib_ote_zone_score", 0.0)), 4),
            "dist_to_nearest_daily_level_atr": round(float(f.get("dist_to_nearest_daily_level_atr", 0.0)), 4),
            "daily_level_side": float(f.get("daily_level_side", 0.0)),
            "bos_displacement_quality": round(float(f.get("bos_displacement_quality", 0.0)), 4),
            "price_vs_session_open_atr": round(float(f.get("price_vs_session_open_atr", 0.0)), 4),
            "ob_freshness_score": round(float(f.get("ob_freshness_score", 0.0)), 4),
            "has_relevant_ob": float(f.get("has_relevant_ob", 0.0)),
            "is_near_asian_extreme": float(f.get("is_near_asian_extreme", 0.0)),
            "consecutive_directional_closes": round(float(f.get("consecutive_directional_closes", 0.0)), 4),
            "has_active_zone": float(f.get("has_active_zone", 0.0)),
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
