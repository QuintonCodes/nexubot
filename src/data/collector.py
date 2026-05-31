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
        with open(self.filename, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # Subtract 1 for the header

    @staticmethod
    def engineer_features(raw_features: dict) -> dict:
        """Transforms raw SMC features into the full engineered feature set."""
        f = raw_features

        active_killzone = int(f.get("active_killzone", 0))
        is_in_fvg = float(f.get("is_in_fvg", 0.0))
        is_in_ifvg = float(f.get("is_in_ifvg", 0.0))
        is_in_orderblock = float(f.get("is_in_orderblock", 0.0))
        distance_to_poi = float(f.get("distance_to_poi", 0.0))
        mitigation_count = float(f.get("mitigation_count", 0.0))
        structural_break = float(f.get("structural_break", 0.0))
        is_htf_aligned = float(f.get("is_htf_aligned", 0.0))
        pd_array_status = float(f.get("pd_array_status", 0.5))
        sweep_depth_atr = float(f.get("sweep_depth_atr", 0.0))
        is_liquidity_swept = float(f.get("is_liquidity_swept", 0.0))

        # Composite score
        score = 0.0
        score += {0: 2.0, 1: 1.5, 2: -0.5, 3: -1.0}.get(active_killzone, 0.0)
        if is_in_fvg == 1:
            score += 2.0
        elif is_in_ifvg == 1:
            score += 0.5
        elif is_in_orderblock == 1:
            score -= 1.0

        if 0.5 <= distance_to_poi < 2.0:
            score += 1.5
        elif distance_to_poi < 0.3:
            score -= 1.5
        elif distance_to_poi >= 4.0:
            score -= 0.5

        mit = int(mitigation_count)
        if mit == 0:
            score += 1.0
        elif mit == 1:
            score += 0.5
        elif mit >= 2:
            score -= 0.5

        if structural_break in (-2.0, 2.0):
            score += 1.0
        elif structural_break in (-1.0, 1.0):
            score += 0.3
        else:
            score -= 0.2

        if is_htf_aligned == 1.0:
            score += 0.5
        elif is_htf_aligned == -1.0:
            score -= 0.3

        return {
            "signal_quality_score": round(score, 2),
            "session_quality_score": {0: 1.0, 1: 0.9, 2: 0.6, 3: 0.55}.get(active_killzone, 1.0),
            "is_optimal_entry_distance": float(0.5 <= distance_to_poi <= 2.5),
            "poi_freshness_score": round(1.0 - (min(mit, 3) / 3.0), 4),
            "log_distance_to_poi": round(float(np.log1p(distance_to_poi)), 4),
            "is_in_fvg": is_in_fvg,
            "is_in_ifvg": is_in_ifvg,
            "is_htf_aligned": is_htf_aligned,
            "is_liquidity_swept_tier": is_liquidity_swept,
            "sweep_depth_atr": sweep_depth_atr,
            "pd_array_status": round(pd_array_status, 4),
            "pd_deviation_from_equilibrium": round(abs(pd_array_status - 0.5), 4),
            "is_inside_poi_flag": float(distance_to_poi < 0.30),
            "zone_overlap_count": float(int(is_in_fvg) + int(is_in_ifvg) + int(is_in_orderblock)),
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
                    df.tail(self.max_rows).to_csv(self.filename, index=False)
                    self._current_row_count = self.max_rows
                    logger.info(f"🧹 Training data pruned to latest {self.max_rows} rows.")
        except Exception as e:
            logger.error(f"Failed to log training data for {symbol}: {e}")
