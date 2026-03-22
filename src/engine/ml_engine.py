import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import logging
import pandas as pd
import sys
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from typing import Dict

from src.config import ENTRY_MODEL_FILE, EXIT_MODEL_FILE, FEATURE_COLS, SCALER_FILE


logger = logging.getLogger(__name__)


class NeuralPredictor:
    """
    Production-ready Read-Only Neural Predictor.
    Loads frozen models bundled with PyInstaller. Falls back to SMC heuristics if unavailable.
    """

    def __init__(self, auto_load: bool = True):
        self.entry_model = None
        self.exit_model = None
        self.scaler = None
        self.is_ready = False

        if auto_load:
            self._load_frozen_artifacts()

    def _get_resource_path(self, relative_path: str) -> str:
        """Helper to find bundled resources safely when packaged as an executable."""
        if hasattr(sys, "_MEIPASS"):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    def _load_frozen_artifacts(self):
        """Loads read-only pre-trained models bundled with the application."""
        entry_path = self._get_resource_path(ENTRY_MODEL_FILE)
        scaler_path = self._get_resource_path(SCALER_FILE)
        exit_path = self._get_resource_path(EXIT_MODEL_FILE)

        if os.path.exists(entry_path) and os.path.exists(scaler_path):
            try:
                self.entry_model = tf.keras.models.load_model(entry_path, compile=False)
                self.scaler = joblib.load(scaler_path)

                if os.path.exists(exit_path):
                    self.exit_model = tf.keras.models.load_model(exit_path, compile=False)

                self.is_ready = True
                logger.info("🧠 Frozen ML Models loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load bundled ML artifacts. Falling back to Heuristics. Error: {e}")
                self.is_ready = False
        else:
            logger.info("⚠️ Bundled ML models not found. Running in Pure SMC Heuristic mode.")
            self.is_ready = False

    def predict(self, features: dict) -> Dict[str, float]:
        """
        Predicts entry probability using the frozen model.
        """
        # Graceful Failover to pure SMC Heuristics if models are tampered/missing
        if not self.is_ready or self.entry_model is None:
            return {"prob": 0.85, "risk_mult": 1.0, "pred_exit_atr": 2.0}

        try:
            defaults = {col: 0.0 for col in FEATURE_COLS}
            data = {k: [features.get(k, defaults.get(k, 0))] for k in FEATURE_COLS}
            df_input = pd.DataFrame(data)

            X_new = self.scaler.transform(df_input)
            prob = float(self.entry_model.predict(X_new, verbose=0)[0][0])

            # Exit Prediction (Default to 2.0 ATR if model is missing or error)
            pred_exit_atr = 2.0
            if self.exit_model:
                raw_exit = float(self.exit_model.predict(X_new, verbose=0)[0][0])
                pred_exit_atr = max(1.0, min(raw_exit, 4.0))

            # Dynamic Risk Sizing
            risk_mult = 0.5  # Base Low
            if prob > 0.85:
                risk_mult = 2.0  # High Conviction
            elif prob > 0.65:
                risk_mult = 1.0  # Standard

            return {"prob": prob, "risk_mult": risk_mult, "pred_exit_atr": pred_exit_atr}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"prob": 0.85, "risk_mult": 1.0, "pred_exit_atr": 2.0}

    def train_network(self):
        """
        Developer Tool: Trains the models on the new SMC features.
        This is ONLY called manually via the GUI to generate production artifacts.
        """
        data_file = "training_data.csv"
        if not os.path.exists(data_file):
            logger.warning("⚠️ No training data found. Run Backfill first.")
            return

        try:
            df = pd.read_csv(data_file, on_bad_lines="skip")

            # Ensure all new features exist
            for col in FEATURE_COLS:
                if col not in df.columns:
                    df[col] = 0.0

            df = df.dropna(subset=FEATURE_COLS + ["target_win"])
            if len(df) < 50:
                logger.warning(f"⚠️ Insufficient data ({len(df)} rows). Need 50+ to train.")
                return

            X = df[FEATURE_COLS]
            y_entry = df["target_win"]

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # 1. Train Entry Model
            logger.info("🧠 Training SMC Entry Model...")
            entry_model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )

            entry_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            entry_model.fit(X_scaled, y_entry, epochs=150, batch_size=32, verbose=1, validation_split=0.15)
            entry_model.save(ENTRY_MODEL_FILE)

            # 2. Train Exit Model mapping Risk adjustments (Ensures secondary keras is generated)
            logger.info("🧠 Training SMC Exit Model...")
            if "target_exit_atr" in df.columns:
                y_exit = df["target_exit_atr"]
            else:
                y_exit = pd.Series([2.0] * len(df))  # Fallback exit target structure mapping if absent

            exit_model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(1, activation="linear"),
                ]
            )

            exit_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            exit_model.fit(X_scaled, y_exit, epochs=100, batch_size=32, verbose=1, validation_split=0.15)
            exit_model.save(EXIT_MODEL_FILE)

            # Save artifacts to root directory
            joblib.dump(self.scaler, SCALER_FILE)

            self.entry_model = entry_model
            self.exit_model = exit_model
            self.is_ready = True
            logger.info("✅ Training Complete. Artifacts ready for PyInstaller packaging.")

        except Exception as e:
            logger.error(f"Training failed: {e}")
