import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from typing import Dict

from src.config import (
    DATA_FILE,
    ENTRY_MODEL_FILE,
    EXIT_MODEL_FILE,
    FEATURE_COLS,
    LEGACY_ENTRY,
    LEGACY_EXIT,
    SCALER_FILE,
)

logger = logging.getLogger(__name__)


class NeuralPredictor:
    """
    Handles both Entry (Classification) and Exit (Regression) Models.
    """

    def __init__(self, auto_load: bool = True):
        self.entry_model = None
        self.exit_model = None
        self.is_ready = False
        self.scaler = None

        if auto_load:
            self._cleanup_legacy()
            self._load_artifacts()

    def _cleanup_legacy(self):
        """Removes old .h5 files to prevent confusion."""
        try:
            if os.path.exists(LEGACY_ENTRY):
                os.remove(LEGACY_ENTRY)
                logger.info("🗑️ Removed legacy entry model (.h5)")
            if os.path.exists(LEGACY_EXIT):
                os.remove(LEGACY_EXIT)
                logger.info("🗑️ Removed legacy exit model (.h5)")
        except Exception:
            pass

    def _delete_artifacts(self):
        """Deletes existing model and scaler files."""
        try:
            if os.path.exists(ENTRY_MODEL_FILE):
                os.remove(ENTRY_MODEL_FILE)
            if os.path.exists(EXIT_MODEL_FILE):
                os.remove(EXIT_MODEL_FILE)
            if os.path.exists(SCALER_FILE):
                os.remove(SCALER_FILE)
            logger.info("🗑️ Corrupt/Old ML artifacts deleted.")
        except Exception as e:
            logger.error(f"Error deleting artifacts: {e}")

    def _load_artifacts(self):
        """Loads model and scaler if they exist."""
        if os.path.exists(ENTRY_MODEL_FILE) and os.path.exists(SCALER_FILE):
            try:
                self.entry_model = tf.keras.models.load_model(ENTRY_MODEL_FILE, compile=False)
                self.entry_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                self.scaler = joblib.load(SCALER_FILE)

                # Check for Exit model (optional if just starting)
                if os.path.exists(EXIT_MODEL_FILE):
                    self.exit_model = tf.keras.models.load_model(EXIT_MODEL_FILE, compile=False)
                    self.exit_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

                if self.entry_model.input_shape[1] != len(FEATURE_COLS):
                    logger.warning("⚠️ Input mismatch. (New Features). Deleting old models.")
                    self._delete_artifacts()
                    self.is_ready = False
                    return

                self.is_ready = True
                logger.info("🧠 ML Models loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load ML artifacts: {e}")
                self._delete_artifacts()
        else:
            logger.info("⚠️ No ML models found. Running in heuristic mode until trained.")

    def predict(self, features: dict) -> Dict[str, float]:
        """
        Predicts entry probability, risk multiplier, and exit ATR multiple.
        """
        if not self.is_ready or self.entry_model is None:
            return {"prob": 0.5, "risk_mult": 1.0, "pred_exit_atr": 2.0}

        try:
            defaults = {
                "rsi": 50,
                "adx": 20,
                "atr": 0,
                "ema_dist": 0,
                "bb_width": 0,
                "vol_ratio": 1.0,
                "htf_trend": 0,
                "dist_to_pivot": 0,
                "hour_norm": 0,
                "day_norm": 0,
                "wick_ratio": 0,
                "dist_ema200": 0,
                "volatility_ratio": 1.0,
                "dist_to_vwap": 0,
                "rolling_acc": 0.5,
            }
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
            return {"prob": 0.5, "risk_mult": 1.0, "pred_exit_atr": 2.0}

    def train_network(self):
        """
        Trains both Entry and Exit models.
        """
        if not os.path.exists(DATA_FILE):
            logger.warning("⚠️ No training data found. Skipping NN training.")
            return

        try:
            df = pd.read_csv(DATA_FILE, on_bad_lines="skip")
            if "target_excursion" not in df.columns:
                df["target_excursion"] = 2.0

            # Ensure all columns exist, fill missing with 0
            for col in FEATURE_COLS:
                if col not in df.columns:
                    df[col] = 0.0

            # Drop rows with missing values
            df = df.dropna(subset=FEATURE_COLS + ["target_win", "target_excursion"])
            if len(df) < 50:
                logger.warning(f"⚠️ Insufficient data ({len(df)} rows). Need 50+ to train.")
                return

            X = df[FEATURE_COLS]
            y_entry = df["target_win"]
            y_exit = df["target_excursion"]  # Float: Max ATR multiples reached

            # Preprocessing
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # compute class weights to correct imbalance
            classes = np.unique(y_entry)
            cw = dict()
            try:
                weights = class_weight.compute_class_weight("balanced", classes=classes, y=y_entry)
                cw = {int(c): float(w) for c, w in zip(classes, weights)}
            except Exception:
                cw = None

            # callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
            ]

            # --- 1. Train Entry Model (Binary Classification) ---
            logger.info("🧠 Training Entry Model...")
            entry_model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            entry_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            entry_model.fit(
                X_scaled,
                y_entry,
                epochs=200,
                batch_size=32,
                verbose=1,
                validation_split=0.15,
                callbacks=callbacks,
                class_weight=cw,
            )
            entry_model.save(ENTRY_MODEL_FILE)
            self.entry_model = entry_model

            # --- 2. Train Exit Model (Regression) ---
            logger.info("🧠 Training Exit Model...")
            exit_model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dense(1, activation="linear"),  # Linear output for regression
                ]
            )
            exit_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
            exit_model.fit(
                X_scaled, y_exit, epochs=200, batch_size=32, verbose=1, validation_split=0.15, callbacks=callbacks
            )
            exit_model.save(EXIT_MODEL_FILE)
            self.exit_model = exit_model

            joblib.dump(self.scaler, SCALER_FILE)
            self.is_ready = True
            logger.info("✅ Training Complete.")

        except Exception as e:
            logger.error(f"Training failed: {e}")
