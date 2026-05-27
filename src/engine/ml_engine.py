import joblib
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Optional, Tuple

from keras.callbacks import EarlyStopping
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from src.config import (
    CALIBRATOR_FILE,
    ENTRY_MODEL_FILE,
    EXIT_MODEL_FILE,
    FEATURE_COLS,
    MIN_RR,
    SCALER_FILE,
    TRAINING_FILE,
)

logger = logging.getLogger(__name__)


class NeuralPredictor:
    """
    Predicts trading entry probabilities and optimal exit targets
    based on SMC structural heuristics.
    """

    def __init__(self, auto_load: bool = False):
        self.entry_model: Optional[tf.keras.Model] = None
        self.exit_model: Optional[tf.keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.is_ready: bool = False

    def _predict_exit(self, X_new: np.ndarray) -> float:
        """Helper to safely predict the exit ATR with boundaries."""
        if not self.exit_model:
            return 3.0

        raw_exit = float(self.exit_model.predict(X_new, verbose=0)[0][0])
        return max(MIN_RR, min(raw_exit, 6.0))

    def _calculate_risk_multiplier(self, probability: float) -> float:
        """Helper to determine position sizing logic based on probability."""
        if probability > 0.80:
            return 2.0  # High Conviction
        elif probability > 0.60:
            return 1.0  # Standard
        return 0.5  # Base Low

    def _load_and_prepare_data(self, data_file: str) -> Optional[pd.DataFrame]:
        """Validates and prepares the initial dataset."""
        if not os.path.exists(data_file):
            logger.warning("⚠️ No training data found. Run Backfill first.")
            return None

        df = pd.read_csv(data_file, on_bad_lines="skip")

        # Ensure all required feature columns exist
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        df = df.dropna(subset=FEATURE_COLS + ["target_win"])

        if len(df) < 50:
            logger.warning(f"⚠️ Insufficient data ({len(df)} rows). Need 50+ to train.")
            return None

        return df

    def _train_entry_model(
        self, X_scaled: np.ndarray, y_entry: pd.Series, early_stop: EarlyStopping
    ) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
        """Constructs and trains the binary classification model for entries."""
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(
            X_scaled, y_entry, epochs=150, batch_size=32, verbose=1, validation_split=0.15, callbacks=[early_stop]
        )

        # Reserve the validation portion for calibration later
        split_idx = int(len(X_scaled) * 0.85)
        X_cal, y_cal = X_scaled[split_idx:], y_entry.values[split_idx:]

        return model, X_cal, y_cal

    def _train_exit_model(self, df: pd.DataFrame, early_stop: EarlyStopping) -> Optional[tf.keras.Model]:
        """Constructs and trains the regression model for trade excursions (winners only)."""
        winning_df = df[df["target_win"] == 1].copy()

        if len(winning_df) < 10:
            logger.warning("⚠️ Insufficient winning trades. Falling back to static exit predictions.")
            return None

        X_winners = winning_df[FEATURE_COLS]
        X_winners_scaled = self.scaler.transform(X_winners)

        # Handle missing excursion target
        if "target_excursion" in winning_df.columns:
            y_exit = winning_df["target_excursion"]
        else:
            y_exit = pd.Series([3.0] * len(winning_df))

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="linear"),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.fit(
            X_winners_scaled,
            y_exit,
            epochs=100,
            batch_size=32,
            verbose=1,
            validation_split=0.15,
            callbacks=[early_stop],
        )

        return model

    def _train_calibrator(self, X_cal: np.ndarray, y_cal: np.ndarray) -> IsotonicRegression:
        """Trains the Isotonic Regression model to align raw probabilities with reality."""
        raw_probs = self.entry_model.predict(X_cal, verbose=0).flatten()
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_cal)
        return calibrator

    def _save_artifacts(self) -> None:
        """Saves models, scalers, and calibrators to disk."""
        if self.entry_model:
            self.entry_model.save(ENTRY_MODEL_FILE)
        if self.exit_model:
            self.exit_model.save(EXIT_MODEL_FILE)

        joblib.dump(self.calibrator, CALIBRATOR_FILE)
        joblib.dump(self.scaler, SCALER_FILE)

    def predict(self, features: dict) -> Dict[str, float]:
        """
        Predicts entry probability, dynamic risk sizing, and optimal exit ATR.
        """
        # Graceful Failover if models are not yet trained
        if not self.is_ready or self.entry_model is None:
            return {"prob": 0.85, "risk_mult": 1.0, "pred_exit_atr": 3.0}

        try:
            # Map incoming features, defaulting to 0 for missing binary flags
            data = {k: [features.get(k, 0.0)] for k in FEATURE_COLS}
            df_input = pd.DataFrame(data)

            # Preprocess and predict entry
            X_new = self.scaler.transform(df_input)
            raw_prob = float(self.entry_model.predict(X_new, verbose=0)[0][0])

            # Calibrate probability if a calibrator is available
            prob = float(self.calibrator.transform([raw_prob])[0]) if self.calibrator else raw_prob

            # Predict exit ATR (Default to 3.0 if model is missing)
            pred_exit_atr = self._predict_exit(X_new)

            # Calculate dynamic risk sizing based on conviction
            risk_mult = self._calculate_risk_multiplier(prob)

            return {"prob": prob, "risk_mult": risk_mult, "pred_exit_atr": pred_exit_atr}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"prob": 0.85, "risk_mult": 1.0, "pred_exit_atr": 3.0}

    def train_network(self) -> None:
        """
        Coordinates the training of the entry model, exit model, and calibrator.
        """
        # 1. Load and prepare data
        df = self._load_and_prepare_data(TRAINING_FILE)
        if df is None:
            return

        try:
            X = df[FEATURE_COLS]
            y_entry = df["target_win"]

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

            # 2. Train Entry Model
            logger.info("🧠 Training SMC Entry Model ...")
            self.entry_model, X_cal, y_cal = self._train_entry_model(X_scaled, y_entry, early_stop)

            # 3. Train Exit Model
            logger.info("🧠 Training SMC Exit Model ...")
            self.exit_model = self._train_exit_model(df, early_stop)

            # 4. Calibrate Predictions
            logger.info("⚖️ Calibrating Probabilities ...")
            self.calibrator = self._train_calibrator(X_cal, y_cal)

            # 5. Save Artifacts
            self._save_artifacts()

            self.is_ready = True
            logger.info("✅ Training Complete. Artifacts ready and models ready.")

        except Exception as e:
            logger.error(f"Training failed: {e}")
