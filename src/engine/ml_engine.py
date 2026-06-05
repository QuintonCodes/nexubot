import joblib
import logging
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import threading
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Optional, Tuple

from src.config import (
    CALIBRATOR_FILE,
    ENTRY_MODEL_FILE,
    EXIT_MODEL_FILE,
    EXIT_SCALER_FILE,
    FEATURE_COLS,
    MIN_RR,
    SCALER_FILE,
    TRAINING_FILE,
)

logger = logging.getLogger(__name__)

MIN_TRAINING_ROWS = 500
MIN_AUC_GATE = 0.56


class NeuralPredictor:
    """
    Predicts trading entry probabilities and optimal exit targets
    based on SMC structural heuristics.
    """

    def __init__(self, auto_load: bool = False):
        self.entry_model: Optional[tf.keras.Model] = None
        self.exit_model: Optional[tf.keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.exit_scaler: Optional[StandardScaler] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.is_ready: bool = False
        self._predict_lock = threading.Lock()

        if auto_load:
            self._load_artifacts()

    def _calculate_risk_multiplier(self, probability: float) -> float:
        """Helper to determine position sizing logic based on probability."""
        if probability > 0.80:
            return 2.0  # High Conviction
        elif probability > 0.60:
            return 1.0  # Standard
        return 0.5  # Base Low

    def _load_artifacts(self) -> None:
        """Loads machine learning pre-trained structures from static files where possible."""
        if os.path.exists(ENTRY_MODEL_FILE) and os.path.exists(SCALER_FILE):
            try:
                self.entry_model = tf.keras.models.load_model(ENTRY_MODEL_FILE)
                self.scaler = joblib.load(SCALER_FILE)
                if os.path.exists(EXIT_MODEL_FILE):
                    self.exit_model = tf.keras.models.load_model(EXIT_MODEL_FILE)
                if os.path.exists(EXIT_SCALER_FILE):
                    self.exit_scaler = joblib.load(EXIT_SCALER_FILE)
                if os.path.exists(CALIBRATOR_FILE):
                    self.calibrator = joblib.load(CALIBRATOR_FILE)
                self.is_ready = True
                logger.info("✅ ML Artifacts loaded successfully on init.")
            except Exception as e:
                logger.error(f"Failed to load ML artifacts: {e}")

    def _load_and_prepare_data(self, data_file: str) -> Optional[pd.DataFrame]:
        """Validates and prepares the initial dataset."""
        if not os.path.exists(data_file):
            logger.warning("⚠️ No training data found. Run Backfill first.")
            return None

        df = pd.read_csv(data_file, on_bad_lines="skip")
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        df = df.dropna(subset=FEATURE_COLS + ["target_win"])
        if len(df) < MIN_TRAINING_ROWS:
            logger.warning(f"⚠️ Insufficient data ({len(df)} rows). Need {MIN_TRAINING_ROWS}+ to train.")
            return None

        return df

    def _predict_exit(self, df_input: pd.DataFrame) -> float:
        """Helper to safely predict the exit ATR with boundaries."""
        if not self.exit_model or not self.exit_scaler:
            return 3.0

        X_new_exit = self.exit_scaler.transform(df_input)
        raw_exit = float(self.exit_model.predict(X_new_exit, verbose=0)[0][0])
        return max(MIN_RR, min(raw_exit, 6.0))

    def _train_entry_model(
        self, X_scaled: np.ndarray, y_entry: np.ndarray, early_stop: EarlyStopping
    ) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
        """Constructs and trains the binary classification model for entries."""
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_scaled, y_entry, test_size=0.15, random_state=42, stratify=y_entry
        )

        classes = np.unique(y_train)
        if len(classes) > 1:
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, weights))
            logger.info(f"⚖️ Applied Dynamic Class Weights: {class_weight_dict}")
        else:
            logger.warning("⚠️ Only one target class found in training split! Falling back to 1.0 weights.")
            class_weight_dict = {0: 1.0, 1: 1.0}

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(
            X_train,
            y_train,
            epochs=150,
            batch_size=32,
            verbose=1,
            validation_split=0.15,
            class_weight=class_weight_dict,
            callbacks=[early_stop],
        )

        return model, X_cal, y_cal

    def _train_exit_model(self, df: pd.DataFrame, early_stop: EarlyStopping) -> Optional[tf.keras.Model]:
        """Constructs and trains the regression model for trade excursions (winners only)."""
        X_all = df[FEATURE_COLS]
        self.exit_scaler = StandardScaler()
        X_all_scaled = self.exit_scaler.fit_transform(X_all)

        y_exit = np.where(df["target_win"] == 1, df["target_excursion"], 0.0)

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="linear"),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.fit(
            X_all_scaled,
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
        if np.std(raw_probs) < 1e-4:
            logger.warning("Raw probabilities have no variance. Skipping calibration.")
            return None

        calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
        calibrator.fit(raw_probs, y_cal)
        return calibrator

    def _save_artifacts(self) -> None:
        """Saves models, scalers, and calibrators to disk."""
        if self.entry_model:
            self.entry_model.save(ENTRY_MODEL_FILE)
        if self.exit_model:
            self.exit_model.save(EXIT_MODEL_FILE)
        if self.scaler:
            joblib.dump(self.scaler, SCALER_FILE)
        if self.exit_scaler:
            joblib.dump(self.exit_scaler, EXIT_SCALER_FILE)
        if self.calibrator:
            joblib.dump(self.calibrator, CALIBRATOR_FILE)

    def predict(self, features: dict) -> Dict[str, float]:
        """Predicts entry probability, dynamic risk sizing, and optimal exit ATR."""
        if not self.is_ready or self.entry_model is None:
            return {"prob": 0.5, "risk_mult": 1.0, "pred_exit_atr": 3.0}

        try:
            data = {k: [features.get(k, 0.0)] for k in FEATURE_COLS}
            df_input = pd.DataFrame(data)

            # Preprocess and predict entry
            with self._predict_lock:
                X_new = self.scaler.transform(df_input)

                if np.any(np.abs(X_new) > 4.0):
                    logger.warning("⚠️ Feature Drift Detected: Extrapolating beyond 4σ from training data bounds.")
                    X_new = np.clip(X_new, -4.0, 4.0)

                raw_prob = float(self.entry_model.predict(X_new, verbose=0)[0][0])

                # Calibrate probability if a calibrator is available
                prob = float(self.calibrator.transform([raw_prob])[0]) if self.calibrator else raw_prob

                # Predict exit ATR (Default to 3.0 if model is missing)
                pred_exit_atr = self._predict_exit(df_input)

            # Calculate dynamic risk sizing based on conviction
            risk_mult = self._calculate_risk_multiplier(prob)

            return {"prob": prob, "risk_mult": risk_mult, "pred_exit_atr": pred_exit_atr}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"prob": 0.5, "risk_mult": 1.0, "pred_exit_atr": 3.0}

    def train_network(self) -> None:
        """Coordinates the training of the entry model, exit model, and calibrator."""
        # 1. Load and prepare data
        df = self._load_and_prepare_data(TRAINING_FILE)
        if df is None:
            return

        expected_sparse_features = [
            "is_htf_aligned",
            "sweep_aligned",
            "sweep_snapback_vel",
            "dist_to_asia_extremes_atr",
        ]

        for col in FEATURE_COLS:
            zero_pct = (df[col] == 0.0).mean()
            if zero_pct > 0.95 and col not in expected_sparse_features:
                logger.warning(f"⚠️ Schema mismatch detected: {col} is >95% zeros. Consider wiping legacy ML data.")

        try:
            X = df[FEATURE_COLS]
            y_entry = df["target_win"].values

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            rf = RandomForestClassifier(n_estimators=50, class_weight="balanced", n_jobs=-1, random_state=42)
            auc_test = cross_val_score(rf, X_scaled, y_entry, cv=3, scoring="roc_auc").mean()
            logger.info(f"🧠 Pre-training signal quality AUC Test: {auc_test:.4f}")

            if auc_test < MIN_AUC_GATE:
                logger.error(f"🛑 Pre-train AUC ({auc_test:.4f}) below threshold ({MIN_AUC_GATE}). Aborting.")
                return

            entry_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)
            exit_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

            # 2. Train Entry Model
            logger.info("🧠 Training SMC Entry Model...")
            self.entry_model, X_cal, y_cal = self._train_entry_model(X_scaled, y_entry, entry_stop)

            val_auc = roc_auc_score(y_cal, self.entry_model.predict(X_cal, verbose=0))
            logger.info(f"📊 Live Validation AUC: {val_auc:.4f}")

            if val_auc < MIN_AUC_GATE:
                logger.error(
                    f"🛑 Post-train Validation AUC ({val_auc:.4f}) below threshold ({MIN_AUC_GATE}). Aborting."
                )
                return

            # 3. Train Exit Model
            logger.info("🧠 Training SMC Exit Model...")
            self.exit_model = self._train_exit_model(df, exit_stop)

            # 4. Calibrate Predictions
            logger.info("⚖️ Calibrating Probabilities...")
            self.calibrator = self._train_calibrator(X_cal, y_cal)

            # 5. Save Artifacts
            self._save_artifacts()

            self.is_ready = True
            logger.info("✅ Training Complete. Artifacts ready and models ready.")

        except Exception as e:
            logger.error(f"Training failed: {e}")
