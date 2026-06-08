import joblib
import logging
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import threading
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Optional, Tuple

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

MIN_TRAINING_ROWS = 500
MIN_AUC_GATE = 0.56


class NeuralPredictor:
    """
    Predicts trading entry probabilities and optimal exit targets
    based on SMC structural heuristics.
    """

    def __init__(self, auto_load: bool = False):
        self.entry_model: Optional[tf.keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.exit_regressor: Optional[GradientBoostingRegressor] = None
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

                # Critical schema version control
                if self.entry_model.input_shape[1] != len(FEATURE_COLS):
                    logger.error(
                        f"🛑 Model input shape mismatch (Model requires: {self.entry_model.input_shape[1]}, Runtime defines: {len(FEATURE_COLS)} features). Retraining is mandatory."
                    )
                    self.is_ready = False
                    self.entry_model = None
                    return

                self.scaler = joblib.load(SCALER_FILE)

                if os.path.exists(CALIBRATOR_FILE):
                    self.calibrator = joblib.load(CALIBRATOR_FILE)
                if os.path.exists(EXIT_MODEL_FILE):
                    self.exit_regressor = joblib.load(EXIT_MODEL_FILE)

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

        # Protect against dataset pollution
        df = df.dropna(subset=FEATURE_COLS + ["target_win"])

        if len(df) < MIN_TRAINING_ROWS:
            logger.warning(f"⚠️ Insufficient data ({len(df)} rows). Need {MIN_TRAINING_ROWS}+ to train.")
            return None

        return df

    def _train_entry_model(
        self, X_scaled: np.ndarray, y_entry: np.ndarray, early_stop: EarlyStopping
    ) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
        """Constructs and trains the binary classification model for entries."""
        split_idx = int(len(X_scaled) * 0.80)
        X_train, X_cal = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_cal = y_entry[:split_idx], y_entry[split_idx:]

        classes = np.unique(y_train)
        if len(classes) > 1:
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, weights))
            logger.info(f"⚖️ Applied Dynamic Class Weights: {class_weight_dict}")
        else:
            logger.warning("⚠️ Only one target class found in training split! Falling back to 1.0 weights.")
            class_weight_dict = {0: 1.0, 1: 1.0}

        # Optimized regularized topology: 24 -> 12 -> 8 -> 1 Architecture
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
                tf.keras.layers.Dense(24, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(12, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
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

    def _train_exit_model(self, df: pd.DataFrame) -> Optional[GradientBoostingRegressor]:
        """Trains localized exit strategy models relying on non-linear interaction patterns."""
        df_wins = df[df["target_win"] == 1].copy()

        if len(df_wins) < 50:
            logger.warning("Not enough winning rows to train exit regressor. System will default.")
            return None

        X = df_wins[FEATURE_COLS]
        y = df_wins["target_excursion"].clip(MIN_RR, 6.0)

        regressor = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        regressor.fit(X, y)
        return regressor

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
        if self.scaler:
            joblib.dump(self.scaler, SCALER_FILE)
        if self.calibrator:
            joblib.dump(self.calibrator, CALIBRATOR_FILE)
        if self.exit_regressor:
            joblib.dump(self.exit_regressor, EXIT_MODEL_FILE)

    def predict(self, features: dict, strategy: str = "Unknown") -> Dict[str, float]:
        """Predicts entry probability, dynamic risk sizing, and optimal exit ATR."""
        if not self.is_ready or self.entry_model is None:
            return {"prob": 0.5, "risk_mult": 1.0, "pred_exit_atr": 3.0}

        try:
            data = {k: [features.get(k, 0.0)] for k in FEATURE_COLS}
            df_input = pd.DataFrame(data)

            # Critical Inference Protection Hook
            df_input = df_input.fillna(0.0)

            # Preprocess and predict entry
            with self._predict_lock:
                X_new = self.scaler.transform(df_input)

                # Feature Distribution Drift Check (Runtime assertion)
                z_scores = np.abs(X_new[0])
                drifted_indices = np.where(z_scores > 4.0)[0]
                if len(drifted_indices) > 0:
                    drifted_features = [FEATURE_COLS[i] for i in drifted_indices]
                    logger.warning(f"⚠️ Feature Drift Detected on {drifted_features}: values exceeded 4σ.")

                raw_prob = float(self.entry_model.predict(X_new, verbose=0)[0][0])

                # Calibrate probability if a calibrator is available
                prob = float(self.calibrator.transform([raw_prob])[0]) if self.calibrator else raw_prob

                # Predict exit ATR (Default to 3.0 if model is missing)
                if self.exit_regressor:
                    pred_exit_atr = float(self.exit_regressor.predict(X_new)[0])
                else:
                    pred_exit_atr = 3.0

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

        try:
            # Upsample targeted strategic categories manually
            daily_sweeps = df[df["strategy"] == "Daily/Asian Sweep"]
            if not daily_sweeps.empty:
                logger.info(
                    f"📈 Applying synthetic oversampling mapping to {len(daily_sweeps)} Daily/Asian Sweep instances..."
                )
                df = pd.concat([df, daily_sweeps, daily_sweeps, daily_sweeps], ignore_index=True)

            X = df[FEATURE_COLS]
            y_entry = df["target_win"].values

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            rf = RandomForestClassifier(n_estimators=50, class_weight="balanced", n_jobs=-1, random_state=42)
            # Use chronological CV split evaluation
            split_idx = int(len(X_scaled) * 0.8)
            rf.fit(X_scaled[:split_idx], y_entry[:split_idx])
            auc_test = roc_auc_score(y_entry[split_idx:], rf.predict_proba(X_scaled[split_idx:])[:, 1])
            logger.info(f"🧠 Pre-training signal quality AUC Test: {auc_test:.4f}")

            if auc_test < MIN_AUC_GATE:
                logger.error(f"🛑 Pre-train AUC ({auc_test:.4f}) below threshold ({MIN_AUC_GATE}). Aborting.")
                return

            # Shortened Patience Thresholds
            entry_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

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
            logger.info("🧠 Generating Dynamic Feature Exit Regression Pattern...")
            self.exit_regressor = self._train_exit_model(df)

            # 4. Calibrate Predictions
            logger.info("⚖️ Calibrating Probabilities...")
            self.calibrator = self._train_calibrator(X_cal, y_cal)

            # 5. Save Artifacts
            self._save_artifacts()

            self.is_ready = True
            logger.info("✅ Training Complete. Artifacts ready and models ready.")

        except Exception as e:
            logger.error(f"Training failed: {e}")
