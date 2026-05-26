import joblib
import logging
import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from typing import Dict

from src.config import ENTRY_MODEL_FILE, EXIT_MODEL_FILE, FEATURE_COLS, SCALER_FILE

logger = logging.getLogger(__name__)


class NeuralPredictor:
    """
    Operates purely on SMC structural heuristics.
    """

    def __init__(self, auto_load: bool = False):
        self.entry_model = None
        self.exit_model = None
        self.scaler = None
        self.is_ready = False

    def predict(self, features: dict) -> Dict[str, float]:
        """
        Predicts entry and exit probability.
        """
        # Graceful Failover if models are not yet trained
        if not self.is_ready or self.entry_model is None:
            return {"prob": 0.85, "risk_mult": 1.0, "pred_exit_atr": 2.0}

        try:
            # Map incoming features, defaulting to 0 for missing binary flags
            data = {k: [features.get(k, 0.0)] for k in FEATURE_COLS}
            df_input = pd.DataFrame(data)

            X_new = self.scaler.transform(df_input)
            prob = float(self.entry_model.predict(X_new, verbose=0)[0][0])

            # Exit Prediction (Default to 2.0 RR equivalent if model is missing)
            pred_exit_atr = 2.0
            if self.exit_model:
                raw_exit = float(self.exit_model.predict(X_new, verbose=0)[0][0])
                pred_exit_atr = max(1.5, min(raw_exit, 4.0))

            # Dynamic Risk Sizing based on Neural Conviction
            risk_mult = 0.5  # Base Low
            if prob > 0.80:
                risk_mult = 2.0  # High Conviction
            elif prob > 0.60:
                risk_mult = 1.0  # Standard

            return {"prob": prob, "risk_mult": risk_mult, "pred_exit_atr": pred_exit_atr}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"prob": 0.85, "risk_mult": 1.0, "pred_exit_atr": 2.0}

    def train_network(self):
        """
        Trains the models on the discrete SMC features using Early Stopping.
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

            # Prevent overfitting on discrete/binary SMC features
            early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

            # 1. Train Entry Model
            logger.info("🧠 Training SMC Entry Model ...")
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
            entry_model.fit(
                X_scaled,
                y_entry,
                epochs=150,
                batch_size=32,
                verbose=1,
                validation_split=0.15,
                callbacks=[early_stop],
            )
            entry_model.save(ENTRY_MODEL_FILE)

            # 2. Train Exit Model mapping Risk adjustments
            logger.info("🧠 Training SMC Exit Model ...")

            # ISOLATE WINNERS FOR EXCURSION REGRESSION
            winning_df = df[df["target_win"] == 1].copy()

            if len(winning_df) < 10:
                logger.warning("⚠️ Insufficient winning trades. Falling back to static exit predictions.")
                exit_model = None
            else:
                X_winners = winning_df[FEATURE_COLS]
                X_winners_scaled = self.scaler.transform(X_winners)

                if "target_excursion" in winning_df.columns:
                    y_exit = winning_df["target_excursion"]
                else:
                    y_exit = pd.Series([2.0] * len(winning_df))

                exit_model = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
                        tf.keras.layers.Dense(16, activation="relu"),
                        tf.keras.layers.Dense(1, activation="linear"),
                    ]
                )

                exit_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
                exit_model.fit(
                    X_winners_scaled,
                    y_exit,
                    epochs=100,
                    batch_size=32,
                    verbose=1,
                    validation_split=0.15,
                    callbacks=[early_stop],
                )
                exit_model.save(EXIT_MODEL_FILE)

            # Save artifacts to root directory
            joblib.dump(self.scaler, SCALER_FILE)

            self.entry_model = entry_model
            self.exit_model = exit_model
            self.is_ready = True
            logger.info("✅ Training Complete. Artifacts ready.")

        except Exception as e:
            logger.error(f"Training failed: {e}")
