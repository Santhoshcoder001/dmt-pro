"""LSTM time-series model module for accident risk forecasting."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger(__name__)


class LSTMRiskModel:
    """Simple sequence model wrapper using TensorFlow Keras LSTM."""

    def __init__(self, sequence_length: int = 8, random_seed: int = 42) -> None:
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        self.model = None

    def _import_keras(self):
        try:
            import tensorflow as tf
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import Dense, LSTM
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TensorFlow is required for LSTMRiskModel. Install tensorflow to use this model."
            ) from exc

        tf.random.set_seed(self.random_seed)
        return Sequential, LSTM, Dense

    def _to_sequences(self, x: np.ndarray, y: np.ndarray | None = None):
        if len(x) <= self.sequence_length:
            raise ValueError("Not enough samples to build LSTM sequences")

        seq_x = []
        seq_y = []
        for idx in range(self.sequence_length, len(x)):
            seq_x.append(x[idx - self.sequence_length : idx])
            if y is not None:
                seq_y.append(y[idx])

        seq_x = np.asarray(seq_x)
        if y is None:
            return seq_x
        return seq_x, np.asarray(seq_y)

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
    ) -> None:
        """Train LSTM on sequence-formatted features and targets."""
        Sequential, LSTM, Dense = self._import_keras()

        x_seq, y_seq = self._to_sequences(features, targets)
        input_dim = x_seq.shape[-1]

        self.model = Sequential(
            [
                LSTM(64, input_shape=(self.sequence_length, input_dim)),
                Dense(32, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )

        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        self.model.fit(x_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0)
        LOGGER.info("Trained LSTM model on %d sequences", len(x_seq))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict risk scores from feature sequences."""
        if self.model is None:
            raise RuntimeError("LSTM model is not trained. Call train() or load() first.")

        x_seq = self._to_sequences(features)
        preds = self.model.predict(x_seq, verbose=0)
        return preds.reshape(-1)

    def save(self, model_path: str | Path) -> Path:
        """Save trained Keras model to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save an untrained LSTM model")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        LOGGER.info("Saved LSTM model to %s", model_path)
        return model_path

    def load(self, model_path: str | Path) -> None:
        """Load Keras model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"LSTM model file not found: {model_path}")

        try:
            from tensorflow.keras.models import load_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TensorFlow is required for LSTMRiskModel. Install tensorflow to use this model."
            ) from exc

        self.model = load_model(model_path)
        LOGGER.info("Loaded LSTM model from %s", model_path)
