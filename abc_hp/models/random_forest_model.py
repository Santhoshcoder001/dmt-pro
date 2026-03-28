"""Random Forest model module for accident risk score prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


LOGGER = logging.getLogger(__name__)


class RandomForestRiskModel:
    """Train and serve a RandomForestRegressor for accident risk scoring."""

    def __init__(
        self,
        feature_columns: list[str],
        target_column: str = "accident_risk_score",
        corrected_risk_column: str = "corrected_risk",
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")

        self.feature_columns = feature_columns
        self.target_column = target_column
        self.corrected_risk_column = corrected_risk_column

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_trained = False

    def _build_input_columns(self, data: pd.DataFrame) -> list[str]:
        """Build model input columns from configured features plus corrected risk."""
        input_columns = list(self.feature_columns)

        if self.corrected_risk_column not in input_columns:
            input_columns.append(self.corrected_risk_column)

        missing = [col for col in input_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing input columns for model: {missing}")

        return input_columns

    def train(self, data: pd.DataFrame) -> None:
        """Train model using features + corrected risk as input."""
        if self.target_column not in data.columns:
            raise ValueError(f"Missing target column: {self.target_column}")

        input_columns = self._build_input_columns(data)
        x_train = data[input_columns]
        y_train = data[self.target_column]

        LOGGER.info("Training RandomForestRegressor on %d rows", len(data))
        self.model.fit(x_train, y_train)
        self._is_trained = True

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict accident risk score for given dataset."""
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() before predict()")

        input_columns = self._build_input_columns(data)
        preds = self.model.predict(data[input_columns])
        return pd.Series(preds, index=data.index, name="predicted_accident_risk_score")

    def save(self, model_path: str | Path) -> None:
        """Persist trained model and metadata to disk."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "corrected_risk_column": self.corrected_risk_column,
            "is_trained": self._is_trained,
        }

        joblib.dump(payload, model_path)
        LOGGER.info("Saved RandomForest model to %s", model_path)

    @classmethod
    def load(cls, model_path: str | Path) -> "RandomForestRiskModel":
        """Load trained model and metadata from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        payload = joblib.load(model_path)

        model_instance = cls(
            feature_columns=payload["feature_columns"],
            target_column=payload["target_column"],
            corrected_risk_column=payload["corrected_risk_column"],
        )
        model_instance.model = payload["model"]
        model_instance._is_trained = payload.get("is_trained", True)

        LOGGER.info("Loaded RandomForest model from %s", model_path)
        return model_instance
