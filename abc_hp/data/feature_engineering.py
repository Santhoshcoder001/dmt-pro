"""Feature engineering module for the ABC-HP system."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class RoadRiskWeights:
    """Configurable weights for road risk score composition."""

    traffic_weight: float = 0.40
    weather_weight: float = 0.35
    infra_weight: float = 0.25


class FeatureEngineering:
    """Creates temporal and contextual features used by downstream models."""

    def __init__(self, risk_weights: Optional[RoadRiskWeights] = None) -> None:
        self.risk_weights = risk_weights or RoadRiskWeights()

    def extract_hour_and_day(self, data: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """Extract hour and day_of_week from the timestamp column."""
        if timestamp_col not in data.columns:
            raise ValueError(f"'{timestamp_col}' column is required to extract temporal features")

        prepared = data.copy()
        prepared[timestamp_col] = pd.to_datetime(prepared[timestamp_col], errors="coerce")

        if prepared[timestamp_col].isna().all():
            raise ValueError("All timestamp values are invalid after datetime conversion")

        prepared["hour"] = prepared[timestamp_col].dt.hour
        prepared["day_of_week"] = prepared[timestamp_col].dt.dayofweek
        return prepared

    def add_traffic_intensity_levels(
        self,
        data: pd.DataFrame,
        traffic_col: str = "traffic_density",
        low_threshold: float = 50.0,
        high_threshold: float = 90.0,
    ) -> pd.DataFrame:
        """Bucket traffic density into low/medium/high intensity levels."""
        if traffic_col not in data.columns:
            raise ValueError(f"'{traffic_col}' column is required to derive traffic intensity")

        if low_threshold >= high_threshold:
            raise ValueError("low_threshold must be less than high_threshold")

        prepared = data.copy()
        bins = [-np.inf, low_threshold, high_threshold, np.inf]
        labels = ["low", "medium", "high"]

        prepared["traffic_intensity_level"] = pd.cut(
            prepared[traffic_col],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        prepared["traffic_intensity_level"] = prepared["traffic_intensity_level"].astype(str)

        # Numeric encoding helps downstream ML pipelines.
        level_map = {"low": 0, "medium": 1, "high": 2}
        prepared["traffic_intensity_score"] = prepared["traffic_intensity_level"].map(level_map).astype(float)
        return prepared

    def add_weather_severity_index(
        self,
        data: pd.DataFrame,
        rainfall_col: str = "rainfall_mm",
        visibility_col: str = "visibility_km",
        temperature_col: str = "temperature_c",
    ) -> pd.DataFrame:
        """Create weather severity index from rainfall, visibility, and temperature deviation."""
        required = [rainfall_col, visibility_col, temperature_col]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing weather columns for severity index: {missing}")

        prepared = data.copy()

        rainfall = prepared[rainfall_col].astype(float)
        visibility = prepared[visibility_col].astype(float)
        temperature = prepared[temperature_col].astype(float)

        rainfall_norm = rainfall / (rainfall.max() + 1e-6)
        visibility_inv = 1 - (visibility / (visibility.max() + 1e-6))
        temp_deviation = np.abs(temperature - 22.0) / 22.0

        prepared["weather_severity_index"] = (
            0.50 * rainfall_norm
            + 0.30 * visibility_inv
            + 0.20 * temp_deviation
        )
        return prepared

    def add_road_risk_score(
        self,
        data: pd.DataFrame,
        traffic_score_col: str = "traffic_intensity_score",
        weather_index_col: str = "weather_severity_index",
        infra_col: str = "infra_risk",
    ) -> pd.DataFrame:
        """Create composite road risk score from traffic, weather, and infrastructure signals."""
        required = [traffic_score_col, weather_index_col, infra_col]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns for road risk score: {missing}")

        prepared = data.copy()

        traffic_norm = prepared[traffic_score_col].astype(float) / 2.0
        weather_norm = prepared[weather_index_col].astype(float)
        infra_norm = prepared[infra_col].astype(float)

        prepared["road_risk_score"] = (
            self.risk_weights.traffic_weight * traffic_norm
            + self.risk_weights.weather_weight * weather_norm
            + self.risk_weights.infra_weight * infra_norm
        )
        return prepared

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply full feature engineering pipeline and return enriched dataset."""
        LOGGER.info("Extracting temporal features")
        prepared = self.extract_hour_and_day(data)

        LOGGER.info("Deriving traffic intensity levels")
        prepared = self.add_traffic_intensity_levels(prepared)

        LOGGER.info("Computing weather severity index")
        prepared = self.add_weather_severity_index(prepared)

        LOGGER.info("Computing road risk score")
        prepared = self.add_road_risk_score(prepared)

        return prepared
