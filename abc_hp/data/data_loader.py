"""Data ingestion module for the ABC-HP system.

This module loads multi-source data, simulates unavailable sources for development,
and merges all sources into one modeling-ready dataframe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


class DataLoader:
    """Load accident data, generate synthetic context, and return a merged dataframe."""

    REQUIRED_COLUMNS = {"timestamp", "latitude", "longitude", "observed_accidents"}

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self._rng = np.random.default_rng(seed=random_seed)

    def load_accident_csv(self, csv_path: str | Path) -> pd.DataFrame:
        """Load accident records from CSV and enforce required schema."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Accident dataset not found: {csv_path}")

        LOGGER.info("Loading accident data from %s", csv_path)
        accidents = pd.read_csv(csv_path)

        if "timestamp" in accidents.columns:
            accidents["timestamp"] = pd.to_datetime(accidents["timestamp"], errors="coerce")

        missing_columns = self.REQUIRED_COLUMNS.difference(accidents.columns)
        if missing_columns:
            raise ValueError(
                "Accident dataset missing required columns: "
                f"{sorted(missing_columns)}"
            )

        accidents = accidents.reset_index(drop=True)
        if "event_id" not in accidents.columns:
            accidents["event_id"] = accidents.index.astype(str)

        LOGGER.info("Loaded %d accident records", len(accidents))
        return accidents

    def generate_synthetic_weather_data(self, base_data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic weather signals aligned to accident events."""
        LOGGER.info("Generating synthetic weather data")
        weather = pd.DataFrame(
            {
                "event_id": base_data["event_id"],
                "timestamp": base_data["timestamp"],
                "temperature_c": self._rng.normal(loc=26.0, scale=5.0, size=len(base_data)),
                "rainfall_mm": self._rng.gamma(shape=1.8, scale=1.2, size=len(base_data)),
                "visibility_km": self._rng.uniform(1.5, 12.0, size=len(base_data)),
            }
        )

        weather["weather_risk"] = (
            0.50 * (weather["rainfall_mm"] / (weather["rainfall_mm"].max() + 1e-6))
            + 0.30 * (1 - weather["visibility_km"] / (weather["visibility_km"].max() + 1e-6))
            + 0.20 * (np.abs(weather["temperature_c"] - 22.0) / 22.0)
        )
        return weather

    def generate_synthetic_traffic_data(self, base_data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic traffic features aligned to accident events."""
        LOGGER.info("Generating synthetic traffic data")
        return pd.DataFrame(
            {
                "event_id": base_data["event_id"],
                "traffic_density": self._rng.uniform(30.0, 120.0, len(base_data)),
                "population_exposure": self._rng.uniform(200.0, 6000.0, len(base_data)),
                "infra_risk": self._rng.uniform(0.1, 1.0, len(base_data)),
            }
        )

    def merge_datasets(
        self,
        accidents: pd.DataFrame,
        weather: pd.DataFrame,
        traffic: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge accident, weather, and traffic sources into one dataframe."""
        LOGGER.info("Merging accident, weather, and traffic datasets")

        merged = accidents.merge(weather.drop(columns=["timestamp"], errors="ignore"), on="event_id", how="left")
        merged = merged.merge(traffic, on="event_id", how="left")

        LOGGER.info("Merged dataset shape: %s", merged.shape)
        return merged

    def load_and_merge(self, accident_csv_path: str | Path) -> pd.DataFrame:
        """Run complete loading pipeline and return merged dataframe."""
        accidents = self.load_accident_csv(accident_csv_path)
        weather = self.generate_synthetic_weather_data(accidents)
        traffic = self.generate_synthetic_traffic_data(accidents)
        return self.merge_datasets(accidents=accidents, weather=weather, traffic=traffic)


class DataIngestionModule(DataLoader):
    """Backward-compatible alias for older pipeline code."""

    def load_accident_data(self, csv_path: str | Path) -> pd.DataFrame:
        return self.load_accident_csv(csv_path)

    def fetch_weather_data(
        self,
        base_data: pd.DataFrame,
        weather_csv_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        if weather_csv_path:
            weather_csv_path = Path(weather_csv_path)
            if not weather_csv_path.exists():
                raise FileNotFoundError(f"Weather dataset not found: {weather_csv_path}")

            LOGGER.info("Loading weather data from %s", weather_csv_path)
            weather = pd.read_csv(weather_csv_path)
            if "timestamp" in weather.columns:
                weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce")

            if "event_id" not in weather.columns:
                weather = weather.merge(base_data[["event_id", "timestamp"]], on="timestamp", how="left")

            weather["weather_risk"] = (
                0.50 * (weather["rainfall_mm"] / (weather["rainfall_mm"].max() + 1e-6))
                + 0.30 * (1 - weather["visibility_km"] / (weather["visibility_km"].max() + 1e-6))
                + 0.20 * (np.abs(weather["temperature_c"] - 22.0) / 22.0)
            )
            return weather

        return self.generate_synthetic_weather_data(base_data)

    def simulate_contextual_data(self, base_data: pd.DataFrame) -> pd.DataFrame:
        return self.generate_synthetic_traffic_data(base_data)

    def build_dataset(
        self,
        accident_csv_path: str | Path,
        weather_csv_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        accidents = self.load_accident_data(accident_csv_path)
        weather = self.fetch_weather_data(accidents, weather_csv_path=weather_csv_path)
        contextual = self.simulate_contextual_data(accidents)
        return self.merge_datasets(accidents=accidents, weather=weather, traffic=contextual)
