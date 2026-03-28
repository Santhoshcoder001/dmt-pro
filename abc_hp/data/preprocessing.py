"""Data preprocessing module for the ABC-HP system."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - optional dependency
    gpd = None

if TYPE_CHECKING:
    import geopandas as gpd_types


LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Container for outputs produced by the preprocessing pipeline."""

    processed_data: pd.DataFrame
    scaler: Optional[StandardScaler]
    spatial_frame: Optional["gpd_types.GeoDataFrame"]


class DataPreprocessor:
    """Preprocesses fused data into model-ready features."""

    def __init__(self, grid_size: float = 0.01) -> None:
        if grid_size <= 0:
            raise ValueError("grid_size must be a positive value")

        self.grid_size = grid_size
        self.scaler: Optional[StandardScaler] = None

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute numeric with median and categorical with mode."""
        LOGGER.info("Handling missing values")
        prepared = data.copy()

        numeric_columns = prepared.select_dtypes(include=[np.number]).columns
        categorical_columns = prepared.select_dtypes(exclude=[np.number]).columns

        for col in numeric_columns:
            if prepared[col].isna().any():
                prepared[col] = prepared[col].fillna(prepared[col].median())

        for col in categorical_columns:
            if prepared[col].isna().any():
                mode_value = prepared[col].mode(dropna=True)
                if not mode_value.empty:
                    prepared[col] = prepared[col].fillna(mode_value.iloc[0])
                else:
                    prepared[col] = prepared[col].fillna("unknown")

        return prepared

    def normalize_features(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """Scale selected numeric features with StandardScaler."""
        if not feature_columns:
            return data

        LOGGER.info("Normalizing %d features", len(feature_columns))
        prepared = data.copy()

        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            prepared[feature_columns] = self.scaler.fit_transform(prepared[feature_columns])
        else:
            prepared[feature_columns] = self.scaler.transform(prepared[feature_columns])

        return prepared

    def add_spatial_grid(
        self,
        data: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> pd.DataFrame:
        """Convert latitude/longitude into stable grid identifiers."""
        if lat_col not in data.columns or lon_col not in data.columns:
            raise ValueError(f"Columns '{lat_col}' and '{lon_col}' are required for grid mapping")

        LOGGER.info("Mapping points to spatial grid with size %.4f", self.grid_size)
        prepared = data.copy()

        # Grid buckets are integer projections of lat/lon based on configured resolution.
        prepared["grid_x"] = np.floor(prepared[lon_col] / self.grid_size).astype(int)
        prepared["grid_y"] = np.floor(prepared[lat_col] / self.grid_size).astype(int)
        prepared["grid_id"] = prepared["grid_y"].astype(str) + "_" + prepared["grid_x"].astype(str)

        return prepared

    def create_spatial_index(
        self,
        data: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> Optional["gpd_types.GeoDataFrame"]:
        """Build a GeoDataFrame and trigger spatial index creation when available."""
        if gpd is None:
            LOGGER.warning("GeoPandas is not installed. Spatial index creation skipped")
            return None

        LOGGER.info("Creating spatial index")
        geo_frame = gpd.GeoDataFrame(
            data.copy(),
            geometry=gpd.points_from_xy(data[lon_col], data[lat_col]),
            crs="EPSG:4326",
        )

        _ = geo_frame.sindex
        return geo_frame

    def preprocess(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        fit_scaler: bool = True,
    ) -> PreprocessingResult:
        """Run full preprocessing workflow for model-ready data."""
        prepared = self.handle_missing_values(data)
        prepared = self.add_spatial_grid(prepared)
        prepared = self.normalize_features(prepared, feature_columns=feature_columns, fit=fit_scaler)
        spatial_frame = self.create_spatial_index(prepared)

        return PreprocessingResult(
            processed_data=prepared,
            scaler=self.scaler,
            spatial_frame=spatial_frame,
        )

    def run_pipeline(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
        fit_scaler: bool = True,
    ) -> pd.DataFrame:
        """Run preprocessing and return only the cleaned dataset."""
        if feature_columns is None:
            excluded = {"event_id", "grid_x", "grid_y", "observed_accidents"}
            feature_columns = [
                col
                for col in data.select_dtypes(include=[np.number]).columns
                if col not in excluded
            ]

        result = self.preprocess(
            data=data,
            feature_columns=feature_columns,
            fit_scaler=fit_scaler,
        )
        return result.processed_data


class PreprocessingPipeline(DataPreprocessor):
    """Friendly alias for preprocessing workflow usage."""

