"""End-to-end ABC-HP pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from abc_hp.config import AppConfig
from abc_hp.data.data_loader import DataLoader
from abc_hp.data.feature_engineering import FeatureEngineering
from abc_hp.data.preprocessing import PreprocessingPipeline
from abc_hp.models.bias_correction import BiasCorrection, BiasWeights
from abc_hp.models.hotspot_detection import HotspotDetector, RiskThresholds
from abc_hp.models.random_forest_model import RandomForestRiskModel
from abc_hp.visualization.hotspot_map import HotspotMapBuilder


LOGGER = logging.getLogger(__name__)


class ABCHPPipeline:
    """Coordinates ingestion, preprocessing, modeling, hotspot detection, and map output."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()

        self.loader = DataLoader(random_seed=self.config.random_state)
        self.preprocessor = PreprocessingPipeline(grid_size=self.config.grid_size)
        self.feature_engineer = FeatureEngineering()

        self.bias_correction = BiasCorrection(
            weights=BiasWeights(
                alpha=self.config.alpha,
                beta=self.config.beta,
                gamma=self.config.gamma,
                delta=self.config.delta,
            ),
            epsilon=self.config.epsilon,
        )

        self.hotspot_detector = HotspotDetector(
            thresholds=RiskThresholds(
                low_max=self.config.low_threshold,
                medium_max=self.config.medium_threshold,
            )
        )
        self.map_builder = HotspotMapBuilder()

        self.model: RandomForestRiskModel | None = None

    def _apply_bias_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        prepared = data.copy()

        prepared["expected_risk"] = self.bias_correction.compute_expected_risk(
            T=prepared["traffic_density"],
            W=prepared["weather_severity_index"],
            P=prepared["population_exposure"],
            I=prepared["infra_risk"],
        )

        prepared["bias_factor"] = self.bias_correction.compute_bias_factor(
            Re=prepared["expected_risk"],
            Ro=prepared["observed_accidents"],
        )

        prepared["corrected_risk"] = self.bias_correction.compute_corrected_risk(
            Ro=prepared["observed_accidents"],
            B=prepared["bias_factor"],
        )

        return prepared

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        featured = self.feature_engineer.transform(data)
        featured = self._apply_bias_correction(featured)

        # Supervised target for baseline RF model.
        featured[self.config.target_column] = (
            0.55 * featured["corrected_risk"]
            + 0.45 * featured["road_risk_score"]
        )

        processed = self.preprocessor.run_pipeline(featured)
        return processed

    def _model_features(self, data: pd.DataFrame) -> list[str]:
        base_features = [
            "hour",
            "day_of_week",
            "traffic_density",
            "traffic_intensity_score",
            "weather_severity_index",
            "road_risk_score",
            "population_exposure",
            "infra_risk",
            "weather_risk",
        ]
        return [col for col in base_features if col in data.columns]

    def train(self, accident_csv_path: str | Path) -> dict[str, object]:
        """Train baseline RandomForest model from accident CSV input."""
        raw_data = self.loader.load_and_merge(accident_csv_path)
        data = self._prepare_features(raw_data)

        model_features = self._model_features(data)
        self.model = RandomForestRiskModel(
            feature_columns=model_features,
            target_column=self.config.target_column,
            corrected_risk_column=self.config.corrected_risk_column,
            n_estimators=self.config.random_forest_estimators,
            random_state=self.config.random_state,
        )
        self.model.train(data)
        self.model.save(self.config.model_path)

        return {
            "rows": len(data),
            "feature_count": len(model_features),
            "model_path": str(self.config.model_path),
        }

    def _ensure_model(self) -> RandomForestRiskModel:
        if self.model is not None:
            return self.model

        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {self.config.model_path}. Run training first."
            )

        self.model = RandomForestRiskModel.load(self.config.model_path)
        return self.model

    def predict(self, accident_csv_path: str | Path) -> pd.DataFrame:
        """Predict accident risk and classify grid-wise hotspots."""
        raw_data = self.loader.load_and_merge(accident_csv_path)
        data = self._prepare_features(raw_data)

        model = self._ensure_model()
        data["predicted_accident_risk_score"] = model.predict(data)

        grid_labels = self.hotspot_detector.classify_gridwise(data)
        result = data.merge(grid_labels[["grid_id", "risk_label"]], on="grid_id", how="left")
        return result

    def generate_map(self, predictions: pd.DataFrame) -> Path:
        """Generate and store hotspot map HTML from prediction output."""
        fmap = self.map_builder.build_map(predictions)
        return self.map_builder.save_map(fmap, self.config.map_output_path)
