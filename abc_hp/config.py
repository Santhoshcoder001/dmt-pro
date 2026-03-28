"""Central configuration for the ABC-HP system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Configuration values used across data, modeling, API, and visualization layers."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    artifact_dir: Path = field(init=False)
    map_output_path: Path = field(init=False)
    model_path: Path = field(init=False)

    # Data and preprocessing
    grid_size: float = 0.01

    # Bias correction weights
    alpha: float = 0.25
    beta: float = 0.25
    gamma: float = 0.25
    delta: float = 0.25
    epsilon: float = 1e-6

    # Hotspot thresholds
    low_threshold: float = 0.3
    medium_threshold: float = 0.7

    # Model parameters
    target_column: str = "accident_risk_score"
    corrected_risk_column: str = "corrected_risk"
    random_forest_estimators: int = 300
    random_state: int = 42

    def __post_init__(self) -> None:
        self.artifact_dir = self.project_root / "artifacts"
        self.map_output_path = self.artifact_dir / "hotspot_map.html"
        self.model_path = self.artifact_dir / "rf_risk_model.joblib"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
