"""Hotspot detection module for threshold-based risk classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class RiskThresholds:
    """Thresholds for risk band classification."""

    low_max: float = 0.3
    medium_max: float = 0.7


class HotspotDetector:
    """Classifies risk scores into low/medium/high and outputs grid-wise labels."""

    def __init__(self, thresholds: RiskThresholds | None = None) -> None:
        self.thresholds = thresholds or RiskThresholds()
        if self.thresholds.low_max >= self.thresholds.medium_max:
            raise ValueError("low_max must be smaller than medium_max")

    def classify_risk_levels(self, risk_scores: pd.Series) -> pd.Series:
        """Apply threshold-based labels: <0.3 low, 0.3-0.7 medium, >0.7 high."""
        low_mask = risk_scores < self.thresholds.low_max
        medium_mask = (risk_scores >= self.thresholds.low_max) & (risk_scores <= self.thresholds.medium_max)

        labels = np.select(
            [low_mask, medium_mask],
            ["low", "medium"],
            default="high",
        )
        return pd.Series(labels, index=risk_scores.index, name="risk_label")

    def classify_gridwise(
        self,
        data: pd.DataFrame,
        risk_col: str = "predicted_accident_risk_score",
        grid_col: str = "grid_id",
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """Aggregate risk scores by grid and assign a single label per grid."""
        if grid_col not in data.columns:
            raise ValueError(f"Missing grid column: {grid_col}")
        if risk_col not in data.columns:
            raise ValueError(f"Missing risk column: {risk_col}")

        if aggregation not in {"mean", "max", "median"}:
            raise ValueError("aggregation must be one of: mean, max, median")

        LOGGER.info("Classifying grid-wise hotspots using %s aggregation", aggregation)
        grouped = (
            data.groupby(grid_col, as_index=False)[risk_col]
            .agg(aggregation)
            .rename(columns={risk_col: "grid_risk_score"})
        )

        grouped["risk_label"] = self.classify_risk_levels(grouped["grid_risk_score"])
        return grouped
