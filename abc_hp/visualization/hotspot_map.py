"""Folium-based hotspot map generation for ABC-HP."""

from __future__ import annotations

import logging
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import HeatMap


LOGGER = logging.getLogger(__name__)


class HotspotMapBuilder:
    """Builds and exports hotspot maps using Folium."""

    def __init__(self, default_zoom: int = 11) -> None:
        self.default_zoom = default_zoom

    def _resolve_center(
        self,
        data: pd.DataFrame,
        lat_col: str,
        lon_col: str,
    ) -> tuple[float, float]:
        if data.empty:
            return (0.0, 0.0)
        return (float(data[lat_col].mean()), float(data[lon_col].mean()))

    def build_map(
        self,
        data: pd.DataFrame,
        risk_col: str = "predicted_accident_risk_score",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        label_col: str = "risk_label",
    ) -> folium.Map:
        """Create a map with heatmap and hotspot markers."""
        required = [risk_col, lat_col, lon_col]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns required to build map: {missing}")

        center_lat, center_lon = self._resolve_center(data, lat_col=lat_col, lon_col=lon_col)
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=self.default_zoom, control_scale=True)

        # Heatmap reflects intensity of predicted risk at each location.
        heat_data = data[[lat_col, lon_col, risk_col]].dropna().values.tolist()
        if heat_data:
            HeatMap(heat_data, radius=12, blur=16, min_opacity=0.2).add_to(fmap)

        marker_data = data.copy()
        if label_col in marker_data.columns:
            marker_data = marker_data[marker_data[label_col].isin(["medium", "high"])]

        for _, row in marker_data.iterrows():
            label = row.get(label_col, "unknown")
            color = "red" if label == "high" else "orange"
            popup = f"Risk: {row[risk_col]:.3f} | Label: {label}"

            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup=popup,
            ).add_to(fmap)

        return fmap

    def save_map(self, fmap: folium.Map, output_path: str | Path) -> Path:
        """Persist generated map as HTML."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(output_path))
        LOGGER.info("Saved hotspot map to %s", output_path)
        return output_path
