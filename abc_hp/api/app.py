"""FastAPI application for ABC-HP inference and training."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from abc_hp.pipeline import ABCHPPipeline
from abc_hp.utils.logging_config import setup_logging


setup_logging()
app = FastAPI(title="ABC-HP API", version="1.0.0")
pipeline = ABCHPPipeline()


@app.get("/predict")
def predict(accident_csv_path: str = Query(..., description="Path to accident CSV file")) -> dict[str, object]:
    """Run hotspot prediction pipeline and return summarized JSON output."""
    try:
        predictions = pipeline.predict(Path(accident_csv_path))
    except Exception as exc:  # pragma: no cover - API boundary
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    preview_cols = [
        "event_id",
        "grid_id",
        "predicted_accident_risk_score",
        "risk_label",
    ]
    preview_cols = [col for col in preview_cols if col in predictions.columns]

    return {
        "rows": len(predictions),
        "columns": list(predictions.columns),
        "preview": predictions[preview_cols].head(20).to_dict(orient="records"),
    }


@app.post("/train")
def train(accident_csv_path: str = Query(..., description="Path to accident CSV file")) -> dict[str, object]:
    """Train and persist baseline RandomForest model."""
    try:
        result = pipeline.train(Path(accident_csv_path))
    except Exception as exc:  # pragma: no cover - API boundary
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Model trained successfully",
        **result,
    }


@app.get("/map")
def generate_map(accident_csv_path: str = Query(..., description="Path to accident CSV file")) -> FileResponse:
    """Generate hotspot map and return HTML file."""
    try:
        predictions = pipeline.predict(Path(accident_csv_path))
        map_path = pipeline.generate_map(predictions)
    except Exception as exc:  # pragma: no cover - API boundary
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(path=map_path, media_type="text/html", filename=map_path.name)
