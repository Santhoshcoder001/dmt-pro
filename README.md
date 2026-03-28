# Adaptive Bias-Corrected Hotspot Prediction (ABC-HP)

ABC-HP is an end-to-end machine learning system for predicting road accident hotspots using bias-corrected risk modeling, spatio-temporal features, and map visualization.

## 1. Project Structure

- `abc_hp/data`: data ingestion, preprocessing, feature engineering
- `abc_hp/models`: bias correction, Random Forest, LSTM, hotspot detection
- `abc_hp/visualization`: Folium hotspot map generation
- `abc_hp/api`: FastAPI endpoints
- `abc_hp/pipeline.py`: end-to-end orchestration class
- `abc_hp/main.py`: CLI entrypoint
- `requirements.txt`: dependencies
- `verify_structure.py`: project structure validator

## 2. Prerequisites

- Python 3.10+
- pip
- Internet access for dependency installation

## 3. Install Dependencies

Run from repository root:

```bash
pip install -r requirements.txt
```

Optional (recommended): create and activate a virtual environment before installing.

## 4. Prepare Input Data

Create an accident CSV file with at least these columns:

- `timestamp`
- `latitude`
- `longitude`
- `observed_accidents`

Example:

```csv
timestamp,latitude,longitude,observed_accidents
2026-01-01 08:15:00,12.9716,77.5946,2
2026-01-01 09:00:00,12.9611,77.6387,1
2026-01-01 18:20:00,12.9352,77.6245,3
```

Save this file anywhere, for example:

```text
data/accidents.csv
```

## 5. Verify Project Structure (Optional)

Use the validator script:

```bash
python verify_structure.py --project-dir abc_hp
```

What it does:

- checks required folders/files
- prints `FOUND` and `MISSING` with colors
- checks file non-empty status
- runs basic Python syntax checks
- writes a report to `check_report.txt`

## 6. Run End-to-End Pipeline (CLI)

### Step 1: Train + Predict + Generate Map

```bash
python -m abc_hp.main data/accidents.csv --train
```

This command:

1. Loads and merges accident + synthetic weather/traffic data
2. Preprocesses and engineers features
3. Applies bias correction (`expected_risk`, `bias_factor`, `corrected_risk`)
4. Trains RandomForest model
5. Predicts risk scores
6. Detects hotspots by grid
7. Generates map HTML

### Step 2: Outputs Generated

- Model artifact: `artifacts/rf_risk_model.joblib`
- Map artifact: `artifacts/hotspot_map.html`

### Step 3: Predict Using Existing Model (Without Retraining)

```bash
python -m abc_hp.main data/accidents.csv
```

Use this after at least one successful `--train` run.

## 7. Run FastAPI Service

Start server:

```bash
uvicorn abc_hp.api.app:app --reload
```

Server default URL:

```text
http://127.0.0.1:8000
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

## 8. API Execution Flow (Step by Step)

### Step 1: Train model

```bash
curl -X POST "http://127.0.0.1:8000/train?accident_csv_path=data/accidents.csv"
```

### Step 2: Get predictions

```bash
curl "http://127.0.0.1:8000/predict?accident_csv_path=data/accidents.csv"
```

### Step 3: Generate and download map

```bash
curl -o hotspot_map.html "http://127.0.0.1:8000/map?accident_csv_path=data/accidents.csv"
```

## 9. Core Pipeline Components

1. Data Loader: loads CSV and synthesizes missing contextual sources
2. Preprocessing: missing value handling, normalization, grid mapping
3. Feature Engineering: temporal + traffic + weather + road risk features
4. Bias Correction: computes `Re`, `B`, and `Rc`
5. Model: RandomForest baseline with save/load support
6. Hotspot Detection: threshold-based labels (`low`, `medium`, `high`)
7. Visualization: Folium heatmap and hotspot markers

## 10. Troubleshooting

### Missing package errors

If imports like `fastapi`, `pandas`, `numpy`, `folium`, or `sklearn` fail:

```bash
pip install -r requirements.txt
```

### Model not found error

If prediction says model file is missing, train first:

```bash
python -m abc_hp.main data/accidents.csv --train
```

### Invalid CSV schema

Ensure required columns exist:

- `timestamp`
- `latitude`
- `longitude`
- `observed_accidents`

## 11. Quick Start (Minimal Commands)

```bash
pip install -r requirements.txt
python -m abc_hp.main data/accidents.csv --train
uvicorn abc_hp.api.app:app --reload
```