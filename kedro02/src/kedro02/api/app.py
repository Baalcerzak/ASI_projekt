import json
import logging
import pickle
import time
from contextlib import asynccontextmanager
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from pycaret import regression as pcr


_default_base = Path(os.environ.get("APP_BASE_DIR", "/app"))
if not _default_base.exists():
    _default_base = Path.cwd()


BASE_DIR = _default_base
MODELS_DIR = BASE_DIR / "data" / "06_models"
REPORTING_DIR = BASE_DIR / "data" / "08_reporting"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_LOG = LOGS_DIR / "predictions_log.jsonl"

# GLOBAL
_model = None
_encoders = None
_selected_features = None
_training_stats = None
_model_comparison = None
_best_model_name = None
_is_pycaret = False

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(str(LOGS_DIR / "predictions.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def load_artifacts():
    global _model, _encoders, _selected_features, _training_stats, _model_comparison, _best_model_name

    comparison_path = REPORTING_DIR / "model_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            _model_comparison = json.load(f)
        best = _model_comparison.get("better_model", "RandomForest")
    else:
        best = "RandomForest"

    _best_model_name = best

    model_map = {
        "RandomForest": "regressor.pickle",
        "GradientBoosting": "gradient_boosting_model.pickle",
        "TunedRandomForest": "tuned_rf_model.pickle",
        "PyCaret": "pycaret_model.pkl",
    }
    model_file = model_map.get(best, "regressor.pickle")
    model_path = MODELS_DIR / model_file

    if not model_path.exists():
        for fname in ["tuned_rf_model.pickle", "regressor.pickle", "gradient_boosting_model.pickle"]:
            p = MODELS_DIR / fname
            if p.exists():
                model_path = p
                break


    if model_path.exists():
        try:
            if best == "PyCaret":
                stem = (MODELS_DIR / "pycaret_model").as_posix()
                _model = pcr.load_model(stem)
                logger.info("Załadowano model PyCaret: pycaret_model.pkl")
            else:
                with open(model_path, "rb") as f:
                    _model = pickle.load(f)
                logger.info(f"Załadowano model: {model_path.name}")
        except Exception as e:
            logger.exception(f"Błąd ładowania modelu ({best}): {e}")
            _model = None
    else:
        logger.warning("Brak wytrenowanego modelu. Uruchom najpierw: kedro run")

    encoders_path = MODELS_DIR / "target_encoders.pickle"
    if encoders_path.exists():
        with open(encoders_path, "rb") as f:
            _encoders = pickle.load(f)

    features_path = MODELS_DIR / "selected_features.pickle"
    if features_path.exists():
        with open(features_path, "rb") as f:
            _selected_features = pickle.load(f)

    stats_path = MODELS_DIR / "training_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            _training_stats = json.load(f)
        logger.info("Załadowano statystyki treningowe (drift detection)")

@asynccontextmanager
async def lifespan(api: FastAPI):
    load_artifacts()
    yield
app = FastAPI(
    title="Car Price Prediction Application API",
    description="Predykcja cen samochodów - model wytrenowany na danych z 2021 roku",
    version="1.0.0",
    lifespan=lifespan,
)

class CarFeatures(BaseModel):
    manufacturer: str = Field(..., example="TOYOTA")
    model: str = Field(..., example="Camry")
    prod_year: int = Field(..., ge=1990, le=2021, example=2018)
    category: str = Field(..., example="Sedan")
    leather_interior: str = Field(..., example="Yes")
    fuel_type: str = Field(..., example="Petrol")
    engine_volume: float = Field(..., gt=0, example=2.5)
    mileage: float = Field(..., ge=0, example=50000)
    cylinders: float = Field(..., gt=0, example=4)
    gear_box_type: str = Field(..., example="Automatic")
    drive_wheels: str = Field(..., example="Front")
    doors: int = Field(..., ge=2, le=6, example=4)
    wheel: str = Field(..., example="Left wheel")
    color: str = Field(..., example="Black")
    airbags: int = Field(..., ge=0, example=8)
    levy: Optional[float] = Field(None, example=1000.0)
    turbo: Optional[int] = Field(0, example=0)

class PredictionRequest(BaseModel):
    cars: List[CarFeatures]

class PredictionResponse(BaseModel):
    predictions: List[float]
    prices_usd: List[float]
    drift_warnings: List[str]
    latency_ms: float
    model_used: str
    timestamp: str

class DriftReport(BaseModel):
    feature: str
    training_mean: float
    input_mean: float
    z_score: float
    drift_detected: bool

LUXURY_BRANDS = {'LEXUS', 'BMW', 'MERCEDES-BENZ', 'PORSCHE', 'JAGUAR', 'LAND ROVER', 'TESLA'}
REFERENCE_YEAR = 2021


def car_to_df(car: CarFeatures) -> pd.DataFrame:
    age = REFERENCE_YEAR - car.prod_year
    mileage_per_year = car.mileage / (age + 1)
    log_mileage = np.log1p(car.mileage)

    row = {
        "Manufacturer": car.manufacturer.upper(),
        "Model": car.model,
        "Prod. year": car.prod_year,
        "Category": car.category,
        "Leather interior": 1 if str(car.leather_interior).strip().lower() in ("yes", "1", "true") else 0,
        "Fuel type": car.fuel_type,
        "Engine volume": car.engine_volume,
        "Mileage": car.mileage,
        "Cylinders": car.cylinders,
        "Gear box type": car.gear_box_type,
        "Drive wheels": car.drive_wheels,
        "Doors": car.doors,
        "Wheel": car.wheel,
        "Color": car.color,
        "Airbags": car.airbags,
        "Levy": car.levy if car.levy is not None else 0.0,
        "Turbo": car.turbo if car.turbo is not None else 0,
        "Age": age,
        "Mileage_per_year": mileage_per_year,
        "Log_Mileage": log_mileage,
        "Age_Mileage": age * log_mileage,
        "Engine_per_cyl": car.engine_volume / (car.cylinders + 0.1),
        "Is_Luxury": int(car.manufacturer.upper() in LUXURY_BRANDS),
        "Is_Automatic": int(car.gear_box_type in ["Automatic", "Tiptronic"]),
        "Is_4x4": int(car.drive_wheels == "4x4"),
        "Airbags_per_door": car.airbags / (car.doors + 1),
        "Safety_Luxury": car.airbags * int(car.manufacturer.upper() in LUXURY_BRANDS),
        "Is_Eco": int(car.fuel_type in ["Hybrid", "LPG", "CNG", "Plug-in Hybrid", "Hydrogen"]),
        "Engine_Size_Group": (
            "Small" if car.engine_volume <= 1.5
            else "Medium" if car.engine_volume <= 2.5
            else "Large" if car.engine_volume <= 4
            else "Very Large"
        ),
        "Age_Group": (
            "New" if age <= 3
            else "Recent" if age <= 7
            else "Old" if age <= 12
            else "Vintage"
        ),
        "Color_Group": car.color if car.color in ["Black", "White", "Silver", "Grey", "Blue"] else "Other",
    }
    return pd.DataFrame([row])


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in encoders.items():
        if col in df.columns:
            df[col] = df[col].map(mapping["map"]).fillna(mapping["global_mean"]).astype(float)
    return df


def detect_drift(df: pd.DataFrame, training_stats: dict, threshold: float = 3.0) -> List[str]:
    warnings = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in training_stats:
            stats = training_stats[col]
            if stats["std"] > 0:
                z = abs((df[col].mean() - stats["mean"]) / stats["std"])
                if z > threshold:
                    warnings.append(
                        f"DRIFT: {col} | input_mean={df[col].mean():.2f} "
                        f"train_mean={stats['mean']:.2f} z={z:.2f}"
                    )
    return warnings


def log_prediction(request_data: dict, predictions: list, drift_warnings: list):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": request_data,
        "predictions": predictions,
        "drift_warnings": drift_warnings,
    }
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ENDPOINTS
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "encoders_loaded": _encoders is not None,
        "training_stats_loaded": _training_stats is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model nie jest załadowany. Uruchom pipeline: kedro run",
        )

    start = time.time()
    dfs = [car_to_df(car) for car in request.cars]
    df = pd.concat(dfs, ignore_index=True)

    drift_warnings = []
    if _training_stats:
        drift_warnings = detect_drift(df, _training_stats)
        if drift_warnings:
            for w in drift_warnings:
                logger.warning(w)

    if _encoders:
        df = apply_encoders(df, _encoders)

    if _selected_features:
        available = [f for f in _selected_features if f in df.columns]
        df = df[available]

    log_preds = _model.predict(df)
    prices = np.expm1(log_preds).tolist()

    latency_ms = (time.time() - start) * 1000

    model_name = _best_model_name or (type(_model).__name__ if _model else "unknown")

    background_tasks.add_task(
        log_prediction,
        [c.dict() for c in request.cars],
        prices,
        drift_warnings,
    )

    logger.info(f"Predykcja: {len(prices)} samochodów, latency={latency_ms:.1f}ms")

    return PredictionResponse(
        predictions=log_preds.tolist(),
        prices_usd=prices,
        drift_warnings=drift_warnings,
        latency_ms=round(latency_ms, 2),
        model_used=model_name,
        timestamp=datetime.utcnow().isoformat(),
    )

@app.get("/drift/report")
def drift_report():
    if not PREDICTIONS_LOG.exists():
        return {"message": "Brak logów predykcji", "drift_events": []}

    drift_events = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("drift_warnings"):
                    drift_events.append({
                        "timestamp": entry["timestamp"],
                        "warnings": entry["drift_warnings"],
                    })
            except json.JSONDecodeError:
                continue

    return {
        "total_predictions_logged": sum(1 for _ in open(PREDICTIONS_LOG)),
        "drift_events_count": len(drift_events),
        "recent_drift_events": drift_events[-10:],
    }

@app.get("/model/info")
def model_info():
    model_class = type(_model).__name__ if _model else None
    pycaret_info = _model_comparison.get("PyCaret", {}) if _model_comparison else {}

    best_metrics = None
    if _model_comparison and _best_model_name:
        best_metrics = _model_comparison.get(_best_model_name)

    info = {
        "best_model_name": _best_model_name,
        "model_class": model_class,
        "model_loaded": _model is not None,
        "selected_features_count": len(_selected_features) if _selected_features else None,
        "selected_features": _selected_features,
        "training_stats_available": _training_stats is not None,
        "best_model_metrics": best_metrics,
        "all_models_comparison": _model_comparison,
        "pycaret_details": pycaret_info,
    }
    return info

@app.get("/predictions/history")
def predictions_history(limit: int = 20):
    if not PREDICTIONS_LOG.exists():
        return {"predictions": []}

    lines = PREDICTIONS_LOG.read_text().strip().split("\n")
    recent = lines[-limit:] if len(lines) >= limit else lines
    entries = []
    for line in recent:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"predictions": entries, "total": len(lines)}
