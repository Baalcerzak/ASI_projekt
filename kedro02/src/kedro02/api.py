# """FastAPI – serwis predykcji cen samochodów.
#
# Uruchomienie:
#     uvicorn kedro02.api:app --host 0.0.0.0 --port 8000 --reload
#
# Endpointy:
#     GET  /health          – status serwisu
#     GET  /model/info      – informacje o załadowanym modelu
#     POST /predict         – predykcja ceny pojedynczego auta
#     POST /predict/batch   – predykcja dla wielu aut
#     GET  /drift           – raport driftu danych
# """
#
# import json
# import logging
# import pickle
# from datetime import datetime
# from pathlib import Path
# from typing import List, Optional
#
# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
#
# # ── Konfiguracja ─────────────────────────────────────────────
# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# MODELS_DIR = BASE_DIR / "data" / "06_models"
# LOGS_DIR = BASE_DIR / "logs"
# LOGS_DIR.mkdir(parents=True, exist_ok=True)
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # ── Aplikacja ─────────────────────────────────────────────────
# app = FastAPI(
#     title="Car Price Prediction API",
#     description="Predykcja cen samochodów – model ML (Random Forest / Gradient Boosting)",
#     version="1.0.0",
# )
#
# # ── Ładowanie modeli przy starcie ─────────────────────────────
# _model = None
# _encoders = None
# _selected_features = None
# _training_stats = None
# _model_name = "unknown"
#
#
# def _load_artifacts():
#     global _model, _encoders, _selected_features, _training_stats, _model_name
#
#     rf_path = MODELS_DIR / "regressor.pickle"
#     gb_path = MODELS_DIR / "gradient_boosting_model.pickle"
#     enc_path = MODELS_DIR / "target_encoders.pickle"
#     feat_path = MODELS_DIR / "selected_features.pickle"
#     stats_path = MODELS_DIR / "training_stats.json"
#
#     # Wybierz model (preferuj GB jeśli istnieje)
#     if gb_path.exists():
#         with open(gb_path, "rb") as f:
#             _model = pickle.load(f)
#         _model_name = "GradientBoostingRegressor"
#     elif rf_path.exists():
#         with open(rf_path, "rb") as f:
#             _model = pickle.load(f)
#         _model_name = "RandomForestRegressor"
#     else:
#         logger.warning("Brak wytrenowanego modelu. Uruchom pipeline Kedro.")
#
#     if enc_path.exists():
#         with open(enc_path, "rb") as f:
#             _encoders = pickle.load(f)
#
#     if feat_path.exists():
#         with open(feat_path, "rb") as f:
#             _selected_features = pickle.load(f)
#
#     if stats_path.exists():
#         with open(stats_path, "r") as f:
#             _training_stats = json.load(f)
#
#     logger.info(f"Załadowano model: {_model_name}")
#
#
# _load_artifacts()
#
#
# # ── Schematy danych ───────────────────────────────────────────
# class CarFeatures(BaseModel):
#     Manufacturer: str = Field(..., example="TOYOTA")
#     Model: str = Field(..., example="Camry")
#     Category: str = Field(..., example="Sedan")
#     Leather_interior: int = Field(1, ge=0, le=1, example=1)
#     Fuel_type: str = Field(..., example="Petrol")
#     Engine_volume: float = Field(..., gt=0, example=2.5)
#     Mileage: float = Field(..., ge=0, example=50000)
#     Cylinders: float = Field(..., gt=0, example=4)
#     Gear_box_type: str = Field(..., example="Automatic")
#     Drive_wheels: str = Field(..., example="Front")
#     Doors: int = Field(4, ge=2, le=6, example=4)
#     Wheel: str = Field("Left wheel", example="Left wheel")
#     Color: str = Field(..., example="Black")
#     Airbags: int = Field(4, ge=0, example=4)
#     Levy: float = Field(0.0, ge=0, example=0.0)
#     Turbo: int = Field(0, ge=0, le=1, example=0)
#     Age: int = Field(..., ge=0, example=5)
#
#
# class PredictionResponse(BaseModel):
#     predicted_price: float
#     predicted_price_log: float
#     model_used: str
#     timestamp: str
#
#
# class BatchPredictionResponse(BaseModel):
#     predictions: List[float]
#     model_used: str
#     count: int
#     timestamp: str
#
#
# # ── Pomocnicze ────────────────────────────────────────────────
# REFERENCE_YEAR = 2021
# LUXURY_BRANDS = {"LEXUS", "BMW", "MERCEDES-BENZ", "PORSCHE", "JAGUAR", "LAND ROVER", "TESLA", "BENTLEY", "MASERATI"}
#
#
# def _features_to_df(car: CarFeatures) -> pd.DataFrame:
#     """Konwertuje CarFeatures na DataFrame z inżynierią cech."""
#     d = {
#         "Manufacturer": car.Manufacturer.upper(),
#         "Model": car.Model,
#         "Category": car.Category,
#         "Leather interior": car.Leather_interior,
#         "Fuel type": car.Fuel_type,
#         "Engine volume": car.Engine_volume,
#         "Mileage": car.Mileage,
#         "Cylinders": car.Cylinders,
#         "Gear box type": car.Gear_box_type,
#         "Drive wheels": car.Drive_wheels,
#         "Doors": car.Doors,
#         "Wheel": car.Wheel,
#         "Color": car.Color,
#         "Airbags": car.Airbags,
#         "Levy": car.Levy,
#         "Turbo": car.Turbo,
#         "Age": car.Age,
#     }
#     df = pd.DataFrame([d])
#
#     # Cechy inżynieryjne
#     df["Mileage_per_year"] = df["Mileage"] / (df["Age"] + 1)
#     df["Log_Mileage"] = np.log1p(df["Mileage"])
#     df["Engine_per_cyl"] = df["Engine volume"] / (df["Cylinders"] + 0.1)
#     df["Is_Luxury"] = df["Manufacturer"].isin(LUXURY_BRANDS).astype(int)
#     df["Is_Automatic"] = df["Gear box type"].isin(["Automatic", "Tiptronic"]).astype(int)
#     df["Is_4x4"] = (df["Drive wheels"] == "4x4").astype(int)
#     df["Is_Eco"] = df["Fuel type"].isin(["Hybrid", "LPG", "CNG", "Plug-in Hybrid", "Hydrogen"]).astype(int)
#     df["Safety_Luxury"] = df["Airbags"] * df["Is_Luxury"]
#
#     return df
#
#
# def _apply_encoders(df: pd.DataFrame) -> pd.DataFrame:
#     """Aplikuje target encoding."""
#     if _encoders is None:
#         return df
#     df = df.copy()
#     for col, enc in _encoders.items():
#         if col in df.columns:
#             df[col] = df[col].map(enc["map"]).fillna(enc["global_mean"]).astype(float)
#     return df
#
#
# def _select_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Wybiera cechy używane przez model."""
#     if _selected_features is None:
#         return df.select_dtypes(include=[np.number])
#     cols = [c for c in _selected_features if c in df.columns]
#     missing = [c for c in _selected_features if c not in df.columns]
#     for c in missing:
#         df[c] = 0.0
#     return df[_selected_features]
#
#
# def _log_prediction(input_data: dict, prediction: float) -> None:
#     """Loguje predykcję do pliku."""
#     log_path = LOGS_DIR / "predictions.log"
#     entry = {
#         "timestamp": datetime.now().isoformat(),
#         "input": input_data,
#         "predicted_price": prediction,
#     }
#     with open(log_path, "a") as f:
#         f.write(json.dumps(entry, ensure_ascii=False) + "\n")
#
#
# # ── Endpointy ─────────────────────────────────────────────────
# @app.get("/health", tags=["System"])
# def health():
#     """Sprawdza status serwisu."""
#     return {
#         "status": "ok",
#         "model_loaded": _model is not None,
#         "model_name": _model_name,
#         "timestamp": datetime.now().isoformat(),
#     }
#
#
# @app.get("/model/info", tags=["System"])
# def model_info():
#     """Zwraca informacje o załadowanym modelu."""
#     if _model is None:
#         raise HTTPException(status_code=503, detail="Model nie jest załadowany.")
#     return {
#         "model_type": _model_name,
#         "n_features": len(_selected_features) if _selected_features else "unknown",
#         "selected_features": _selected_features,
#         "encoders_available": _encoders is not None,
#         "training_stats_available": _training_stats is not None,
#     }
#
#
# @app.post("/predict", response_model=PredictionResponse, tags=["Predykcja"])
# def predict(car: CarFeatures):
#     """Przewiduje cenę pojedynczego samochodu."""
#     if _model is None:
#         raise HTTPException(status_code=503, detail="Model nie jest załadowany. Uruchom pipeline Kedro.")
#
#     try:
#         df = _features_to_df(car)
#         df = _apply_encoders(df)
#         df = _select_features(df)
#
#         log_price = float(_model.predict(df)[0])
#         price = float(np.expm1(log_price))
#
#         _log_prediction(car.model_dump(), price)
#
#         return PredictionResponse(
#             predicted_price=round(price, 2),
#             predicted_price_log=round(log_price, 6),
#             model_used=_model_name,
#             timestamp=datetime.now().isoformat(),
#         )
#     except Exception as e:
#         logger.error(f"Błąd predykcji: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predykcja"])
# def predict_batch(cars: List[CarFeatures]):
#     """Przewiduje ceny dla listy samochodów."""
#     if _model is None:
#         raise HTTPException(status_code=503, detail="Model nie jest załadowany.")
#     if len(cars) > 1000:
#         raise HTTPException(status_code=400, detail="Maksymalnie 1000 rekordów na raz.")
#
#     try:
#         dfs = [_features_to_df(car) for car in cars]
#         df = pd.concat(dfs, ignore_index=True)
#         df = _apply_encoders(df)
#         df = _select_features(df)
#
#         log_prices = _model.predict(df)
#         prices = [round(float(np.expm1(p)), 2) for p in log_prices]
#
#         return BatchPredictionResponse(
#             predictions=prices,
#             model_used=_model_name,
#             count=len(prices),
#             timestamp=datetime.now().isoformat(),
#         )
#     except Exception as e:
#         logger.error(f"Błąd predykcji batch: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.get("/drift", tags=["Monitoring"])
# def drift_report():
#     """Zwraca statystyki treningowe do wykrywania driftu danych."""
#     if _training_stats is None:
#         raise HTTPException(status_code=404, detail="Brak statystyk treningowych.")
#     return {
#         "training_stats": _training_stats,
#         "description": "Porównaj statystyki nowych danych z training_stats aby wykryć drift.",
#         "timestamp": datetime.now().isoformat(),
#     }
