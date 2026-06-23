# Kedro02 – Predykcja cen samochodów

Projekt zaliczeniowy z zakresu MLOps. Przewidywanie cen samochodów używanych na podstawie danych z Kaggle (2021).

**Dataset:** [Car Prices Dataset – Kaggle](https://www.kaggle.com/datasets/sidharth178/car-prices-dataset/data)  
**Problem:** Regresja – przewidywanie ceny samochodu (USD)  
**Modele:** Random Forest, Gradient Boosting, AutoGluon (AutoML)  
**Stack:** Kedro · MLflow · AutoGluon · FastAPI · Docker

---

## Spis treści

1. [Struktura projektu](#struktura-projektu)
2. [Opis danych](#opis-danych)
3. [Pipeline ML](#pipeline-ml)
4. [Uruchomienie](#uruchomienie)
5. [API](#api)
6. [MLflow – śledzenie eksperymentów](#mlflow)
7. [Docker](#docker)
8. [Wyniki](#wyniki)
9. [Monitoring](#monitoring)

---

## Struktura projektu

```
kedro02/
├── conf/
│   ├── base/
│   │   ├── catalog.yml          # Definicje datasetów Kedro
│   │   └── parameters.yml       # Parametry pipeline'u
│   └── local/
│       └── credentials.yml      # Lokalne dane dostępowe (nie commituj!)
├── data/
│   ├── 01_raw/                  # Surowe dane (train.csv)
│   ├── 02_intermediate/         # Dane po czyszczeniu
│   ├── 03_primary/              # Dane po preprocessingu
│   ├── 04_feature/              # Podział train/test
│   ├── 05_model_input/          # Dane po encodingu i selekcji cech
│   ├── 06_models/               # Wytrenowane modele i artefakty
│   └── 08_reporting/            # Metryki i raporty
├── notebooks/
│   └── 01_car_price_baseline.ipynb   # EDA + model bazowy
├── src/
│   └── kedro02/
│       ├── nodes/
│       │   ├── preprocessing.py  # Węzły czyszczenia i preprocessingu
│       │   └── modeling.py       # Węzły treningu i ewaluacji
│       ├── pipelines/
│       │   └── pipeline.py       # Definicja pipeline'u Kedro
│       ├── pipeline_registry.py  # Rejestr pipeline'ów
│       └── api.py                # FastAPI – serwis predykcji
├── logs/
│   └── predictions.log           # Logi predykcji API
├── Dockerfile                    # Obraz Docker dla API
├── pyproject.toml                # Konfiguracja projektu
└── README.md
```

---

## Opis danych

| Kolumna | Opis |
|---------|------|
| `Price` | Cena samochodu (USD) – zmienna docelowa |
| `Manufacturer` | Producent (np. TOYOTA, BMW) |
| `Model` | Model samochodu |
| `Category` | Kategoria (Sedan, Jeep, Hatchback, ...) |
| `Prod. year` | Rok produkcji |
| `Fuel type` | Typ paliwa (Petrol, Diesel, Hybrid, ...) |
| `Engine volume` | Pojemność silnika (litry) |
| `Mileage` | Przebieg (km) |
| `Cylinders` | Liczba cylindrów |
| `Gear box type` | Skrzynia biegów (Automatic, Manual, ...) |
| `Drive wheels` | Napęd (Front, Rear, 4x4) |
| `Doors` | Liczba drzwi |
| `Airbags` | Liczba poduszek powietrznych |
| `Leather interior` | Skórzana tapicerka (Yes/No) |
| `Color` | Kolor nadwozia |
| `Levy` | Podatek importowy |

**Cechy inżynieryjne** tworzone w pipeline:
- `Age` = 2021 − rok produkcji
- `Mileage_per_year` = Mileage / (Age + 1)
- `Log_Mileage` = log(Mileage + 1)
- `Engine_per_cyl` = Engine volume / (Cylinders + 0.1)
- `Is_Luxury`, `Is_Automatic`, `Is_4x4`, `Is_Eco`, `Turbo`
- `Safety_Luxury` = Airbags × Is_Luxury

---

## Pipeline ML

Pipeline składa się z węzłów Kedro pogrupowanych w etapy:

```
train_raw
    │
    ▼
[preprocessing]
clean_data_node          ← czyszczenie, filtrowanie outlierów
    │
    ├──► save_training_stats_node   ← statystyki do monitoringu driftu
    │
    ▼
preprocess_data_node     ← log-transformacja ceny
    │
    ▼
split_data_node          ← podział 80/20 train/test
    │
    ▼
[feature_engineering]
create_target_encoders_node  ← target encoding dla zmiennych kategorycznych
    │
    ├──► apply_encoders_train_node
    └──► apply_encoders_test_node
              │
              ▼
         combine_train/test_node   ← łączenie X + y
              │
              ▼
         select_features_node      ← selekcja cech (SelectFromModel RF)
              │
    ┌─────────┴──────────┬──────────────┐
    ▼                    ▼              ▼
[training]          [tuning]        [automl]
train_rf_node   tune_hyperparams_node  train_automl_node
train_gb_node        │                     │
    │                ▼                     ▼
    ▼         evaluate_tuned_node   evaluate_automl_node
evaluate_rf_node                          │
evaluate_gb_node                          │
    │                    │                │
    └────────────────────┴────────────────┘
                         ▼
                 compare_models_node  ← wybór najlepszego modelu
```

### Uruchomienie pełnego pipeline'u

```bash
cd kedro02
kedro run
```

### Uruchomienie wybranego etapu (tag)

```bash
kedro run --tags preprocessing
kedro run --tags feature_engineering
kedro run --tags training
kedro run --tags tuning
kedro run --tags automl
kedro run --tags evaluation
```

### Uruchomienie wybranego węzła

```bash
kedro run --node clean_data_node
```

---

## Uruchomienie

### Wymagania

```bash
pip install kedro>=0.19.0 kedro-datasets pandas numpy scikit-learn mlflow pyarrow matplotlib seaborn fastapi "uvicorn[standard]" "pydantic>=2.0"
```

Lub z pliku konfiguracyjnego projektu:

```bash
cd kedro02
pip install -e .
```

### Dane

Dane treningowe (`train.csv`) z folderu `dataset-start/` są już skopiowane do `data/01_raw/train.csv`.

### Pipeline Kedro

```bash
cd kedro02
kedro run
```

### Notebook (model bazowy)

```bash
cd kedro02
jupyter notebook notebooks/01_car_price_baseline.ipynb
```

---

## API

> ⚠️ **Uwaga:** Moduł FastAPI (`src/kedro02/api.py`) jest obecnie zakomentowany i nieaktywny. Poniższy opis przedstawia planowaną funkcjonalność.

FastAPI serwuje predykcje wytrenowanego modelu.

### Uruchomienie lokalne

```bash
cd kedro02
uvicorn kedro02.api:app --host 0.0.0.0 --port 8000 --reload
```

Dokumentacja Swagger: http://localhost:8000/docs

### Endpointy

| Metoda | Endpoint | Opis |
|--------|----------|------|
| GET | `/health` | Status serwisu |
| GET | `/model/info` | Informacje o modelu |
| POST | `/predict` | Predykcja ceny jednego auta |
| POST | `/predict/batch` | Predykcja dla wielu aut (max 1000) |
| GET | `/drift` | Statystyki treningowe (monitoring driftu) |

### Przykład zapytania

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Manufacturer": "TOYOTA",
    "Model": "Camry",
    "Category": "Sedan",
    "Leather_interior": 1,
    "Fuel_type": "Petrol",
    "Engine_volume": 2.5,
    "Mileage": 50000,
    "Cylinders": 4,
    "Gear_box_type": "Automatic",
    "Drive_wheels": "Front",
    "Doors": 4,
    "Wheel": "Left wheel",
    "Color": "Black",
    "Airbags": 4,
    "Levy": 0,
    "Turbo": 0,
    "Age": 5
  }'
```

### Przykładowa odpowiedź

```json
{
  "predicted_price": 14250.50,
  "predicted_price_log": 9.564,
  "model_used": "GradientBoostingRegressor",
  "timestamp": "2024-01-15T12:00:00"
}
```

---

## MLflow

Eksperymenty są śledzone lokalnie w folderze `mlruns/`.

```bash
cd kedro02
mlflow ui --port 5000
```

UI dostępne pod: http://localhost:5000

Logowane metryki:
- `cv_r2_mean`, `cv_r2_std` – cross-validation R²
- `train_r2`, `train_mae` – metryki treningowe
- `r2`, `mae`, `rmse` – metryki testowe

---


## Docker

### Budowanie obrazu

```bash
cd kedro02
docker build -t car-price-api .
```

### Uruchomienie kontenera

```bash
docker run -p 8000:8000 car-price-api
```

API dostępne pod: http://localhost:8000

---

## Wyniki

| Model | R² (test) | MAE (USD) | RMSE (USD) |
|-------|-----------|-----------|------------|
| Random Forest | 0.744 | 5 402 | 12 559 |
| Gradient Boosting | **0.751** | 5 433 | **11 401** |
| RF (po tuningu) | – | – | – |
| GB (po tuningu) | – | – | – |
| AutoGluon | – | – | – |

> Najlepszy model wybierany jest automatycznie przez węzeł `compare_models_node` na podstawie R² i RMSE.

Transformacja logarytmiczna ceny znacząco poprawia jakość modelu.  
Najważniejsze cechy: wiek auta, przebieg, pojemność silnika, marka.

---

## Monitoring

Predykcje API są logowane do `logs/predictions.log` (format JSON Lines).  
Endpoint `/drift` zwraca statystyki danych treningowych do porównania z danymi produkcyjnymi.
