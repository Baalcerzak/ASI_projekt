import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

REFERENCE_YEAR = 2021
LUXURY_BRANDS = [
    "LEXUS", "BMW", "MERCEDES-BENZ", "PORSCHE",
    "JAGUAR", "LAND ROVER", "TESLA", "BENTLEY", "MASERATI",
]

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    logger.info(f"Wczytano dane z pliku dataset: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
    return df

def clean_data(
    df: pd.DataFrame,
    reference_year: int = REFERENCE_YEAR,
    price_min: int = 500,
    price_max: int = 300000,
    mileage_max: int = 1_500_000,
    luxury_brands: list = None,
) -> pd.DataFrame:
    if luxury_brands is None:
        luxury_brands = LUXURY_BRANDS

    df = df.copy()
    df = df.drop_duplicates()
    df = df.drop(columns=["ID"], errors="ignore")

    df["Price"] = df["Price"].replace("-", np.nan)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])
    df = df[(df["Price"] >= price_min) & (df["Price"] <= price_max)]

    df["Levy"] = df["Levy"].replace("-", np.nan)
    df["Levy"] = pd.to_numeric(df["Levy"], errors="coerce")
    df["Levy"] = df["Levy"].fillna(df["Levy"].median())

    if "Engine volume" in df.columns:
        df["Turbo"] = df["Engine volume"].astype(str).str.contains("Turbo").astype(int)
        df["Engine volume"] = (
            df["Engine volume"].astype(str).str.replace(" Turbo", "", regex=False)
        )
        df["Engine volume"] = pd.to_numeric(df["Engine volume"], errors="coerce")
        df["Engine volume"] = df["Engine volume"].fillna(df["Engine volume"].median())

    if "Mileage" in df.columns:
        df["Mileage"] = (
            df["Mileage"].astype(str).str.replace(" km", "", regex=False)
        )
        df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
        df["Mileage"] = df["Mileage"].fillna(df["Mileage"].median())
        df = df[df["Mileage"] <= mileage_max]

    if "Cylinders" in df.columns:
        df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce")
        df["Cylinders"] = df["Cylinders"].fillna(df["Cylinders"].median())

    if "Doors" in df.columns:
        df["Doors"] = df["Doors"].replace({"04-May": 4, "02-Mar": 2, ">5": 5})
        df["Doors"] = pd.to_numeric(df["Doors"], errors="coerce").fillna(4).astype(int)

    if "Leather interior" in df.columns:
        df["Leather interior"] = (df["Leather interior"] == "Yes").astype(int)

    if "Prod. year" in df.columns:
        df["Age"] = (reference_year - df["Prod. year"]).clip(lower=0)
        df = df.drop(columns=["Prod. year"])

    if "Mileage" in df.columns and "Age" in df.columns:
        df["Mileage_per_year"] = df["Mileage"] / (df["Age"] + 1)
        df["Log_Mileage"] = np.log1p(df["Mileage"])

    if "Engine volume" in df.columns and "Cylinders" in df.columns:
        df["Engine_per_cyl"] = df["Engine volume"] / (df["Cylinders"] + 0.1)

    if "Manufacturer" in df.columns:
        df["Is_Luxury"] = df["Manufacturer"].isin(luxury_brands).astype(int)

    if "Gear box type" in df.columns:
        df["Is_Automatic"] = df["Gear box type"].isin(["Automatic", "Tiptronic"]).astype(int)

    if "Drive wheels" in df.columns:
        df["Is_4x4"] = (df["Drive wheels"] == "4x4").astype(int)

    if "Fuel type" in df.columns:
        df["Is_Eco"] = df["Fuel type"].isin(
            ["Hybrid", "LPG", "CNG", "Plug-in Hybrid", "Hydrogen"]
        ).astype(int)

    if "Airbags" in df.columns and "Is_Luxury" in df.columns:
        df["Safety_Luxury"] = df["Airbags"] * df["Is_Luxury"]

    logger.info(f"Po czyszczeniu: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
    return df


def save_training_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "q25": float(df[col].quantile(0.25)),
            "q75": float(df[col].quantile(0.75)),
        }
    logger.info(f"Zapisano statystyki dla {len(stats)} kolumn")
    return stats


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Price"] = np.log1p(df["Price"])
    df = df.dropna()
    logger.info(f"Po preprocessingu: {df.shape[0]} wierszy")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df.drop(columns=["Price"], errors="ignore")
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Podział: train={X_train.shape[0]}, test={X_test.shape[0]}"
    )
    return X_train, X_test, y_train.to_frame(), y_test.to_frame()


def create_target_encoders(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    smoothing_factor: int = 15,
) -> dict:
    y = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train
    global_mean = float(y.mean())
    encoders = {}

    for col in X_train.select_dtypes(include=["object"]).columns:
        agg = y.groupby(X_train[col]).agg(["count", "mean"])
        counts = agg["count"]
        means = agg["mean"]
        smooth = (counts * means + smoothing_factor * global_mean) / (counts + smoothing_factor)
        encoders[col] = {
            "map": smooth.to_dict(),
            "global_mean": global_mean,
        }

    logger.info(f"Utworzono encodery dla {len(encoders)} kolumn kategorycznych")
    return encoders


def apply_target_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, enc in encoders.items():
        if col in df.columns:
            df[col] = df[col].map(enc["map"]).fillna(enc["global_mean"]).astype(float)
    return df
