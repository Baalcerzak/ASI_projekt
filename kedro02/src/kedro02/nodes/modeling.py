import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import randint, uniform

logger = logging.getLogger(__name__)



def _setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)



def _compute_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "r2": float(r2_score(y_true_log, y_pred_log)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
    }



def select_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    random_state: int = 42,
) -> tuple:
    X_train = train_data.drop(columns=["Price"])
    y_train = train_data["Price"]
    X_test = test_data.drop(columns=["Price"])

    selector_model = RandomForestRegressor(
        n_estimators=100, random_state=random_state, n_jobs=-1
    )
    selector_model.fit(X_train, y_train)

    selector = SelectFromModel(selector_model, prefit=True, threshold="median")
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)

    selected_cols = X_train.columns[selector.get_support()].tolist()
    logger.info(f"Wybrano {len(selected_cols)} cech z {X_train.shape[1]}")

    train_sel = pd.DataFrame(X_train_sel, columns=selected_cols, index=X_train.index)
    train_sel["Price"] = y_train.values

    test_sel = pd.DataFrame(X_test_sel, columns=selected_cols, index=X_test.index)
    test_sel["Price"] = test_data["Price"].values

    return train_sel, test_sel, selected_cols



def train_random_forest(
    train_data: pd.DataFrame,
    rf_n_estimators: int = 500,
    rf_max_depth: int = 15,
    rf_min_samples_split: int = 10,
    rf_min_samples_leaf: int = 5,
    rf_max_features: float = 0.5,
    random_state: int = 42,
    mlflow_experiment_name: str = "car_price_prediction",
    mlflow_tracking_uri: str = "mlruns",
) -> RandomForestRegressor:
    X_train = train_data.drop(columns=["Price"])
    y_train = train_data["Price"]

    _setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name)

    with mlflow.start_run(run_name="RandomForest"):
        params = {
            "n_estimators": rf_n_estimators,
            "max_depth": rf_max_depth,
            "min_samples_split": rf_min_samples_split,
            "min_samples_leaf": rf_min_samples_leaf,
            "max_features": rf_max_features,
            "random_state": random_state,
        }
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])

        model = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            max_features=rf_max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        mlflow.log_metric("cv_r2_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_r2_std", float(cv_scores.std()))

        y_pred_train = model.predict(X_train)
        train_metrics = _compute_metrics(y_train.values, y_pred_train)
        mlflow.log_metric("train_r2", train_metrics["r2"])
        mlflow.log_metric("train_mae", train_metrics["mae"])

        mlflow.sklearn.log_model(model, "random_forest_model")
        logger.info(
            f"RF wytrenowany | CV R²={cv_scores.mean():.4f} ± {cv_scores.std():.4f} "
            f"| Train MAE={train_metrics['mae']:.0f}"
        )

    return model



def train_gradient_boosting(
    train_data: pd.DataFrame,
    gb_n_estimators: int = 500,
    gb_learning_rate: float = 0.05,
    gb_max_depth: int = 5,
    gb_subsample: float = 0.8,
    random_state: int = 42,
    mlflow_experiment_name: str = "car_price_prediction",
    mlflow_tracking_uri: str = "mlruns",
) -> GradientBoostingRegressor:
    X_train = train_data.drop(columns=["Price"])
    y_train = train_data["Price"]

    _setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name)

    with mlflow.start_run(run_name="GradientBoosting"):
        params = {
            "n_estimators": gb_n_estimators,
            "learning_rate": gb_learning_rate,
            "max_depth": gb_max_depth,
            "subsample": gb_subsample,
            "random_state": random_state,
        }
        mlflow.log_params(params)
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])

        model = GradientBoostingRegressor(
            n_estimators=gb_n_estimators,
            learning_rate=gb_learning_rate,
            max_depth=gb_max_depth,
            subsample=gb_subsample,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        train_metrics = _compute_metrics(y_train.values, y_pred_train)
        mlflow.log_metric("train_r2", train_metrics["r2"])
        mlflow.log_metric("train_mae", train_metrics["mae"])

        mlflow.sklearn.log_model(model, "gradient_boosting_model")
        logger.info(
            f"GB wytrenowany | Train R²={train_metrics['r2']:.4f} "
            f"| Train MAE={train_metrics['mae']:.0f}"
        )

    return model



def evaluate_model(
    model,
    test_data: pd.DataFrame,
    mlflow_experiment_name: str = "car_price_prediction",
    mlflow_tracking_uri: str = "mlruns",
) -> dict:
    X_test = test_data.drop(columns=["Price"])
    y_test = test_data["Price"]

    y_pred = model.predict(X_test)
    metrics = _compute_metrics(y_test.values, y_pred)
    metrics["model_type"] = type(model).__name__

    _setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name)
    with mlflow.start_run(run_name=f"Evaluate_{type(model).__name__}"):
        mlflow.log_param("model_type", type(model).__name__)
        for k, v in metrics.items():
            if isinstance(v, float):
                mlflow.log_metric(k, v)

    logger.info(
        f"Ewaluacja {type(model).__name__}: "
        f"R²={metrics['r2']:.4f}, MAE={metrics['mae']:.0f}, RMSE={metrics['rmse']:.0f}"
    )
    return metrics



def compare_models(
    metrics_rf: dict,
    metrics_gb: dict,
    metrics_tuned_rf: dict = None,
    metrics_automl: dict = None,
    mlflow_experiment_name: str = "car_price_prediction",
    mlflow_tracking_uri: str = "mlruns",
) -> dict:
    candidates = {
        "RandomForest": metrics_rf,
        "GradientBoosting": metrics_gb,
    }
    if metrics_tuned_rf and metrics_tuned_rf.get("r2", 0) > 0:
        candidates["TunedRandomForest"] = metrics_tuned_rf
    if metrics_automl and metrics_automl.get("r2", 0) > 0:
        automl_name = metrics_automl.get("model_type", "AutoML")
        candidates[automl_name] = metrics_automl

    better = max(candidates, key=lambda k: candidates[k]["r2"])

    comparison = {**candidates, "better_model": better}

    _setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name)
    with mlflow.start_run(run_name="Model_Comparison"):
        mlflow.log_param("better_model", better)
        for name, m in candidates.items():
            mlflow.log_metric(f"{name.lower()}_r2", m["r2"])
            mlflow.log_metric(f"{name.lower()}_mae", m["mae"])

    r2_summary = ", ".join(f"{k} R²={v['r2']:.4f}" for k, v in candidates.items())
    logger.info(f"Porównanie: {r2_summary} → Najlepszy: {better}")
    return comparison



def tune_hyperparameters(
    train_data: pd.DataFrame,
    tuning_n_iter: int = 20,
    tuning_cv: int = 3,
    random_state: int = 42,
    mlflow_experiment_name: str = "car_price_prediction",
    mlflow_tracking_uri: str = "mlruns",
) -> RandomForestRegressor:
    X_train = train_data.drop(columns=["Price"])
    y_train = train_data["Price"]

    param_dist = {
        "n_estimators": randint(100, 600),
        "max_depth": randint(5, 25),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": uniform(0.3, 0.6),
    }

    base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=tuning_n_iter,
        cv=tuning_cv,
        scoring="r2",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    _setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name)
    with mlflow.start_run(run_name="HyperparameterTuning_RF"):
        search.fit(X_train, y_train)
        best_params = search.best_params_
        best_score = search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "TunedRandomForestRegressor")
        mlflow.log_param("tuning_n_iter", tuning_n_iter)
        mlflow.log_param("tuning_cv", tuning_cv)
        mlflow.log_metric("best_cv_r2", float(best_score))
        mlflow.sklearn.log_model(search.best_estimator_, "tuned_rf_model")

        logger.info(
            f"Strojenie RF zakończone | Best CV R²={best_score:.4f} | Params: {best_params}"
        )

    return search.best_estimator_



def train_automl(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    automl_time_limit: int = 120,
    automl_presets: str = "medium_quality",
    mlflow_experiment_name: str = "car_price_prediction",
    mlflow_tracking_uri: str = "mlruns",
) -> dict:
    try:
        from pycaret import regression as pcr
    except Exception as _pycaret_err:
        logger.warning("PyCaret niedostępny (%s: %s). Pomijam węzeł AutoML.", type(_pycaret_err).__name__, _pycaret_err)
        return {"r2": 0.0, "mae": float("inf"), "rmse": float("inf"), "model_type": "PyCaret_skipped"}

    X_test = test_data.drop(columns=["Price"])
    y_test = test_data["Price"]

    pcr.setup(
        data=train_data.copy(),
        target="Price",
        session_id=42,
        fold=5,
        fold_strategy="kfold",
        verbose=False,
        log_experiment=False,
    )

    best_model = pcr.compare_models(sort="R2", turbo=True)
    final_model = pcr.finalize_model(best_model)

    preds_df = pcr.predict_model(final_model, data=X_test.copy())
    y_pred = preds_df["prediction_label"].values

    metrics = _compute_metrics(y_test.values, y_pred)
    metrics["model_type"] = "PyCaret"

    leaderboard = pcr.pull()
    if leaderboard is not None and not leaderboard.empty:
        best_model_name = str(leaderboard.iloc[0].get("Model", "unknown"))
    else:
        best_model_name = "unknown"
    metrics["best_automl_model"] = best_model_name

    _setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name)
    with mlflow.start_run(run_name="AutoML_PyCaret"):
        mlflow.log_param("model_type", "PyCaret")
        mlflow.log_param("automl_presets", automl_presets)
        mlflow.log_param("automl_time_limit", automl_time_limit)
        mlflow.log_param("best_automl_model", best_model_name)
        for k, v in metrics.items():
            if isinstance(v, float):
                mlflow.log_metric(k, v)

    logger.info(
        f"PyCaret zakończony | R²={metrics['r2']:.4f}, MAE={metrics['mae']:.0f} "
        f"| Najlepszy model: {best_model_name}"
    )
    return metrics
