import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline

from kedro02.nodes.preprocessing import (
    clean_data,
    preprocess_data,
    split_data,
    create_target_encoders,
    apply_target_encoders,
    save_training_stats,
)
from kedro02.nodes.modeling import (
    select_features,
    train_random_forest,
    train_gradient_boosting,
    tune_hyperparameters,
    train_automl,
    evaluate_model,
    compare_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=[
                    "train_raw",
                    "params:reference_year",
                    "params:price_min",
                    "params:price_max",
                    "params:mileage_max",
                    "params:luxury_brands",
                ],
                outputs="cleaned_data",
                name="clean_data_node",
                tags=["preprocessing"],
            ),
            node(
                func=save_training_stats,
                inputs="cleaned_data",
                outputs="training_stats",
                name="save_training_stats_node",
                tags=["preprocessing", "monitoring"],
            ),
            node(
                func=preprocess_data,
                inputs="cleaned_data",
                outputs="preprocessed_data",
                name="preprocess_data_node",
                tags=["preprocessing"],
            ),
            node(
                func=split_data,
                inputs=[
                    "preprocessed_data",
                    "params:test_size",
                    "params:random_state",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
                tags=["preprocessing"],
            ),
            node(
                func=create_target_encoders,
                inputs=["X_train", "y_train", "params:smoothing_factor"],
                outputs="target_encoders",
                name="create_target_encoders_node",
                tags=["feature_engineering"],
            ),
            node(
                func=apply_target_encoders,
                inputs=["X_train", "target_encoders"],
                outputs="X_train_encoded",
                name="apply_encoders_train_node",
                tags=["feature_engineering"],
            ),
            node(
                func=apply_target_encoders,
                inputs=["X_test", "target_encoders"],
                outputs="X_test_encoded",
                name="apply_encoders_test_node",
                tags=["feature_engineering"],
            ),
            node(
                func=lambda X, y: pd.concat(
                    [X, y.rename(columns={y.columns[0]: "Price"})], axis=1
                ),
                inputs=["X_train_encoded", "y_train"],
                outputs="preprocessed_train",
                name="combine_train_node",
                tags=["feature_engineering"],
            ),
            node(
                func=lambda X, y: pd.concat(
                    [X, y.rename(columns={y.columns[0]: "Price"})], axis=1
                ),
                inputs=["X_test_encoded", "y_test"],
                outputs="preprocessed_test",
                name="combine_test_node",
                tags=["feature_engineering"],
            ),
            node(
                func=select_features,
                inputs=["preprocessed_train", "preprocessed_test", "params:random_state"],
                outputs=["selected_train", "selected_test", "selected_features"],
                name="select_features_node",
                tags=["feature_engineering"],
            ),
            node(
                func=train_random_forest,
                inputs=[
                    "selected_train",
                    "params:rf_n_estimators",
                    "params:rf_max_depth",
                    "params:rf_min_samples_split",
                    "params:rf_min_samples_leaf",
                    "params:rf_max_features",
                    "params:random_state",
                    "params:mlflow_experiment_name",
                    "params:mlflow_tracking_uri",
                ],
                outputs="regressor",
                name="train_random_forest_node",
                tags=["training"],
            ),
            node(
                func=train_gradient_boosting,
                inputs=[
                    "selected_train",
                    "params:gb_n_estimators",
                    "params:gb_learning_rate",
                    "params:gb_max_depth",
                    "params:gb_subsample",
                    "params:random_state",
                    "params:mlflow_experiment_name",
                    "params:mlflow_tracking_uri",
                ],
                outputs="gradient_boosting_model",
                name="train_gradient_boosting_node",
                tags=["training"],
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "regressor",
                    "selected_test",
                    "params:mlflow_experiment_name",
                    "params:mlflow_tracking_uri",
                ],
                outputs="metrics_rf",
                name="evaluate_rf_node",
                tags=["evaluation"],
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "gradient_boosting_model",
                    "selected_test",
                    "params:mlflow_experiment_name",
                    "params:mlflow_tracking_uri",
                ],
                outputs="metrics_gb",
                name="evaluate_gb_node",
                tags=["evaluation"],
            ),
            node(
                func=tune_hyperparameters,
                inputs=[
                    "selected_train",
                    "params:tuning_n_iter",
                    "params:tuning_cv",
                    "params:random_state",
                    "params:mlflow_experiment_name",
                    "params:mlflow_tracking_uri",
                ],
                outputs="tuned_rf_model",
                name="tune_hyperparameters_node",
                tags=["training", "tuning"],
            ),
            node(
                func=train_automl,
                inputs=[
                    "selected_train",
                    "selected_test",
                    "params:automl_time_limit",
                    "params:automl_presets",
                    "params:mlflow_experiment_name",
                    "params:mlflow_tracking_uri",
                ],
                outputs="metrics_automl",
                name="train_automl_node",
                tags=["training", "automl"],
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "tuned_rf_model",
                    "selected_test",
                    "params:mlflow_experiment_name",
                    "params:mlflow_tracking_uri",
                ],
                outputs="metrics_tuned_rf",
                name="evaluate_tuned_rf_node",
                tags=["evaluation"],
            ),
            node(
                func=compare_models,
                inputs=[
                    "metrics_rf",
                    "metrics_gb",
                    "metrics_tuned_rf",
                    "metrics_automl",
                    "params:mlflow_experiment_name",
                    "params:mlflow_tracking_uri",
                ],
                outputs="model_comparison",
                name="compare_models_node",
                tags=["evaluation"],
            ),
        ]
    )
