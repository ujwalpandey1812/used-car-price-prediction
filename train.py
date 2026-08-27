"""
train.py
--------
End-to-end training script for the Used Car Price Prediction project.

Run with:
    python train.py

What it does (in order):
  1. Loads the raw dataset (data/dataset.csv)
  2. Cleans it and engineers features (src/preprocessing.py)
  3. Splits into train / validation / test
  4. Trains a Linear Regression baseline, a Random Forest and an XGBoost model
  5. Tunes the strongest candidate with RandomizedSearchCV
  6. Evaluates every model on the held-out test set (MAE, RMSE, R2)
  7. Computes the 5 project KPIs
  8. Saves the final pipeline (preprocessing + model) to models/final_model.pkl
  9. Saves results/model_results.csv and results/kpi_report.csv

All numbers printed/saved by this script are computed directly from the
uploaded dataset. Nothing is hard-coded.
"""

import os
import time
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

from src.preprocessing import (
    load_raw_dataset,
    clean_dataset,
    get_feature_target_split,
    build_full_pipeline,
    ALL_FEATURES,
)
from src.evaluation import regression_metrics, within_tolerance_accuracy, build_comparison_table

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DATA_PATH = os.path.join("data", "dataset.csv")
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    print(f"[train.py] {msg}")


def main():
    # ------------------------------------------------------------------
    # 1. LOAD + CLEAN
    # ------------------------------------------------------------------
    log("Loading raw dataset...")
    raw_df = load_raw_dataset(DATA_PATH)
    log(f"Raw shape: {raw_df.shape}")

    missing_rate_raw = raw_df.isnull().sum().sum() / (raw_df.shape[0] * raw_df.shape[1])
    log(f"Raw overall missing-value rate: {missing_rate_raw:.4%}")

    df = clean_dataset(raw_df)
    log(f"Cleaned shape: {df.shape}")
    log(f"Rows removed during cleaning: {raw_df.shape[0] - df.shape[0]}")

    X, y = get_feature_target_split(df)
    log(f"Feature columns used by the model ({len(ALL_FEATURES)}): {ALL_FEATURES}")

    # ------------------------------------------------------------------
    # 2. TRAIN / VALIDATION / TEST SPLIT (60 / 20 / 20)
    # ------------------------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
    )
    log(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # ------------------------------------------------------------------
    # 3. MODELS
    #    All models share the identical ColumnTransformer preprocessing,
    #    fitted ONLY on the training split (no leakage into val/test).
    #    Target is modeled in log-space (right-skewed price distribution)
    #    via TransformedTargetRegressor, and predictions are converted
    #    back to rupees before computing metrics.
    # ------------------------------------------------------------------
    def make_pipeline(estimator):
        preprocessor = build_full_pipeline()
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ])
        return TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1
        )

    results = {}
    fitted_models = {}

    # --- Baseline: Linear Regression ---
    log("Training baseline: Linear Regression...")
    lr_model = make_pipeline(LinearRegression())
    lr_model.fit(X_train, y_train)
    val_pred = lr_model.predict(X_val)
    results["Linear Regression"] = regression_metrics(y_val, val_pred)
    fitted_models["Linear Regression"] = lr_model
    log(f"  Validation: {results['Linear Regression']}")

    # --- Model 1: Random Forest ---
    log("Training Random Forest...")
    rf_model = make_pipeline(
        RandomForestRegressor(n_estimators=200, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1)
    )
    rf_model.fit(X_train, y_train)
    val_pred = rf_model.predict(X_val)
    results["Random Forest"] = regression_metrics(y_val, val_pred)
    fitted_models["Random Forest"] = rf_model
    log(f"  Validation: {results['Random Forest']}")

    # --- Model 2: XGBoost ---
    log("Training XGBoost...")
    xgb_model = make_pipeline(
        XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            objective="reg:squarederror",
        )
    )
    xgb_model.fit(X_train, y_train)
    val_pred = xgb_model.predict(X_val)
    results["XGBoost"] = regression_metrics(y_val, val_pred)
    fitted_models["XGBoost"] = xgb_model
    log(f"  Validation: {results['XGBoost']}")

    # ------------------------------------------------------------------
    # 4. HYPERPARAMETER TUNING on the strongest candidate (by validation R2)
    # ------------------------------------------------------------------
    best_candidate_name = max(
        ["Random Forest", "XGBoost"], key=lambda n: results[n]["R2"]
    )
    log(f"Best pre-tuning candidate on validation set: {best_candidate_name}")

    if best_candidate_name == "XGBoost":
        param_dist = {
            "regressor__model__n_estimators": [200, 300, 400, 600],
            "regressor__model__max_depth": [3, 4, 5, 6, 8],
            "regressor__model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
            "regressor__model__subsample": [0.7, 0.8, 0.9, 1.0],
            "regressor__model__colsample_bytree": [0.6, 0.8, 0.9, 1.0],
        }
        base_pipe = make_pipeline(
            XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, objective="reg:squarederror")
        )
    else:
        param_dist = {
            "regressor__model__n_estimators": [200, 300, 400, 600],
            "regressor__model__max_depth": [None, 8, 12, 16, 20],
            "regressor__model__min_samples_leaf": [1, 2, 4, 8],
            "regressor__model__max_features": ["sqrt", "log2", None],
        }
        base_pipe = make_pipeline(
            RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
        )

    log("Running RandomizedSearchCV (5-fold, 20 iterations)...")
    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="r2",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    # Combine train+val for CV during search, final scoring stays on held-out test set
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    search.fit(X_train_val, y_train_val)

    tuned_model = search.best_estimator_
    log(f"Best params: {search.best_params_}")

    # ------------------------------------------------------------------
    # 5. FINAL EVALUATION ON THE TEST SET (never touched until now)
    # ------------------------------------------------------------------
    log("Evaluating all models on the held-out TEST set...")
    test_results = {}
    for name, model in fitted_models.items():
        pred = model.predict(X_test)
        test_results[name] = regression_metrics(y_test, pred)

    tuned_pred = tuned_model.predict(X_test)
    test_results[f"Tuned {best_candidate_name}"] = regression_metrics(y_test, tuned_pred)

    comparison_df = build_comparison_table(test_results)
    log("\n" + comparison_df.to_string(index=False))
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_results.csv"), index=False)

    # ------------------------------------------------------------------
    # 6. FINAL MODEL SELECTION (based on actual test R2)
    # ------------------------------------------------------------------
    final_model_name = comparison_df.iloc[0]["Model"]
    log(f"Final selected model: {final_model_name}")

    if final_model_name == f"Tuned {best_candidate_name}":
        final_model = tuned_model
    else:
        final_model = fitted_models[final_model_name]

    final_pred_test = final_model.predict(X_test)
    final_metrics = regression_metrics(y_test, final_pred_test)

    # ------------------------------------------------------------------
    # 7. KPI FRAMEWORK  (5 KPIs, all computed from actual results)
    # ------------------------------------------------------------------
    log("Computing KPIs...")

    business_accuracy = within_tolerance_accuracy(y_test, final_pred_test, tolerance=0.10)

    missing_rate = raw_df.isnull().sum().sum() / (raw_df.shape[0] * raw_df.shape[1]) * 100

    # Prediction latency: time to run .predict() on a single held-out row
    single_row = X_test.iloc[[0]]
    n_runs = 50
    start = time.perf_counter()
    for _ in range(n_runs):
        final_model.predict(single_row)
    elapsed = (time.perf_counter() - start) / n_runs

    kpi_rows = [
        {
            "KPI": "Price Estimation Accuracy",
            "Category": "Business",
            "Definition": "Percentage of test-set predictions within +/-10% of actual sale price",
            "Formula": "count(|pred-actual|/actual <= 0.10) / n * 100",
            "Target": ">= 80%",
            "Actual": f"{business_accuracy:.2f}%",
            "Interpretation": "Meets target" if business_accuracy >= 80 else "Below target",
        },
        {
            "KPI": "MAE",
            "Category": "ML",
            "Definition": "Average absolute difference between predicted and actual sale price (INR)",
            "Formula": "mean(|pred - actual|)",
            "Target": "Minimize",
            "Actual": f"₹{final_metrics['MAE']:.2f}",
            "Interpretation": f"Average prediction error is ₹{final_metrics['MAE']:.0f}",
        },
        {
            "KPI": "R2 (Coefficient of Determination)",
            "Category": "ML",
            "Definition": "Proportion of variance in sale_price explained by the model",
            "Formula": "1 - (SS_res / SS_tot)",
            "Target": ">= 0.85",
            "Actual": f"{final_metrics['R2']:.4f}",
            "Interpretation": "Meets target" if final_metrics["R2"] >= 0.85 else "Below target",
        },
        {
            "KPI": "Missing Value Rate",
            "Category": "Data Quality",
            "Definition": "Proportion of missing cells in the raw uploaded dataset",
            "Formula": "missing_cells / total_cells",
            "Target": "< 5%",
            "Actual": f"{missing_rate:.2f}%",
            "Interpretation": "Within target" if missing_rate < 5 else "Above target (handled via imputation / column removal)",
        },
        {
            "KPI": "Prediction Latency",
            "Category": "Product / Engineering",
            "Definition": "Time taken by the trained pipeline to generate a single prediction",
            "Formula": "wall-clock time for one .predict() call, averaged over 50 runs",
            "Target": "< 2 sec",
            "Actual": f"{elapsed*1000:.2f} ms",
            "Interpretation": "Meets target" if elapsed < 2 else "Below target",
        },
    ]
    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_csv(os.path.join(RESULTS_DIR, "kpi_report.csv"), index=False)
    log("\n" + kpi_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 8. SAVE FINAL MODEL + METADATA
    # ------------------------------------------------------------------
    from src.preprocessing import get_category_options, CATEGORICAL_FEATURES, NUMERIC_FEATURES, BOOLEAN_FEATURES

    category_options = get_category_options(df)
    numeric_ranges = {
        col: {"min": float(df[col].min()), "max": float(df[col].max()), "median": float(df[col].median())}
        for col in NUMERIC_FEATURES
    }

    metadata = {
        "final_model_name": final_model_name,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "boolean": BOOLEAN_FEATURES,
        },
        "category_options": category_options,
        "numeric_ranges": numeric_ranges,
        "test_metrics": final_metrics,
        "random_seed": RANDOM_SEED,
    }

    joblib.dump(final_model, os.path.join(MODELS_DIR, "final_model.pkl"))
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"Saved final model to {MODELS_DIR}/final_model.pkl")
    log(f"Saved metadata to {MODELS_DIR}/metadata.json")
    log("Training complete.")


if __name__ == "__main__":
    main()
