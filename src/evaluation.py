"""
evaluation.py
-------------
Regression evaluation utilities shared across models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE and R^2 for a set of true vs predicted values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def within_tolerance_accuracy(y_true, y_pred, tolerance: float = 0.10) -> float:
    """
    Business KPI helper: percentage of predictions that fall within
    +/- `tolerance` (default 10%) of the actual sale price.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    pct_error = np.abs(y_pred - y_true) / y_true
    within = (pct_error <= tolerance).mean() * 100
    return within


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    results: dict of {model_name: {"MAE":..., "RMSE":..., "R2":...}}
    Returns a tidy DataFrame sorted by R2 descending.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "MAE": round(metrics["MAE"], 2),
            "RMSE": round(metrics["RMSE"], 2),
            "R2": round(metrics["R2"], 4),
        })
    df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return df
