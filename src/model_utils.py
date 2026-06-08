"""Model evaluation utilities for MPF regression."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def clip_mpf(values):
    """Clip MPF predictions to the physically valid interval [0, 1]."""
    return np.clip(values, 0.0, 1.0)


def evaluate_regression(y_true, y_pred):
    """Return MSE, RMSE, MAE and R² for regression predictions."""
    y_pred = clip_mpf(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
