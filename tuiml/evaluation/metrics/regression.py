"""
Regression evaluation metrics.

Equivalent to Weka's Evaluation class regression methods.
"""

from typing import Optional, Union
import numpy as np
from tuiml.base.metrics import check_consistent_length, safe_divide

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE). Equivalent to Weka's Evaluation.meanAbsoluteError()."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, squared: bool = True) -> float:
    """Compute Mean Squared Error (MSE). Equivalent to Weka's Evaluation.rootMeanSquaredError() squared."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse if squared else np.sqrt(mse))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE). Equivalent to Weka's Evaluation.rootMeanSquaredError()."""
    return mean_squared_error(y_true, y_pred, squared=False)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 (coefficient of determination). Equivalent to Weka's Evaluation.correlationCoefficient() squared."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - safe_divide(ss_res, ss_tot))

def relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Relative Absolute Error (RAE). Equivalent to Weka's Evaluation.relativeAbsoluteError()."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mae = np.sum(np.abs(y_true - y_pred))
    mae_mean = np.sum(np.abs(y_true - np.mean(y_true)))
    return float(safe_divide(mae, mae_mean) * 100)

def root_relative_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Relative Squared Error (RRSE). Equivalent to Weka's Evaluation.rootRelativeSquaredError()."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mse = np.sum((y_true - y_pred) ** 2)
    mse_mean = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(np.sqrt(safe_divide(mse, mse_mean)) * 100)

def correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute correlation coefficient. Equivalent to Weka's Evaluation.correlationCoefficient()."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0
