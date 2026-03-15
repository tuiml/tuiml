"""Gradient Boosting frameworks (XGBoost, CatBoost, LightGBM)."""

# Optional imports (require external libraries)
try:
    from tuiml.algorithms.gradient_boosting.xgboost import XGBoostClassifier, XGBoostRegressor
except Exception as e:
    print(f"Failed to import XGBoost: {e}")
    XGBoostClassifier = None
    XGBoostRegressor = None

try:
    from tuiml.algorithms.gradient_boosting.catboost import CatBoostClassifier, CatBoostRegressor
except Exception as e:
    print(f"Failed to import CatBoost: {e}")
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    from tuiml.algorithms.gradient_boosting.lightgbm import LightGBMClassifier, LightGBMRegressor
except Exception as e:
    print(f"Failed to import LightGBM: {e}")
    LightGBMClassifier = None
    LightGBMRegressor = None

__all__ = [
    "XGBoostClassifier",
    "XGBoostRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "LightGBMClassifier",
    "LightGBMRegressor",
]
