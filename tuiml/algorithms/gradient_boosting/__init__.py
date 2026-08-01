"""Gradient boosting frameworks (XGBoost, CatBoost, LightGBM).

Boosted tree ensembles, each fitting the next tree to the previous ensemble's
errors. These are usually the strongest option on tabular data, at the cost
of longer training and more hyperparameters than a single tree.

Algorithms
----------
- **XGBoostClassifier / XGBoostRegressor:** Regularised boosting; the
  general-purpose default of the three.
- **LightGBMClassifier / LightGBMRegressor:** Leaf-wise growth with
  histogram binning; fastest on wide or large data.
- **CatBoostClassifier / CatBoostRegressor:** Ordered boosting with native
  categorical handling; strongest when categorical columns dominate.

Notes
-----
These wrap the three upstream libraries, which are required dependencies of
TuiML rather than optional extras. Unlike most of :mod:`tuiml.algorithms`,
the implementations are not native.

Trees are scale-invariant, so feature scaling gains nothing here. Set
``random_state`` (or pass ``random_seed`` to :func:`tuiml.train`) for
reproducible fits — all three sample rows and columns while boosting.
"""

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
