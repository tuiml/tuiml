"""
Imputation transformers for handling missing values.

Available:
    - SimpleImputer: Mean/median/mode/constant imputation (WEKA: ReplaceMissingValues)
    - KNNImputer: K-nearest neighbors imputation
"""

from tuiml.preprocessing.imputation.simple_imputer import SimpleImputer
from tuiml.preprocessing.imputation.knn_imputer import KNNImputer

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

if SKLEARN_AVAILABLE:
    from tuiml.preprocessing.imputation.sklearn_imputer import SklearnSimpleImputer
    from tuiml.preprocessing.imputation.sklearn_iterative_imputer import SklearnIterativeImputer

__all__ = [
    "SimpleImputer",
    "KNNImputer",
]

if SKLEARN_AVAILABLE:
    __all__.extend([
        "SklearnSimpleImputer",
        "SklearnIterativeImputer",
    ])

