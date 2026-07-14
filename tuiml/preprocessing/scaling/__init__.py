"""
Scaling transformers for numerical feature normalization.

Available:
    - MinMaxScaler: Min-max scaling (WEKA: Normalize)
    - StandardScaler: Z-score normalization (WEKA: Standardize)
    - CenterScaler: Mean centering (WEKA: Center)
    - SklearnStandardScaler: Scikit-Learn StandardScaler
    - SklearnMinMaxScaler: Scikit-Learn MinMaxScaler
"""

from tuiml.preprocessing.scaling.normalize import MinMaxScaler
from tuiml.preprocessing.scaling.standardize import StandardScaler
from tuiml.preprocessing.scaling.center import CenterScaler

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

if SKLEARN_AVAILABLE:
    from tuiml.preprocessing.scaling.sklearn_standard import SklearnStandardScaler
    from tuiml.preprocessing.scaling.sklearn_minmax import SklearnMinMaxScaler
    from tuiml.preprocessing.scaling.sklearn_robust import SklearnRobustScaler
    from tuiml.preprocessing.scaling.sklearn_normalizer import SklearnNormalizer
    from tuiml.preprocessing.scaling.sklearn_binarizer import SklearnBinarizer
    from tuiml.preprocessing.scaling.sklearn_poly import SklearnPolynomialFeatures

__all__ = [
    "MinMaxScaler",
    "StandardScaler",
    "CenterScaler",
]

if SKLEARN_AVAILABLE:
    __all__.extend([
        "SklearnStandardScaler",
        "SklearnMinMaxScaler",
        "SklearnRobustScaler",
        "SklearnNormalizer",
        "SklearnBinarizer",
        "SklearnPolynomialFeatures",
    ])

