"""Curated scikit-learn preprocessing wrappers.

Registered under ``sklearn.<ClassName>`` hub keys (e.g. ``sklearn.StandardScaler``),
so they coexist with the native TuiML preprocessors of the same name.
"""

from typing import Any, Dict, Optional

from tuiml.base.preprocessing import Transformer
from tuiml.sklearn._base import _SklearnTransformerMixin, sk_transformer


@sk_transformer(tags=["scaling"])
class StandardScaler(_SklearnTransformerMixin, Transformer):
    """scikit-learn StandardScaler."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std

    def _build_estimator(self):
        from sklearn.preprocessing import StandardScaler as _Est
        return _Est(with_mean=self.with_mean, with_std=self.with_std)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"with_mean": {"type": "boolean", "default": True},
                "with_std": {"type": "boolean", "default": True}}


@sk_transformer(tags=["scaling"])
class MinMaxScaler(_SklearnTransformerMixin, Transformer):
    """scikit-learn MinMaxScaler."""

    def __init__(self, feature_range: tuple = (0, 1)):
        super().__init__()
        self.feature_range = feature_range

    def _build_estimator(self):
        from sklearn.preprocessing import MinMaxScaler as _Est
        return _Est(feature_range=self.feature_range)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {}


@sk_transformer(tags=["scaling", "robust"])
class RobustScaler(_SklearnTransformerMixin, Transformer):
    """scikit-learn RobustScaler."""

    def __init__(self, with_centering: bool = True, with_scaling: bool = True):
        super().__init__()
        self.with_centering = with_centering
        self.with_scaling = with_scaling

    def _build_estimator(self):
        from sklearn.preprocessing import RobustScaler as _Est
        return _Est(with_centering=self.with_centering, with_scaling=self.with_scaling)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {}


@sk_transformer(tags=["imputation"])
class SimpleImputer(_SklearnTransformerMixin, Transformer):
    """scikit-learn SimpleImputer."""

    def __init__(self, strategy: str = "mean", fill_value: Optional[Any] = None):
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value

    def _build_estimator(self):
        from sklearn.impute import SimpleImputer as _Est
        return _Est(strategy=self.strategy, fill_value=self.fill_value)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"strategy": {"type": "string", "default": "mean",
                             "enum": ["mean", "median", "most_frequent", "constant"]}}


@sk_transformer(tags=["imputation", "iterative"])
class IterativeImputer(_SklearnTransformerMixin, Transformer):
    """scikit-learn IterativeImputer (multivariate)."""

    def __init__(self, max_iter: int = 10, random_state: Optional[int] = None):
        super().__init__()
        self.max_iter = max_iter
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer as _Est
        return _Est(max_iter=self.max_iter, random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"max_iter": {"type": "integer", "default": 10}}


@sk_transformer(tags=["discretization"])
class KBinsDiscretizer(_SklearnTransformerMixin, Transformer):
    """scikit-learn KBinsDiscretizer."""

    def __init__(self, n_bins: int = 5, encode: str = "ordinal",
                 strategy: str = "quantile"):
        super().__init__()
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

    def _build_estimator(self):
        from sklearn.preprocessing import KBinsDiscretizer as _Est
        return _Est(n_bins=self.n_bins, encode=self.encode, strategy=self.strategy)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"n_bins": {"type": "integer", "default": 5},
                "strategy": {"type": "string", "default": "quantile",
                             "enum": ["uniform", "quantile", "kmeans"]}}


@sk_transformer(tags=["generation", "polynomial"])
class PolynomialFeatures(_SklearnTransformerMixin, Transformer):
    """scikit-learn PolynomialFeatures."""

    def __init__(self, degree: int = 2, include_bias: bool = True):
        super().__init__()
        self.degree = degree
        self.include_bias = include_bias

    def _build_estimator(self):
        from sklearn.preprocessing import PolynomialFeatures as _Est
        return _Est(degree=self.degree, include_bias=self.include_bias)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"degree": {"type": "integer", "default": 2}}


__all__ = [
    "StandardScaler", "MinMaxScaler", "RobustScaler", "SimpleImputer",
    "IterativeImputer", "KBinsDiscretizer", "PolynomialFeatures",
]
