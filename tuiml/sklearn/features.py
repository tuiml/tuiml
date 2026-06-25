"""Curated scikit-learn feature selection & extraction wrappers.

Registered under ``sklearn.<ClassName>`` hub keys.
"""

from typing import Any, Dict, Optional

from tuiml.base.features import FeatureSelector, FeatureExtractor
from tuiml.sklearn._base import (
    _SklearnSelectorMixin,
    _SklearnExtractorMixin,
    sk_feature_selector,
    sk_feature_extractor,
)


# =============================================================================
# Feature selection
# =============================================================================

@sk_feature_selector(tags=["univariate", "statistical"])
class SelectKBest(_SklearnSelectorMixin, FeatureSelector):
    """scikit-learn SelectKBest (ANOVA F-test by default)."""

    def __init__(self, k: int = 10, score_func: str = "f_classif"):
        super().__init__(k=k)
        self.score_func = score_func

    def _build_estimator(self):
        from sklearn.feature_selection import (
            SelectKBest as _Est, f_classif, f_regression, mutual_info_classif,
        )
        funcs = {"f_classif": f_classif, "f_regression": f_regression,
                 "mutual_info_classif": mutual_info_classif}
        return _Est(score_func=funcs.get(self.score_func, f_classif), k=self.k)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"k": {"type": "integer", "default": 10},
                "score_func": {"type": "string", "default": "f_classif",
                               "enum": ["f_classif", "f_regression",
                                        "mutual_info_classif"]}}


@sk_feature_selector(tags=["variance", "unsupervised"])
class VarianceThreshold(_SklearnSelectorMixin, FeatureSelector):
    """scikit-learn VarianceThreshold."""

    def __init__(self, threshold: float = 0.0):
        super().__init__(threshold=threshold)

    def _build_estimator(self):
        from sklearn.feature_selection import VarianceThreshold as _Est
        return _Est(threshold=self.threshold)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"threshold": {"type": "number", "default": 0.0}}


# =============================================================================
# Feature extraction
# =============================================================================

@sk_feature_extractor(tags=["dimensionality-reduction", "linear"])
class PCAExtractor(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn PCA."""

    def __init__(self, n_components: Optional[Any] = None,
                 random_state: Optional[int] = None):
        super().__init__(n_components=n_components)
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.decomposition import PCA as _Est
        return _Est(n_components=self.n_components, random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"n_components": {"type": ["integer", "number", "null"],
                                 "default": None}}


@sk_feature_extractor(tags=["dimensionality-reduction", "sparse"])
class TruncatedSVD(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn TruncatedSVD (LSA)."""

    def __init__(self, n_components: int = 2, random_state: Optional[int] = None):
        super().__init__(n_components=n_components)
        self.random_state = random_state

    def _build_estimator(self):
        from sklearn.decomposition import TruncatedSVD as _Est
        return _Est(n_components=self.n_components, random_state=self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"n_components": {"type": "integer", "default": 2}}


__all__ = ["SelectKBest", "VarianceThreshold", "PCAExtractor", "TruncatedSVD"]
