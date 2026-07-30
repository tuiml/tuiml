"""Shared machinery for the scikit-learn bridge.

This module centralizes everything the ``tuiml.sklearn`` wrappers need:

* **Namespaced registration decorators** (``sk_classifier`` etc.) that register
  each wrapper into the hub under a ``sklearn.<ClassName>`` key. The namespaced
  key keeps the wrappers from colliding with the native algorithms of the same
  name (e.g. ``RandomForestClassifier``), so the wrapper classes themselves can
  use clean, prefix-free names.
* **Wrapper mixins** that implement ``fit`` / ``predict`` / ``transform`` by
  delegating to a lazily-built scikit-learn estimator. Each concrete wrapper
  only needs to implement ``_build_estimator()`` and declare its constructor
  parameters.

scikit-learn is an *optional* dependency: importing this module never requires
it. The dependency is checked once, at ``fit`` time, and a clear, actionable
``ImportError`` is raised if it is missing.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml.base.algorithms import classifier, regressor, clusterer
from tuiml.base.preprocessing import transformer
from tuiml.base.features import feature_selector, feature_extractor

#: Hub-registry namespace prefix. The hub key for a wrapper named ``Foo`` is
#: ``"sklearn.Foo"`` while the Python class stays ``Foo``.
NAMESPACE = "sklearn"

#: pip extra that provides the backing dependency: ``pip install tuiml[sklearn]``.
_EXTRA = "sklearn"


def _ensure_sklearn(cls_name: str) -> None:
    """Raise an actionable error if scikit-learn is not importable.

    Parameters
    ----------
    cls_name : str
        Name of the wrapper class, used in the error message.
    """
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without sklearn
        raise ImportError(
            f"{cls_name} requires scikit-learn, which is not installed. "
            f"Install it with:  pip install 'tuiml[{_EXTRA}]'"
        ) from exc


def _prepare(X: Any) -> Any:
    """Prepare ``X`` for the backing estimator without destroying its dtype.

    Numeric input is cast to float, which is what the numeric scikit-learn
    estimators expect. Anything else is passed through untouched so that
    scikit-learn's own validation applies: a blanket ``astype(float)`` breaks
    text pipelines (``CountVectorizer``) and mixed-dtype frames
    (``ColumnTransformer``), both of which scikit-learn handles natively.

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    prepared : array-like
        ``X`` as a float array when numeric, otherwise unchanged.
    """
    # pandas objects go through as-is so column names survive for estimators
    # that use them (ColumnTransformer, feature_names_in_).
    if hasattr(X, "iloc"):
        return X
    arr = np.asarray(X)
    # bool / int / uint / float / complex -> safe to cast
    if arr.dtype.kind in "biufc":
        return arr.astype(float, copy=False)
    return arr


# =============================================================================
# Namespaced registration decorators
# =============================================================================

def _namespaced(base_decorator):
    """Wrap a base tuiml decorator so it registers under ``sklearn.<ClassName>``.

    Parameters
    ----------
    base_decorator : callable
        One of the tuiml registration decorators (``classifier``,
        ``regressor``, ``transformer``, ...).

    Returns
    -------
    callable
        A decorator factory mirroring the base decorator's keyword arguments.
    """

    def factory(tags: Optional[List[str]] = None, version: str = "1.0.0"):
        def decorate(cls):
            key = f"{NAMESPACE}.{cls.__name__}"
            merged_tags = list(tags or [])
            if NAMESPACE not in merged_tags:
                merged_tags.append(NAMESPACE)
            return base_decorator(name=key, tags=merged_tags, version=version)(cls)

        return decorate

    return factory


sk_classifier = _namespaced(classifier)
sk_regressor = _namespaced(regressor)
sk_clusterer = _namespaced(clusterer)
sk_transformer = _namespaced(transformer)
sk_feature_selector = _namespaced(feature_selector)
sk_feature_extractor = _namespaced(feature_extractor)


# =============================================================================
# Wrapper mixins
# =============================================================================

class _SklearnBackedMixin:
    """Implements ``fit`` / ``predict`` by delegating to a scikit-learn model.

    Concrete subclasses must implement :meth:`_build_estimator`, which returns a
    fresh (unfitted) scikit-learn estimator configured from the wrapper's
    constructor parameters.
    """

    _requires = "scikit-learn"
    _extra = _EXTRA

    def _build_estimator(self):
        """Return a fresh, configured scikit-learn estimator."""
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Any":
        """Fit the backing scikit-learn estimator."""
        _ensure_sklearn(type(self).__name__)
        self.model_ = self._build_estimator()
        X = _prepare(X)
        if y is not None:
            self.model_.fit(X, y)
        else:
            self.model_.fit(X)
        # Surface the common scikit-learn fitted attributes on the wrapper.
        for attr in ("classes_", "n_features_in_", "feature_importances_"):
            if hasattr(self.model_, attr):
                setattr(self, attr, getattr(self.model_, attr))
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the backing estimator."""
        self._check_is_fitted()
        return self.model_.predict(_prepare(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities when the backing estimator supports it.

        Falls back to the base-class one-hot behavior otherwise.
        """
        self._check_is_fitted()
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(_prepare(X))
        return super().predict_proba(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return constructor parameters (scikit-learn ``clone`` compatible).

        Excludes fitted attributes (trailing underscore) and private state.
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not k.endswith("_")
        }


class _SklearnClustererMixin(_SklearnBackedMixin):
    """Clustering variant: populates ``labels_`` and handles transductive models."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Any":
        """Fit and capture cluster labels."""
        _ensure_sklearn(type(self).__name__)
        self.model_ = self._build_estimator()
        X = _prepare(X)
        if hasattr(self.model_, "fit_predict"):
            labels = self.model_.fit_predict(X)
        else:
            self.model_.fit(X)
            labels = getattr(self.model_, "labels_", None)
        self.labels_ = labels
        self.n_clusters_ = int(len(set(labels))) if labels is not None else None
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign clusters to ``X`` (re-fits transductive models)."""
        self._check_is_fitted()
        X = _prepare(X)
        if hasattr(self.model_, "predict"):
            return self.model_.predict(X)
        # Transductive estimators (DBSCAN, Agglomerative) cannot assign new
        # points; reproduce scikit-learn's fit_predict semantics.
        return self.model_.fit_predict(X)


class _SklearnTransformerMixin(_SklearnBackedMixin):
    """Transformer variant: delegates ``transform`` to the backing estimator."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform ``X`` with the fitted estimator."""
        self._check_is_fitted()
        return self.model_.transform(_prepare(X))


class _SklearnExtractorMixin(_SklearnTransformerMixin):
    """Feature-extractor variant (same surface as the transformer mixin)."""


class _SklearnSelectorMixin(_SklearnBackedMixin):
    """Feature-selector variant: populates the base selector's index state.

    Overrides ``fit`` to run the backing scikit-learn selector and record
    ``_selected_indices`` / ``_feature_scores`` so the base class's
    ``transform`` and ``get_support`` work unchanged.
    """

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Required by the abstract base; scores are captured during fit."""
        return getattr(self, "_feature_scores", np.zeros(np.asarray(X).shape[1]))

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Any":
        """Fit the backing selector and capture selected feature indices."""
        _ensure_sklearn(type(self).__name__)
        self.model_ = self._build_estimator()
        X = _prepare(X)
        self.model_.fit(X, y)
        support = self.model_.get_support(indices=True)
        self._selected_indices = np.sort(np.asarray(support))
        scores = getattr(self.model_, "scores_", None)
        if scores is None:
            scores = np.zeros(X.shape[1])
        self._feature_scores = np.asarray(scores, dtype=float)
        self._is_fitted = True
        return self
