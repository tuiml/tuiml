"""scikit-learn imputation wrappers.

Generated from the ``PREPROCESSING_IMPUTATION`` table in :mod:`tuiml.sklearn.specs`.
Registered under ``sklearn.<ClassName>`` hub keys, mirroring the native
TuiML ``preprocessing/imputation`` family.
"""


from typing import Any, Dict, List

from tuiml.base.preprocessing import Transformer
from tuiml.sklearn._base import (
    _SklearnTransformerMixin,
    sk_transformer,
)
from tuiml.sklearn._spec import build_estimator, derive_schema
from tuiml.sklearn import _overrides

from tuiml.sklearn.specs import PREPROCESSING_IMPUTATION as _SPECS

_BY_NAME = {s.name: s for s in _SPECS}


@sk_transformer(tags=['imputation', 'impute'])
class IterativeImputer(_SklearnTransformerMixin, Transformer):
    """scikit-learn IterativeImputer (hub key ``sklearn.IterativeImputer``).

    Wraps :class:`sklearn.impute._iterative.IterativeImputer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: max_iter, random_state.
    """

    _SPEC = _BY_NAME['IterativeImputer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return _overrides.iterative_imputer(
            self._SPEC.target, self._params, 'IterativeImputer'
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return derive_schema(
            cls._SPEC.target, cls._SPEC.highlight, cls._SPEC.exclude
        )

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return list(cls._SPEC.capabilities)


@sk_transformer(tags=['imputation', 'impute'])
class KNNImputer(_SklearnTransformerMixin, Transformer):
    """scikit-learn KNNImputer (hub key ``sklearn.KNNImputer``).

    Wraps :class:`sklearn.impute._knn.KNNImputer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_neighbors, weights.
    """

    _SPEC = _BY_NAME['KNNImputer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'KNNImputer')

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return derive_schema(
            cls._SPEC.target, cls._SPEC.highlight, cls._SPEC.exclude
        )

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return list(cls._SPEC.capabilities)


@sk_transformer(tags=['imputation', 'impute', 'sklearn-extended'])
class MissingIndicator(_SklearnTransformerMixin, Transformer):
    """scikit-learn MissingIndicator (hub key ``sklearn.MissingIndicator``).

    Wraps :class:`sklearn.impute._base.MissingIndicator`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['MissingIndicator']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'MissingIndicator')

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return derive_schema(
            cls._SPEC.target, cls._SPEC.highlight, cls._SPEC.exclude
        )

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return list(cls._SPEC.capabilities)


@sk_transformer(tags=['imputation', 'impute'])
class SimpleImputer(_SklearnTransformerMixin, Transformer):
    """scikit-learn SimpleImputer (hub key ``sklearn.SimpleImputer``).

    Wraps :class:`sklearn.impute._base.SimpleImputer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: strategy, fill_value.
    """

    _SPEC = _BY_NAME['SimpleImputer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'SimpleImputer')

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return derive_schema(
            cls._SPEC.target, cls._SPEC.highlight, cls._SPEC.exclude
        )

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return list(cls._SPEC.capabilities)


__all__ = ['IterativeImputer', 'KNNImputer', 'MissingIndicator', 'SimpleImputer']
