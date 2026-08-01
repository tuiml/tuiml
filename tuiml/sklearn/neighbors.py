"""scikit-learn neighbors wrappers.

Generated from the ``NEIGHBORS`` table in :mod:`tuiml.sklearn.specs`.
Registered under ``sklearn.<ClassName>`` hub keys, mirroring the native
TuiML ``neighbors`` family.
"""


from typing import Any, Dict, List

from tuiml.base.algorithms import Classifier, Regressor
from tuiml.base.preprocessing import Transformer
from tuiml.sklearn._base import (
    _SklearnBackedMixin,
    _SklearnTransformerMixin,
    sk_classifier,
    sk_regressor,
    sk_transformer,
)
from tuiml.sklearn._spec import build_estimator, derive_schema

from tuiml.sklearn.specs import NEIGHBORS as _SPECS

_BY_NAME = {s.name: s for s in _SPECS}


@sk_classifier(tags=['neighbors', 'instance-based'])
class KNeighborsClassifier(_SklearnBackedMixin, Classifier):
    """scikit-learn KNeighborsClassifier (hub key ``sklearn.KNeighborsClassifier``).

    Wraps :class:`sklearn.neighbors._classification.KNeighborsClassifier`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_neighbors, weights, metric.
    """

    _SPEC = _BY_NAME['KNeighborsClassifier']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'KNeighborsClassifier')

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


@sk_classifier(tags=['neighbors', 'sklearn-extended'])
class NearestCentroid(_SklearnBackedMixin, Classifier):
    """scikit-learn NearestCentroid (hub key ``sklearn.NearestCentroid``).

    Wraps :class:`sklearn.neighbors._nearest_centroid.NearestCentroid`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['NearestCentroid']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'NearestCentroid')

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


@sk_classifier(tags=['neighbors', 'instance-based', 'sklearn-extended'])
class RadiusNeighborsClassifier(_SklearnBackedMixin, Classifier):
    """scikit-learn RadiusNeighborsClassifier (hub key ``sklearn.RadiusNeighborsClassifier``).

    Wraps :class:`sklearn.neighbors._classification.RadiusNeighborsClassifier`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['RadiusNeighborsClassifier']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'RadiusNeighborsClassifier')

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


@sk_regressor(tags=['neighbors', 'instance-based'])
class KNeighborsRegressor(_SklearnBackedMixin, Regressor):
    """scikit-learn KNeighborsRegressor (hub key ``sklearn.KNeighborsRegressor``).

    Wraps :class:`sklearn.neighbors._regression.KNeighborsRegressor`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_neighbors, weights, metric.
    """

    _SPEC = _BY_NAME['KNeighborsRegressor']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'KNeighborsRegressor')

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


@sk_regressor(tags=['neighbors', 'instance-based', 'sklearn-extended'])
class RadiusNeighborsRegressor(_SklearnBackedMixin, Regressor):
    """scikit-learn RadiusNeighborsRegressor (hub key ``sklearn.RadiusNeighborsRegressor``).

    Wraps :class:`sklearn.neighbors._regression.RadiusNeighborsRegressor`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['RadiusNeighborsRegressor']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'RadiusNeighborsRegressor')

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


@sk_transformer(tags=['neighbors', 'instance-based', 'sklearn-extended'])
class KNeighborsTransformer(_SklearnTransformerMixin, Transformer):
    """scikit-learn KNeighborsTransformer (hub key ``sklearn.KNeighborsTransformer``).

    Wraps :class:`sklearn.neighbors._graph.KNeighborsTransformer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['KNeighborsTransformer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'KNeighborsTransformer')

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


@sk_transformer(tags=['neighbors', 'sklearn-extended'])
class NeighborhoodComponentsAnalysis(_SklearnTransformerMixin, Transformer):
    """scikit-learn NeighborhoodComponentsAnalysis (hub key ``sklearn.NeighborhoodComponentsAnalysis``).

    Wraps :class:`sklearn.neighbors._nca.NeighborhoodComponentsAnalysis`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['NeighborhoodComponentsAnalysis']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'NeighborhoodComponentsAnalysis')

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


@sk_transformer(tags=['neighbors', 'instance-based', 'sklearn-extended'])
class RadiusNeighborsTransformer(_SklearnTransformerMixin, Transformer):
    """scikit-learn RadiusNeighborsTransformer (hub key ``sklearn.RadiusNeighborsTransformer``).

    Wraps :class:`sklearn.neighbors._graph.RadiusNeighborsTransformer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['RadiusNeighborsTransformer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'RadiusNeighborsTransformer')

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


__all__ = ['KNeighborsClassifier', 'NearestCentroid', 'RadiusNeighborsClassifier', 'KNeighborsRegressor', 'RadiusNeighborsRegressor', 'KNeighborsTransformer', 'NeighborhoodComponentsAnalysis', 'RadiusNeighborsTransformer']
