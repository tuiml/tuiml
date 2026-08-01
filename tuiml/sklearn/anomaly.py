"""scikit-learn anomaly wrappers.

Generated from the ``ANOMALY`` table in :mod:`tuiml.sklearn.specs`.
Registered under ``sklearn.<ClassName>`` hub keys, mirroring the native
TuiML ``anomaly`` family.
"""


from typing import Any, Dict, List

from tuiml.base.algorithms import Classifier
from tuiml.sklearn._base import (
    _SklearnBackedMixin,
    sk_classifier,
)
from tuiml.sklearn._spec import build_estimator, derive_schema

from tuiml.sklearn.specs import ANOMALY as _SPECS

_BY_NAME = {s.name: s for s in _SPECS}


@sk_classifier(tags=['anomaly', 'covariance', 'anomaly-detection'])
class EllipticEnvelope(_SklearnBackedMixin, Classifier):
    """scikit-learn EllipticEnvelope (hub key ``sklearn.EllipticEnvelope``).

    Wraps :class:`sklearn.covariance._elliptic_envelope.EllipticEnvelope`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['EllipticEnvelope']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'EllipticEnvelope')

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


@sk_classifier(tags=['anomaly', 'ensemble', 'anomaly-detection'])
class IsolationForest(_SklearnBackedMixin, Classifier):
    """scikit-learn IsolationForest (hub key ``sklearn.IsolationForest``).

    Wraps :class:`sklearn.ensemble._iforest.IsolationForest`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_estimators, contamination, random_state.
    """

    _SPEC = _BY_NAME['IsolationForest']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'IsolationForest')

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


@sk_classifier(tags=['anomaly', 'neighbors', 'anomaly-detection'])
class LocalOutlierFactor(_SklearnBackedMixin, Classifier):
    """scikit-learn LocalOutlierFactor (hub key ``sklearn.LocalOutlierFactor``).

    Wraps :class:`sklearn.neighbors._lof.LocalOutlierFactor`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_neighbors, contamination.
    """

    _SPEC = _BY_NAME['LocalOutlierFactor']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'LocalOutlierFactor')

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


@sk_classifier(tags=['anomaly', 'svm', 'anomaly-detection'])
class OneClassSVM(_SklearnBackedMixin, Classifier):
    """scikit-learn OneClassSVM (hub key ``sklearn.OneClassSVM``).

    Wraps :class:`sklearn.svm._classes.OneClassSVM`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: kernel, nu, gamma.
    """

    _SPEC = _BY_NAME['OneClassSVM']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'OneClassSVM')

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


@sk_classifier(tags=['anomaly', 'linear-model', 'anomaly-detection', 'svm', 'sklearn-extended'])
class SGDOneClassSVM(_SklearnBackedMixin, Classifier):
    """scikit-learn SGDOneClassSVM (hub key ``sklearn.SGDOneClassSVM``).

    Wraps :class:`sklearn.linear_model._stochastic_gradient.SGDOneClassSVM`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['SGDOneClassSVM']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'SGDOneClassSVM')

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


__all__ = ['EllipticEnvelope', 'IsolationForest', 'LocalOutlierFactor', 'OneClassSVM', 'SGDOneClassSVM']
