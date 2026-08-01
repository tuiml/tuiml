"""scikit-learn svm wrappers.

Generated from the ``SVM`` table in :mod:`tuiml.sklearn.specs`.
Registered under ``sklearn.<ClassName>`` hub keys, mirroring the native
TuiML ``svm`` family.
"""


from typing import Any, Dict, List

from tuiml.base.algorithms import Classifier, Regressor
from tuiml.sklearn._base import (
    _SklearnBackedMixin,
    sk_classifier,
    sk_regressor,
)
from tuiml.sklearn._spec import build_estimator, derive_schema
from tuiml.sklearn import _overrides

from tuiml.sklearn.specs import SVM as _SPECS

_BY_NAME = {s.name: s for s in _SPECS}


@sk_classifier(tags=['svm'])
class LinearSVC(_SklearnBackedMixin, Classifier):
    """scikit-learn LinearSVC (hub key ``sklearn.LinearSVC``).

    Wraps :class:`sklearn.svm._classes.LinearSVC`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: C, loss, max_iter.
    """

    _SPEC = _BY_NAME['LinearSVC']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'LinearSVC')

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


@sk_classifier(tags=['svm', 'sklearn-extended'])
class NuSVC(_SklearnBackedMixin, Classifier):
    """scikit-learn NuSVC (hub key ``sklearn.NuSVC``).

    Wraps :class:`sklearn.svm._classes.NuSVC`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['NuSVC']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'NuSVC')

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


@sk_classifier(tags=['svm'])
class SVC(_SklearnBackedMixin, Classifier):
    """scikit-learn SVC (hub key ``sklearn.SVC``).

    Wraps :class:`sklearn.svm._classes.SVC`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: C, kernel, gamma, probability.
    """

    _SPEC = _BY_NAME['SVC']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return _overrides.svc_with_optional_calibration(
            self._SPEC.target, self._params, 'SVC'
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


@sk_regressor(tags=['svm'])
class LinearSVR(_SklearnBackedMixin, Regressor):
    """scikit-learn LinearSVR (hub key ``sklearn.LinearSVR``).

    Wraps :class:`sklearn.svm._classes.LinearSVR`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['LinearSVR']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'LinearSVR')

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


@sk_regressor(tags=['svm', 'sklearn-extended'])
class NuSVR(_SklearnBackedMixin, Regressor):
    """scikit-learn NuSVR (hub key ``sklearn.NuSVR``).

    Wraps :class:`sklearn.svm._classes.NuSVR`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['NuSVR']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'NuSVR')

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


@sk_regressor(tags=['svm'])
class SVR(_SklearnBackedMixin, Regressor):
    """scikit-learn SVR (hub key ``sklearn.SVR``).

    Wraps :class:`sklearn.svm._classes.SVR`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: C, kernel, epsilon.
    """

    _SPEC = _BY_NAME['SVR']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'SVR')

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


__all__ = ['LinearSVC', 'NuSVC', 'SVC', 'LinearSVR', 'NuSVR', 'SVR']
