"""scikit-learn neural wrappers.

Generated from the ``NEURAL`` table in :mod:`tuiml.sklearn.specs`.
Registered under ``sklearn.<ClassName>`` hub keys, mirroring the native
TuiML ``neural`` family.
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

from tuiml.sklearn.specs import NEURAL as _SPECS

_BY_NAME = {s.name: s for s in _SPECS}


@sk_classifier(tags=['neural', 'neural-network'])
class MLPClassifier(_SklearnBackedMixin, Classifier):
    """scikit-learn MLPClassifier (hub key ``sklearn.MLPClassifier``).

    Wraps :class:`sklearn.neural_network._multilayer_perceptron.MLPClassifier`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: hidden_layer_sizes, activation, max_iter.
    """

    _SPEC = _BY_NAME['MLPClassifier']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'MLPClassifier')

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


@sk_regressor(tags=['neural', 'neural-network'])
class MLPRegressor(_SklearnBackedMixin, Regressor):
    """scikit-learn MLPRegressor (hub key ``sklearn.MLPRegressor``).

    Wraps :class:`sklearn.neural_network._multilayer_perceptron.MLPRegressor`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: hidden_layer_sizes, activation, max_iter.
    """

    _SPEC = _BY_NAME['MLPRegressor']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'MLPRegressor')

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


@sk_transformer(tags=['neural', 'neural-network', 'sklearn-extended'])
class BernoulliRBM(_SklearnTransformerMixin, Transformer):
    """scikit-learn BernoulliRBM (hub key ``sklearn.BernoulliRBM``).

    Wraps :class:`sklearn.neural_network._rbm.BernoulliRBM`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['BernoulliRBM']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'BernoulliRBM')

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


__all__ = ['MLPClassifier', 'MLPRegressor', 'BernoulliRBM']
