"""scikit-learn scaling wrappers.

Generated from the ``PREPROCESSING_SCALING`` table in :mod:`tuiml.sklearn.specs`.
Registered under ``sklearn.<ClassName>`` hub keys, mirroring the native
TuiML ``preprocessing/scaling`` family.
"""


from typing import Any, Dict, List

from tuiml.base.preprocessing import Transformer
from tuiml.sklearn._base import (
    _SklearnTransformerMixin,
    sk_transformer,
)
from tuiml.sklearn._spec import build_estimator, derive_schema

from tuiml.sklearn.specs import PREPROCESSING_SCALING as _SPECS

_BY_NAME = {s.name: s for s in _SPECS}


@sk_transformer(tags=['scaling', 'preprocessing', 'sklearn-extended'])
class Binarizer(_SklearnTransformerMixin, Transformer):
    """scikit-learn Binarizer (hub key ``sklearn.Binarizer``).

    Wraps :class:`sklearn.preprocessing._data.Binarizer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['Binarizer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'Binarizer')

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


@sk_transformer(tags=['scaling', 'preprocessing', 'sklearn-extended'])
class FunctionTransformer(_SklearnTransformerMixin, Transformer):
    """scikit-learn FunctionTransformer (hub key ``sklearn.FunctionTransformer``).

    Wraps :class:`sklearn.preprocessing._function_transformer.FunctionTransformer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['FunctionTransformer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'FunctionTransformer')

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


@sk_transformer(tags=['scaling', 'preprocessing'])
class KBinsDiscretizer(_SklearnTransformerMixin, Transformer):
    """scikit-learn KBinsDiscretizer (hub key ``sklearn.KBinsDiscretizer``).

    Wraps :class:`sklearn.preprocessing._discretization.KBinsDiscretizer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_bins, encode, strategy.
    """

    _SPEC = _BY_NAME['KBinsDiscretizer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'KBinsDiscretizer')

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


@sk_transformer(tags=['scaling', 'preprocessing'])
class MaxAbsScaler(_SklearnTransformerMixin, Transformer):
    """scikit-learn MaxAbsScaler (hub key ``sklearn.MaxAbsScaler``).

    Wraps :class:`sklearn.preprocessing._data.MaxAbsScaler`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['MaxAbsScaler']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'MaxAbsScaler')

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


@sk_transformer(tags=['scaling', 'preprocessing'])
class MinMaxScaler(_SklearnTransformerMixin, Transformer):
    """scikit-learn MinMaxScaler (hub key ``sklearn.MinMaxScaler``).

    Wraps :class:`sklearn.preprocessing._data.MinMaxScaler`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: feature_range.
    """

    _SPEC = _BY_NAME['MinMaxScaler']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'MinMaxScaler')

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


@sk_transformer(tags=['scaling', 'preprocessing'])
class Normalizer(_SklearnTransformerMixin, Transformer):
    """scikit-learn Normalizer (hub key ``sklearn.Normalizer``).

    Wraps :class:`sklearn.preprocessing._data.Normalizer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['Normalizer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'Normalizer')

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


@sk_transformer(tags=['scaling', 'preprocessing', 'encoding'])
class OneHotEncoder(_SklearnTransformerMixin, Transformer):
    """scikit-learn OneHotEncoder (hub key ``sklearn.OneHotEncoder``).

    Wraps :class:`sklearn.preprocessing._encoders.OneHotEncoder`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: handle_unknown, drop, sparse_output.
    """

    _SPEC = _BY_NAME['OneHotEncoder']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'OneHotEncoder')

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


@sk_transformer(tags=['scaling', 'preprocessing', 'encoding'])
class OrdinalEncoder(_SklearnTransformerMixin, Transformer):
    """scikit-learn OrdinalEncoder (hub key ``sklearn.OrdinalEncoder``).

    Wraps :class:`sklearn.preprocessing._encoders.OrdinalEncoder`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['OrdinalEncoder']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'OrdinalEncoder')

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


@sk_transformer(tags=['scaling', 'preprocessing'])
class PolynomialFeatures(_SklearnTransformerMixin, Transformer):
    """scikit-learn PolynomialFeatures (hub key ``sklearn.PolynomialFeatures``).

    Wraps :class:`sklearn.preprocessing._polynomial.PolynomialFeatures`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: degree, interaction_only, include_bias.
    """

    _SPEC = _BY_NAME['PolynomialFeatures']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'PolynomialFeatures')

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


@sk_transformer(tags=['scaling', 'preprocessing'])
class PowerTransformer(_SklearnTransformerMixin, Transformer):
    """scikit-learn PowerTransformer (hub key ``sklearn.PowerTransformer``).

    Wraps :class:`sklearn.preprocessing._data.PowerTransformer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['PowerTransformer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'PowerTransformer')

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


@sk_transformer(tags=['scaling', 'preprocessing'])
class QuantileTransformer(_SklearnTransformerMixin, Transformer):
    """scikit-learn QuantileTransformer (hub key ``sklearn.QuantileTransformer``).

    Wraps :class:`sklearn.preprocessing._data.QuantileTransformer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['QuantileTransformer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'QuantileTransformer')

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


@sk_transformer(tags=['scaling', 'preprocessing'])
class RobustScaler(_SklearnTransformerMixin, Transformer):
    """scikit-learn RobustScaler (hub key ``sklearn.RobustScaler``).

    Wraps :class:`sklearn.preprocessing._data.RobustScaler`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: with_centering, with_scaling, quantile_range.
    """

    _SPEC = _BY_NAME['RobustScaler']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'RobustScaler')

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


@sk_transformer(tags=['scaling', 'preprocessing', 'sklearn-extended'])
class SplineTransformer(_SklearnTransformerMixin, Transformer):
    """scikit-learn SplineTransformer (hub key ``sklearn.SplineTransformer``).

    Wraps :class:`sklearn.preprocessing._polynomial.SplineTransformer`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['SplineTransformer']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'SplineTransformer')

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


@sk_transformer(tags=['scaling', 'preprocessing'])
class StandardScaler(_SklearnTransformerMixin, Transformer):
    """scikit-learn StandardScaler (hub key ``sklearn.StandardScaler``).

    Wraps :class:`sklearn.preprocessing._data.StandardScaler`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: with_mean, with_std.
    """

    _SPEC = _BY_NAME['StandardScaler']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'StandardScaler')

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


@sk_transformer(tags=['scaling', 'preprocessing', 'encoding', 'sklearn-extended'])
class TargetEncoder(_SklearnTransformerMixin, Transformer):
    """scikit-learn TargetEncoder (hub key ``sklearn.TargetEncoder``).

    Wraps :class:`sklearn.preprocessing._target_encoder.TargetEncoder`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['TargetEncoder']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'TargetEncoder')

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


__all__ = ['Binarizer', 'FunctionTransformer', 'KBinsDiscretizer', 'MaxAbsScaler', 'MinMaxScaler', 'Normalizer', 'OneHotEncoder', 'OrdinalEncoder', 'PolynomialFeatures', 'PowerTransformer', 'QuantileTransformer', 'RobustScaler', 'SplineTransformer', 'StandardScaler', 'TargetEncoder']
