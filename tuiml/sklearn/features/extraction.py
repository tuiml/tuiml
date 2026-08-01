"""scikit-learn extraction wrappers.

Generated from the ``FEATURES_EXTRACTION`` table in :mod:`tuiml.sklearn.specs`.
Registered under ``sklearn.<ClassName>`` hub keys, mirroring the native
TuiML ``features/extraction`` family.
"""


from typing import Any, Dict, List

from tuiml.base.features import FeatureExtractor
from tuiml.sklearn._base import (
    _SklearnExtractorMixin,
    sk_feature_extractor,
)
from tuiml.sklearn._spec import build_estimator, derive_schema

from tuiml.sklearn.specs import FEATURES_EXTRACTION as _SPECS

_BY_NAME = {s.name: s for s in _SPECS}


@sk_feature_extractor(tags=['extraction', 'kernel-approximation', 'sklearn-extended'])
class AdditiveChi2Sampler(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn AdditiveChi2Sampler (hub key ``sklearn.AdditiveChi2Sampler``).

    Wraps :class:`sklearn.kernel_approximation.AdditiveChi2Sampler`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['AdditiveChi2Sampler']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'AdditiveChi2Sampler')

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


@sk_feature_extractor(tags=['extraction', 'decomposition', 'sklearn-extended'])
class DictionaryLearning(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn DictionaryLearning (hub key ``sklearn.DictionaryLearning``).

    Wraps :class:`sklearn.decomposition._dict_learning.DictionaryLearning`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['DictionaryLearning']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'DictionaryLearning')

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


@sk_feature_extractor(tags=['extraction', 'decomposition', 'sklearn-extended'])
class FactorAnalysis(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn FactorAnalysis (hub key ``sklearn.FactorAnalysis``).

    Wraps :class:`sklearn.decomposition._factor_analysis.FactorAnalysis`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['FactorAnalysis']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'FactorAnalysis')

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


@sk_feature_extractor(tags=['extraction', 'decomposition'])
class FastICA(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn FastICA (hub key ``sklearn.FastICA``).

    Wraps :class:`sklearn.decomposition._fastica.FastICA`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_components, random_state.
    """

    _SPEC = _BY_NAME['FastICA']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'FastICA')

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


@sk_feature_extractor(tags=['extraction', 'random-projection', 'sklearn-extended'])
class GaussianRandomProjection(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn GaussianRandomProjection (hub key ``sklearn.GaussianRandomProjection``).

    Wraps :class:`sklearn.random_projection.GaussianRandomProjection`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['GaussianRandomProjection']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'GaussianRandomProjection')

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


@sk_feature_extractor(tags=['extraction', 'decomposition', 'sklearn-extended'])
class IncrementalPCA(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn IncrementalPCA (hub key ``sklearn.IncrementalPCA``).

    Wraps :class:`sklearn.decomposition._incremental_pca.IncrementalPCA`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['IncrementalPCA']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'IncrementalPCA')

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


@sk_feature_extractor(tags=['extraction', 'manifold', 'sklearn-extended'])
class Isomap(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn Isomap (hub key ``sklearn.Isomap``).

    Wraps :class:`sklearn.manifold._isomap.Isomap`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['Isomap']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'Isomap')

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


@sk_feature_extractor(tags=['extraction', 'decomposition'])
class KernelPCA(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn KernelPCA (hub key ``sklearn.KernelPCA``).

    Wraps :class:`sklearn.decomposition._kernel_pca.KernelPCA`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_components, kernel, gamma.
    """

    _SPEC = _BY_NAME['KernelPCA']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'KernelPCA')

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


@sk_feature_extractor(tags=['extraction', 'decomposition', 'sklearn-extended'])
class LatentDirichletAllocation(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn LatentDirichletAllocation (hub key ``sklearn.LatentDirichletAllocation``).

    Wraps :class:`sklearn.decomposition._lda.LatentDirichletAllocation`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['LatentDirichletAllocation']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'LatentDirichletAllocation')

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


@sk_feature_extractor(tags=['extraction', 'manifold', 'sklearn-extended'])
class LocallyLinearEmbedding(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn LocallyLinearEmbedding (hub key ``sklearn.LocallyLinearEmbedding``).

    Wraps :class:`sklearn.manifold._locally_linear.LocallyLinearEmbedding`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['LocallyLinearEmbedding']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'LocallyLinearEmbedding')

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


@sk_feature_extractor(tags=['extraction', 'decomposition', 'sklearn-extended'])
class MiniBatchDictionaryLearning(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn MiniBatchDictionaryLearning (hub key ``sklearn.MiniBatchDictionaryLearning``).

    Wraps :class:`sklearn.decomposition._dict_learning.MiniBatchDictionaryLearning`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['MiniBatchDictionaryLearning']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'MiniBatchDictionaryLearning')

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


@sk_feature_extractor(tags=['extraction', 'decomposition', 'sklearn-extended'])
class MiniBatchNMF(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn MiniBatchNMF (hub key ``sklearn.MiniBatchNMF``).

    Wraps :class:`sklearn.decomposition._nmf.MiniBatchNMF`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['MiniBatchNMF']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'MiniBatchNMF')

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


@sk_feature_extractor(tags=['extraction', 'decomposition', 'sklearn-extended'])
class MiniBatchSparsePCA(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn MiniBatchSparsePCA (hub key ``sklearn.MiniBatchSparsePCA``).

    Wraps :class:`sklearn.decomposition._sparse_pca.MiniBatchSparsePCA`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['MiniBatchSparsePCA']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'MiniBatchSparsePCA')

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


@sk_feature_extractor(tags=['extraction', 'decomposition'])
class NMF(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn NMF (hub key ``sklearn.NMF``).

    Wraps :class:`sklearn.decomposition._nmf.NMF`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_components, init, random_state.
    """

    _SPEC = _BY_NAME['NMF']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'NMF')

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


@sk_feature_extractor(tags=['extraction', 'kernel-approximation', 'sklearn-extended'])
class Nystroem(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn Nystroem (hub key ``sklearn.Nystroem``).

    Wraps :class:`sklearn.kernel_approximation.Nystroem`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['Nystroem']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'Nystroem')

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


@sk_feature_extractor(tags=['extraction', 'decomposition'])
class PCAExtractor(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn PCA (hub key ``sklearn.PCAExtractor``).

    Wraps :class:`sklearn.decomposition._pca.PCA`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_components, whiten, random_state.
    """

    _SPEC = _BY_NAME['PCAExtractor']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'PCAExtractor')

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


@sk_feature_extractor(tags=['extraction', 'kernel-approximation', 'sklearn-extended'])
class PolynomialCountSketch(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn PolynomialCountSketch (hub key ``sklearn.PolynomialCountSketch``).

    Wraps :class:`sklearn.kernel_approximation.PolynomialCountSketch`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['PolynomialCountSketch']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'PolynomialCountSketch')

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


@sk_feature_extractor(tags=['extraction', 'kernel-approximation', 'sklearn-extended'])
class RBFSampler(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn RBFSampler (hub key ``sklearn.RBFSampler``).

    Wraps :class:`sklearn.kernel_approximation.RBFSampler`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['RBFSampler']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'RBFSampler')

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


@sk_feature_extractor(tags=['extraction', 'kernel-approximation', 'sklearn-extended'])
class SkewedChi2Sampler(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn SkewedChi2Sampler (hub key ``sklearn.SkewedChi2Sampler``).

    Wraps :class:`sklearn.kernel_approximation.SkewedChi2Sampler`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['SkewedChi2Sampler']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'SkewedChi2Sampler')

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


@sk_feature_extractor(tags=['extraction', 'decomposition', 'sklearn-extended'])
class SparsePCA(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn SparsePCA (hub key ``sklearn.SparsePCA``).

    Wraps :class:`sklearn.decomposition._sparse_pca.SparsePCA`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['SparsePCA']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'SparsePCA')

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


@sk_feature_extractor(tags=['extraction', 'random-projection', 'sklearn-extended'])
class SparseRandomProjection(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn SparseRandomProjection (hub key ``sklearn.SparseRandomProjection``).

    Wraps :class:`sklearn.random_projection.SparseRandomProjection`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['SparseRandomProjection']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'SparseRandomProjection')

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


@sk_feature_extractor(tags=['extraction', 'decomposition', 'svm'])
class TruncatedSVD(_SklearnExtractorMixin, FeatureExtractor):
    """scikit-learn TruncatedSVD (hub key ``sklearn.TruncatedSVD``).

    Wraps :class:`sklearn.decomposition._truncated_svd.TruncatedSVD`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_components, algorithm, random_state.
    """

    _SPEC = _BY_NAME['TruncatedSVD']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'TruncatedSVD')

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


__all__ = ['AdditiveChi2Sampler', 'DictionaryLearning', 'FactorAnalysis', 'FastICA', 'GaussianRandomProjection', 'IncrementalPCA', 'Isomap', 'KernelPCA', 'LatentDirichletAllocation', 'LocallyLinearEmbedding', 'MiniBatchDictionaryLearning', 'MiniBatchNMF', 'MiniBatchSparsePCA', 'NMF', 'Nystroem', 'PCAExtractor', 'PolynomialCountSketch', 'RBFSampler', 'SkewedChi2Sampler', 'SparsePCA', 'SparseRandomProjection', 'TruncatedSVD']
