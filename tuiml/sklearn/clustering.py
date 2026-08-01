"""scikit-learn clustering wrappers.

Generated from the ``CLUSTERING`` table in :mod:`tuiml.sklearn.specs`.
Registered under ``sklearn.<ClassName>`` hub keys, mirroring the native
TuiML ``clustering`` family.
"""


from typing import Any, Dict, List

from tuiml.base.algorithms import Clusterer
from tuiml.sklearn._base import (
    _SklearnClustererMixin,
    sk_clusterer,
)
from tuiml.sklearn._spec import build_estimator, derive_schema

from tuiml.sklearn.specs import CLUSTERING as _SPECS

_BY_NAME = {s.name: s for s in _SPECS}


@sk_clusterer(tags=['clustering', 'cluster', 'sklearn-extended'])
class AffinityPropagation(_SklearnClustererMixin, Clusterer):
    """scikit-learn AffinityPropagation (hub key ``sklearn.AffinityPropagation``).

    Wraps :class:`sklearn.cluster._affinity_propagation.AffinityPropagation`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['AffinityPropagation']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'AffinityPropagation')

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


@sk_clusterer(tags=['clustering', 'cluster'])
class AgglomerativeClustering(_SklearnClustererMixin, Clusterer):
    """scikit-learn AgglomerativeClustering (hub key ``sklearn.AgglomerativeClustering``).

    Wraps :class:`sklearn.cluster._agglomerative.AgglomerativeClustering`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_clusters, linkage, metric.
    """

    _SPEC = _BY_NAME['AgglomerativeClustering']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'AgglomerativeClustering')

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


@sk_clusterer(tags=['clustering', 'mixture', 'sklearn-extended'])
class BayesianGaussianMixture(_SklearnClustererMixin, Clusterer):
    """scikit-learn BayesianGaussianMixture (hub key ``sklearn.BayesianGaussianMixture``).

    Wraps :class:`sklearn.mixture._bayesian_mixture.BayesianGaussianMixture`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['BayesianGaussianMixture']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'BayesianGaussianMixture')

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


@sk_clusterer(tags=['clustering', 'cluster'])
class Birch(_SklearnClustererMixin, Clusterer):
    """scikit-learn Birch (hub key ``sklearn.Birch``).

    Wraps :class:`sklearn.cluster._birch.Birch`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['Birch']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'Birch')

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


@sk_clusterer(tags=['clustering', 'cluster', 'sklearn-extended'])
class BisectingKMeans(_SklearnClustererMixin, Clusterer):
    """scikit-learn BisectingKMeans (hub key ``sklearn.BisectingKMeans``).

    Wraps :class:`sklearn.cluster._bisect_k_means.BisectingKMeans`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['BisectingKMeans']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'BisectingKMeans')

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


@sk_clusterer(tags=['clustering', 'cluster'])
class DBSCAN(_SklearnClustererMixin, Clusterer):
    """scikit-learn DBSCAN (hub key ``sklearn.DBSCAN``).

    Wraps :class:`sklearn.cluster._dbscan.DBSCAN`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: eps, min_samples, metric.
    """

    _SPEC = _BY_NAME['DBSCAN']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'DBSCAN')

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


@sk_clusterer(tags=['clustering', 'mixture'])
class GaussianMixture(_SklearnClustererMixin, Clusterer):
    """scikit-learn GaussianMixture (hub key ``sklearn.GaussianMixture``).

    Wraps :class:`sklearn.mixture._gaussian_mixture.GaussianMixture`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_components, covariance_type, random_state.
    """

    _SPEC = _BY_NAME['GaussianMixture']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'GaussianMixture')

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


@sk_clusterer(tags=['clustering', 'cluster', 'sklearn-extended'])
class HDBSCAN(_SklearnClustererMixin, Clusterer):
    """scikit-learn HDBSCAN (hub key ``sklearn.HDBSCAN``).

    Wraps :class:`sklearn.cluster._hdbscan.hdbscan.HDBSCAN`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['HDBSCAN']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'HDBSCAN')

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


@sk_clusterer(tags=['clustering', 'cluster'])
class KMeans(_SklearnClustererMixin, Clusterer):
    """scikit-learn KMeans (hub key ``sklearn.KMeans``).

    Wraps :class:`sklearn.cluster._kmeans.KMeans`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.

    Commonly set: n_clusters, init, n_init, random_state.
    """

    _SPEC = _BY_NAME['KMeans']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'KMeans')

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


@sk_clusterer(tags=['clustering', 'cluster'])
class MeanShift(_SklearnClustererMixin, Clusterer):
    """scikit-learn MeanShift (hub key ``sklearn.MeanShift``).

    Wraps :class:`sklearn.cluster._mean_shift.MeanShift`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['MeanShift']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'MeanShift')

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


@sk_clusterer(tags=['clustering', 'cluster', 'sklearn-extended'])
class MiniBatchKMeans(_SklearnClustererMixin, Clusterer):
    """scikit-learn MiniBatchKMeans (hub key ``sklearn.MiniBatchKMeans``).

    Wraps :class:`sklearn.cluster._kmeans.MiniBatchKMeans`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['MiniBatchKMeans']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'MiniBatchKMeans')

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


@sk_clusterer(tags=['clustering', 'cluster', 'sklearn-extended'])
class OPTICS(_SklearnClustererMixin, Clusterer):
    """scikit-learn OPTICS (hub key ``sklearn.OPTICS``).

    Wraps :class:`sklearn.cluster._optics.OPTICS`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['OPTICS']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'OPTICS')

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


@sk_clusterer(tags=['clustering', 'cluster'])
class SpectralClustering(_SklearnClustererMixin, Clusterer):
    """scikit-learn SpectralClustering (hub key ``sklearn.SpectralClustering``).

    Wraps :class:`sklearn.cluster._spectral.SpectralClustering`. Accepts that
    estimator's constructor parameters as keyword arguments; call
    :meth:`get_parameter_schema` for the full list with types and
    defaults derived from the installed scikit-learn.
    """

    _SPEC = _BY_NAME['SpectralClustering']

    def __init__(self, **params: Any):
        super().__init__()
        self._params = {**self._SPEC.defaults, **params}
        for key, value in self._params.items():
            setattr(self, key, value)

    def _build_estimator(self):
        """Construct the backing scikit-learn estimator."""
        return build_estimator(self._SPEC.target, self._params, 'SpectralClustering')

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


__all__ = ['AffinityPropagation', 'AgglomerativeClustering', 'BayesianGaussianMixture', 'Birch', 'BisectingKMeans', 'DBSCAN', 'GaussianMixture', 'HDBSCAN', 'KMeans', 'MeanShift', 'MiniBatchKMeans', 'OPTICS', 'SpectralClustering']
