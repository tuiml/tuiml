"""Meta-clusterer that applies a filter before clustering."""

import numpy as np
from typing import Dict, List, Any, Optional, Type
from tuiml.base.algorithms import Clusterer, clusterer
from tuiml.hub import registry

@clusterer(tags=["meta", "filter", "preprocessing"], version="1.0.0")
class FilteredClusterer(Clusterer):
    """FilteredClusterer meta-clusterer.

    A **meta-clusterer** that wraps a base clusterer and a **preprocessor**
    (filter). The filter is applied to the training data before it is passed
    to the base clusterer, enabling standardized or transformed clustering
    pipelines.

    Overview
    --------
    The meta-clusterer follows a two-stage pipeline:

    1. **Filter stage**: Apply the preprocessor's ``fit_transform()`` to the
       training data (e.g., standardization, PCA)
    2. **Cluster stage**: Fit the base clustering algorithm on the transformed data
    3. For prediction, apply ``transform()`` then ``predict()``

    Theory
    ------
    FilteredClusterer implements a composition of transformations:

    .. math::
        \\hat{y} = f_{\\text{cluster}}(g_{\\text{filter}}(X))

    where :math:`g_{\\text{filter}}` is any feature transformation and
    :math:`f_{\\text{cluster}}` is the base clustering algorithm. This is
    particularly useful when the base clusterer is sensitive to feature
    scaling (e.g., K-Means with Euclidean distance).

    Parameters
    ----------
    base_clusterer : str or Clusterer type, default='KMeansClusterer'
        The base clustering algorithm to use.
    filter : str, object, or type, optional, default=None
        The preprocessor to apply. Can be a hub registry name or an
        object with fit_transform() and transform() methods.
    filter_params : dict, optional, default=None
        Parameters to pass to the filter if it's specified as a string
        or type.

    Attributes
    ----------
    clusterer_ : Clusterer
        The fitted base clusterer instance.
    filter_ : object
        The fitted filter instance.
    n_clusters_ : int
        The number of clusters found by the base clusterer.
    labels_ : np.ndarray
        Labels assigned to training data.

    Notes
    -----
    **Complexity:**

    - Training/Prediction: Complexity of the chosen filter plus the
      complexity of the chosen base clusterer.

    **When to use FilteredClusterer:**

    - When features have different scales and the base clusterer is scale-sensitive
    - When dimensionality reduction is needed before clustering
    - To build reproducible clustering pipelines with preprocessing
    - When comparing clustering results with and without preprocessing

    References
    ----------
    .. [Witten2016] Witten, I.H., Frank, E., Hall, M.A. & Pal, C.J. (2016).
           **Data Mining: Practical Machine Learning Tools and Techniques.**
           *Morgan Kaufmann*, 4th Edition.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.KMeansClusterer` : Default base clusterer for FilteredClusterer.
    :class:`~tuiml.algorithms.clustering.GaussianMixtureClusterer` : Probabilistic base clusterer option.

    Examples
    --------
    Clustering with standardization preprocessing:

    >>> from tuiml.algorithms.clustering import FilteredClusterer
    >>> fc = FilteredClusterer(base_clusterer='KMeansClusterer', filter='StandardScaler')
    >>> fc.fit(X)
    >>> labels = fc.predict(X_new)
    """

    def __init__(self, base_clusterer: Any = 'KMeansClusterer',
                 filter: Any = None, filter_params: Optional[Dict] = None):
        """Initialize FilteredClusterer.

        Parameters
        ----------
        base_clusterer : Any, default='KMeansClusterer'
            Base clustering algorithm.
        filter : Any, default=None
            Filter/Preprocessor.
        filter_params : dict, optional
            Parameters for the filter.
        """
        super().__init__()
        self.base_clusterer = base_clusterer
        self.filter = filter
        self.filter_params = filter_params or {}
        self.clusterer_ = None
        self.filter_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "base_clusterer": {"type": "string", "default": "KMeansClusterer"},
            "filter": {"type": "string", "default": None},
            "filter_params": {"type": "object", "default": {}},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal"]

    def _get_base_class(self) -> Type[Clusterer]:
        """Resolve the base clusterer from a string name or type.

        Returns
        -------
        cls : Type[Clusterer]
            The resolved clusterer class.
        """
        if isinstance(self.base_clusterer, str):
            return registry.get(self.base_clusterer)
        elif isinstance(self.base_clusterer, type):
            return self.base_clusterer
        raise ValueError(f"Invalid base_clusterer: {self.base_clusterer}")

    def _get_filter(self):
        """Resolve and instantiate the filter from a string, type, or object.

        Returns
        -------
        filter_instance : object or None
            An instantiated filter with ``fit_transform()`` and ``transform()``
            methods, or None if no filter is specified.
        """
        if self.filter is None:
            return None
        if isinstance(self.filter, str):
            try:
                filter_class = registry.get(self.filter)
                return filter_class(**self.filter_params)
            except (KeyError, ValueError):
                return _StandardScaler()
        elif hasattr(self.filter, 'fit_transform'):
            return self.filter
        elif isinstance(self.filter, type):
            return self.filter(**self.filter_params)
        return None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FilteredClusterer":
        """Fit the filter and then the clusterer.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray, optional
            Target values (ignored for clustering).

        Returns
        -------
        self : FilteredClusterer
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        self.filter_ = self._get_filter()
        X_filtered = self.filter_.fit_transform(X) if self.filter_ else X
        
        base_class = self._get_base_class()
        self.clusterer_ = base_class()
        self.clusterer_.fit(X_filtered)
        
        self.n_clusters_ = self.clusterer_.n_clusters_
        self.labels_ = self.clusterer_.labels_
        self.cluster_centers_ = self.clusterer_.cluster_centers_
        
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for filtered data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to cluster.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        X_filtered = self.filter_.transform(X) if self.filter_ else X
        return self.clusterer_.predict(X_filtered)

class _StandardScaler:
    """Simple standardization helper."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, X):
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_

    def transform(self, X):
        return (X - self.mean_) / self.std_
