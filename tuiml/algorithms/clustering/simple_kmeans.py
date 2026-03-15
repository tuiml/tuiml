"""K-Means clustering with multiple initialization methods."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Clusterer, clusterer
from tuiml.algorithms.clustering.distance import pairwise_distances

@clusterer(tags=["partitional", "centroid-based"], version="1.0.0")
class KMeansClusterer(Clusterer):
    r"""
    K-Means clustering algorithm.

    Partitions data into :math:`k` clusters by iteratively assigning points to
    the nearest **centroid** and updating centroids to the mean of assigned
    points. The goal is to minimize the total **inertia** (within-cluster sum
    of squared distances).

    Overview
    --------
    The algorithm follows Lloyd's iterative refinement procedure:

    1. **Initialize** :math:`k` centroids using the chosen method (k-means++, random, or farthest)
    2. **Assign** each point to the nearest centroid
    3. **Update** each centroid to the mean (or median for Manhattan) of its assigned points
    4. Repeat steps 2--3 until convergence or ``max_iter`` is reached
    5. Select the best result across ``n_init`` independent runs

    Theory
    ------
    K-Means minimizes the within-cluster sum of squares (inertia):

    .. math::
        J = \sum_{i=1}^{n} \min_{\mu_j \in C} \| x_i - \mu_j \|^2

    The **k-means++** initialization selects the next center :math:`\mu_i`
    with probability proportional to :math:`D(x)^2`, the squared distance
    to the nearest existing center:

    .. math::
        P(\mu_i = x) = \\frac{D(x)^2}{\sum_{x' \in X} D(x')^2}

    This provides an :math:`O(\log k)` competitive approximation to the
    optimal k-means solution.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form as well as the number of
        centroids to generate.
    init : {"k-means++", "random", "farthest"}, default="k-means++"
        Method for initialization:

        - ``"k-means++"`` — Smart initialization that selects centers far from each other
        - ``"random"`` — Randomly selects :math:`k` observations from the data
        - ``"farthest"`` — Farthest-first traversal initialization
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init runs in terms of inertia.
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    distance : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric to use.
    random_state : int, optional, default=None
        Determines random number generation for centroid initialization.

    Attributes
    ----------
    cluster_centers_ : np.ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : np.ndarray of shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot k \cdot m \cdot i)` where :math:`n` is samples,
      :math:`k` is clusters, :math:`m` is features, and :math:`i` is iterations.
    - Space: :math:`O((n+k) \cdot m)`.

    **When to use KMeansClusterer:**

    - When the number of clusters is known or can be estimated
    - When clusters are roughly spherical and similar in size
    - Large datasets (linear complexity per iteration)
    - As a baseline clustering method before trying more complex approaches

    References
    ----------
    .. [Arthur2007] Arthur, D., & Vassilvitskii, S. (2007).
           **k-means++: The advantages of careful seeding.**
           *Proceedings of the eighteenth annual ACM-SIAM symposium on
           Discrete algorithms*, pp. 1027-1035.

    .. [Lloyd1982] Lloyd, S. P. (1982).
           **Least squares quantization in PCM.**
           *IEEE Transactions on Information Theory*, 28(2), pp. 129-137.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.FarthestFirstClusterer` : Fast center initialization often used with K-Means.
    :class:`~tuiml.algorithms.clustering.GaussianMixtureClusterer` : Soft probabilistic extension of K-Means.
    :class:`~tuiml.algorithms.clustering.FilteredClusterer` : Meta-clusterer for preprocessing before K-Means.

    Examples
    --------
    Basic K-Means clustering:

    >>> import numpy as np
    >>> from tuiml.algorithms.clustering import KMeansClusterer
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeansClusterer(n_clusters=2, random_state=0)
    >>> kmeans.fit(X)
    >>> kmeans.labels_
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([0, 1], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[1., 2.], [10., 2.]])
    """

    def __init__(self, n_clusters: int = 2,
                 init: str = 'k-means++',
                 max_iter: int = 300,
                 n_init: int = 10,
                 tol: float = 1e-4,
                 distance: str = 'euclidean',
                 random_state: Optional[int] = None):
        """Initialize KMeansClusterer with clustering parameters.

        Parameters
        ----------
        n_clusters : int, default=2
            Number of clusters.
        init : str, default='k-means++'
            Initialization method.
        max_iter : int, default=300
            Maximum iterations.
        n_init : int, default=10
            Number of initializations.
        tol : float, default=1e-4
            Convergence tolerance.
        distance : str, default='euclidean'
            Distance metric.
        random_state : int, optional
            Random seed.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.distance = distance
        self.random_state = random_state
        self.inertia_ = None
        self.n_iter_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_clusters": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Number of clusters"
            },
            "init": {
                "type": "string",
                "default": "k-means++",
                "enum": ["random", "k-means++", "farthest"],
                "description": "Initialization method"
            },
            "max_iter": {
                "type": "integer",
                "default": 300,
                "minimum": 1,
                "description": "Maximum iterations"
            },
            "n_init": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "description": "Number of initializations"
            },
            "tol": {
                "type": "number",
                "default": 1e-4,
                "minimum": 0,
                "description": "Convergence tolerance"
            },
            "distance": {
                "type": "string",
                "default": "euclidean",
                "enum": ["euclidean", "manhattan"],
                "description": "Distance metric"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "missing_values"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * k * d * i) time, O(n * d + k * d) space"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Arthur, D. & Vassilvitskii, S. (2007). k-means++: the advantages "
            "of careful seeding. SODA '07, 1027-1035."
        ]

    def _init_random(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Initialize centroids randomly.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        centers : np.ndarray of shape (n_clusters, n_features)
            Randomly selected initial centroids.
        """
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _init_kmeans_pp(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Initialize centroids using k-means++ algorithm.

        Selects initial centroids with probability proportional to
        squared distance from the nearest existing centroid.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        centers : np.ndarray of shape (n_clusters, n_features)
            Initial centroids selected via k-means++.
        """
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features))

        # Choose first center randomly
        idx = rng.integers(n_samples)
        centers[0] = X[idx]

        # Choose remaining centers
        for i in range(1, self.n_clusters):
            # Compute distances to nearest center
            distances = pairwise_distances(X, centers[:i], self.distance)
            min_distances = np.min(distances, axis=1)

            # Square distances for weighting
            weights = min_distances ** 2
            weights /= weights.sum()

            # Sample next center
            idx = rng.choice(n_samples, p=weights)
            centers[i] = X[idx]

        return centers

    def _init_farthest(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Initialize centroids using farthest-first traversal.

        Each new centroid is the point farthest from all existing centroids.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        centers : np.ndarray of shape (n_clusters, n_features)
            Initial centroids selected via farthest-first traversal.
        """
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features))

        # Choose first center randomly
        idx = rng.integers(n_samples)
        centers[0] = X[idx]

        # Choose remaining centers
        for i in range(1, self.n_clusters):
            distances = pairwise_distances(X, centers[:i], self.distance)
            min_distances = np.min(distances, axis=1)
            idx = np.argmax(min_distances)
            centers[i] = X[idx]

        return centers

    def _compute_labels(self, X: np.ndarray) -> tuple:
        """Assign points to nearest centroid.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points to assign.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Index of the nearest centroid for each point.
        inertia : float
            Sum of squared distances to the nearest centroid.
        """
        distances = pairwise_distances(X, self.cluster_centers_, self.distance)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.min(distances, axis=1) ** 2)
        return labels, inertia

    def _update_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids to mean of assigned points.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        labels : np.ndarray of shape (n_samples,)
            Current cluster assignments.

        Returns
        -------
        centers : np.ndarray of shape (n_clusters, n_features)
            Updated centroids.
        """
        n_features = X.shape[1]
        centers = np.zeros((self.n_clusters, n_features))

        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                if self.distance == 'manhattan':
                    # Use median for Manhattan distance
                    centers[k] = np.median(X[mask], axis=0)
                else:
                    centers[k] = np.mean(X[mask], axis=0)
            else:
                # Empty cluster: reinitialize to a random point
                centers[k] = X[np.random.randint(len(X))]

        return centers

    def _single_run(self, X: np.ndarray, rng: np.random.Generator) -> tuple:
        """Run a single K-Means optimization from one random initialization.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        centers : np.ndarray of shape (n_clusters, n_features)
            Final cluster centers.
        labels : np.ndarray of shape (n_samples,)
            Final cluster assignments.
        inertia : float
            Final inertia value.
        n_iter : int
            Number of iterations performed.
        """
        # Initialize centroids
        if self.init == 'k-means++':
            centers = self._init_kmeans_pp(X, rng)
        elif self.init == 'farthest':
            centers = self._init_farthest(X, rng)
        else:
            centers = self._init_random(X, rng)

        self.cluster_centers_ = centers

        # Iterate
        for iteration in range(self.max_iter):
            # Assign labels
            labels, inertia = self._compute_labels(X)

            # Update centers
            new_centers = self._update_centers(X, labels)

            # Check convergence
            center_shift = np.sum((new_centers - self.cluster_centers_) ** 2)
            self.cluster_centers_ = new_centers

            if center_shift < self.tol:
                break

        return self.cluster_centers_.copy(), labels, inertia, iteration + 1

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        """Compute k-means clustering.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        self : KMeansClusterer
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]

        # Handle case where n_clusters > n_samples
        if self.n_clusters > n_samples:
            self.n_clusters = n_samples

        # Set up random number generator
        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_n_iter = 0

        # Run multiple initializations
        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._single_run(X, rng)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self.n_clusters_ = self.n_clusters
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        labels, _ = self._compute_labels(X)
        return labels

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster 
        centers.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return pairwise_distances(X, self.cluster_centers_, self.distance)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"KMeansClusterer(n_clusters={self.n_clusters_}, "
                   f"inertia={self.inertia_:.2f}, n_iter={self.n_iter_})")
        return f"KMeansClusterer(n_clusters={self.n_clusters}, init='{self.init}')"
