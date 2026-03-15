"""K-Nearest Neighbors classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import Counter

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.algorithms.neighbors.search import (
    NearestNeighborSearch, LinearNNSearch, KDTree, BallTree
)
from tuiml._cpp_ext import neighbors as _cpp_nn

@classifier(tags=["lazy", "instance-based", "knn"], version="1.0.0")
class KNearestNeighborsClassifier(Classifier):
    """K-Nearest Neighbors classifier using **instance-based lazy learning**.

    KNearestNeighborsClassifier classifies instances based on **similarity** to
    training examples. For each test instance, it finds the :math:`k` nearest
    training instances and predicts the **majority class** among them. The
    algorithm is "lazy" because it defers computation until prediction time,
    storing all training instances rather than building an explicit model.

    Also known as IBk (Instance-Based learning algorithm k).

    Overview
    --------
    The algorithm operates in the following steps:

    1. Store all training instances during ``fit()`` (no model is built).
    2. For a new query point, compute the distance to every training instance
       (or use an accelerated search structure such as a KD-tree or Ball Tree).
    3. Select the :math:`k` closest training instances.
    4. Assign weights to the neighbors according to the chosen weighting scheme.
    5. Predict the class with the highest aggregated weight among the neighbors.

    Theory
    ------
    Given a query point :math:`x`, the predicted class is:

    .. math::
        \\hat{y} = \\arg\\max_{c \\in C} \\sum_{i \\in N_k(x)} w_i \\cdot \\mathbb{1}(y_i = c)

    where:

    - :math:`N_k(x)` — The set of :math:`k` nearest neighbors of :math:`x`
    - :math:`w_i` — Weight assigned to neighbor :math:`i`
    - :math:`C` — The set of all classes

    **Weighting schemes:**

    - *Uniform:* :math:`w_i = 1`
    - *Inverse distance:* :math:`w_i = 1 / d(x, x_i)`
    - *Similarity:* :math:`w_i = 1 / (1 + d(x, x_i))`

    The default distance metric is **Euclidean distance**:

    .. math::
        d(x, x_i) = \\sqrt{\\sum_{j=1}^{m} (x_j - x_{i,j})^2}

    Parameters
    ----------
    k : int, default=1
        Number of neighbors to use.
    distance_weighting : {'uniform', 'distance', 'similarity'}, default='uniform'
        How to weight neighbors:

        - ``'uniform'`` — All neighbors weighted equally.
        - ``'distance'`` — Weight by inverse of distance :math:`1/d`.
        - ``'similarity'`` — Weight by similarity :math:`1/(1+d)`.
    search_algorithm : {'brute', 'kd_tree', 'ball_tree'}, default='brute'
        Algorithm for finding neighbors:

        - ``'brute'`` — Brute force search.
        - ``'kd_tree'`` — KD-tree for faster search in low dimensions.
        - ``'ball_tree'`` — Ball tree for higher-dimensional or non-Euclidean data.
    cross_validate : bool, default=False
        If True, use leave-one-out cross-validation to automatically
        select the optimal :math:`k`.
    mean_squared : bool, default=False
        If True, use the square of the weights specified by
        ``distance_weighting``.
    leaf_size : int, default=30
        Leaf size for tree-based search algorithms (KD-tree and Ball Tree).

    Attributes
    ----------
    X_train_ : np.ndarray
        Training features stored for lazy learning.
    y_train_ : np.ndarray
        Training labels stored for lazy learning.
    classes_ : np.ndarray
        Unique class labels discovered during :meth:`fit`.
    search_ : NearestNeighborSearch
        The search structure instance used for neighbor queries.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(1)` (instances are simply stored)
    - Prediction (brute force): :math:`O(n \\cdot m)` per query where :math:`n` = number of training samples, :math:`m` = number of features
    - Prediction (KD-tree, average case): :math:`O(m \\cdot \\log n)` per query
    - Space: :math:`O(n \\cdot m)` for storing all training instances

    **When to use KNearestNeighborsClassifier:**

    - Small to medium datasets where training time must be near-zero
    - Decision boundaries are highly irregular or non-linear
    - New training instances arrive incrementally (online learning)
    - When an interpretable, non-parametric baseline is needed
    - Low-dimensional feature spaces (especially with tree-based search)

    References
    ----------
    .. [Aha1991] Aha, D.W., Kibler, D. and Albert, M.K. (1991).
           **Instance-Based Learning Algorithms.**
           *Machine Learning*, 6, pp. 37-66.
           DOI: `10.1007/BF00153759 <https://doi.org/10.1007/BF00153759>`_

    .. [Cover1967] Cover, T.M. and Hart, P.E. (1967).
           **Nearest Neighbor Pattern Classification.**
           *IEEE Transactions on Information Theory*, 13(1), pp. 21-27.
           DOI: `10.1109/TIT.1967.1053964 <https://doi.org/10.1109/TIT.1967.1053964>`_

    .. [Dudani1976] Dudani, S.A. (1976).
           **The Distance-Weighted k-Nearest-Neighbor Rule.**
           *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-6(4), pp. 325-327.
           DOI: `10.1109/TSMC.1976.5408784 <https://doi.org/10.1109/TSMC.1976.5408784>`_

    See Also
    --------
    :class:`~tuiml.algorithms.neighbors.KStarClassifier` : Entropy-based instance classifier using transformation complexity.
    :class:`~tuiml.algorithms.neighbors.LocallyWeightedLearningRegressor` : Instance-based regression with local models.

    Examples
    --------
    Basic classification with distance-weighted voting:

    >>> from tuiml.algorithms.neighbors import KNearestNeighborsClassifier
    >>> from tuiml.datasets import load_iris
    >>> X, y = load_iris()
    >>> clf = KNearestNeighborsClassifier(k=3, distance_weighting='distance')
    >>> clf.fit(X, y)
    KNearestNeighborsClassifier(k=3, n_train=150, weighting='distance')
    >>> clf.predict(X[:1])
    array([0])
    """

    def __init__(self, k: int = 1,
                 distance_weighting: str = 'uniform',
                 search_algorithm: str = 'brute',
                 cross_validate: bool = False,
                 mean_squared: bool = False,
                 leaf_size: int = 30):
        """Initialize KNearestNeighborsClassifier.

        Parameters
        ----------
        k : int, default=1
            Number of neighbors.
        distance_weighting : str, default='uniform'
            Weighting scheme: ``'uniform'``, ``'distance'``, or ``'similarity'``.
        search_algorithm : str, default='brute'
            Search backend: ``'brute'``, ``'kd_tree'``, or ``'ball_tree'``.
        cross_validate : bool, default=False
            Use leave-one-out cross-validation to select k.
        mean_squared : bool, default=False
            Use squared weights for distance weighting.
        leaf_size : int, default=30
            Leaf size for tree-based search algorithms.
        """
        super().__init__()
        self.k = k
        self.distance_weighting = distance_weighting
        self.search_algorithm = search_algorithm
        self.cross_validate = cross_validate
        self.mean_squared = mean_squared
        self.leaf_size = leaf_size
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self._n_features = None
        self.search_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "k": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Number of nearest neighbors to use"
            },
            "distance_weighting": {
                "type": "string",
                "default": "uniform",
                "enum": ["uniform", "distance", "similarity"],
                "description": "Method for weighting neighbors"
            },
            "search_algorithm": {
                "type": "string",
                "default": "brute",
                "enum": ["brute", "kd_tree", "ball_tree"],
                "description": "Algorithm for finding neighbors"
            },
            "cross_validate": {
                "type": "boolean",
                "default": False,
                "description": "Use cross-validation to select k"
            },
            "mean_squared": {
                "type": "boolean",
                "default": False,
                "description": "Use mean squared distance for weighting"
            },
            "leaf_size": {
                "type": "integer",
                "default": 30,
                "minimum": 1,
                "description": "Leaf size for tree-based search algorithms"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return [
            "numeric",
            "nominal",
            "missing_values",
            "binary_class",
            "multiclass"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(1) training, O(n * m) prediction per sample"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Aha, D.W., Kibler, D., & Albert, M.K. (1991). Instance-Based "
            "Learning Algorithms. Machine Learning, 6, 37-66."
        ]

    def _create_search(self) -> NearestNeighborSearch:
        """Create the search algorithm based on configuration.

        Returns
        -------
        search : NearestNeighborSearch
            An instance of the selected search backend.
        """
        if self.search_algorithm == 'kd_tree':
            return KDTree(leaf_size=self.leaf_size)
        elif self.search_algorithm == 'ball_tree':
            return BallTree(leaf_size=self.leaf_size)
        else:  # 'brute' or default
            return LinearNNSearch()

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two instances.

        Missing values (NaN) are ignored during computation.

        Parameters
        ----------
        x1 : np.ndarray of shape (n_features,)
            First instance.
        x2 : np.ndarray of shape (n_features,)
            Second instance.

        Returns
        -------
        distance : float
            Euclidean distance, or ``np.inf`` if no valid features remain.
        """
        # Handle missing values by ignoring them
        valid_mask = ~(np.isnan(x1) | np.isnan(x2))
        if not np.any(valid_mask):
            return np.inf

        diff = x1[valid_mask] - x2[valid_mask]
        return np.sqrt(np.sum(diff ** 2))

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Compute distances from a query point to all training instances.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query point.

        Returns
        -------
        distances : np.ndarray of shape (n_samples,)
            Euclidean distance from ``x`` to each training instance.
        """
        distances = np.zeros(len(self.X_train_))
        for i, x_train in enumerate(self.X_train_):
            distances[i] = self._euclidean_distance(x, x_train)
        return distances

    def _get_neighbors(self, x: np.ndarray) -> tuple:
        """Find the k nearest neighbors for a single query point.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query point.

        Returns
        -------
        neighbor_indices : np.ndarray of shape (k,)
            Indices of the k nearest neighbors in the training data.
        neighbor_distances : np.ndarray of shape (k,)
            Corresponding distances to the neighbors.
        """
        # Use the search algorithm if available
        if self.search_ is not None:
            neighbor_distances, neighbor_indices = self.search_.query(x, self.k)
            return neighbor_indices, neighbor_distances

        # Fallback to brute force
        distances = self._compute_distances(x)

        # Get indices of k smallest distances
        if self.k >= len(distances):
            neighbor_indices = np.arange(len(distances))
        else:
            neighbor_indices = np.argpartition(distances, self.k)[:self.k]

        neighbor_distances = distances[neighbor_indices]

        # Sort by distance
        sorted_order = np.argsort(neighbor_distances)
        neighbor_indices = neighbor_indices[sorted_order]
        neighbor_distances = neighbor_distances[sorted_order]

        return neighbor_indices, neighbor_distances

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute weights for neighbors based on their distances.

        Parameters
        ----------
        distances : np.ndarray of shape (k,)
            Distances from the query point to each neighbor.

        Returns
        -------
        weights : np.ndarray of shape (k,)
            Weight assigned to each neighbor.
        """
        if self.distance_weighting == 'uniform':
            return np.ones(len(distances))

        elif self.distance_weighting == 'distance':
            # Inverse distance weighting
            # Handle zero distance (exact match)
            weights = np.where(distances > 0, 1.0 / distances, 1e10)
            if self.mean_squared:
                weights = weights ** 2
            return weights

        elif self.distance_weighting == 'similarity':
            # Similarity weighting
            weights = 1.0 / (1.0 + distances)
            if self.mean_squared:
                weights = weights ** 2
            return weights

        else:
            return np.ones(len(distances))

    def _cross_validate_k(self, X: np.ndarray, y: np.ndarray) -> int:
        """Use leave-one-out cross-validation to find the optimal k.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        best_k : int
            The value of k (between 1 and 20) that maximizes LOO accuracy.
        """
        n_samples = len(X)
        max_k = min(20, n_samples - 1)  # Test k from 1 to 20

        best_k = 1
        best_accuracy = 0

        for k in range(1, max_k + 1):
            correct = 0

            for i in range(n_samples):
                # Leave one out
                X_loo = np.delete(X, i, axis=0)
                y_loo = np.delete(y, i)

                # Temporarily set training data
                self.X_train_ = X_loo
                self.y_train_ = y_loo
                self.k = k

                # Predict
                pred = self._predict_single(X[i])
                if pred == y[i]:
                    correct += 1

            accuracy = correct / n_samples
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        return best_k

    def _predict_single(self, x: np.ndarray) -> Any:
        """Predict the class label for a single instance.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query instance.

        Returns
        -------
        label : Any
            The predicted class label.
        """
        neighbor_indices, neighbor_distances = self._get_neighbors(x)
        weights = self._compute_weights(neighbor_distances)

        # Get neighbor classes
        neighbor_classes = self.y_train_[neighbor_indices]

        # Weighted voting
        class_weights = {}
        for i, cls in enumerate(neighbor_classes):
            class_weights[cls] = class_weights.get(cls, 0) + weights[i]

        return max(class_weights, key=class_weights.get)

    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single instance.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query instance.

        Returns
        -------
        proba : np.ndarray of shape (n_classes,)
            Normalized class probabilities derived from weighted voting.
        """
        neighbor_indices, neighbor_distances = self._get_neighbors(x)
        weights = self._compute_weights(neighbor_distances)

        # Get neighbor classes
        neighbor_classes = self.y_train_[neighbor_indices]

        # Weighted voting
        class_weights = {cls: 0.0 for cls in self.classes_}
        for i, cls in enumerate(neighbor_classes):
            class_weights[cls] += weights[i]

        # Normalize to probabilities
        total_weight = sum(class_weights.values())
        proba = np.array([class_weights[cls] / total_weight
                          for cls in self.classes_])

        return proba

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighborsClassifier":
        """Fit the KNearestNeighborsClassifier classifier by storing the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : KNearestNeighborsClassifier
            Returns the instance itself.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self._n_features = X.shape[1]

        # Cross-validate to find optimal k if requested
        if self.cross_validate:
            self.k = self._cross_validate_k(X, y)

        # Build the search structure
        self.search_ = self._create_search()
        self.search_.build(X)

        self._is_fitted = True
        return self

    def _batch_predict_core(self, X: np.ndarray):
        """Batch neighbor query + weight computation via C++.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query samples (already validated).

        Returns
        -------
        distances : np.ndarray of shape (n_samples, k)
            Distances to k nearest neighbors.
        neighbor_labels : np.ndarray of shape (n_samples, k)
            Labels of k nearest neighbors.
        weights : np.ndarray of shape (n_samples, k)
            Weights for each neighbor.
        """
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        distances, indices = self.search_.query_batch(X_c, self.k)
        neighbor_labels = self.y_train_[indices]

        # Vectorised weight computation
        if self.distance_weighting == 'distance':
            weights = np.where(distances > 0, 1.0 / distances, 1e10)
        elif self.distance_weighting == 'similarity':
            weights = 1.0 / (1.0 + distances)
        else:
            weights = np.ones_like(distances)

        if self.mean_squared and self.distance_weighting != 'uniform':
            weights = weights ** 2

        return distances, neighbor_labels, weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the provided samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The samples to predict.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            The predicted class labels.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        distances, neighbor_labels, weights = self._batch_predict_core(X)

        # Vectorised weighted voting
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_scores = np.zeros((n_samples, n_classes))

        for c_idx, c in enumerate(self.classes_):
            mask = neighbor_labels == c
            class_scores[:, c_idx] = np.sum(weights * mask, axis=1)

        pred_indices = np.argmax(class_scores, axis=1)
        return self.classes_[pred_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the provided samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The samples to predict.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            The class probabilities.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        distances, neighbor_labels, weights = self._batch_predict_core(X)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_scores = np.zeros((n_samples, n_classes))

        for c_idx, c in enumerate(self.classes_):
            mask = neighbor_labels == c
            class_scores[:, c_idx] = np.sum(weights * mask, axis=1)

        # Normalise to probabilities
        totals = class_scores.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return class_scores / totals

    def update(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighborsClassifier":
        """Add new instances to the training set (online learning).

        Parameters
        ----------
        X : np.ndarray
            New training features.
        y : np.ndarray
            New training labels.

        Returns
        -------
        self : KNearestNeighborsClassifier
            Returns the instance with updated training data.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self._is_fitted:
            self.X_train_ = np.vstack([self.X_train_, X])
            self.y_train_ = np.concatenate([self.y_train_, y])
            # Update classes if new ones appear
            self.classes_ = np.unique(self.y_train_)
            # Rebuild the search structure with updated data
            self.search_ = self._create_search()
            self.search_.build(self.X_train_)
        else:
            self.fit(X, y)

        return self

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"KNearestNeighborsClassifier(k={self.k}, n_train={len(self.X_train_)}, "
                   f"weighting='{self.distance_weighting}')")
        return f"KNearestNeighborsClassifier(k={self.k}, distance_weighting='{self.distance_weighting}')"


@regressor(tags=["lazy", "instance-based", "knn"], version="1.0.0")
class KNearestNeighborsRegressor(Regressor):
    """K-Nearest Neighbors regressor using **instance-based lazy learning**.

    KNearestNeighborsRegressor predicts continuous target values based on
    **similarity** to training examples. For each test instance, it finds the
    :math:`k` nearest training instances and predicts the **weighted average**
    of their target values. The algorithm is "lazy" because it defers
    computation until prediction time, storing all training instances rather
    than building an explicit model.

    Overview
    --------
    The algorithm operates in the following steps:

    1. Store all training instances during ``fit()`` (no model is built).
    2. For a new query point, compute the distance to every training instance
       (or use an accelerated search structure such as a KD-tree or Ball Tree).
    3. Select the :math:`k` closest training instances.
    4. Assign weights to the neighbors according to the chosen weighting scheme.
    5. Predict the **weighted average** of the neighbor target values.

    Theory
    ------
    Given a query point :math:`x`, the predicted value is:

    .. math::
        \\hat{y} = \\frac{\\sum_{i \\in N_k(x)} w_i \\cdot y_i}{\\sum_{i \\in N_k(x)} w_i}

    where:

    - :math:`N_k(x)` -- The set of :math:`k` nearest neighbors of :math:`x`
    - :math:`w_i` -- Weight assigned to neighbor :math:`i`
    - :math:`y_i` -- Target value of neighbor :math:`i`

    **Weighting schemes:**

    - *Uniform:* :math:`w_i = 1`
    - *Inverse distance:* :math:`w_i = 1 / d(x, x_i)`
    - *Similarity:* :math:`w_i = 1 / (1 + d(x, x_i))`

    Parameters
    ----------
    k : int, default=1
        Number of neighbors to use.
    distance_weighting : {'uniform', 'distance', 'similarity'}, default='uniform'
        How to weight neighbors:

        - ``'uniform'`` -- All neighbors weighted equally.
        - ``'distance'`` -- Weight by inverse of distance :math:`1/d`.
        - ``'similarity'`` -- Weight by similarity :math:`1/(1+d)`.
    search_algorithm : {'brute', 'kd_tree', 'ball_tree'}, default='brute'
        Algorithm for finding neighbors:

        - ``'brute'`` -- Brute force search.
        - ``'kd_tree'`` -- KD-tree for faster search in low dimensions.
        - ``'ball_tree'`` -- Ball tree for higher-dimensional data.
    cross_validate : bool, default=False
        If True, use leave-one-out cross-validation to automatically
        select the optimal :math:`k` using MSE.
    mean_squared : bool, default=False
        If True, use the square of the weights specified by
        ``distance_weighting``.
    leaf_size : int, default=30
        Leaf size for tree-based search algorithms (KD-tree and Ball Tree).

    Attributes
    ----------
    X_train_ : np.ndarray
        Training features stored for lazy learning.
    y_train_ : np.ndarray
        Training target values stored for lazy learning.
    search_ : NearestNeighborSearch
        The search structure instance used for neighbor queries.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(1)` (instances are simply stored)
    - Prediction (brute force): :math:`O(n \\cdot m)` per query
    - Prediction (KD-tree, average case): :math:`O(m \\cdot \\log n)` per query
    - Space: :math:`O(n \\cdot m)` for storing all training instances

    **When to use KNearestNeighborsRegressor:**

    - Small to medium datasets where training time must be near-zero
    - Non-linear relationships between features and target
    - When an interpretable, non-parametric baseline is needed
    - Low-dimensional feature spaces (especially with tree-based search)

    References
    ----------
    .. [Aha1991] Aha, D.W., Kibler, D. and Albert, M.K. (1991).
           **Instance-Based Learning Algorithms.**
           *Machine Learning*, 6, pp. 37-66.
           DOI: `10.1007/BF00153759 <https://doi.org/10.1007/BF00153759>`_

    .. [Cover1967] Cover, T.M. and Hart, P.E. (1967).
           **Nearest Neighbor Pattern Classification.**
           *IEEE Transactions on Information Theory*, 13(1), pp. 21-27.
           DOI: `10.1109/TIT.1967.1053964 <https://doi.org/10.1109/TIT.1967.1053964>`_

    See Also
    --------
    :class:`~tuiml.algorithms.neighbors.KNearestNeighborsClassifier` : KNN for classification tasks.
    :class:`~tuiml.algorithms.neighbors.LocallyWeightedLearningRegressor` : Instance-based regression with local models.

    Examples
    --------
    Basic regression with distance-weighted averaging:

    >>> from tuiml.algorithms.neighbors import KNearestNeighborsRegressor
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([1.0, 2.1, 2.9, 4.0, 5.1])
    >>> reg = KNearestNeighborsRegressor(k=3, distance_weighting='distance')
    >>> reg.fit(X, y)
    KNearestNeighborsRegressor(k=3, n_train=5, weighting='distance')
    >>> reg.predict(np.array([[2.5]]))
    array([...])
    """

    def __init__(self, k: int = 1,
                 distance_weighting: str = 'uniform',
                 search_algorithm: str = 'brute',
                 cross_validate: bool = False,
                 mean_squared: bool = False,
                 leaf_size: int = 30):
        """Initialize KNearestNeighborsRegressor.

        Parameters
        ----------
        k : int, default=1
            Number of neighbors.
        distance_weighting : str, default='uniform'
            Weighting scheme: ``'uniform'``, ``'distance'``, or ``'similarity'``.
        search_algorithm : str, default='brute'
            Search backend: ``'brute'``, ``'kd_tree'``, or ``'ball_tree'``.
        cross_validate : bool, default=False
            Use leave-one-out cross-validation to select k using MSE.
        mean_squared : bool, default=False
            Use squared weights for distance weighting.
        leaf_size : int, default=30
            Leaf size for tree-based search algorithms.
        """
        super().__init__()
        self.k = k
        self.distance_weighting = distance_weighting
        self.search_algorithm = search_algorithm
        self.cross_validate = cross_validate
        self.mean_squared = mean_squared
        self.leaf_size = leaf_size
        self.X_train_ = None
        self.y_train_ = None
        self._n_features = None
        self.search_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "k": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Number of nearest neighbors to use"
            },
            "distance_weighting": {
                "type": "string",
                "default": "uniform",
                "enum": ["uniform", "distance", "similarity"],
                "description": "Method for weighting neighbors"
            },
            "search_algorithm": {
                "type": "string",
                "default": "brute",
                "enum": ["brute", "kd_tree", "ball_tree"],
                "description": "Algorithm for finding neighbors"
            },
            "cross_validate": {
                "type": "boolean",
                "default": False,
                "description": "Use cross-validation to select k"
            },
            "mean_squared": {
                "type": "boolean",
                "default": False,
                "description": "Use mean squared distance for weighting"
            },
            "leaf_size": {
                "type": "integer",
                "default": 30,
                "minimum": 1,
                "description": "Leaf size for tree-based search algorithms"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return regressor capabilities."""
        return ["numeric", "numeric_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(1) training, O(n * m) prediction per sample"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Aha, D.W., Kibler, D., & Albert, M.K. (1991). Instance-Based "
            "Learning Algorithms. Machine Learning, 6, 37-66."
        ]

    def _create_search(self) -> NearestNeighborSearch:
        """Create the search algorithm based on configuration.

        Returns
        -------
        search : NearestNeighborSearch
            An instance of the selected search backend.
        """
        if self.search_algorithm == 'kd_tree':
            return KDTree(leaf_size=self.leaf_size)
        elif self.search_algorithm == 'ball_tree':
            return BallTree(leaf_size=self.leaf_size)
        else:
            return LinearNNSearch()

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two instances.

        Missing values (NaN) are ignored during computation.

        Parameters
        ----------
        x1 : np.ndarray of shape (n_features,)
            First instance.
        x2 : np.ndarray of shape (n_features,)
            Second instance.

        Returns
        -------
        distance : float
            Euclidean distance, or ``np.inf`` if no valid features remain.
        """
        valid_mask = ~(np.isnan(x1) | np.isnan(x2))
        if not np.any(valid_mask):
            return np.inf
        diff = x1[valid_mask] - x2[valid_mask]
        return np.sqrt(np.sum(diff ** 2))

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Compute distances from a query point to all training instances.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query point.

        Returns
        -------
        distances : np.ndarray of shape (n_samples,)
            Euclidean distance from ``x`` to each training instance.
        """
        distances = np.zeros(len(self.X_train_))
        for i, x_train in enumerate(self.X_train_):
            distances[i] = self._euclidean_distance(x, x_train)
        return distances

    def _get_neighbors(self, x: np.ndarray) -> tuple:
        """Find the k nearest neighbors for a single query point.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query point.

        Returns
        -------
        neighbor_indices : np.ndarray of shape (k,)
            Indices of the k nearest neighbors in the training data.
        neighbor_distances : np.ndarray of shape (k,)
            Corresponding distances to the neighbors.
        """
        if self.search_ is not None:
            neighbor_distances, neighbor_indices = self.search_.query(x, self.k)
            return neighbor_indices, neighbor_distances

        distances = self._compute_distances(x)

        if self.k >= len(distances):
            neighbor_indices = np.arange(len(distances))
        else:
            neighbor_indices = np.argpartition(distances, self.k)[:self.k]

        neighbor_distances = distances[neighbor_indices]

        sorted_order = np.argsort(neighbor_distances)
        neighbor_indices = neighbor_indices[sorted_order]
        neighbor_distances = neighbor_distances[sorted_order]

        return neighbor_indices, neighbor_distances

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute weights for neighbors based on their distances.

        Parameters
        ----------
        distances : np.ndarray of shape (k,)
            Distances from the query point to each neighbor.

        Returns
        -------
        weights : np.ndarray of shape (k,)
            Weight assigned to each neighbor.
        """
        if self.distance_weighting == 'uniform':
            return np.ones(len(distances))
        elif self.distance_weighting == 'distance':
            weights = np.where(distances > 0, 1.0 / distances, 1e10)
            if self.mean_squared:
                weights = weights ** 2
            return weights
        elif self.distance_weighting == 'similarity':
            weights = 1.0 / (1.0 + distances)
            if self.mean_squared:
                weights = weights ** 2
            return weights
        else:
            return np.ones(len(distances))

    def _cross_validate_k(self, X: np.ndarray, y: np.ndarray) -> int:
        """Use leave-one-out cross-validation to find the optimal k via MSE.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training target values.

        Returns
        -------
        best_k : int
            The value of k (between 1 and 20) that minimizes LOO MSE.
        """
        n_samples = len(X)
        max_k = min(20, n_samples - 1)

        best_k = 1
        best_mse = np.inf

        for k in range(1, max_k + 1):
            mse_sum = 0.0

            for i in range(n_samples):
                X_loo = np.delete(X, i, axis=0)
                y_loo = np.delete(y, i)

                self.X_train_ = X_loo
                self.y_train_ = y_loo
                self.k = k

                pred = self._predict_single(X[i])
                mse_sum += (pred - y[i]) ** 2

            mse = mse_sum / n_samples
            if mse < best_mse:
                best_mse = mse
                best_k = k

        return best_k

    def _predict_single(self, x: np.ndarray) -> float:
        """Predict the target value for a single instance.

        Computes the weighted average of the target values of the k
        nearest neighbors.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query instance.

        Returns
        -------
        prediction : float
            The predicted target value.
        """
        neighbor_indices, neighbor_distances = self._get_neighbors(x)
        weights = self._compute_weights(neighbor_distances)

        neighbor_targets = self.y_train_[neighbor_indices]
        return np.sum(weights * neighbor_targets) / np.sum(weights)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighborsRegressor":
        """Fit the regressor by storing the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNearestNeighborsRegressor
            Returns the instance itself.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_train_ = X
        self.y_train_ = y
        self._n_features = X.shape[1]

        if self.cross_validate:
            self.k = self._cross_validate_k(X, y)

        self.search_ = self._create_search()
        self.search_.build(X)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for the provided samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The samples to predict.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            The predicted target values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        distances, indices = self.search_.query_batch(X_c, self.k)
        neighbor_targets = self.y_train_[indices]

        # Vectorised weight computation
        if self.distance_weighting == 'distance':
            weights = np.where(distances > 0, 1.0 / distances, 1e10)
        elif self.distance_weighting == 'similarity':
            weights = 1.0 / (1.0 + distances)
        else:
            weights = np.ones_like(distances)

        if self.mean_squared and self.distance_weighting != 'uniform':
            weights = weights ** 2

        # Weighted average
        return np.sum(weights * neighbor_targets, axis=1) / np.sum(weights, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the R-squared (coefficient of determination) score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            R-squared score.
        """
        self._check_is_fitted()
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def update(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighborsRegressor":
        """Add new instances to the training set (online learning).

        Parameters
        ----------
        X : np.ndarray
            New training features.
        y : np.ndarray
            New training target values.

        Returns
        -------
        self : KNearestNeighborsRegressor
            Returns the instance with updated training data.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self._is_fitted:
            self.X_train_ = np.vstack([self.X_train_, X])
            self.y_train_ = np.concatenate([self.y_train_, y])
            self.search_ = self._create_search()
            self.search_.build(self.X_train_)
        else:
            self.fit(X, y)

        return self

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"KNearestNeighborsRegressor(k={self.k}, n_train={len(self.X_train_)}, "
                   f"weighting='{self.distance_weighting}')")
        return f"KNearestNeighborsRegressor(k={self.k}, distance_weighting='{self.distance_weighting}')"
