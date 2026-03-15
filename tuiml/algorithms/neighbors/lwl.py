"""Locally Weighted Learning implementation."""

import numpy as np
from typing import Dict, List, Any, Optional
from copy import deepcopy

from tuiml.base.algorithms import Regressor, regressor
from tuiml._cpp_ext import distance as _cpp_dist

@regressor(tags=["neighbors", "regression", "local", "lazy"], version="1.0.0")
class LocallyWeightedLearningRegressor(Regressor):
    """Locally Weighted Learning for regression using **distance-based kernels**.

    LocallyWeightedLearningRegressor is an instance-based algorithm that assigns
    **weights** to training instances based on their distance to each query point.
    A **local weighted linear regression** model is then trained using these
    weighted instances to make a prediction for that specific point.

    Also known as LWL (Locally Weighted Learning).

    Overview
    --------
    The algorithm operates in the following steps:

    1. Store and standardize all training instances during ``fit()``.
    2. For a new query :math:`x`, compute the Euclidean distance to every
       training instance in the standardized feature space.
    3. Select the :math:`k` nearest neighbors (or use all instances if
       :math:`k = -1`).
    4. Compute a weight for each selected instance using the chosen kernel
       function.
    5. Fit a **weighted least-squares** linear regression using the weighted
       instances.
    6. Predict the target value for the query point using the local model.

    Theory
    ------
    The prediction for a query point :math:`x` is obtained by solving the
    weighted least-squares problem:

    .. math::
        \\hat{\\beta} = (X^T W X + \\lambda I)^{-1} X^T W y

    where:

    - :math:`X` — Design matrix of (optionally selected) training instances
    - :math:`W` — Diagonal matrix of kernel weights :math:`\\text{diag}(w_1, \\ldots, w_n)`
    - :math:`y` — Target vector
    - :math:`\\lambda` — Small ridge regularization term for numerical stability

    **Kernel functions:**

    - *Linear:* :math:`w_i = \\max(0,\\; 1 - d_i / d_{\\max})`
    - *Inverse:* :math:`w_i = 1 / (d_i + \\epsilon)`
    - *Gaussian:* :math:`w_i = \\exp(-d_i^2 / (2 \\sigma^2))`

    where :math:`d_i = \\|x - x_i\\|` and :math:`\\sigma = d_{\\max} / 3`.

    Parameters
    ----------
    k : int, default=-1
        Number of nearest neighbors to use for local learning.
        If ``-1``, all training instances are used.
    weight_kernel : {'linear', 'inverse', 'gaussian'}, default='linear'
        Kernel function used to calculate weights from distances:

        - ``'linear'`` — :math:`w = 1 - d / d_{\\max}`
        - ``'inverse'`` — :math:`w = 1 / (d + \\epsilon)`
        - ``'gaussian'`` — :math:`w = \\exp(-d^2 / (2\\sigma^2))`
    use_all_attributes : bool, default=True
        Whether to use all attributes for distance calculation.

    Attributes
    ----------
    X_train_ : np.ndarray
        Standardized training features stored for lazy learning.
    y_train_ : np.ndarray
        Training target values.
    n_features_ : int
        Number of features in the training data.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m)` for standardization, where :math:`n` = samples, :math:`m` = features
    - Prediction: :math:`O(n \\cdot m + m^3)` per query (distance computation plus weighted least-squares solve)
    - Space: :math:`O(n \\cdot m)` for storing training data

    **When to use LocallyWeightedLearningRegressor:**

    - Non-linear regression problems where a global model is inadequate
    - When the relationship between features and target varies across
      the feature space
    - Problems where local smoothness can be assumed
    - When interpretability at the local level (linear coefficients) is
      useful

    References
    ----------
    .. [Atkeson1997] Atkeson, C.G., Moore, A.W. and Schaal, S. (1997).
           **Locally Weighted Learning.**
           *Artificial Intelligence Review*, 11(1-5), pp. 11-73.
           DOI: `10.1023/A:1006559212014 <https://doi.org/10.1023/A:1006559212014>`_

    .. [Frank2003] Frank, E., Hall, M. and Pfahringer, B. (2003).
           **Locally Weighted Naive Bayes.**
           *Proceedings of the 19th Conference on Uncertainty in AI*, pp. 249-256.

    See Also
    --------
    :class:`~tuiml.algorithms.neighbors.KNearestNeighborsClassifier` : Instance-based classification with k-nearest neighbors.
    :class:`~tuiml.algorithms.neighbors.KStarClassifier` : Entropy-based instance classifier.

    Examples
    --------
    Locally weighted regression on a simple linear relationship:

    >>> from tuiml.algorithms.neighbors import LocallyWeightedLearningRegressor
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([2, 4, 6, 8])
    >>> reg = LocallyWeightedLearningRegressor(k=2, weight_kernel='gaussian')
    >>> reg.fit(X, y)
    LocallyWeightedLearningRegressor(k=2, kernel='gaussian', n_train=4)
    >>> reg.predict([[2.5]])
    array([5.])
    """

    def __init__(
        self,
        k: int = -1,
        weight_kernel: str = "linear",
        use_all_attributes: bool = True,
    ):
        """Initialize LocallyWeightedLearningRegressor.

        Parameters
        ----------
        k : int, default=-1
            Number of neighbors (``-1`` for all instances).
        weight_kernel : str, default='linear'
            Kernel function: ``'linear'``, ``'inverse'``, or ``'gaussian'``.
        use_all_attributes : bool, default=True
            Use all attributes for distance calculation.
        """
        super().__init__()
        self.k = k
        self.weight_kernel = weight_kernel
        self.use_all_attributes = use_all_attributes

        # Fitted attributes
        self.X_train_ = None
        self.y_train_ = None
        self.n_features_ = None
        self._X_std = None
        self._X_mean = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "k": {
                "type": "integer",
                "default": -1,
                "minimum": -1,
                "description": "Number of neighbors (-1 for all)"
            },
            "weight_kernel": {
                "type": "string",
                "default": "linear",
                "enum": ["linear", "inverse", "gaussian"],
                "description": "Kernel function for weighting"
            },
            "use_all_attributes": {
                "type": "boolean",
                "default": True,
                "description": "Use all attributes for distance"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return [
            "numeric",
            "numeric_class"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m) per prediction, O(n * m) storage"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Atkeson, C., Moore, A., & Schaal, S. (1996). Locally weighted "
            "learning. AI Review, 11, 11-73.",
            "Frank, E., Hall, M., & Pfahringer, B. (2003). Locally Weighted "
            "Naive Bayes. 19th Conf. on Uncertainty in AI, 249-256."
        ]

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances from a query point to all training instances.

        The query point is standardized before computing distances in
        the standardized feature space.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query point (in original scale).

        Returns
        -------
        distances : np.ndarray of shape (n_samples,)
            Euclidean distance from the standardized query to each
            standardized training instance.
        """
        # Standardize query point
        x_std = (x - self._X_mean) / self._X_std

        # Compute Euclidean distances
        diff = self.X_train_ - x_std
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        return distances

    def _compute_weights(self, distances: np.ndarray, k: int) -> np.ndarray:
        """Compute instance weights based on distances and the kernel function.

        Parameters
        ----------
        distances : np.ndarray of shape (n_samples,)
            Distances from the query point to each training instance.
        k : int
            Number of nearest neighbors to consider. If ``k >= n_samples``,
            all instances are used.

        Returns
        -------
        weights : np.ndarray of shape (n_samples,)
            Kernel-derived weights for each training instance. Instances
            outside the :math:`k`-nearest neighborhood receive zero weight.
        """
        n = len(distances)
        weights = np.zeros(n)

        # Select k nearest neighbors if specified
        if k > 0 and k < n:
            # Get indices of k nearest neighbors
            k_indices = np.argpartition(distances, k)[:k]
            mask = np.zeros(n, dtype=bool)
            mask[k_indices] = True
            distances_subset = distances[mask]
        else:
            mask = np.ones(n, dtype=bool)
            distances_subset = distances

        # Compute max distance for normalization
        max_dist = np.max(distances_subset)
        if max_dist == 0:
            max_dist = 1.0

        # Apply kernel
        if self.weight_kernel == "linear":
            # Linear decay: w = 1 - d/max_d
            weights[mask] = 1 - distances[mask] / max_dist
            weights = np.maximum(weights, 0)

        elif self.weight_kernel == "inverse":
            # Inverse distance: w = 1 / (d + epsilon)
            epsilon = 1e-6
            weights[mask] = 1.0 / (distances[mask] + epsilon)

        elif self.weight_kernel == "gaussian":
            # Gaussian kernel: w = exp(-d^2 / (2 * bandwidth^2))
            bandwidth = max_dist / 3.0  # Heuristic bandwidth
            if bandwidth == 0:
                bandwidth = 1.0
            weights[mask] = np.exp(-distances[mask] ** 2 / (2 * bandwidth ** 2))

        else:
            raise ValueError(f"Unknown weight kernel: {self.weight_kernel}")

        return weights

    def _weighted_regression(self, x: np.ndarray, weights: np.ndarray) -> float:
        """Perform weighted linear regression for a single query point.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            Query point (in original scale; standardized internally).
        weights : np.ndarray of shape (n_samples,)
            Instance weights derived from the kernel function.

        Returns
        -------
        prediction : float
            Predicted target value. Falls back to weighted mean if
            the number of valid samples is insufficient or if the
            linear solve fails.
        """
        # Filter to instances with non-zero weights
        valid = weights > 1e-10
        if not np.any(valid):
            return np.mean(self.y_train_)

        X_valid = self.X_train_[valid]
        y_valid = self.y_train_[valid]
        w_valid = weights[valid]

        n_samples = len(y_valid)
        n_features = X_valid.shape[1]

        # If not enough samples for full regression, use weighted mean
        if n_samples < n_features + 1:
            return np.average(y_valid, weights=w_valid)

        # Weighted linear regression
        # Minimize: sum(w_i * (y_i - (X_i @ beta + b))^2)

        # Add intercept column
        X_bias = np.column_stack([np.ones(n_samples), X_valid])

        # Create diagonal weight matrix
        W = np.diag(w_valid)

        try:
            # Weighted least squares: (X'WX)^-1 X'Wy
            XtW = X_bias.T @ W
            XtWX = XtW @ X_bias
            XtWy = XtW @ y_valid

            # Add small ridge for stability
            ridge = 1e-6
            XtWX += ridge * np.eye(n_features + 1)

            beta = np.linalg.solve(XtWX, XtWy)

            # Standardize query point
            x_std = (x - self._X_mean) / self._X_std
            x_bias = np.concatenate([[1], x_std])

            prediction = x_bias @ beta
            return prediction

        except np.linalg.LinAlgError:
            # Fallback to weighted mean
            return np.average(y_valid, weights=w_valid)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LocallyWeightedLearningRegressor":
        """Fit the LocallyWeightedLearningRegressor regressor by storing the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LocallyWeightedLearningRegressor
            Returns the instance itself.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, self.n_features_ = X.shape

        # Standardize features
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        self._X_std[self._X_std == 0] = 1.0

        self.X_train_ = (X - self._X_mean) / self._X_std
        self.y_train_ = y

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for the provided samples.

        Uses C++ pairwise Euclidean distance for the distance matrix,
        then performs per-query weighted regression.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The samples to predict.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            The predicted values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        n_train = len(self.y_train_)
        k = self.k if self.k > 0 else n_train

        # Standardise query points and compute distances via C++
        X_std = (X - self._X_mean) / self._X_std
        X_std_c = np.ascontiguousarray(X_std, dtype=np.float64)
        Xt_c = np.ascontiguousarray(self.X_train_, dtype=np.float64)
        dist_matrix = _cpp_dist.euclidean(X_std_c, Xt_c)  # (n_query, n_train)

        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            distances = dist_matrix[i]
            weights = self._compute_weights(distances, k)
            predictions[i] = self._weighted_regression(X[i], weights)

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R-squared score."""
        self._check_is_fitted()
        y_pred = self.predict(X)
        y = np.asarray(y)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            n_train = len(self.y_train_)
            k_str = "all" if self.k < 0 else self.k
            return f"LocallyWeightedLearningRegressor(k={k_str}, kernel='{self.weight_kernel}', n_train={n_train})"
        return f"LocallyWeightedLearningRegressor(k={self.k}, kernel='{self.weight_kernel}')"
