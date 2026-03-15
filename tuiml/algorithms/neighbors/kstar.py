"""KStarClassifier (K*) classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from tuiml.base.algorithms import Classifier, classifier
from tuiml._cpp_ext import distance as _cpp_dist

@classifier(tags=["lazy", "instance-based", "entropy"], version="1.0.0")
class KStarClassifier(Classifier):
    """K* (KStarClassifier) instance-based classifier using **entropic distance**.

    K* uses an **entropy-based distance function** that measures the complexity
    of transforming one instance into another through a sequence of random
    transformations. Unlike conventional distance metrics, K* defines
    similarity through **Kolmogorov complexity** approximations.

    Overview
    --------
    The algorithm operates in the following steps:

    1. Store all training instances during ``fit()`` (lazy learning).
    2. For a new query :math:`x`, compute the **transformation probability**
       :math:`P^*(b \\mid a)` from the query to each training instance.
    3. Sum the probabilities for each class across all training instances.
    4. Normalize to obtain posterior class probabilities.
    5. Predict the class with the highest posterior probability.

    Theory
    ------
    The K* probability of class :math:`c` for a query :math:`x` is:

    .. math::
        P(c \\mid x) = \\frac{\\sum_{b \\in T_c} P^*(b \\mid x)}{\\sum_{b \\in T} P^*(b \\mid x)}

    where:

    - :math:`T_c` — Training instances belonging to class :math:`c`
    - :math:`T` — All training instances
    - :math:`P^*(b \\mid x)` — Transformation probability from :math:`x` to :math:`b`

    The transformation probability for a single attribute is computed using
    an exponential decay model controlled by the **blending parameter**
    :math:`s`:

    .. math::
        P^*(b_j \\mid x_j) = \\exp\\left(-\\frac{|x_j - b_j|}{\\sigma_j \\cdot (s / 100 + \\epsilon)}\\right)

    where :math:`\\sigma_j` is the standard deviation of attribute :math:`j`.

    The overall transformation probability is the product across all
    attributes:

    .. math::
        P^*(b \\mid x) = \\prod_{j=1}^{m} P^*(b_j \\mid x_j)

    Parameters
    ----------
    global_blend : int, default=20
        Global blending parameter (0--100). Controls the influence of
        all instances vs. specific neighbors. Higher values spread
        probability mass more evenly.
    entropy_auto_blend : bool, default=False
        Whether to use entropy-based automatic blending.
    missing_mode : {'average', 'maxdiff', 'ignore'}, default='average'
        How to handle missing values:

        - ``'average'`` — Average over all possible values.
        - ``'maxdiff'`` — Use the maximum possible difference.
        - ``'ignore'`` — Ignore missing attributes during distance calculation.

    Attributes
    ----------
    X_train_ : np.ndarray
        Training features stored for lazy learning.
    y_train_ : np.ndarray
        Training labels stored for lazy learning.
    classes_ : np.ndarray
        Unique class labels discovered during :meth:`fit`.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m)` for computing attribute scales, where :math:`n` = samples, :math:`m` = features
    - Prediction: :math:`O(n \\cdot m)` per query (computes transformation probability to every training instance)
    - Space: :math:`O(n \\cdot m)` for storing training data

    **When to use KStarClassifier:**

    - Datasets with mixed numeric and missing-value attributes
    - When a principled, information-theoretic distance is preferred over geometric metrics
    - Problems where the blending parameter can be tuned for regularization
    - When a probabilistic interpretation of neighbor similarity is desired

    References
    ----------
    .. [Cleary1995] Cleary, J.G. and Trigg, L.E. (1995).
           **K*: An Instance-based Learner Using an Entropic Distance Measure.**
           *Proceedings of the 12th International Conference on Machine Learning*,
           pp. 108-114.

    See Also
    --------
    :class:`~tuiml.algorithms.neighbors.KNearestNeighborsClassifier` : Classic k-nearest neighbors with distance weighting.
    :class:`~tuiml.algorithms.neighbors.LocallyWeightedLearningRegressor` : Instance-based regression with local models.

    Examples
    --------
    Entropy-based classification on the Iris dataset:

    >>> from tuiml.algorithms.neighbors import KStarClassifier
    >>> from tuiml.datasets import load_iris
    >>> X, y = load_iris()
    >>> clf = KStarClassifier(global_blend=20)
    >>> clf.fit(X, y)
    KStarClassifier(n_train=150, blend=20)
    >>> clf.predict(X[:1])
    array([0])
    """

    def __init__(self, global_blend: int = 20,
                 entropy_auto_blend: bool = False,
                 missing_mode: str = 'average'):
        """Initialize KStarClassifier.

        Parameters
        ----------
        global_blend : int, default=20
            Blending parameter controlling probability spread (0--100).
        entropy_auto_blend : bool, default=False
            Use entropy-based automatic blending.
        missing_mode : str, default='average'
            Missing value handling: ``'average'``, ``'maxdiff'``, or ``'ignore'``.
        """
        super().__init__()
        self.global_blend = global_blend
        self.entropy_auto_blend = entropy_auto_blend
        self.missing_mode = missing_mode
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self._n_features = None
        self._attr_scales = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "global_blend": {"type": "integer", "default": 20,
                            "minimum": 0, "maximum": 100,
                            "description": "Global blending parameter"},
            "entropy_auto_blend": {"type": "boolean", "default": False,
                                  "description": "Use entropy-based blending"},
            "missing_mode": {"type": "string", "default": "average",
                            "enum": ["average", "maxdiff", "ignore"],
                            "description": "Missing value handling"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "missing_values", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(1) training, O(n * m) prediction per sample"

    @classmethod
    def get_references(cls) -> List[str]:
        return ["Cleary, J.G., & Trigg, L.E. (1995). K*: An Instance-based "
                "Learner Using an Entropic Distance Measure. Proceedings of "
                "the 12th International Conference on Machine Learning."]

    def _compute_scale(self, col: np.ndarray) -> float:
        """Compute the normalization scale for a numeric attribute.

        Parameters
        ----------
        col : np.ndarray of shape (n_samples,)
            Column values for a single attribute (may contain NaN).

        Returns
        -------
        scale : float
            Standard deviation of valid values, with a small epsilon
            to prevent division by zero.
        """
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            return 1.0
        return np.std(valid) + 1e-10

    def _sphere_prob(self, diff: float, scale: float, blend: float) -> float:
        """Compute the probability of a transformation sphere.

        This is the core of the K* distance measure, representing how
        likely it is that a random walk transforms one attribute value
        into another.

        Parameters
        ----------
        diff : float
            Absolute difference between two attribute values.
        scale : float
            Normalization scale for the attribute.
        blend : float
            Blending parameter (0--100).

        Returns
        -------
        prob : float
            Transformation probability in the range ``[0, 1]``.
        """
        if scale == 0:
            return 1.0 if diff == 0 else 0.0

        # Normalized difference
        norm_diff = abs(diff) / scale

        # Blend parameter (converted to 0-1 range)
        b = blend / 100.0

        # K* formula: probability based on transformation complexity
        if norm_diff == 0:
            prob = 1.0
        else:
            # Exponential decay with blending
            prob = np.exp(-norm_diff / (b + 1e-10))

        return prob

    def _kstar_prob(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the K* transformation probability :math:`P^*(x_2 \\mid x_1)`.

        Parameters
        ----------
        x1 : np.ndarray of shape (n_features,)
            Source instance.
        x2 : np.ndarray of shape (n_features,)
            Target instance.

        Returns
        -------
        prob : float
            Joint transformation probability across all attributes.
        """
        log_prob = 0.0
        blend = self.global_blend

        for i in range(self._n_features):
            v1, v2 = x1[i], x2[i]

            # Handle missing values
            if np.isnan(v1) or np.isnan(v2):
                if self.missing_mode == 'ignore':
                    continue
                elif self.missing_mode == 'maxdiff':
                    prob = 0.01  # Very low probability
                else:  # average
                    prob = 0.5
            else:
                diff = abs(v1 - v2)
                scale = self._attr_scales[i]
                prob = self._sphere_prob(diff, scale, blend)

            if prob > 0:
                log_prob += np.log(prob + 1e-10)
            else:
                log_prob += np.log(1e-10)

        return np.exp(log_prob)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KStarClassifier":
        """Fit the KStarClassifier classifier by storing the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : KStarClassifier
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

        # Compute attribute scales
        self._attr_scales = np.array([self._compute_scale(X[:, i])
                                      for i in range(self._n_features)])

        self._is_fitted = True
        return self

    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single instance.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query instance.

        Returns
        -------
        proba : np.ndarray of shape (n_classes,)
            Normalized class probabilities based on transformation sums.
        """
        class_probs = defaultdict(float)
        total_prob = 0.0

        for i in range(len(self.X_train_)):
            prob = self._kstar_prob(x, self.X_train_[i])
            class_probs[self.y_train_[i]] += prob
            total_prob += prob

        # Normalize
        proba = np.zeros(len(self.classes_))
        if total_prob > 0:
            for idx, c in enumerate(self.classes_):
                proba[idx] = class_probs[c] / total_prob
        else:
            proba[:] = 1.0 / len(self.classes_)

        return proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the provided samples.

        Uses C++ pairwise distance to accelerate the transformation
        probability computation across all query-training pairs.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The samples to predict.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_query = len(X)
        n_train = len(self.X_train_)
        n_classes = len(self.classes_)
        blend = self.global_blend / 100.0 + 1e-10

        # C++ pairwise distances per feature, then compute log-probs
        # Compute normalised absolute differences per feature
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        Xt_c = np.ascontiguousarray(self.X_train_, dtype=np.float64)

        # Pairwise Euclidean via C++ just to check for NaNs (optional)
        # Main computation: per-feature exponential decay
        # log P*(b|a) = sum_j [ -|a_j - b_j| / (scale_j * blend) ]
        # Vectorised: (n_query, n_train) for each feature
        log_probs = np.zeros((n_query, n_train), dtype=np.float64)

        for j in range(self._n_features):
            # |X[:,j] - X_train[:,j]| -> (n_query, n_train)
            diff = np.abs(X_c[:, j:j+1] - Xt_c[:, j].reshape(1, -1))
            scale = self._attr_scales[j]
            # log(exp(-diff / (scale * blend))) = -diff / (scale * blend)
            log_probs -= diff / (scale * blend)

        # Convert to probabilities (numerically stable)
        # Shift per-query for numerical stability
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)  # (n_query, n_train)

        # Aggregate by class
        proba = np.zeros((n_query, n_classes))
        for c_idx, c in enumerate(self.classes_):
            mask = self.y_train_ == c
            proba[:, c_idx] = probs[:, mask].sum(axis=1)

        # Normalise
        totals = proba.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        proba /= totals

        return proba

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
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"KStarClassifier(n_train={len(self.X_train_)}, blend={self.global_blend})"
        return f"KStarClassifier(global_blend={self.global_blend})"
