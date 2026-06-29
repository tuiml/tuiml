"""Categorical Naive Bayes classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier


@classifier(tags=["bayes", "probabilistic", "categorical", "discrete"], version="1.0.0")
class CategoricalNBClassifier(Classifier):
    """Categorical Naive Bayes classifier for **discrete / nominal** features.

    Suitable for data whose features are **categorical** — each feature takes
    one of a finite set of integer-coded values. A separate **categorical
    distribution** is estimated per feature and class, making it the natural
    Naive Bayes variant for nominal data (unlike the Gaussian variant, which
    assumes continuous features).

    Overview
    --------
    The algorithm classifies samples through the following steps:

    1. For each class, estimate the **prior** :math:`P(c)` from label frequencies.
    2. For each (class, feature) pair, build a **category probability table**
       by counting how often each category value occurs, with additive
       (Laplace) smoothing.
    3. At prediction time, sum the **log probabilities** of the observed
       category in every feature plus the log prior.
    4. Return the class with the highest posterior.

    Theory
    ------
    Assuming conditional independence between features, the posterior for
    class :math:`c` given a sample :math:`\\mathbf{x} = (x_1, \\ldots, x_m)` is:

    .. math::

        P(c \\mid \\mathbf{x}) \\propto P(c) \\prod_{j=1}^{m} P(x_j \\mid c)

    With **Laplace smoothing** (parameter :math:`\\alpha`), each categorical
    likelihood is estimated as:

    .. math::

        \\hat{P}(x_j = v \\mid c) =
            \\frac{N_{cjv} + \\alpha}{N_c + \\alpha \\, K_j}

    where :math:`N_{cjv}` is the count of value :math:`v` for feature
    :math:`j` in class :math:`c`, :math:`N_c` is the number of samples in
    class :math:`c`, and :math:`K_j` is the number of categories of feature
    :math:`j`.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter. ``0`` disables
        smoothing.
    min_categories : int, list of int or None, default=None
        Minimum number of categories per feature. When ``None``, the number
        of categories of feature :math:`j` is inferred as ``max(X[:, j]) + 1``.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels encountered during :meth:`fit`.
    class_log_prior_ : np.ndarray of shape (n_classes,)
        Smoothed log prior probability of each class.
    category_log_prob_ : list of np.ndarray
        Per-feature arrays of shape ``(n_classes, n_categories_j)`` holding
        ``log P(x_j = v | c)``.
    n_categories_ : np.ndarray of shape (n_features,)
        Number of categories assumed for each feature.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m)` where :math:`n` = samples, :math:`m` = features
    - Prediction: :math:`O(n \\cdot m \\cdot c)` where :math:`c` = classes

    **When to use CategoricalNBClassifier:**

    - Features are nominal / categorical (integer-coded)
    - Continuous features have been discretised into bins
    - A fast, interpretable probabilistic baseline is desired

    Category values must be non-negative integers ``0 .. K_j - 1``. Values
    outside the range seen during fitting fall back to the smoothed
    zero-count probability.

    References
    ----------
    .. [Manning2008] Manning, C.D., Raghavan, P. and Schutze, H. (2008).
           **Introduction to Information Retrieval.**
           *Cambridge University Press*, Chapter 13.

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.NaiveBayesClassifier` : Gaussian/KDE Naive Bayes for continuous features.
    :class:`~tuiml.algorithms.bayesian.NaiveBayesMultinomialClassifier` : Multinomial Naive Bayes for count features.

    Examples
    --------
    Classification with integer-coded categorical features:

    >>> from tuiml.algorithms.bayesian import CategoricalNBClassifier
    >>> import numpy as np
    >>>
    >>> # Two categorical features (values 0..2 and 0..1)
    >>> X = np.array([[0, 1], [1, 0], [2, 1], [1, 1], [0, 0]])
    >>> y = np.array([0, 1, 1, 1, 0])
    >>>
    >>> clf = CategoricalNBClassifier(alpha=1.0)
    >>> clf.fit(X, y)
    CategoricalNBClassifier(alpha=1.0, classes=[0, 1])
    >>> clf.predict([[0, 1]])
    array([0])
    """

    def __init__(self, alpha: float = 1.0, min_categories: Any = None):
        """Initialize the Categorical Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, default=1.0
            Additive (Laplace) smoothing parameter.
        min_categories : int, list of int or None, default=None
            Minimum number of categories per feature.
        """
        super().__init__()
        self.alpha = alpha
        self.min_categories = min_categories
        self.classes_ = None
        self.class_log_prior_ = None
        self.category_log_prob_ = None
        self.n_categories_ = None
        self._n_features = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "alpha": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "description": "Additive (Laplace/Lidstone) smoothing parameter"
            },
            "min_categories": {
                "type": ["integer", "array", "null"],
                "default": None,
                "description": "Minimum number of categories per feature"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return [
            "nominal",
            "binary_class",
            "multiclass"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m) training, O(n * m * c) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Manning, C.D., Raghavan, P. & Schutze, H. (2008). Introduction "
            "to Information Retrieval. Cambridge University Press, Chapter 13."
        ]

    def _resolve_n_categories(self, X: np.ndarray) -> np.ndarray:
        """Determine the number of categories per feature.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Integer-coded categorical features.

        Returns
        -------
        n_categories : np.ndarray of shape (n_features,)
            Number of categories assumed for each feature.
        """
        observed = X.max(axis=0).astype(int) + 1
        observed = np.maximum(observed, 1)
        if self.min_categories is None:
            return observed
        if np.isscalar(self.min_categories):
            floor = np.full(self._n_features, int(self.min_categories))
        else:
            floor = np.asarray(self.min_categories, dtype=int)
        return np.maximum(observed, floor)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CategoricalNBClassifier":
        """Fit the Categorical Naive Bayes classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Integer-coded categorical features (non-negative).
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : CategoricalNBClassifier
            The fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X = X.astype(int)
        if np.any(X < 0):
            raise ValueError("CategoricalNB requires non-negative integer-coded features")

        n_samples, self._n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.n_categories_ = self._resolve_n_categories(X)

        # Smoothed log prior
        class_count = np.array([np.sum(y == c) for c in self.classes_], dtype=np.float64)
        smoothed_cc = class_count + self.alpha
        self.class_log_prior_ = np.log(smoothed_cc / smoothed_cc.sum())

        # Per-feature category probability tables.
        self.category_log_prob_ = []
        for j in range(self._n_features):
            k = int(self.n_categories_[j])
            counts = np.zeros((n_classes, k), dtype=np.float64)
            for ci, c in enumerate(self.classes_):
                col = X[y == c, j]
                # Clip stray values into range so bincount stays bounded.
                col = np.clip(col, 0, k - 1)
                counts[ci, :] = np.bincount(col, minlength=k)
            smoothed = counts + self.alpha
            log_prob = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
            self.category_log_prob_.append(log_prob)

        self._is_fitted = True
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute the joint log-likelihood per sample and class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Integer-coded categorical features.

        Returns
        -------
        jll : np.ndarray of shape (n_samples, n_classes)
            Joint log-likelihood ``log P(class, features)``.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        jll = np.tile(self.class_log_prior_, (n_samples, 1))
        for j in range(self._n_features):
            k = int(self.n_categories_[j])
            col = np.clip(X[:, j].astype(int), 0, k - 1)
            # category_log_prob_[j] is (n_classes, k); gather per sample -> (n_samples, n_classes)
            jll += self.category_log_prob_[j][:, col].T
        return jll

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Integer-coded categorical features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Integer-coded categorical features.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Normalized class probabilities; columns follow ``classes_``.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        jll = self._joint_log_likelihood(X)
        log_prob_max = np.max(jll, axis=1, keepdims=True)
        proba = np.exp(jll - log_prob_max)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"CategoricalNBClassifier(alpha={self.alpha}, classes={list(self.classes_)})"
        return f"CategoricalNBClassifier(alpha={self.alpha})"
