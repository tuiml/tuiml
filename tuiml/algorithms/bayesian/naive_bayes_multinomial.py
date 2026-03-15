"""Multinomial Naive Bayes classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["bayes", "probabilistic", "text", "multinomial"], version="1.0.0")
class NaiveBayesMultinomialClassifier(Classifier):
    """Multinomial Naive Bayes classifier for **text** and **discrete** data.

    The multinomial Naive Bayes classifier is suitable for classification
    with **discrete features** (e.g., word counts for text classification).
    It models the probability of a document belonging to a class as the
    product of the probabilities of each word in the document given that class.

    Overview
    --------
    The algorithm classifies documents through the following steps:

    1. Count the **frequency** of each feature (word) for every class
    2. Apply **additive smoothing** to avoid zero probabilities
    3. Compute **log class priors** and **log feature likelihoods**
    4. At prediction time, compute the joint log-likelihood and return
       the class with the highest score

    Theory
    ------
    The multinomial model assumes features are generated from a
    **multinomial distribution**. The posterior for class :math:`c` given
    document :math:`\\mathbf{x} = (x_1, \\ldots, x_m)` is:

    .. math::

        P(c \\mid \\mathbf{x}) \\propto P(c) \\prod_{i=1}^{m} P(w_i \\mid c)^{x_i}

    where :math:`x_i` is the count of feature :math:`i` in the document.

    With **Laplace smoothing** (parameter :math:`\\alpha`), the feature
    likelihood is estimated as:

    .. math::

        \\hat{P}(w_i \\mid c) = \\frac{N_{ic} + \\alpha}{N_c + \\alpha \\, m}

    where :math:`N_{ic}` is the total count of feature :math:`i` in class
    :math:`c`, :math:`N_c` is the total count of all features in class
    :math:`c`, and :math:`m` is the vocabulary size.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter.
        Set to 0 for no smoothing. Smoothing prevents zero probabilities
        for features not seen in the training data.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels encountered during ``fit``.
    class_prior_ : np.ndarray of shape (n_classes,)
        Log prior probability of each class: ``log P(class)``.
    feature_log_prob_ : np.ndarray of shape (n_classes, n_features)
        Log probability of features given class: ``log P(feature | class)``.
    class_count_ : np.ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting.
    feature_count_ : np.ndarray of shape (n_classes, n_features)
        Aggregate feature counts encountered for each class.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m)` where :math:`n` = samples, :math:`m` = features
    - Prediction: :math:`O(m \\cdot c)` per sample where :math:`c` = classes

    **When to use NaiveBayesMultinomialClassifier:**

    - Text classification (e.g., spam filtering, sentiment analysis)
    - Features are word counts, term frequencies, or TF-IDF values
    - High-dimensional sparse data (large vocabularies)
    - Online / incremental learning via ``partial_fit``

    While theoretically designed for integer counts, this classifier often
    works well with fractional counts such as TF-IDF weighted values.

    References
    ----------
    .. [McCallum1998] McCallum, A. and Nigam, K. (1998).
           **A Comparison of Event Models for Naive Bayes Text Classification.**
           *AAAI-98 Workshop on Learning for Text Categorization*, pp. 41-48.

    .. [Manning2008] Manning, C.D., Raghavan, P. and Schutze, H. (2008).
           **Introduction to Information Retrieval.**
           *Cambridge University Press*, Chapter 13.

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.NaiveBayesClassifier` : Gaussian/KDE-based Naive Bayes for continuous features.
    :class:`~tuiml.algorithms.bayesian.BayesianNetworkClassifier` : Bayesian network with learned structure.

    Examples
    --------
    Basic text-like classification with word counts:

    >>> from tuiml.algorithms.bayesian import NaiveBayesMultinomialClassifier
    >>> import numpy as np
    >>>
    >>> # Word count features for 4 documents
    >>> X = np.array([[2, 1, 0], [1, 2, 0], [0, 1, 2], [0, 2, 1]])
    >>> y = np.array([0, 0, 1, 1])
    >>>
    >>> # Fit with Laplace smoothing
    >>> clf = NaiveBayesMultinomialClassifier(alpha=1.0)
    >>> clf.fit(X, y)
    NaiveBayesMultinomialClassifier(alpha=1.0, classes=[0, 1])
    >>> clf.predict([[3, 0, 0]])
    array([0])
    """

    def __init__(self, alpha: float = 1.0):
        """Initialize Multinomial Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, default=1.0
            Additive smoothing parameter (Laplace smoothing).
        """
        super().__init__()
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None
        self.class_count_ = None
        self.feature_count_ = None
        self._n_features = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "alpha": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "description": "Additive (Laplace/Lidstone) smoothing parameter"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return [
            "numeric",
            "binary_class",
            "multiclass"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m) training, O(m * c) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "McCallum, A., & Nigam, K. (1998). A comparison of event models "
            "for Naive Bayes text classification. AAAI-98 Workshop on "
            "Learning for Text Categorization."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesMultinomialClassifier":
        """Fit the Multinomial Naive Bayes classifier to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Features should be 
            non-negative (frequencies, counts, or TF-IDF).
        y : array-like of shape (n_samples,)
            Target values (class labels).

        Returns
        -------
        self : NaiveBayesMultinomialClassifier
            Returns the fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Check for negative values
        if np.any(X < 0):
            raise ValueError("Multinomial Naive Bayes requires non-negative features")

        n_samples, self._n_features = X.shape

        # Find classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Initialize counts
        self.class_count_ = np.zeros(n_classes)
        self.feature_count_ = np.zeros((n_classes, self._n_features))

        # Compute feature counts for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_[idx] = X_c.shape[0]
            self.feature_count_[idx, :] = np.sum(X_c, axis=0)

        # Compute class prior (log probability)
        self.class_prior_ = np.log(self.class_count_ / n_samples)

        # Compute feature log probabilities with smoothing
        # P(feature|class) = (count(feature, class) + alpha) /
        #                    (total_count(class) + alpha * n_features)
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)

        self._is_fitted = True
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Calculate the joint log-likelihood :math:`\\log P(\\text{class}, \\text{features})`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix (counts or frequencies).

        Returns
        -------
        jll : np.ndarray of shape (n_samples, n_classes)
            Joint log-likelihood for each sample and class.
        """
        # P(class, features) = P(class) * P(features|class)
        # log P(class, features) = log P(class) + sum(count_i * log P(feature_i|class))
        return X @ self.feature_log_prob_.T + self.class_prior_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : np.ndarray of shape (n_samples,)
            Predicted target values for X.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : np.ndarray of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in 
            ``self.classes_``.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        jll = self._joint_log_likelihood(X)

        # Log-sum-exp trick for numerical stability
        log_prob_max = np.max(jll, axis=1, keepdims=True)
        jll_shifted = jll - log_prob_max
        proba = np.exp(jll_shifted)
        proba /= np.sum(proba, axis=1, keepdims=True)

        return proba

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict log class probabilities for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        log_proba : np.ndarray of shape (n_samples, n_classes)
            Log class probabilities. Columns correspond to
            classes in ``self.classes_``.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        jll = self._joint_log_likelihood(X)

        # Normalize to get log probabilities
        log_prob_max = np.max(jll, axis=1, keepdims=True)
        jll_shifted = jll - log_prob_max
        log_sum = np.log(np.sum(np.exp(jll_shifted), axis=1, keepdims=True))
        return jll - log_prob_max - log_sum

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    classes: Optional[np.ndarray] = None) -> "NaiveBayesMultinomialClassifier":
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        y : array-like of shape (n_samples,)
            Target values (class labels).
        classes : np.ndarray or None, default=None
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        Returns
        -------
        self : NaiveBayesMultinomialClassifier
            Returns the fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not self._is_fitted:
            # First call - initialize
            if classes is None:
                self.classes_ = np.unique(y)
            else:
                self.classes_ = np.asarray(classes)

            n_classes = len(self.classes_)
            self._n_features = X.shape[1]
            self.class_count_ = np.zeros(n_classes)
            self.feature_count_ = np.zeros((n_classes, self._n_features))

        # Update counts
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_[idx] += X_c.shape[0]
            self.feature_count_[idx, :] += np.sum(X_c, axis=0)

        # Recompute probabilities
        total_samples = np.sum(self.class_count_)
        self.class_prior_ = np.log(self.class_count_ / total_samples)

        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)

        self._is_fitted = True
        return self

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"NaiveBayesMultinomialClassifier(alpha={self.alpha}, classes={list(self.classes_)})"
        return f"NaiveBayesMultinomialClassifier(alpha={self.alpha})"
