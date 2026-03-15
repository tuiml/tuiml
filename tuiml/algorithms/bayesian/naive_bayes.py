"""Naive Bayes classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier
from tuiml.algorithms.bayesian.estimators import (
    NormalEstimator,
    KernelEstimator,
    DiscreteEstimator
)

@classifier(tags=["bayes", "probabilistic", "fast"], version="1.1.0")
class NaiveBayesClassifier(Classifier):
    """Naive Bayes classifier using **pluggable probability estimators**.

    Naive Bayes classifiers are a family of **probabilistic classifiers**
    based on applying **Bayes' theorem** with strong independence assumptions
    between the features. This implementation supports both Gaussian and
    kernel density estimation for modeling numeric feature distributions.

    Overview
    --------
    The algorithm classifies instances through the following steps:

    1. Compute the **prior probability** of each class from training labels
    2. For each feature, fit a probability estimator (Gaussian or KDE) per class
    3. At prediction time, compute the **posterior** for each class as the
       product of the prior and the per-feature likelihoods
    4. Return the class with the highest posterior probability

    Theory
    ------
    Classification is based on **Bayes' theorem**. The posterior probability
    of class :math:`c` given a feature vector :math:`\\mathbf{x}` is:

    .. math::

        P(c \\mid \\mathbf{x}) = \\frac{P(c) \\, P(\\mathbf{x} \\mid c)}{P(\\mathbf{x})}

    Under the **naive independence assumption**, the likelihood factorises:

    .. math::

        P(\\mathbf{x} \\mid c) = \\prod_{i=1}^{m} P(x_i \\mid c)

    so the decision rule becomes:

    .. math::

        \\hat{c} = \\arg\\max_{c} \\; P(c) \\prod_{i=1}^{m} P(x_i \\mid c)

    Each factor :math:`P(x_i \\mid c)` is estimated by a pluggable density
    estimator -- either a **Gaussian** (normal) distribution or a **kernel
    density estimator** (KDE).

    Parameters
    ----------
    use_kernel_estimator : bool, default=False
        If ``True``, use kernel density estimation for numeric attributes
        instead of assuming a Gaussian distribution. Kernel estimation is
        more flexible but computationally more expensive.
    use_laplace : bool, default=True
        If ``True``, apply Laplace (add-one) smoothing to class priors.
        This prevents zero probabilities for classes with few samples.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels discovered during :meth:`fit`.
    class_prior_ : np.ndarray of shape (n_classes,)
        Prior probability of each class, i.e., ``P(class)``.
        Computed from training data, optionally with Laplace smoothing.
    estimators_ : list of list
        2D list of probability estimators indexed as
        ``estimators_[class_idx][feature_idx]``. Each estimator models
        the conditional distribution ``P(feature|class)``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m \\cdot c)` where :math:`n` = samples, :math:`m` = features, :math:`c` = classes
    - Prediction: :math:`O(m \\cdot c)` per sample (Gaussian); :math:`O(n \\cdot m \\cdot c)` per sample (KDE)

    **When to use NaiveBayesClassifier:**

    - When you need a fast, simple baseline classifier
    - Text classification and spam filtering
    - When features are approximately independent given the class
    - Datasets with missing values (NaN values are handled gracefully)
    - When interpretable class probabilities are desired

    The classifier uses log-probabilities internally for numerical stability.
    For numeric features, the default is to use a Gaussian distribution.
    Set ``use_kernel_estimator=True`` for non-Gaussian or multimodal data.

    References
    ----------
    .. [John1995] John, G.H. and Langley, P. (1995).
           **Estimating Continuous Distributions in Bayesian Classifiers.**
           *Proceedings of the 11th Conference on Uncertainty in Artificial Intelligence*,
           pp. 338-345.

    .. [Domingos1997] Domingos, P. and Pazzani, M. (1997).
           **On the Optimality of the Simple Bayesian Classifier under Zero-One Loss.**
           *Machine Learning*, 29(2-3), 103-130.
           DOI: `10.1023/A:1007413511361 <https://doi.org/10.1023/A:1007413511361>`_

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.NaiveBayesMultinomialClassifier` : Multinomial variant for discrete/count data.
    :class:`~tuiml.algorithms.bayesian.BayesianNetworkClassifier` : Bayesian network with learned structure.
    :class:`~tuiml.algorithms.bayesian.estimators.NormalEstimator` : Gaussian probability estimator used by default.
    :class:`~tuiml.algorithms.bayesian.estimators.KernelEstimator` : Kernel density estimator for flexible distributions.

    Examples
    --------
    Basic classification with default settings:

    >>> from tuiml.algorithms.bayesian import NaiveBayesClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample training data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
    >>> y_train = np.array([0, 0, 0, 1, 1, 1])
    >>>
    >>> # Fit and predict with Gaussian estimators
    >>> clf = NaiveBayesClassifier()
    >>> clf.fit(X_train, y_train)
    NaiveBayesClassifier(classes=[0, 1], estimator=Normal)
    >>> clf.predict([[2, 3]])
    array([0])
    >>> clf.predict_proba([[2, 3]])  # doctest: +SKIP
    array([[0.99, 0.01]])

    Using kernel density estimation for non-Gaussian data:

    >>> clf_kde = NaiveBayesClassifier(use_kernel_estimator=True)
    >>> clf_kde.fit(X_train, y_train)
    NaiveBayesClassifier(classes=[0, 1], estimator=Kernel)
    """

    def __init__(self, use_kernel_estimator: bool = False,
                 use_laplace: bool = True):
        """
        Initialize the Naive Bayes classifier.

        Parameters
        ----------
        use_kernel_estimator : bool, default=False
            Whether to use kernel density estimation for modeling
            numeric feature distributions. If ``False`` (default),
            a Gaussian distribution is assumed.
        use_laplace : bool, default=True
            Whether to apply Laplace smoothing to class prior
            probabilities to avoid zero probabilities.
        """
        super().__init__()
        self.use_kernel_estimator = use_kernel_estimator
        self.use_laplace = use_laplace
        self.classes_ = None
        self.class_prior_ = None
        self.estimators_: List[List] = []  # [class_idx][feature_idx]
        self._epsilon = 1e-9
        self._n_features = 0

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """
        Return the JSON schema for classifier parameters.

        This schema is used for LLM tool calling and parameter validation.
        It follows JSON Schema specification for describing the expected
        parameter types and defaults.

        Returns
        -------
        dict of str to dict
            A dictionary mapping parameter names to their schema definitions.
            Each schema includes ``type``, ``default``, and ``description``.

        Examples
        --------
        >>> schema = NaiveBayesClassifier.get_parameter_schema()
        >>> schema['use_kernel_estimator']['type']
        'boolean'
        """
        return {
            "use_kernel_estimator": {
                "type": "boolean",
                "default": False,
                "description": "Use kernel density estimation for numeric attributes"
            },
            "use_laplace": {
                "type": "boolean",
                "default": True,
                "description": "Use Laplace smoothing for class priors"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """
        Return the data capabilities supported by this classifier.

        Capabilities indicate what types of data and problem settings
        the classifier can handle.

        Returns
        -------
        list of str
            List of capability identifiers:

            - ``"numeric"``: Handles numeric/continuous features
            - ``"missing_values"``: Handles missing values (NaN)
            - ``"binary_class"``: Supports binary classification
            - ``"multiclass"``: Supports multi-class classification

        Examples
        --------
        >>> caps = NaiveBayesClassifier.get_capabilities()
        >>> 'multiclass' in caps
        True
        """
        return [
            "numeric",
            "missing_values",
            "binary_class",
            "multiclass"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """
        Return the computational complexity of the algorithm.

        Returns
        -------
        str
            A string describing time complexity for training and prediction,
            where:

            - ``n`` = number of training samples
            - ``m`` = number of features
            - ``c`` = number of classes

        Examples
        --------
        >>> NaiveBayesClassifier.get_complexity()
        'O(n * m * c) training, O(m * c) prediction'
        """
        return "O(n * m * c) training, O(m * c) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """
        Return academic references for the algorithm.

        Returns
        -------
        list of str
            List of citation strings in standard academic format.
        """
        return [
            "John, G.H., & Langley, P. (1995). Estimating Continuous "
            "Distributions in Bayesian Classifiers. Proceedings of the "
            "11th Conference on Uncertainty in Artificial Intelligence."
        ]

    def _create_estimator(self):
        """
        Create a probability estimator based on configuration.

        Returns
        -------
        NormalEstimator or KernelEstimator
            A new estimator instance. Returns :class:`KernelEstimator`
            if ``use_kernel_estimator=True``, otherwise :class:`NormalEstimator`.
        """
        if self.use_kernel_estimator:
            return KernelEstimator()
        else:
            return NormalEstimator()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesClassifier":
        """
        Fit the Naive Bayes classifier to the training data.

        This method computes the class priors and trains probability
        estimators for each feature-class combination.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix. Can be a NumPy array or any
            array-like that can be converted to a NumPy array.
            Missing values (NaN) are allowed and handled gracefully.
        y : array-like of shape (n_samples,)
            Target class labels for the training samples.

        Returns
        -------
        self : NaiveBayesClassifier
            Returns the fitted classifier instance for method chaining.

        Notes
        -----
        After fitting, the following attributes are available:

        - ``classes_``: The unique class labels
        - ``class_prior_``: The prior probability of each class
        - ``estimators_``: Trained probability estimators for each
          class-feature combination

        Examples
        --------
        >>> clf = NaiveBayesClassifier()
        >>> clf.fit(X_train, y_train)
        >>> clf.classes_
        array([0, 1])
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self._n_features = n_features

        # Find classes and compute priors
        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)

        # Apply Laplace smoothing to priors if requested
        if self.use_laplace:
            self.class_prior_ = (class_counts + 1) / (n_samples + n_classes)
        else:
            self.class_prior_ = class_counts / n_samples

        # Create estimators for each class and each feature
        self.estimators_ = []
        
        for class_idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            class_estimators = []
            
            for feature_idx in range(n_features):
                # Create estimator for this class-feature combination
                estimator = self._create_estimator()
                
                # Get feature values for this class
                feature_values = X_c[:, feature_idx]
                
                # Add non-NaN values to the estimator
                for val in feature_values:
                    if not np.isnan(val):
                        estimator.add_value(val)
                
                class_estimators.append(estimator)
            
            self.estimators_.append(class_estimators)

        self._is_fitted = True
        return self

    def _calculate_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate log-likelihood of features for each class.

        Computes the log of the conditional probability of features
        given each class using the trained estimators.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix for which to compute likelihoods.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Log-likelihood values. Entry ``[i, j]`` contains
            ``log P(X[i] | class_j)``.

        Notes
        -----
        Missing values (NaN) contribute 0 to the log-likelihood,
        effectively treating them as uninformative.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_likelihood = np.zeros((n_samples, n_classes))

        for class_idx in range(n_classes):
            class_log_prob = np.zeros(n_samples)
            
            for feature_idx in range(self._n_features):
                estimator = self.estimators_[class_idx][feature_idx]
                
                for sample_idx in range(n_samples):
                    val = X[sample_idx, feature_idx]
                    
                    if np.isnan(val):
                        # Missing value contributes 0 to log probability
                        continue
                    
                    prob = estimator.get_probability(val)
                    
                    # Avoid log(0)
                    if prob > 0:
                        class_log_prob[sample_idx] += np.log(prob)
                    else:
                        class_log_prob[sample_idx] += np.log(self._epsilon)
            
            log_likelihood[:, class_idx] = class_log_prob

        return log_likelihood

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Returns the class with the highest posterior probability
        for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix of samples to classify.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted class label for each sample.

        Raises
        ------
        RuntimeError
            If the classifier has not been fitted yet.

        Examples
        --------
        >>> clf = NaiveBayesClassifier()
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> predictions.shape
        (100,)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Calculate posterior = log prior + log likelihood
        log_likelihood = self._calculate_log_likelihood(X)
        log_prior = np.log(self.class_prior_)
        log_posterior = log_likelihood + log_prior

        # Return class with highest posterior
        return self.classes_[np.argmax(log_posterior, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class membership probabilities for input samples.

        Returns normalized posterior probabilities for each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix of samples to classify.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Probability of each class for each sample. Columns
            correspond to classes in ``self.classes_``. Each row
            sums to 1.0.

        Raises
        ------
        RuntimeError
            If the classifier has not been fitted yet.

        Notes
        -----
        Probabilities are computed using the log-sum-exp trick for
        numerical stability, avoiding underflow with small probabilities.

        Examples
        --------
        >>> clf = NaiveBayesClassifier()
        >>> clf.fit(X_train, y_train)
        >>> probas = clf.predict_proba(X_test)
        >>> probas[0]  # Probabilities for first sample
        array([0.95, 0.05])
        >>> probas[0].sum()  # Probabilities sum to 1
        1.0
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Calculate log posterior
        log_likelihood = self._calculate_log_likelihood(X)
        log_prior = np.log(self.class_prior_)
        log_posterior = log_likelihood + log_prior

        # Convert to probabilities using log-sum-exp trick for numerical stability
        log_posterior_max = np.max(log_posterior, axis=1, keepdims=True)
        log_posterior_shifted = log_posterior - log_posterior_max
        posterior = np.exp(log_posterior_shifted)
        posterior /= np.sum(posterior, axis=1, keepdims=True)

        return posterior

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict log-probabilities of class membership for input samples.

        This is the logarithm of the values returned by :meth:`predict_proba`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix of samples to classify.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Log-probability of each class for each sample. Columns
            correspond to classes in ``self.classes_``.

        Raises
        ------
        RuntimeError
            If the classifier has not been fitted yet.

        Notes
        -----
        A small epsilon is added before taking the log to avoid
        ``log(0)`` for probabilities very close to zero.

        Examples
        --------
        >>> clf = NaiveBayesClassifier()
        >>> clf.fit(X_train, y_train)
        >>> log_probas = clf.predict_log_proba(X_test)
        >>> log_probas[0]
        array([-0.05, -2.99])
        """
        return np.log(self.predict_proba(X) + self._epsilon)

    def __repr__(self) -> str:
        """
        Return a string representation of the classifier.

        Returns
        -------
        str
            A concise string showing the classifier state.
            If fitted, shows the discovered classes and estimator type.
            If not fitted, shows the configuration parameters.

        Examples
        --------
        >>> clf = NaiveBayesClassifier()
        >>> repr(clf)
        'NaiveBayesClassifier(use_kernel_estimator=False)'
        >>> clf.fit(X, y)
        >>> repr(clf)
        'NaiveBayesClassifier(classes=[0, 1], estimator=Normal)'
        """
        if self._is_fitted:
            estimator_type = "Kernel" if self.use_kernel_estimator else "Normal"
            return f"NaiveBayesClassifier(classes={list(self.classes_)}, estimator={estimator_type})"
        return f"NaiveBayesClassifier(use_kernel_estimator={self.use_kernel_estimator})"
