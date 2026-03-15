"""RegressionByDiscretization implementation."""

import numpy as np
from typing import Dict, List, Any, Optional
from copy import deepcopy

from tuiml.base.algorithms import Regressor, Classifier, regressor

@regressor(tags=["ensemble", "regression", "meta"], version="1.0.0")
class RegressionByDiscretization(Regressor):
    """Regression by Discretization.

    Converts a regression problem to classification by discretizing the target 
    variable into bins, training a classifier, and then predicting continuous 
    values using the predicted class probabilities and bin centers.

    Parameters
    ----------
    base_classifier : Classifier, optional
        The classifier to use. If None, a simple Naive Bayes-like classifier 
        is used.
    n_bins : int, default=10
        The number of bins for discretization.
    use_equal_frequency : bool, default=True
        Whether to use equal-frequency (vs equal-width) binning.

    Attributes
    ----------
    classifier_ : Classifier
        The fitted base classifier.
    bin_edges_ : np.ndarray
        The edges of the bins used for discretization.
    bin_centers_ : np.ndarray
        The center value of each bin.

    Examples
    --------
    >>> from tuiml.algorithms.ensemble import RegressionByDiscretization
    >>> reg = RegressionByDiscretization(n_bins=10)
    >>> reg.fit(X_train, y_train)
    RegressionByDiscretization(...)
    >>> predictions = reg.predict(X_test)

    References
    ----------
    .. [1] Frank, E., & Witten, I. H. (1999). Making better use of global 
           discretization. Proceedings of the 16th International Conference 
           on Machine Learning (ICML), 115-123.
    """

    def __init__(
        self,
        base_classifier: Optional[Classifier] = None,
        n_bins: int = 10,
        use_equal_frequency: bool = True,
    ):
        """
        Initialize RegressionByDiscretization.

        Args:
            base_classifier: Classifier to use
            n_bins: Number of bins
            use_equal_frequency: Use equal-frequency binning
        """
        super().__init__()
        self.base_classifier = base_classifier
        self.n_bins = n_bins
        self.use_equal_frequency = use_equal_frequency

        # Fitted attributes
        self.classifier_ = None
        self.bin_edges_ = None
        self.bin_centers_ = None
        self.n_features_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_bins": {
                "type": "integer",
                "default": 10,
                "minimum": 2,
                "description": "Number of bins for discretization"
            },
            "use_equal_frequency": {
                "type": "boolean",
                "default": True,
                "description": "Use equal-frequency (vs equal-width) binning"
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
        return "O(base_classifier_complexity)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Frank, E., & Witten, I. H. (1999). Making better use of global "
            "discretization. ICML, 115-123."
        ]

    def _create_classifier(self) -> Classifier:
        """Create a new instance of the base classifier."""
        if self.base_classifier is not None:
            return deepcopy(self.base_classifier)

        # Default: Simple Naive Bayes-like classifier
        return _SimpleNBClassifier()

    def _discretize(self, y: np.ndarray) -> tuple:
        """
        Discretize continuous target values into bins.

        Returns:
            Tuple of (bin_indices, bin_edges, bin_centers)
        """
        n_samples = len(y)

        if self.use_equal_frequency:
            # Equal-frequency binning: each bin has ~same number of samples
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = np.percentile(y, percentiles)
            # Handle duplicate edges
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.array([y.min() - 0.5, y.max() + 0.5])
        else:
            # Equal-width binning
            y_min, y_max = y.min(), y.max()
            if y_min == y_max:
                bin_edges = np.array([y_min - 0.5, y_max + 0.5])
            else:
                bin_edges = np.linspace(y_min, y_max, self.n_bins + 1)

        # Compute bin centers (used for prediction)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Assign each sample to a bin
        bin_indices = np.digitize(y, bin_edges[1:-1])

        return bin_indices, bin_edges, bin_centers

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionByDiscretization":
        """Fit the RegressionByDiscretization model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RegressionByDiscretization
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, self.n_features_ = X.shape

        # Discretize target
        y_discrete, self.bin_edges_, self.bin_centers_ = self._discretize(y)

        # Train classifier
        self.classifier_ = self._create_classifier()
        self.classifier_.fit(X, y_discrete)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]

        # Get class probabilities from classifier
        try:
            probabilities = self.classifier_.predict_proba(X)
        except (AttributeError, NotImplementedError):
            # Classifier doesn't support probabilities, use hard prediction
            predictions = self.classifier_.predict(X)
            return self.bin_centers_[predictions.astype(int)]

        # Compute expected value using bin centers
        # E[y] = sum(P(bin_i) * center_i)
        n_bins = len(self.bin_centers_)

        # Handle case where probabilities have fewer classes than bins
        if probabilities.shape[1] < n_bins:
            # Expand probabilities with zeros
            expanded = np.zeros((n_samples, n_bins))
            expanded[:, :probabilities.shape[1]] = probabilities
            probabilities = expanded

        # Handle case where probabilities have more classes
        probabilities = probabilities[:, :n_bins]

        # Compute expected value
        predictions = probabilities @ self.bin_centers_

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R-squared score.

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
        y = np.asarray(y)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            actual_bins = len(self.bin_centers_)
            return f"RegressionByDiscretization(n_bins={actual_bins})"
        return f"RegressionByDiscretization(n_bins={self.n_bins})"

class _SimpleNBClassifier:
    """
    Simple Naive Bayes-like classifier for default use.

    Assumes Gaussian distribution for continuous features.
    """

    def __init__(self):
        self._is_fitted = False
        self.classes_ = None
        self.class_priors_ = None
        self.means_ = None
        self.vars_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SimpleNBClassifier":
        """Fit the classifier."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_priors_[i] = len(X_c) / len(y)
            self.means_[i] = np.mean(X_c, axis=0)
            self.vars_[i] = np.var(X_c, axis=0) + 1e-9  # Avoid zero variance

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Compute log-probabilities
        log_probs = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # Log prior
            log_prior = np.log(self.class_priors_[i])

            # Log-likelihood (Gaussian)
            diff = X - self.means_[i]
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.vars_[i]) + diff ** 2 / self.vars_[i],
                axis=1
            )

            log_probs[:, i] = log_prior + log_likelihood

        # Convert to probabilities (softmax)
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
