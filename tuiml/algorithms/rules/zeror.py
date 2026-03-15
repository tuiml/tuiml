"""ZeroRuleClassifier classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter

from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["rules", "baseline", "simple"], version="1.0.0")
class ZeroRuleClassifier(Classifier):
    """Zero Rule classifier that predicts the **most frequent class**,
    ignoring all **input features**.

    ZeroRuleClassifier is the simplest classification method which relies
    only on the target distribution and ignores all predictors. It simply
    predicts the majority class (mode) for classification tasks. This
    classifier is useful as a **baseline** -- any classifier that cannot
    beat ZeroRuleClassifier should be discarded.

    Overview
    --------
    The algorithm operates as follows:

    1. During training, compute the frequency of each class label
    2. Store the majority class (the class with the highest count)
    3. At prediction time, return the majority class for every instance

    Theory
    ------
    The prediction for all instances is the mode of the training labels:

    .. math::
        \\hat{y} = \\arg\\max_{c \\in C} \\sum_{i=1}^{n} \\mathbb{1}[y_i = c]

    The expected error rate of ZeroRuleClassifier equals the proportion of
    non-majority instances:

    .. math::
        \\text{Error} = 1 - \\frac{\\max_{c} n_c}{n}

    where :math:`n_c` is the count of class :math:`c` and :math:`n` is
    the total number of training instances.

    Attributes
    ----------
    majority_class_ : any
        The most frequent class in training data.
    class_counts_ : dict
        Count of each class in training data.
    classes_ : np.ndarray
        Unique classes in training data.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n)` where :math:`n` = number of samples
    - Prediction: :math:`O(1)` per sample

    **When to use ZeroRuleClassifier:**

    - As the absolute baseline for any classification task
    - To establish the minimum acceptable accuracy threshold
    - When comparing classifier performance (any useful model must beat ZeroR)
    - For sanity-checking evaluation pipelines

    References
    ----------
    .. [Witten2011] Witten, I.H., Frank, E. and Hall, M.A. (2011).
           **Data Mining: Practical Machine Learning Tools and Techniques.**
           *Morgan Kaufmann*, 3rd Edition.
           DOI: `10.1016/C2009-0-19715-5 <https://doi.org/10.1016/C2009-0-19715-5>`_

    See Also
    --------
    :class:`~tuiml.algorithms.rules.OneRuleClassifier` : Single-attribute rule learner (next simplest).
    :class:`~tuiml.algorithms.rules.DecisionTableClassifier` : Multi-attribute lookup table classifier.

    Examples
    --------
    Basic usage as a baseline classifier:

    >>> from tuiml.algorithms.rules import ZeroRuleClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    >>> y_train = np.array([0, 0, 1, 1, 1])
    >>>
    >>> # Fit the model
    >>> clf = ZeroRuleClassifier()
    >>> clf.fit(X_train, y_train)
    ZeroRuleClassifier(...)
    >>> # Predicts majority class (1) for all instances
    >>> predictions = clf.predict(X_train)
    >>> print(predictions)
    [1 1 1 1 1]
    """

    def __init__(self):
        """Initialize ZeroRuleClassifier classifier."""
        super().__init__()
        self.majority_class_ = None
        self.class_counts_ = None
        self.classes_ = None
        self._n_classes = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema (ZeroRuleClassifier has no parameters)."""
        return {}

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
        return "O(n) training, O(1) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Witten, I.H., Frank, E., & Hall, M.A. (2011). "
            "Data Mining: Practical Machine Learning Tools and Techniques. "
            "Morgan Kaufmann, 3rd Edition."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ZeroRuleClassifier":
        """Fit the ZeroRuleClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features. Ignored by ZeroRuleClassifier.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : ZeroRuleClassifier
            Returns the fitted instance.
        """
        y = np.asarray(y)

        # Find unique classes and their counts
        self.classes_, counts = np.unique(y, return_counts=True)
        self._n_classes = len(self.classes_)
        self.class_counts_ = dict(zip(self.classes_, counts))

        # Find majority class
        self.majority_class_ = self.classes_[np.argmax(counts)]

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the majority class for all samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Array of predictions containing the majority class value.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        n_samples = X.shape[0] if X.ndim > 1 else len(X)
        return np.full(n_samples, self.majority_class_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities based on training distribution.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        n_samples = X.shape[0] if X.ndim > 1 else len(X)

        # Calculate probabilities from training class distribution
        total = sum(self.class_counts_.values())
        probs = np.array([self.class_counts_[c] / total for c in self.classes_])

        # Return same probabilities for all samples
        return np.tile(probs, (n_samples, 1))

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"ZeroRuleClassifier(majority_class={self.majority_class_})"
        return "ZeroRuleClassifier()"
