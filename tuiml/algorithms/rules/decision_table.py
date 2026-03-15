"""DecisionTableClassifier classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter
from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["rules", "table", "interpretable"], version="1.0.0")
class DecisionTableClassifier(Classifier):
    """Decision Table classifier using **feature subset selection** and a
    **lookup table** for majority-class prediction.

    DecisionTableClassifier builds a simple decision table majority classifier.
    It uses a best-first search to find a subset of features that minimizes
    error when using a **lookup table** approach for classification.

    Overview
    --------
    The algorithm constructs a decision table through the following steps:

    1. Discretize all continuous features into bins
    2. Perform feature subset selection using best-first or greedy search
    3. Evaluate each candidate subset via leave-one-out cross-validation
    4. Build a lookup table mapping discretized feature patterns to majority classes
    5. At prediction time, discretize the input and look up the majority class

    Theory
    ------
    The decision table selects a feature subset :math:`S` that maximizes
    classification accuracy under leave-one-out evaluation:

    .. math::
        S^* = \\arg\\max_{S \\subseteq F} \\text{Acc}_{\\text{LOO}}(S)

    For a given subset, the table maps each unique pattern :math:`p` of
    discretized feature values to the majority class:

    .. math::
        \\hat{y}(p) = \\arg\\max_{c \\in C} \\sum_{i=1}^{n} \\mathbb{1}[x_i^S = p \\wedge y_i = c]

    where :math:`x_i^S` is the projection of instance :math:`i` onto
    feature subset :math:`S`, and :math:`C` is the set of class labels.

    Parameters
    ----------
    search_method : {'best_first', 'greedy'}, default='best_first'
        The search algorithm used to find the best feature subset.
    cross_val_folds : int, default=1
        Number of folds for cross-validation during feature selection.
    use_ibk : bool, default=False
        Whether to use IBk (k-Nearest Neighbors) as the underlying classifier.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    selected_features_ : list of int
        Indices of the features selected for the decision table.
    decision_table_ : dict
        The learned lookup table mapping feature patterns to classes.
    classes_ : np.ndarray
        Unique class labels encountered during training.
    default_class_ : any
        Default class prediction for patterns not in the table.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n^2 \\cdot m)` where :math:`n` = number of samples
      and :math:`m` = number of features (due to leave-one-out evaluation
      over candidate subsets)
    - Prediction: :math:`O(m)` per sample (table lookup)

    **When to use DecisionTableClassifier:**

    - When maximum interpretability is required
    - Low-dimensional datasets with a small number of informative features
    - When the decision boundary can be captured by a few discrete patterns
    - As a baseline before trying more complex rule learners

    References
    ----------
    .. [Kohavi1995] Kohavi, R. (1995).
           **The Power of Decision Tables.**
           *Proceedings of the 8th European Conference on Machine Learning (ECML)*, pp. 174-189.
           DOI: `10.1007/3-540-59286-5_57 <https://doi.org/10.1007/3-540-59286-5_57>`_

    See Also
    --------
    :class:`~tuiml.algorithms.rules.OneRuleClassifier` : Single-attribute rule learner.
    :class:`~tuiml.algorithms.rules.ZeroRuleClassifier` : Majority-class baseline classifier.
    :class:`~tuiml.algorithms.rules.RIPPERClassifier` : RIPPER propositional rule learner.

    Examples
    --------
    Basic usage for classification with feature subset selection:

    >>> from tuiml.algorithms.rules import DecisionTableClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    >>> y_train = np.array([0, 0, 1, 1, 1])
    >>>
    >>> # Fit the model
    >>> clf = DecisionTableClassifier(search_method='best_first')
    >>> clf.fit(X_train, y_train)
    DecisionTableClassifier(...)
    >>> predictions = clf.predict(X_train)
    """

    def __init__(self, search_method: str = 'best_first',
                 cross_val_folds: int = 1,
                 use_ibk: bool = False,
                 random_state: Optional[int] = None):
        """Initialize DecisionTableClassifier.

        Parameters
        ----------
        search_method : str, default='best_first'
            The search algorithm used to find the best feature subset.
        cross_val_folds : int, default=1
            Number of folds for cross-validation during feature selection.
        use_ibk : bool, default=False
            Whether to use IBk as the underlying classifier.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__()
        self.search_method = search_method
        self.cross_val_folds = cross_val_folds
        self.use_ibk = use_ibk
        self.random_state = random_state
        self.selected_features_ = None
        self.decision_table_ = None
        self.classes_ = None
        self.default_class_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "search_method": {"type": "string", "default": "best_first",
                            "enum": ["best_first", "greedy"]},
            "cross_val_folds": {"type": "integer", "default": 1, "minimum": 1},
            "use_ibk": {"type": "boolean", "default": False},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "binary_class", "multiclass"]

    def _discretize(self, X: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Discretize continuous features into integer bin indices.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.
        n_bins : int, default=10
            Maximum number of bins for discretization.

        Returns
        -------
        X_disc : np.ndarray of shape (n_samples, n_features)
            Discretized feature matrix with integer bin indices.
        """
        X_disc = np.zeros_like(X, dtype=int)
        for j in range(X.shape[1]):
            col = X[:, j]
            if len(np.unique(col)) > n_bins:
                bins = np.percentile(col[~np.isnan(col)], 
                                    np.linspace(0, 100, n_bins + 1))
                bins = np.unique(bins)
                X_disc[:, j] = np.digitize(col, bins[1:-1])
            else:
                _, X_disc[:, j] = np.unique(col, return_inverse=True)
        return X_disc

    def _evaluate_subset(self, X: np.ndarray, y: np.ndarray,
                        features: List[int]) -> float:
        """Evaluate a feature subset using leave-one-out cross-validation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Discretized feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target labels.
        features : list of int
            Indices of the candidate feature subset.

        Returns
        -------
        accuracy : float
            Leave-one-out accuracy for the given feature subset.
        """
        if len(features) == 0:
            return 0.0
        
        X_sub = X[:, features]
        n_samples = len(y)
        correct = 0
        
        for i in range(n_samples):
            # Leave-one-out
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False
            
            key = tuple(X_sub[i])
            matches = [j for j in range(n_samples) if mask[j] and 
                      tuple(X_sub[j]) == key]
            
            if matches:
                votes = Counter(y[matches])
                pred = votes.most_common(1)[0][0]
            else:
                pred = self.default_class_
            
            if pred == y[i]:
                correct += 1
        
        return correct / n_samples

    def _best_first_search(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Perform best-first forward search over feature subsets.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Discretized feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        best_features : list of int
            Indices of the selected feature subset.
        """
        n_features = X.shape[1]
        best_features = []
        best_score = 0.0
        
        # Start with empty set and add features greedily
        remaining = list(range(n_features))
        
        for _ in range(min(n_features, 5)):  # Limit search depth
            best_candidate = None
            best_candidate_score = best_score
            
            for f in remaining:
                candidate = best_features + [f]
                score = self._evaluate_subset(X, y, candidate)
                
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = f
            
            if best_candidate is not None:
                best_features.append(best_candidate)
                remaining.remove(best_candidate)
                best_score = best_candidate_score
            else:
                break
        
        return best_features if best_features else [0]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTableClassifier":
        """Fit the DecisionTableClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : DecisionTableClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        self.classes_ = np.unique(y)
        self.default_class_ = Counter(y).most_common(1)[0][0]
        
        # Discretize features
        X_disc = self._discretize(X)
        
        # Feature selection
        self.selected_features_ = self._best_first_search(X_disc, y)
        
        # Build decision table
        X_selected = X_disc[:, self.selected_features_]
        self.decision_table_ = {}
        
        for i in range(len(y)):
            key = tuple(X_selected[i])
            if key not in self.decision_table_:
                self.decision_table_[key] = []
            self.decision_table_[key].append(y[i])
        
        # Convert to majority class
        for key in self.decision_table_:
            self.decision_table_[key] = Counter(self.decision_table_[key]).most_common(1)[0][0]
        
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        X_disc = self._discretize(X)
        X_selected = X_disc[:, self.selected_features_]
        
        predictions = [self.decision_table_.get(tuple(X_selected[i]), self.default_class_) for i in range(len(X))]
        return np.array(predictions, dtype=self.classes_.dtype)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities (one-hot encoded hard predictions).
        """
        preds = self.predict(X)
        proba = np.zeros((len(preds), len(self.classes_)))
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0]
            if len(idx) > 0:
                proba[i, idx[0]] = 1.0
        return proba
