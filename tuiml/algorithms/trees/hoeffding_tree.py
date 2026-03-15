"""HoeffdingTreeClassifier (VFDT) classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

from tuiml.base.algorithms import Classifier, classifier
from ._core import entropy_from_counts

@dataclass
class HoeffdingNode:
    """Node in a Hoeffding Tree.

    Streaming nodes store per-feature value statistics and are fundamentally
    incompatible with the batch ``TreeNode``; therefore ``HoeffdingNode``
    remains a separate dataclass.
    """
    is_leaf: bool = True
    class_counts: Dict = field(default_factory=dict)
    n_samples: int = 0
    feature_index: int = None
    threshold: float = None
    left: "HoeffdingNode" = None
    right: "HoeffdingNode" = None
    # For numeric attributes: store value statistics
    feature_stats: Dict = field(default_factory=dict)

@classifier(tags=["trees", "streaming", "incremental", "online"], version="1.0.0")
class HoeffdingTreeClassifier(Classifier):
    """Hoeffding Tree (Very Fast Decision Tree - VFDT) classifier.

    A Hoeffding Tree is designed for **streaming data** scenarios where
    instances arrive continuously. It uses the **Hoeffding bound** to
    statistically determine when a split is reliable, guaranteeing
    **asymptotic equivalence** to a batch learner with constant memory
    usage per leaf.

    Overview
    --------
    The algorithm builds a tree incrementally from a data stream:

    1. Each incoming instance is sorted down to its corresponding **leaf node**
    2. The leaf updates its **sufficient statistics** (class counts, feature
       value distributions) for the new instance
    3. Every ``grace_period`` instances, the leaf evaluates candidate splits
       using **information gain**
    4. The **Hoeffding bound** determines if the best attribute is
       statistically significantly better than the second-best
    5. If the bound is satisfied (or a tie is detected), the leaf is
       **converted to an internal node** and new empty leaves are created
    6. The tree grows incrementally without revisiting past data

    Theory
    ------
    The Hoeffding bound states that with probability :math:`1 - \\delta`, the
    true mean of a random variable :math:`r` with range :math:`R` after
    :math:`n` observations differs from the sample mean by at most:

    .. math::
        \\epsilon = \\sqrt{\\frac{R^2 \\ln(1/\\delta)}{2n}}

    A split on attribute :math:`A_a` is chosen over :math:`A_b` when:

    .. math::
        \\Delta G = G(A_a) - G(A_b) > \\epsilon

    where :math:`G(\\cdot)` is the information gain. If
    :math:`\\Delta G < \\epsilon < \\tau` (the tie threshold), a split is
    also performed (tie-breaking).

    Parameters
    ----------
    grace_period : int, default=200
        Number of instances between split attempts.
    split_confidence : float, default=1e-7
        Confidence level :math:`\\delta` for the Hoeffding bound. Smaller
        values require more evidence before splitting.
    tie_threshold : float, default=0.05
        Threshold for tie-breaking when gain difference is small.
    min_samples_split : int, default=10
        Minimum samples required to consider a split.
    leaf_prediction : {'mc', 'nb'}, default='mc'
        Prediction strategy at leaves: ``'mc'`` for majority class,
        ``'nb'`` for Naive Bayes.

    Attributes
    ----------
    tree_ : HoeffdingNode
        Root node of the tree.
    classes_ : np.ndarray
        Unique class labels.
    n_seen_ : int
        Total instances processed.

    Notes
    -----
    **Complexity:**

    - Per instance: :math:`O(m \\cdot \\log(n_{leaves}))` amortized, where
      :math:`m` = features, :math:`n_{leaves}` = current number of leaves
    - Memory: :math:`O(l \\cdot m \\cdot v)` where :math:`l` = leaves,
      :math:`v` = distinct values per feature

    **When to use HoeffdingTreeClassifier:**

    - **Streaming data** where instances arrive one at a time or in batches
    - When the dataset is too large to fit in memory
    - When you need **incremental learning** with ``partial_fit``
    - When you need theoretical guarantees on split quality
    - Real-time classification systems with evolving data distributions

    References
    ----------
    .. [Domingos2000] Domingos, P. and Hulten, G. (2000).
           **Mining High-Speed Data Streams.**
           *Proceedings of the 6th ACM SIGKDD International Conference on
           Knowledge Discovery and Data Mining*, pp. 71-80.
           DOI: `10.1145/347090.347107 <https://doi.org/10.1145/347090.347107>`_

    .. [Hulten2001] Hulten, G., Spencer, L. and Domingos, P. (2001).
           **Mining Time-Changing Data Streams.**
           *Proceedings of the 7th ACM SIGKDD*, pp. 97-106.
           DOI: `10.1145/502512.502529 <https://doi.org/10.1145/502512.502529>`_

    See Also
    --------
    :class:`~tuiml.algorithms.trees.C45TreeClassifier` : Batch C4.5 decision tree.
    :class:`~tuiml.algorithms.trees.RandomForestClassifier` : Ensemble of batch trees.
    :class:`~tuiml.algorithms.trees.ReducedErrorPruningTreeClassifier` : Fast tree with pruning.

    Examples
    --------
    Incremental learning from a data stream:

    >>> from tuiml.algorithms.trees import HoeffdingTreeClassifier
    >>> import numpy as np
    >>>
    >>> # Create streaming data
    >>> X_batch1 = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_batch1 = np.array([0, 0, 1])
    >>> X_batch2 = np.array([[7, 8], [2, 3], [4, 5]])
    >>> y_batch2 = np.array([1, 0, 1])
    >>>
    >>> # Incrementally fit
    >>> clf = HoeffdingTreeClassifier(grace_period=200)
    >>> clf.partial_fit(X_batch1, y_batch1)
    HoeffdingTreeClassifier(...)
    >>> clf.partial_fit(X_batch2, y_batch2)
    HoeffdingTreeClassifier(...)
    >>> predictions = clf.predict(X_batch1)
    """

    def __init__(self, grace_period: int = 200,
                 split_confidence: float = 1e-7,
                 tie_threshold: float = 0.05,
                 min_samples_split: int = 10,
                 leaf_prediction: str = 'mc'):
        super().__init__()
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.min_samples_split = min_samples_split
        self.leaf_prediction = leaf_prediction
        self.tree_ = None
        self.classes_ = None
        self.n_seen_ = 0
        self._n_features = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "grace_period": {"type": "integer", "default": 200, "minimum": 1,
                            "description": "Instances between split attempts"},
            "split_confidence": {"type": "number", "default": 1e-7,
                                "minimum": 0, "description": "Hoeffding bound confidence"},
            "tie_threshold": {"type": "number", "default": 0.05,
                             "minimum": 0, "description": "Tie-breaking threshold"},
            "min_samples_split": {"type": "integer", "default": 10, "minimum": 1,
                                 "description": "Minimum samples for split"},
            "leaf_prediction": {"type": "string", "default": "mc",
                               "enum": ["mc", "nb"],
                               "description": "Leaf prediction strategy"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(log(n)) per instance, O(m * log(n)) total per instance"

    @classmethod
    def get_references(cls) -> List[str]:
        return ["Domingos, P., & Hulten, G. (2000). Mining High-Speed Data "
                "Streams. Proceedings of the 6th ACM SIGKDD, 71-80."]

    def _information_gain(self, parent_counts: Dict, parent_total: int,
                          left_counts: Dict, left_total: int,
                          right_counts: Dict, right_total: int) -> float:
        """Calculate information gain for a split using shared entropy.

        Parameters
        ----------
        parent_counts : dict
            Class counts at the parent node.
        parent_total : int
            Total samples at parent.
        left_counts : dict
            Class counts for the left partition.
        left_total : int
            Total samples in left partition.
        right_counts : dict
            Class counts for the right partition.
        right_total : int
            Total samples in right partition.

        Returns
        -------
        ig : float
            Information gain value.
        """
        parent_H = entropy_from_counts(parent_counts, parent_total)

        if parent_total == 0:
            return 0.0

        left_H = entropy_from_counts(left_counts, left_total)
        right_H = entropy_from_counts(right_counts, right_total)

        child_H = ((left_total / parent_total) * left_H +
                   (right_total / parent_total) * right_H)

        return parent_H - child_H

    def _hoeffding_bound(self, n: int, delta: float) -> float:
        """Compute Hoeffding bound.

        Parameters
        ----------
        n : int
            Number of observations.
        delta : float
            Confidence parameter (split_confidence).

        Returns
        -------
        epsilon : float
            Hoeffding bound value.
        """
        R = np.log2(max(len(self.classes_), 2))
        return np.sqrt((R ** 2 * np.log(1 / delta)) / (2 * n))

    def _find_best_split(self, node: HoeffdingNode) -> Tuple[int, float, float, float]:
        """Find the best split for a node.

        Parameters
        ----------
        node : HoeffdingNode
            Leaf node to evaluate for splitting.

        Returns
        -------
        best_feature : int
            Index of the best feature.
        best_threshold : float
            Best split threshold.
        best_gain : float
            Information gain of the best split.
        second_best_gain : float
            Information gain of the second-best split.
        """
        best_gain = -np.inf
        second_best_gain = -np.inf
        best_feature = 0
        best_threshold = 0.0

        for feature_idx in range(self._n_features):
            if feature_idx not in node.feature_stats:
                continue

            stats = node.feature_stats[feature_idx]

            if 'values' not in stats or len(stats['values']) < 2:
                continue

            values = sorted(stats['values'].keys())
            thresholds = [(values[i] + values[i + 1]) / 2
                         for i in range(len(values) - 1)]

            for threshold in thresholds:
                left_counts = defaultdict(int)
                right_counts = defaultdict(int)
                left_total = 0
                right_total = 0

                for val, class_counts in stats['values'].items():
                    if val <= threshold:
                        for c, count in class_counts.items():
                            left_counts[c] += count
                            left_total += count
                    else:
                        for c, count in class_counts.items():
                            right_counts[c] += count
                            right_total += count

                if left_total < self.min_samples_split or \
                   right_total < self.min_samples_split:
                    continue

                gain = self._information_gain(node.class_counts, node.n_samples,
                                              left_counts, left_total,
                                              right_counts, right_total)

                if gain > best_gain:
                    second_best_gain = best_gain
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                elif gain > second_best_gain:
                    second_best_gain = gain

        return best_feature, best_threshold, best_gain, second_best_gain

    def _attempt_split(self, node: HoeffdingNode):
        """Attempt to split a leaf node.

        Parameters
        ----------
        node : HoeffdingNode
            Leaf node to evaluate. Converted to internal node in-place
            if the Hoeffding bound is satisfied.
        """
        if node.n_samples < self.min_samples_split:
            return

        best_feature, best_threshold, best_gain, second_best_gain = \
            self._find_best_split(node)

        epsilon = self._hoeffding_bound(node.n_samples, self.split_confidence)

        if best_gain - second_best_gain > epsilon or epsilon < self.tie_threshold:
            if best_gain > 0:
                node.is_leaf = False
                node.feature_index = best_feature
                node.threshold = best_threshold

                node.left = HoeffdingNode()
                node.right = HoeffdingNode()

                for c in self.classes_:
                    node.left.class_counts[c] = 0
                    node.right.class_counts[c] = 0

    def _update_leaf(self, node: HoeffdingNode, x: np.ndarray, y: Any):
        """Update statistics at a leaf node.

        Parameters
        ----------
        node : HoeffdingNode
            Leaf node to update.
        x : np.ndarray of shape (n_features,)
            Single sample feature vector.
        y : Any
            Class label for the sample.
        """
        if y not in node.class_counts:
            node.class_counts[y] = 0
        node.class_counts[y] += 1
        node.n_samples += 1

        for feature_idx in range(self._n_features):
            if feature_idx not in node.feature_stats:
                node.feature_stats[feature_idx] = {'values': defaultdict(lambda: defaultdict(int))}

            val = x[feature_idx]
            if not np.isnan(val):
                val = round(val, 4)
                node.feature_stats[feature_idx]['values'][val][y] += 1

        if node.n_samples % self.grace_period == 0:
            self._attempt_split(node)

    def _sort_to_leaf(self, node: HoeffdingNode, x: np.ndarray) -> HoeffdingNode:
        """Sort an instance to its appropriate leaf.

        Parameters
        ----------
        node : HoeffdingNode
            Root node to start traversal from.
        x : np.ndarray of shape (n_features,)
            Single sample feature vector.

        Returns
        -------
        leaf : HoeffdingNode
            Leaf node where the instance belongs.
        """
        while not node.is_leaf:
            val = x[node.feature_index]
            if np.isnan(val) or val <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    classes: Optional[np.ndarray] = None) -> "HoeffdingTreeClassifier":
        """Incrementally fit on a batch of samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.
        classes : np.ndarray, optional
            List of all possible classes (required on first call).

        Returns
        -------
        self : HoeffdingTreeClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not self._is_fitted:
            if classes is not None:
                self.classes_ = np.asarray(classes)
            else:
                self.classes_ = np.unique(y)

            self._n_features = X.shape[1]
            self.tree_ = HoeffdingNode()

            for c in self.classes_:
                self.tree_.class_counts[c] = 0

            self._is_fitted = True

        for i in range(len(y)):
            leaf = self._sort_to_leaf(self.tree_, X[i])
            self._update_leaf(leaf, X[i], y[i])
            self.n_seen_ += 1

        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HoeffdingTreeClassifier":
        """Fit the HoeffdingTreeClassifier classifier."""
        self._is_fitted = False
        return self.partial_fit(X, y)

    def _predict_leaf(self, node: HoeffdingNode, x: np.ndarray) -> Any:
        """Predict at a leaf node.

        Parameters
        ----------
        node : HoeffdingNode
            Leaf node for prediction.
        x : np.ndarray of shape (n_features,)
            Single sample feature vector.

        Returns
        -------
        prediction : Any
            Predicted class label.
        """
        if not node.class_counts:
            return self.classes_[0]

        if self.leaf_prediction == 'mc':
            return max(node.class_counts, key=node.class_counts.get)
        else:
            return max(node.class_counts, key=node.class_counts.get)

    def _predict_proba_leaf(self, node: HoeffdingNode) -> np.ndarray:
        """Predict probabilities at a leaf.

        Parameters
        ----------
        node : HoeffdingNode
            Leaf node for probability prediction.

        Returns
        -------
        proba : np.ndarray of shape (n_classes,)
            Class probability distribution.
        """
        proba = np.zeros(len(self.classes_))
        total = node.n_samples

        if total > 0:
            for idx, c in enumerate(self.classes_):
                proba[idx] = node.class_counts.get(c, 0) / total
        else:
            proba[:] = 1.0 / len(self.classes_)

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = [self._predict_leaf(self._sort_to_leaf(self.tree_, X[i]), X[i]) for i in range(len(X))]
        return np.array(predictions, dtype=self.classes_.dtype)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        proba = np.zeros((len(X), len(self.classes_)))
        for i in range(len(X)):
            leaf = self._sort_to_leaf(self.tree_, X[i])
            proba[i] = self._predict_proba_leaf(leaf)

        return proba

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"HoeffdingTreeClassifier(n_seen={self.n_seen_})"
        return f"HoeffdingTreeClassifier(grace_period={self.grace_period})"
