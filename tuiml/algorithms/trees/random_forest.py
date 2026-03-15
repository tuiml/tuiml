"""RandomForestClassifier and RandomForestRegressor using C++ tree builders."""

import numpy as np
from typing import Dict, List, Any, Optional
from joblib import Parallel, delayed

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.algorithms.trees._core import (
    TreeNode,
    TreeConfig,
    build_classifier_tree,
    build_regressor_tree,
    compute_max_features,
)
from tuiml.algorithms.trees._core_dispatch import (
    predict_batch_cpp as _predict_batch_cpp,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _build_single_classifier_tree(X, y, config, seed):
    """Build one classification tree on a bootstrap sample.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Full training features.
    y : np.ndarray of shape (n_samples,)
        Integer-encoded labels.
    config : TreeConfig
        Tree building configuration.
    seed : int
        Random seed for this tree.

    Returns
    -------
    tree : TreeNode
        Root of the fitted tree.
    indices : np.ndarray
        Bootstrap sample indices (for OOB computation).
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    indices = rng.choice(n, size=n, replace=True)
    X_boot, y_boot = X[indices], y[indices]
    tree = build_classifier_tree(X_boot, y_boot, config, rng)
    return tree, indices


def _build_single_regressor_tree(X, y, config, seed):
    """Build one regression tree on a bootstrap sample.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Full training features.
    y : np.ndarray of shape (n_samples,)
        Target values.
    config : TreeConfig
        Tree building configuration.
    seed : int
        Random seed for this tree.

    Returns
    -------
    tree : TreeNode
        Root of the fitted tree.
    indices : np.ndarray
        Bootstrap sample indices (for OOB computation).
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    indices = rng.choice(n, size=n, replace=True)
    X_boot, y_boot = X[indices], y[indices]
    tree = build_regressor_tree(X_boot, y_boot, config, rng)
    return tree, indices


def _predict_tree_proba(tree, X, n_classes):
    """Get probability predictions from a single tree for all samples.

    Parameters
    ----------
    tree : TreeNode
        Fitted tree root.
    X : np.ndarray of shape (n_samples, n_features)
        Input samples.
    n_classes : int
        Number of classes.

    Returns
    -------
    proba : np.ndarray of shape (n_samples, n_classes)
        Class probabilities from this tree.
    """
    raw = _predict_batch_cpp(tree._cpp_flat, X)
    # Ensure correct width (C++ may return fewer columns)
    if raw.shape[1] < n_classes:
        padded = np.zeros((raw.shape[0], n_classes), dtype=np.float64)
        padded[:, :raw.shape[1]] = raw
        return padded
    return raw[:, :n_classes]


def _predict_tree_regression(tree, X):
    """Get regression predictions from a single tree for all samples.

    Parameters
    ----------
    tree : TreeNode
        Fitted tree root.
    X : np.ndarray of shape (n_samples, n_features)
        Input samples.

    Returns
    -------
    preds : np.ndarray of shape (n_samples,)
        Predictions from this tree.
    """
    raw = _predict_batch_cpp(tree._cpp_flat, X)
    return raw[:, 0]


def _compute_feature_importances(trees, n_features, mode='classifier'):
    """Compute feature importances across all trees.

    Uses impurity-based importance when node statistics are available
    (Python-built trees). Falls back to split-count importance when
    nodes lack ``n_samples``/``impurity`` (C++-built trees).

    Parameters
    ----------
    trees : list of TreeNode
        Fitted trees.
    n_features : int
        Number of features.
    mode : str
        ``'classifier'`` or ``'regressor'``.

    Returns
    -------
    importances : np.ndarray of shape (n_features,)
        Normalized feature importances.
    """
    importances = np.zeros(n_features, dtype=np.float64)

    for tree in trees:
        _accumulate_importances(tree, importances)

    # If impurity-based importances are all zero (e.g. C++ trees with
    # n_samples=0), fall back to split-count importance.
    if importances.sum() == 0:
        for tree in trees:
            _accumulate_split_counts(tree, importances)

    total = importances.sum()
    if total > 0:
        importances /= total
    return importances


def _accumulate_importances(node, importances):
    """Recursively accumulate weighted impurity decrease per feature.

    Parameters
    ----------
    node : TreeNode or None
        Current node.
    importances : np.ndarray
        Accumulator array (modified in place).
    """
    if node is None or node.is_leaf:
        return

    left_imp = 0.0
    right_imp = 0.0
    n_left = 0
    n_right = 0

    if node.left is not None:
        left_imp = node.left.impurity
        n_left = node.left.n_samples
    if node.right is not None:
        right_imp = node.right.impurity
        n_right = node.right.n_samples

    n_total = node.n_samples if node.n_samples > 0 else (n_left + n_right)
    if n_total > 0:
        weighted_decrease = (
            n_total * node.impurity
            - n_left * left_imp
            - n_right * right_imp
        )
        if weighted_decrease > 0 and 0 <= node.feature_index < len(importances):
            importances[node.feature_index] += weighted_decrease

    _accumulate_importances(node.left, importances)
    _accumulate_importances(node.right, importances)


def _accumulate_split_counts(node, importances):
    """Count splits per feature (fallback for C++-built trees).

    Parameters
    ----------
    node : TreeNode or None
        Current node.
    importances : np.ndarray
        Accumulator array (modified in place).
    """
    if node is None or node.is_leaf:
        return
    if 0 <= node.feature_index < len(importances):
        importances[node.feature_index] += 1.0
    _accumulate_split_counts(node.left, importances)
    _accumulate_split_counts(node.right, importances)


@classifier(tags=["trees", "ensemble", "random", "bagging"], version="1.0.0")
class RandomForestClassifier(Classifier):
    """Random Forest classifier - ensemble of random trees.

    Random Forest is an **ensemble method** that fits multiple randomized
    decision trees on **bootstrapped subsamples** of the dataset and uses
    **majority voting** (or probability averaging) to improve predictive
    accuracy and control overfitting.

    Overview
    --------
    The Random Forest algorithm works as follows:

    1. For each of the :math:`T` trees, draw a **bootstrap sample**
       (sampling with replacement) of size :math:`n` from the training data
    2. Build a fully-grown :class:`~tuiml.algorithms.trees.RandomTreeClassifier`
       on each bootstrap sample, selecting :math:`k` random features at each split
    3. For prediction, aggregate results via **majority voting** (classification)
       or **averaging** (probability estimation)
    4. Optionally compute the **out-of-bag (OOB) score** using samples not
       included in each tree's bootstrap

    Theory
    ------
    Each tree :math:`h_t` is trained on a bootstrap sample :math:`S_t`. The
    ensemble prediction is determined by majority vote:

    .. math::
        \\hat{y}(x) = \\arg\\max_c \\sum_{t=1}^{T} \\mathbb{1}[h_t(x) = c]

    The **generalization error** of a Random Forest is bounded by:

    .. math::
        PE^* \\leq \\bar{\\rho} \\cdot \\frac{(1 - s^2)}{s^2}

    where :math:`\\bar{\\rho}` is the mean correlation between trees and
    :math:`s` is the strength (margin) of individual trees.

    The **out-of-bag error** is computed using each sample's predictions only
    from trees that did not include it in their bootstrap:

    .. math::
        OOB_{error} = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbb{1}[\\hat{y}_{OOB}(x_i) \\neq y_i]

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_features : {'sqrt', 'log2'}, int or float, default='sqrt'
        Number of features to consider at each split.
    max_depth : int, optional
        Maximum depth of the trees. None means unlimited.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    bootstrap : bool, default=True
        Whether to use bootstrap samples when building trees.
    oob_score : bool, default=False
        Whether to calculate out-of-bag score.
    random_state : int, optional
        Random seed for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs (-1 for all CPUs).
    criterion : str, default='gini'
        Splitting criterion (``'gini'`` or ``'entropy'``).

    Attributes
    ----------
    estimators_ : list of TreeNode
        Fitted tree root nodes.
    classes_ : np.ndarray
        Unique class labels.
    n_features_ : int
        Number of features seen during fit.
    oob_score_ : float
        Out-of-bag score (if ``oob_score=True``).
    feature_importances_ : np.ndarray or None
        Feature importance scores (impurity-based).

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot k \\cdot \\log(n))` where :math:`T` =
      n_estimators, :math:`n` = samples, :math:`k` = max_features
    - Prediction: :math:`O(T \\cdot \\log(n))` per sample

    **When to use RandomForestClassifier:**

    - When you need a **robust, general-purpose** classifier
    - High-dimensional datasets where feature selection is implicit
    - When **out-of-bag error** estimation is desired (no separate validation set)
    - When training can be **parallelized** across multiple cores
    - When individual tree interpretability is less important than accuracy

    References
    ----------
    .. [Breiman2001] Breiman, L. (2001).
           **Random Forests.**
           *Machine Learning*, 45(1), pp. 5-32.
           DOI: `10.1023/A:1010933404324 <https://doi.org/10.1023/A:1010933404324>`_

    .. [Breiman1996] Breiman, L. (1996).
           **Bagging Predictors.**
           *Machine Learning*, 24(2), pp. 123-140.
           DOI: `10.1007/BF00058655 <https://doi.org/10.1007/BF00058655>`_

    See Also
    --------
    :class:`~tuiml.algorithms.trees.RandomTreeClassifier` : Single randomized tree (base learner).
    :class:`~tuiml.algorithms.trees.C45TreeClassifier` : Deterministic C4.5 decision tree.
    :class:`~tuiml.algorithms.trees.LogisticModelTreeClassifier` : Tree with logistic regression at leaves.

    Examples
    --------
    Basic usage for classification with OOB score:

    >>> from tuiml.algorithms.trees import RandomForestClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>>
    >>> # Fit a random forest
    >>> clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
    >>> clf.fit(X, y)
    RandomForestClassifier(...)
    >>> predictions = clf.predict(X)
    """

    def __init__(self, n_estimators: int = 100,
                 max_features: Any = 'sqrt',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 criterion: str = 'gini'):
        """Initialize RandomForestClassifier classifier.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of trees.
        max_features : {'sqrt', 'log2'}, int or float, default='sqrt'
            Features to consider at each split.
        max_depth : int or None, default=None
            Maximum tree depth.
        min_samples_split : int, default=2
            Minimum samples to split a node.
        min_samples_leaf : int, default=1
            Minimum samples at leaf node.
        bootstrap : bool, default=True
            Use bootstrap samples.
        oob_score : bool, default=False
            Calculate out-of-bag score.
        random_state : int or None, default=None
            Random seed.
        n_jobs : int, default=1
            Number of parallel jobs.
        criterion : str, default='gini'
            Splitting criterion.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.criterion = criterion
        self.estimators_ = None
        self.classes_ = None
        self.n_features_ = None
        self.oob_score_ = None
        self.feature_importances_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Number of trees in the forest"
            },
            "max_features": {
                "type": ["string", "integer", "number"],
                "default": "sqrt",
                "description": "Number of features to consider at each split"
            },
            "max_depth": {
                "type": "integer",
                "default": None,
                "minimum": 1,
                "description": "Maximum depth of trees"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "minimum": 2,
                "description": "Minimum samples required to split a node"
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Minimum samples required at a leaf node"
            },
            "bootstrap": {
                "type": "boolean",
                "default": True,
                "description": "Whether to use bootstrap samples"
            },
            "oob_score": {
                "type": "boolean",
                "default": False,
                "description": "Whether to calculate out-of-bag score"
            },
            "random_state": {
                "type": "integer",
                "default": None,
                "description": "Random seed for reproducibility"
            },
            "n_jobs": {
                "type": "integer",
                "default": 1,
                "description": "Number of parallel jobs (-1 for all CPUs)"
            },
            "criterion": {
                "type": "string",
                "default": "gini",
                "enum": ["gini", "entropy"],
                "description": "Splitting criterion"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return [
            "numeric",
            "missing_values",
            "binary_class",
            "multiclass"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m * log(n) * T) training, O(log(n) * T) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Fit the Random Forest classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : RandomForestClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Encode labels to integer indices
        if not np.issubdtype(y.dtype, np.integer) or not np.array_equal(
            self.classes_, np.arange(n_classes)
        ):
            self._label_map = {label: i for i, label in enumerate(self.classes_)}
            y_encoded = np.array([self._label_map[label] for label in y])
        else:
            self._label_map = None
            y_encoded = y

        max_features_int = compute_max_features(self.max_features, self.n_features_)

        config = TreeConfig(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            max_features=max_features_int,
            n_classes=n_classes,
        )

        # Generate per-tree seeds for reproducibility
        master_rng = np.random.RandomState(self.random_state)
        seeds = master_rng.randint(0, 2**31, size=self.n_estimators)

        if self.bootstrap:
            # Build trees with bootstrap sampling, optionally in parallel
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_build_single_classifier_tree)(X, y_encoded, config, int(s))
                for s in seeds
            )
            self.estimators_ = [r[0] for r in results]
            bootstrap_indices = [r[1] for r in results]
        else:
            # No bootstrap: each tree sees all data (still randomized via max_features)
            def _build_no_bootstrap(seed):
                rng = np.random.RandomState(seed)
                tree = build_classifier_tree(X, y_encoded, config, rng)
                return tree

            trees = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_build_no_bootstrap)(int(s)) for s in seeds
            )
            self.estimators_ = trees
            bootstrap_indices = None

        # Feature importances
        self.feature_importances_ = _compute_feature_importances(
            self.estimators_, self.n_features_, mode='classifier'
        )

        # OOB score
        if self.oob_score and self.bootstrap and bootstrap_indices is not None:
            self.oob_score_ = self._compute_oob_score(
                X, y_encoded, bootstrap_indices, n_classes
            )

        self._is_fitted = True
        return self

    def _compute_oob_score(self, X, y, bootstrap_indices, n_classes):
        """Compute out-of-bag classification accuracy.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Integer-encoded labels.
        bootstrap_indices : list of np.ndarray
            Bootstrap indices for each tree.
        n_classes : int
            Number of classes.

        Returns
        -------
        oob_accuracy : float
            Out-of-bag accuracy score.
        """
        n_samples = len(y)
        oob_proba = np.zeros((n_samples, n_classes), dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int32)

        for tree, indices in zip(self.estimators_, bootstrap_indices):
            # Samples NOT in this tree's bootstrap
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[indices] = True
            oob_mask = ~in_bag

            if not np.any(oob_mask):
                continue

            X_oob = X[oob_mask]
            proba = _predict_tree_proba(tree, X_oob, n_classes)
            oob_proba[oob_mask] += proba
            oob_counts[oob_mask] += 1

        # Only score samples that were OOB for at least one tree
        valid = oob_counts > 0
        if not np.any(valid):
            return 0.0

        oob_pred = np.argmax(oob_proba[valid], axis=1)
        return float(np.mean(oob_pred == y[valid]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Uses probability averaging across all trees, then argmax.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Returns the mean class probabilities of the trees.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        avg_proba = np.zeros((n_samples, n_classes), dtype=np.float64)

        for tree in self.estimators_:
            avg_proba += _predict_tree_proba(tree, X, n_classes)

        avg_proba /= len(self.estimators_)
        return avg_proba

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            oob_str = f", oob_score={self.oob_score_:.4f}" if self.oob_score_ is not None else ""
            return (f"RandomForestClassifier(n_estimators={self.n_estimators}, "
                   f"n_features={self.n_features_}{oob_str})")
        return f"RandomForestClassifier(n_estimators={self.n_estimators})"


@regressor(tags=["trees", "ensemble", "random", "bagging"], version="1.0.0")
class RandomForestRegressor(Regressor):
    """Random Forest regressor - ensemble of random regression trees.

    Random Forest regressor is an **ensemble method** that fits multiple
    randomized regression trees on **bootstrapped subsamples** and uses
    **averaging** of predictions to improve accuracy and control
    overfitting. Each base tree uses **variance reduction** (MSE) as its
    splitting criterion.

    Overview
    --------
    The Random Forest regression algorithm works as follows:

    1. For each of the :math:`T` trees, draw a **bootstrap sample**
       (sampling with replacement) of size :math:`n` from the training data
    2. Build a fully-grown :class:`~tuiml.algorithms.trees.RandomTreeRegressor`
       on each bootstrap sample, selecting :math:`k` random features at each split
    3. For prediction, **average** the outputs of all trees
    4. Optionally compute the **out-of-bag (OOB) R-squared** using samples not
       included in each tree's bootstrap

    Theory
    ------
    Each tree :math:`h_t` is trained on a bootstrap sample :math:`S_t`. The
    ensemble prediction is the mean of individual tree predictions:

    .. math::
        \\hat{y}(x) = \\frac{1}{T} \\sum_{t=1}^{T} h_t(x)

    The **out-of-bag R-squared** is computed using each sample's predictions
    only from trees that did not include it in their bootstrap:

    .. math::
        R^2_{OOB} = 1 - \\frac{\\sum_{i=1}^{n} (y_i - \\hat{y}_{OOB}(x_i))^2}
                              {\\sum_{i=1}^{n} (y_i - \\bar{y})^2}

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_features : {'sqrt', 'log2'}, int or float, default='sqrt'
        Number of features to consider at each split.
    max_depth : int, optional
        Maximum depth of the trees. None means unlimited.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    bootstrap : bool, default=True
        Whether to use bootstrap samples when building trees.
    oob_score : bool, default=False
        Whether to calculate out-of-bag R-squared score.
    random_state : int, optional
        Random seed for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs (-1 for all CPUs).
    criterion : str, default='squared_error'
        Splitting criterion (``'squared_error'`` or ``'friedman_mse'``).

    Attributes
    ----------
    estimators_ : list of TreeNode
        Fitted tree root nodes.
    n_features_ : int
        Number of features seen during fit.
    oob_score_ : float
        Out-of-bag R-squared score (if ``oob_score=True``).
    feature_importances_ : np.ndarray or None
        Feature importance scores (impurity-based).

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot k \\cdot \\log(n))` where :math:`T` =
      n_estimators, :math:`n` = samples, :math:`k` = max_features
    - Prediction: :math:`O(T \\cdot \\log(n))` per sample

    **When to use RandomForestRegressor:**

    - When you need a **robust, general-purpose** regressor
    - High-dimensional datasets where feature selection is implicit
    - When **out-of-bag R-squared** estimation is desired
    - When training can be **parallelized** across multiple cores

    References
    ----------
    .. [Breiman2001] Breiman, L. (2001).
           **Random Forests.**
           *Machine Learning*, 45(1), pp. 5-32.
           DOI: `10.1023/A:1010933404324 <https://doi.org/10.1023/A:1010933404324>`_

    See Also
    --------
    :class:`~tuiml.algorithms.trees.RandomTreeRegressor` : Single randomized regression tree (base learner).
    :class:`~tuiml.algorithms.trees.RandomForestClassifier` : Classifier counterpart.
    :class:`~tuiml.algorithms.trees.C45TreeRegressor` : Deterministic regression tree.

    Examples
    --------
    >>> from tuiml.algorithms.trees import RandomForestRegressor
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
    >>> y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5])
    >>> reg = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
    >>> reg.fit(X, y)
    RandomForestRegressor(...)
    >>> predictions = reg.predict(X)
    """

    def __init__(self, n_estimators: int = 100,
                 max_features: Any = 'sqrt',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 random_state: Optional[int] = None,
                 n_jobs: int = 1,
                 criterion: str = 'squared_error'):
        """Initialize RandomForestRegressor.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of trees.
        max_features : {'sqrt', 'log2'}, int or float, default='sqrt'
            Features to consider at each split.
        max_depth : int or None, default=None
            Maximum tree depth.
        min_samples_split : int, default=2
            Minimum samples to split a node.
        min_samples_leaf : int, default=1
            Minimum samples at leaf node.
        bootstrap : bool, default=True
            Use bootstrap samples.
        oob_score : bool, default=False
            Calculate out-of-bag R-squared score.
        random_state : int or None, default=None
            Random seed.
        n_jobs : int, default=1
            Number of parallel jobs.
        criterion : str, default='squared_error'
            Splitting criterion.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.criterion = criterion
        self.estimators_ = None
        self.n_features_ = None
        self.oob_score_ = None
        self.feature_importances_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Number of trees in the forest"
            },
            "max_features": {
                "type": ["string", "integer", "number"],
                "default": "sqrt",
                "description": "Number of features to consider at each split"
            },
            "max_depth": {
                "type": "integer",
                "default": None,
                "minimum": 1,
                "description": "Maximum depth of trees"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "minimum": 2,
                "description": "Minimum samples required to split a node"
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Minimum samples required at a leaf node"
            },
            "bootstrap": {
                "type": "boolean",
                "default": True,
                "description": "Whether to use bootstrap samples"
            },
            "oob_score": {
                "type": "boolean",
                "default": False,
                "description": "Whether to calculate out-of-bag R-squared score"
            },
            "random_state": {
                "type": "integer",
                "default": None,
                "description": "Random seed for reproducibility"
            },
            "n_jobs": {
                "type": "integer",
                "default": 1,
                "description": "Number of parallel jobs (-1 for all CPUs)"
            },
            "criterion": {
                "type": "string",
                "default": "squared_error",
                "enum": ["squared_error", "friedman_mse"],
                "description": "Splitting criterion"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return regressor capabilities."""
        return ["numeric", "numeric_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m * log(n) * T) training, O(log(n) * T) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        """Fit the Random Forest regressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RandomForestRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, self.n_features_ = X.shape

        max_features_int = compute_max_features(self.max_features, self.n_features_)

        config = TreeConfig(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            max_features=max_features_int,
        )

        # Generate per-tree seeds for reproducibility
        master_rng = np.random.RandomState(self.random_state)
        seeds = master_rng.randint(0, 2**31, size=self.n_estimators)

        if self.bootstrap:
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_build_single_regressor_tree)(X, y, config, int(s))
                for s in seeds
            )
            self.estimators_ = [r[0] for r in results]
            bootstrap_indices = [r[1] for r in results]
        else:
            def _build_no_bootstrap(seed):
                rng = np.random.RandomState(seed)
                tree = build_regressor_tree(X, y, config, rng)
                return tree

            trees = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_build_no_bootstrap)(int(s)) for s in seeds
            )
            self.estimators_ = trees
            bootstrap_indices = None

        # Feature importances
        self.feature_importances_ = _compute_feature_importances(
            self.estimators_, self.n_features_, mode='regressor'
        )

        # OOB score
        if self.oob_score and self.bootstrap and bootstrap_indices is not None:
            self.oob_score_ = self._compute_oob_score(X, y, bootstrap_indices)

        self._is_fitted = True
        return self

    def _compute_oob_score(self, X, y, bootstrap_indices):
        """Compute out-of-bag R-squared score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.
        bootstrap_indices : list of np.ndarray
            Bootstrap indices for each tree.

        Returns
        -------
        oob_r2 : float
            Out-of-bag R-squared score.
        """
        n_samples = len(y)
        oob_sum = np.zeros(n_samples, dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int32)

        for tree, indices in zip(self.estimators_, bootstrap_indices):
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[indices] = True
            oob_mask = ~in_bag

            if not np.any(oob_mask):
                continue

            X_oob = X[oob_mask]
            preds = _predict_tree_regression(tree, X_oob)
            oob_sum[oob_mask] += preds
            oob_counts[oob_mask] += 1

        valid = oob_counts > 0
        if not np.any(valid):
            return 0.0

        oob_pred = oob_sum[valid] / oob_counts[valid]
        y_valid = y[valid]
        ss_res = np.sum((y_valid - oob_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples.

        Averages predictions across all trees.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        avg = np.zeros(n_samples, dtype=np.float64)

        for tree in self.estimators_:
            avg += _predict_tree_regression(tree, X)

        avg /= len(self.estimators_)
        return avg

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R-squared score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        r2 : float
            R-squared score.
        """
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            oob_str = f", oob_score={self.oob_score_:.4f}" if self.oob_score_ is not None else ""
            return (f"RandomForestRegressor(n_estimators={self.n_estimators}, "
                   f"n_features={self.n_features_}{oob_str})")
        return f"RandomForestRegressor(n_estimators={self.n_estimators})"
