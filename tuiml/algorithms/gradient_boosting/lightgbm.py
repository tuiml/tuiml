"""LightGBM (Light Gradient Boosting Machine) implementation."""

import numpy as np
from typing import Dict, List, Any, Optional
import warnings

from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    lgb = None
    LIGHTGBM_AVAILABLE = False

@classifier(tags=["gradient-boosting", "lightgbm", "distributed"], version="1.0.0")
class LightGBMClassifier(Classifier):
    """LightGBM classifier for distributed, high-performance **gradient boosting**.

    LightGBM is a gradient boosting framework that uses **leaf-wise tree growth**
    and **histogram-based** split finding. It is designed for **distributed**
    training with fast speed, low memory usage, and support for billion-level data.

    Overview
    --------
    The algorithm builds an ensemble of decision trees using leaf-wise growth:

    1. Initialize the model with a constant prediction (e.g., log-odds for
       classification)
    2. For each boosting iteration, compute the **negative gradient** of the
       loss function for every training sample
    3. Build histograms of feature values using **gradient-based one-side
       sampling (GOSS)** and **exclusive feature bundling (EFB)**
    4. Grow the tree **leaf-wise** by splitting the leaf with the highest
       gain, rather than level-wise, controlled by ``num_leaves``
    5. Add the new tree to the ensemble, scaled by the learning rate
    6. Repeat until the specified number of boosting rounds is reached

    Theory
    ------
    At each boosting round :math:`t`, LightGBM minimizes:

    .. math::
        \\mathcal{L}^{(t)} = \\sum_{i=1}^{n} l(y_i, \\hat{y}_i^{(t-1)} + f_t(x_i))
        + \\Omega(f_t)

    where the regularization term is:

    .. math::
        \\Omega(f) = \\gamma T + \\frac{1}{2} \\lambda \\sum_{j=1}^{T} w_j^2
        + \\alpha \\sum_{j=1}^{T} |w_j|

    **GOSS** keeps all instances with large gradients and randomly samples from
    instances with small gradients, multiplying them by :math:`\\frac{1-a}{b}`
    to compensate:

    .. math::
        \\tilde{V}_j(d) = \\frac{1}{n} \\left( \\sum_{x_i \\in A_l} g_i
        + \\frac{1-a}{b} \\sum_{x_i \\in B_l} g_i \\right)^2

    where :math:`A` is the set of top-:math:`a` instances and :math:`B` is
    sampled from the remaining instances with ratio :math:`b`.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations (trees to build).

    max_depth : int, default=-1
        Maximum tree depth. ``-1`` means no limit.

    learning_rate : float, default=0.1
        Step size shrinkage to prevent overfitting.

    num_leaves : int, default=31
        Maximum number of leaves in one tree.

    subsample : float, default=1.0
        Fraction of samples to randomly sample for each tree.

    colsample_bytree : float, default=1.0
        Fraction of features to randomly sample for each tree.

    reg_alpha : float, default=0.0
        L1 regularization term on weights.

    reg_lambda : float, default=0.0
        L2 regularization term on weights.

    min_child_samples : int, default=20
        Minimum number of samples required in a leaf node.

    min_split_gain : float, default=0.0
        Minimum loss reduction required to make a further partition.

    verbose : int, default=-1
        Verbosity level. ``-1``: Quiet, ``0``: Warnings, ``1``: Info.

    random_state : int, optional
        Seed for reproducibility.

    Attributes
    ----------
    model_ : lgb.LGBMClassifier
        The underlying fitted LightGBM model object.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels discovered during ``fit()``.

    n_classes_ : int
        Number of unique classes discovered during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot d \\cdot L)` where :math:`T` = n_estimators,
      :math:`n` = n_samples, :math:`d` = n_features, :math:`L` = num_leaves.
      In practice, GOSS and EFB significantly reduce effective :math:`n` and :math:`d`.
    - Prediction: :math:`O(T \\cdot L)` per sample

    **When to use LightGBMClassifier:**

    - **Large-scale** classification tasks where training speed is critical
    - High-dimensional datasets where **exclusive feature bundling** reduces cost
    - When memory efficiency is important (histogram-based approach)
    - Distributed training scenarios across multiple machines

    References
    ----------
    .. [Ke2017] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W.,
           Ma, W., Ye, Q. and Liu, T.Y. (2017).
           **LightGBM: A Highly Efficient Gradient Boosting Decision Tree.**
           *Advances in Neural Information Processing Systems (NeurIPS)*, 30,
           pp. 3146-3154.

    .. [Friedman2001] Friedman, J.H. (2001).
           **Greedy Function Approximation: A Gradient Boosting Machine.**
           *The Annals of Statistics*, 29(5), pp. 1189-1232.
           DOI: `10.1214/aos/1013203451 <https://doi.org/10.1214/aos/1013203451>`_

    See Also
    --------
    :class:`~tuiml.algorithms.gradient_boosting.LightGBMRegressor` : LightGBM for regression tasks.
    :class:`~tuiml.algorithms.gradient_boosting.XGBoostClassifier` : XGBoost classifier with second-order optimization.
    :class:`~tuiml.algorithms.gradient_boosting.CatBoostClassifier` : CatBoost classifier with native categorical support.

    Examples
    --------
    Train a LightGBM classifier on a binary classification task:

    >>> from tuiml.algorithms.gradient_boosting import LightGBMClassifier
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y_train = np.array([0, 0, 1, 1])
    >>> clf = LightGBMClassifier(n_estimators=100, num_leaves=31)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_train)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_child_samples: int = 20,
        min_split_gain: float = 0.0,
        verbose: int = -1,
        random_state: Optional[int] = None
    ):
        """Initialize LightGBMClassifier.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting iterations.
        max_depth : int, default=-1
            Maximum tree depth. -1 means no limit.
        learning_rate : float, default=0.1
            Step size shrinkage.
        num_leaves : int, default=31
            Maximum number of leaves in one tree.
        subsample : float, default=1.0
            Fraction of samples per tree.
        colsample_bytree : float, default=1.0
            Fraction of features per tree.
        reg_alpha : float, default=0.0
            L1 regularization term.
        reg_lambda : float, default=0.0
            L2 regularization term.
        min_child_samples : int, default=20
            Minimum samples required in a leaf.
        min_split_gain : float, default=0.0
            Minimum loss reduction for a partition.
        verbose : int, default=-1
            Verbosity level.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install it with: pip install lightgbm"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.min_split_gain = min_split_gain
        self.verbose = verbose
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None
        self.n_classes_ = None
        self._objective_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for algorithm parameters."""
        return {
            'n_estimators': {'type': int, 'default': 100, 'range': (1, 10000)},
            'max_depth': {'type': int, 'default': -1, 'range': (-1, 20)},
            'learning_rate': {'type': float, 'default': 0.1, 'range': (0.001, 1.0)},
            'num_leaves': {'type': int, 'default': 31, 'range': (2, 131072)},
            'subsample': {'type': float, 'default': 1.0, 'range': (0.1, 1.0)},
            'colsample_bytree': {'type': float, 'default': 1.0, 'range': (0.1, 1.0)},
            'reg_alpha': {'type': float, 'default': 0.0, 'range': (0.0, 10.0)},
            'reg_lambda': {'type': float, 'default': 0.0, 'range': (0.0, 10.0)},
            'min_child_samples': {'type': int, 'default': 20, 'range': (1, 1000)},
            'min_split_gain': {'type': float, 'default': 0.0, 'range': (0.0, 10.0)},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "O(n_estimators * n_samples * num_leaves)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Ke et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS."
        ]

    def _build_params(self, objective: str) -> Dict[str, Any]:
        """Build the native LightGBM parameter dict."""
        params: Dict[str, Any] = {
            'objective': objective,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.colsample_bytree,
            'bagging_fraction': self.subsample,
            'bagging_freq': 1 if self.subsample < 1.0 else 0,
            'lambda_l1': self.reg_alpha,
            'lambda_l2': self.reg_lambda,
            'min_data_in_leaf': self.min_child_samples,
            'min_gain_to_split': self.min_split_gain,
            'verbose': self.verbose,
        }
        if objective == 'multiclass':
            params['num_class'] = int(self.n_classes_)
        if self.random_state is not None:
            params['seed'] = int(self.random_state)
        return params

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMClassifier":
        """Fit the LightGBM classifier to training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.

        y : ndarray of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : LightGBMClassifier
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Encode labels to 0..n_classes-1
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([class_to_idx[c] for c in y], dtype=np.int32)

        objective = 'binary' if self.n_classes_ <= 2 else 'multiclass'
        params = self._build_params(objective)

        dtrain = lgb.Dataset(X, label=y_encoded, free_raw_data=False)
        self.model_ = lgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
        )
        self._objective_ = objective

        importances = self.model_.feature_importance(importance_type='gain').astype(float)
        total = importances.sum()
        if total > 0:
            importances = importances / total
        self.feature_importances_ = importances

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")

        proba = self.predict_proba(np.asarray(X))
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict probabilities for.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probability matrix.
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X)
        raw = self.model_.predict(X)
        if self._objective_ == 'binary':
            raw = np.asarray(raw).reshape(-1)
            return np.column_stack([1.0 - raw, raw])
        return np.asarray(raw)

    def __repr__(self) -> str:
        return (
            f"LightGBMClassifier(n_estimators={self.n_estimators}, "
            f"num_leaves={self.num_leaves}, learning_rate={self.learning_rate})"
        )

@regressor(tags=["gradient-boosting", "lightgbm", "distributed"], version="1.0.0")
class LightGBMRegressor(Regressor):
    """LightGBM regressor for distributed, high-performance **gradient boosting**
    on continuous targets.

    Implementation of the LightGBM regression algorithm using **leaf-wise tree
    growth** and **histogram-based** split finding for fast, memory-efficient
    training.

    Overview
    --------
    The regression variant follows the same leaf-wise boosting procedure:

    1. Initialize predictions with a constant value (e.g., mean of targets)
    2. For each boosting iteration, compute the **negative gradient** of the
       loss (e.g., squared error) for every training sample
    3. Build feature histograms using **GOSS** (gradient-based one-side
       sampling) and **EFB** (exclusive feature bundling)
    4. Grow the tree **leaf-wise** by splitting the leaf with the highest
       gain, controlled by ``num_leaves``
    5. Add the new tree to the ensemble, scaled by the learning rate
    6. Repeat until the specified number of boosting rounds is reached

    Theory
    ------
    For the default squared-error objective, the loss for sample :math:`i` is:

    .. math::
        l(y_i, \\hat{y}_i) = \\frac{1}{2}(y_i - \\hat{y}_i)^2

    The regularized objective at round :math:`t` is:

    .. math::
        \\mathcal{L}^{(t)} = \\sum_{i=1}^{n} l(y_i, \\hat{y}_i^{(t-1)} + f_t(x_i))
        + \\gamma T + \\frac{1}{2} \\lambda \\sum_{j=1}^{T} w_j^2
        + \\alpha \\sum_{j=1}^{T} |w_j|

    The optimal leaf weight for leaf :math:`j` is:

    .. math::
        w_j^* = -\\frac{\\sum_{i \\in I_j} g_i}{|I_j| + \\lambda}

    where :math:`g_i` is the gradient, :math:`\\lambda` is the L2 regularization
    term (``reg_lambda``), and :math:`\\alpha` is the L1 term (``reg_alpha``).

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations (trees to build).

    max_depth : int, default=-1
        Maximum tree depth. ``-1`` means no limit.

    learning_rate : float, default=0.1
        Step size shrinkage to prevent overfitting.

    num_leaves : int, default=31
        Maximum number of leaves in one tree.

    subsample : float, default=1.0
        Fraction of samples to randomly sample for each tree.

    colsample_bytree : float, default=1.0
        Fraction of features to randomly sample for each tree.

    reg_alpha : float, default=0.0
        L1 regularization term on weights.

    reg_lambda : float, default=0.0
        L2 regularization term on weights.

    min_child_samples : int, default=20
        Minimum number of samples required in a leaf node.

    min_split_gain : float, default=0.0
        Minimum loss reduction required to make a split.

    verbose : int, default=-1
        Verbosity level. ``-1``: Quiet, ``0``: Warnings, ``1``: Info.

    random_state : int, optional
        Seed for reproducibility.

    Attributes
    ----------
    model_ : lgb.LGBMRegressor
        The underlying fitted LightGBM regressor object.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot d \\cdot L)` where :math:`T` = n_estimators,
      :math:`n` = n_samples, :math:`d` = n_features, :math:`L` = num_leaves.
      GOSS and EFB reduce effective :math:`n` and :math:`d` in practice.
    - Prediction: :math:`O(T \\cdot L)` per sample

    **When to use LightGBMRegressor:**

    - **Large-scale** regression tasks where training speed is critical
    - High-dimensional datasets where **exclusive feature bundling** reduces cost
    - When memory efficiency is important (histogram-based approach)
    - Distributed training scenarios across multiple machines

    References
    ----------
    .. [Ke2017] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W.,
           Ma, W., Ye, Q. and Liu, T.Y. (2017).
           **LightGBM: A Highly Efficient Gradient Boosting Decision Tree.**
           *Advances in Neural Information Processing Systems (NeurIPS)*, 30,
           pp. 3146-3154.

    .. [Friedman2001] Friedman, J.H. (2001).
           **Greedy Function Approximation: A Gradient Boosting Machine.**
           *The Annals of Statistics*, 29(5), pp. 1189-1232.
           DOI: `10.1214/aos/1013203451 <https://doi.org/10.1214/aos/1013203451>`_

    See Also
    --------
    :class:`~tuiml.algorithms.gradient_boosting.LightGBMClassifier` : LightGBM for classification tasks.
    :class:`~tuiml.algorithms.gradient_boosting.XGBoostRegressor` : XGBoost regressor with second-order optimization.
    :class:`~tuiml.algorithms.gradient_boosting.CatBoostRegressor` : CatBoost regressor with native categorical support.

    Examples
    --------
    Train a LightGBM regressor on a simple regression task:

    >>> from tuiml.algorithms.gradient_boosting import LightGBMRegressor
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y_train = np.array([1.5, 3.5, 5.5, 7.5])
    >>> reg = LightGBMRegressor(n_estimators=100, learning_rate=0.05)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_train)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_child_samples: int = 20,
        min_split_gain: float = 0.0,
        verbose: int = -1,
        random_state: Optional[int] = None
    ):
        """Initialize LightGBMRegressor.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting iterations.
        max_depth : int, default=-1
            Maximum tree depth. -1 means no limit.
        learning_rate : float, default=0.1
            Step size shrinkage.
        num_leaves : int, default=31
            Maximum number of leaves in one tree.
        subsample : float, default=1.0
            Fraction of samples per tree.
        colsample_bytree : float, default=1.0
            Fraction of features per tree.
        reg_alpha : float, default=0.0
            L1 regularization term.
        reg_lambda : float, default=0.0
            L2 regularization term.
        min_child_samples : int, default=20
            Minimum samples required in a leaf.
        min_split_gain : float, default=0.0
            Minimum loss reduction for a split.
        verbose : int, default=-1
            Verbosity level.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install it with: pip install lightgbm"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.min_split_gain = min_split_gain
        self.verbose = verbose
        self.random_state = random_state

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for algorithm parameters."""
        return {
            'n_estimators': {'type': int, 'default': 100, 'range': (1, 10000)},
            'max_depth': {'type': int, 'default': -1, 'range': (-1, 20)},
            'learning_rate': {'type': float, 'default': 0.1, 'range': (0.001, 1.0)},
            'num_leaves': {'type': int, 'default': 31, 'range': (2, 131072)},
            'subsample': {'type': float, 'default': 1.0, 'range': (0.1, 1.0)},
            'colsample_bytree': {'type': float, 'default': 1.0, 'range': (0.1, 1.0)},
            'reg_alpha': {'type': float, 'default': 0.0, 'range': (0.0, 10.0)},
            'reg_lambda': {'type': float, 'default': 0.0, 'range': (0.0, 10.0)},
            'min_child_samples': {'type': int, 'default': 20, 'range': (1, 1000)},
            'min_split_gain': {'type': float, 'default': 0.0, 'range': (0.0, 10.0)},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "nominal", "missing_values"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "O(n_estimators * n_samples * num_leaves)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Ke et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMRegressor":
        """Fit the LightGBM regressor to training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.

        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LightGBMRegressor
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)

        params: Dict[str, Any] = {
            'objective': 'regression',
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.colsample_bytree,
            'bagging_fraction': self.subsample,
            'bagging_freq': 1 if self.subsample < 1.0 else 0,
            'lambda_l1': self.reg_alpha,
            'lambda_l2': self.reg_lambda,
            'min_data_in_leaf': self.min_child_samples,
            'min_gain_to_split': self.min_split_gain,
            'verbose': self.verbose,
        }
        if self.random_state is not None:
            params['seed'] = int(self.random_state)

        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        self.model_ = lgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
        )

        importances = self.model_.feature_importance(importance_type='gain').astype(float)
        total = importances.sum()
        if total > 0:
            importances = importances / total
        self.feature_importances_ = importances

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X)
        return np.asarray(self.model_.predict(X))

    def __repr__(self) -> str:
        return (
            f"LightGBMRegressor(n_estimators={self.n_estimators}, "
            f"num_leaves={self.num_leaves}, learning_rate={self.learning_rate})"
        )
