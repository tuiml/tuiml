"""XGBoost (eXtreme Gradient Boosting) implementation."""

import numpy as np
from typing import Dict, List, Any, Optional
import warnings

from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBOOST_AVAILABLE = False

@classifier(tags=["gradient-boosting", "xgboost", "ensemble"], version="1.0.0")
class XGBoostClassifier(Classifier):
    """XGBoost classifier for high-performance **gradient boosting**.

    XGBoost (eXtreme Gradient Boosting) uses a **regularized gradient boosting**
    framework to build an ensemble of decision trees with **sparsity awareness**
    and cache-aware access patterns to achieve state-of-the-art results.

    Overview
    --------
    The algorithm builds an additive ensemble of decision trees:

    1. Initialize the model with a constant prediction (e.g., log-odds for classification)
    2. For each boosting round, compute the **negative gradient** (pseudo-residuals) and
       **second-order Hessian** of the loss function for every training sample
    3. Fit a new regression tree to the negative gradient using an
       approximate split-finding algorithm with **weighted quantile sketch**
    4. Prune the tree using the :math:`\\gamma` (minimum loss reduction) threshold
    5. Add the new tree to the ensemble, scaled by the **learning rate** :math:`\\eta`
    6. Repeat until the specified number of boosting rounds is reached

    Theory
    ------
    At boosting round :math:`t`, XGBoost minimizes the regularized objective:

    .. math::
        \\mathcal{L}^{(t)} = \\sum_{i=1}^{n} l(y_i, \\hat{y}_i^{(t-1)} + f_t(x_i))
        + \\Omega(f_t)

    where the regularization term is:

    .. math::
        \\Omega(f) = \\gamma T + \\frac{1}{2} \\lambda \\sum_{j=1}^{T} w_j^2
        + \\alpha \\sum_{j=1}^{T} |w_j|

    Using a second-order Taylor expansion, the objective becomes:

    .. math::
        \\tilde{\\mathcal{L}}^{(t)} \\approx \\sum_{i=1}^{n}
        \\left[ g_i f_t(x_i) + \\frac{1}{2} h_i f_t^2(x_i) \\right] + \\Omega(f_t)

    where :math:`g_i = \\partial_{\\hat{y}} l(y_i, \\hat{y}_i^{(t-1)})` and
    :math:`h_i = \\partial^2_{\\hat{y}} l(y_i, \\hat{y}_i^{(t-1)})` are the
    first and second order gradients. The optimal leaf weight for leaf :math:`j` is:

    .. math::
        w_j^* = -\\frac{\\sum_{i \\in I_j} g_i}{\\sum_{i \\in I_j} h_i + \\lambda}

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds (trees to build).

    max_depth : int, default=6
        Maximum tree depth for base learners.

    learning_rate : float, default=0.3
        Step size shrinkage used in update to prevent overfitting (eta).

    subsample : float, default=1.0
        Fraction of training samples to randomly sample for each tree.

    colsample_bytree : float, default=1.0
        Fraction of columns to randomly sample for each tree.

    min_child_weight : float, default=1.0
        Minimum sum of instance weight (hessian) needed in a child.

    gamma : float, default=0.0
        Minimum loss reduction required to make a further partition.

    reg_alpha : float, default=0.0
        L1 regularization term on weights.

    reg_lambda : float, default=1.0
        L2 regularization term on weights.

    objective : str, default="binary:logistic"
        Learning objective. Automatically switches to ``"multi:softprob"``
        for multi-class tasks.

    random_state : int, optional
        Seed for reproducibility.

    Attributes
    ----------
    model_ : xgb.XGBClassifier
        The underlying fitted XGBoost model object.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels discovered during ``fit()``.

    n_classes_ : int
        Number of unique classes discovered during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot d \\cdot D)` where :math:`T` = n_estimators,
      :math:`n` = n_samples, :math:`d` = n_features, :math:`D` = max_depth
    - Prediction: :math:`O(T \\cdot D)` per sample

    **When to use XGBoostClassifier:**

    - Structured / tabular classification tasks with moderate-to-large datasets
    - When you need built-in handling of **missing values**
    - Competitions and benchmarks where predictive accuracy is paramount
    - When L1 and L2 regularization are needed to control model complexity
    - Datasets that benefit from second-order gradient optimization

    References
    ----------
    .. [Chen2016] Chen, T. and Guestrin, C. (2016).
           **XGBoost: A Scalable Tree Boosting System.**
           *Proceedings of the 22nd ACM SIGKDD International Conference on
           Knowledge Discovery and Data Mining*, pp. 785-794.
           DOI: `10.1145/2939672.2939785 <https://doi.org/10.1145/2939672.2939785>`_

    .. [Friedman2001] Friedman, J.H. (2001).
           **Greedy Function Approximation: A Gradient Boosting Machine.**
           *The Annals of Statistics*, 29(5), pp. 1189-1232.
           DOI: `10.1214/aos/1013203451 <https://doi.org/10.1214/aos/1013203451>`_

    See Also
    --------
    :class:`~tuiml.algorithms.gradient_boosting.XGBoostRegressor` : XGBoost for regression tasks.
    :class:`~tuiml.algorithms.gradient_boosting.LightGBMClassifier` : LightGBM classifier with leaf-wise growth.
    :class:`~tuiml.algorithms.gradient_boosting.CatBoostClassifier` : CatBoost classifier with native categorical support.

    Examples
    --------
    Train an XGBoost classifier on a binary classification task:

    >>> from tuiml.algorithms.gradient_boosting import XGBoostClassifier
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y_train = np.array([0, 0, 1, 1])
    >>> clf = XGBoostClassifier(n_estimators=100, learning_rate=0.1)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_train)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: str = 'binary:logistic',
        random_state: Optional[int] = None
    ):
        """Initialize XGBoostClassifier.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting rounds.
        max_depth : int, default=6
            Maximum tree depth for base learners.
        learning_rate : float, default=0.3
            Step size shrinkage (eta).
        subsample : float, default=1.0
            Fraction of training samples per tree.
        colsample_bytree : float, default=1.0
            Fraction of columns per tree.
        min_child_weight : float, default=1.0
            Minimum sum of instance weight in a child.
        gamma : float, default=0.0
            Minimum loss reduction for a partition.
        reg_alpha : float, default=0.0
            L1 regularization term.
        reg_lambda : float, default=1.0
            L2 regularization term.
        objective : str, default='binary:logistic'
            Learning objective.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()

        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None
        self.n_classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for algorithm parameters."""
        return {
            'n_estimators': {'type': int, 'default': 100, 'range': (1, 1000)},
            'max_depth': {'type': int, 'default': 6, 'range': (1, 20)},
            'learning_rate': {'type': float, 'default': 0.3, 'range': (0.01, 1.0)},
            'subsample': {'type': float, 'default': 1.0, 'range': (0.1, 1.0)},
            'colsample_bytree': {'type': float, 'default': 1.0, 'range': (0.1, 1.0)},
            'min_child_weight': {'type': float, 'default': 1.0, 'range': (0.0, 10.0)},
            'gamma': {'type': float, 'default': 0.0, 'range': (0.0, 10.0)},
            'reg_alpha': {'type': float, 'default': 0.0, 'range': (0.0, 10.0)},
            'reg_lambda': {'type': float, 'default': 1.0, 'range': (0.0, 10.0)},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric", "nominal", "missing_values",
            "binary_class", "multiclass",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "O(n_estimators * n_samples * n_features * max_depth)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        """Fit the XGBoost classifier to training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.

        y : ndarray of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : XGBoostClassifier
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Determine objective based on number of classes
        if self.objective == 'binary:logistic' and self.n_classes_ > 2:
            objective = 'multi:softprob'
        else:
            objective = self.objective

        # Initialize XGBoost classifier
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=objective,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        # Fit the model
        self.model_.fit(X, y)
        self.feature_importances_ = self.model_.feature_importances_

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
        
        X = np.asarray(X)
        return self.model_.predict(X)

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
        return self.model_.predict_proba(X)

    def __repr__(self) -> str:
        return (
            f"XGBoostClassifier(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, learning_rate={self.learning_rate})"
        )

@regressor(tags=["gradient-boosting", "xgboost", "ensemble"], version="1.0.0")
class XGBoostRegressor(Regressor):
    """XGBoost regressor for high-performance **gradient boosting** on continuous targets.

    Implementation of the XGBoost regression algorithm using **regularized
    gradient boosted decision trees** with second-order optimization.

    Overview
    --------
    The regression variant follows the same additive training procedure:

    1. Initialize predictions with a constant value (e.g., mean of targets)
    2. For each boosting round, compute the **gradient** and **Hessian** of the
       squared-error (or custom) loss for every training sample
    3. Fit a regression tree to the negative gradient using approximate
       split finding with **weighted quantile sketch**
    4. Prune the tree using the :math:`\\gamma` threshold and regularization
    5. Update predictions by adding the new tree scaled by learning rate :math:`\\eta`
    6. Repeat until all boosting rounds are completed

    Theory
    ------
    For the default squared-error objective, the loss for sample :math:`i` is:

    .. math::
        l(y_i, \\hat{y}_i) = \\frac{1}{2}(y_i - \\hat{y}_i)^2

    The gradients are :math:`g_i = \\hat{y}_i - y_i` and :math:`h_i = 1`.
    The regularized objective at round :math:`t` is:

    .. math::
        \\mathcal{L}^{(t)} = \\sum_{i=1}^{n} l(y_i, \\hat{y}_i^{(t-1)} + f_t(x_i))
        + \\gamma T + \\frac{1}{2} \\lambda \\sum_{j=1}^{T} w_j^2
        + \\alpha \\sum_{j=1}^{T} |w_j|

    where :math:`T` is the number of leaves, :math:`w_j` are leaf weights,
    :math:`\\lambda` is L2 regularization, and :math:`\\alpha` is L1 regularization.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds (trees to build).

    max_depth : int, default=6
        Maximum tree depth for base learners.

    learning_rate : float, default=0.3
        Step size shrinkage (eta) to prevent overfitting.

    subsample : float, default=1.0
        Fraction of training samples to randomly sample for each tree.

    colsample_bytree : float, default=1.0
        Fraction of columns to randomly sample for each tree.

    min_child_weight : float, default=1.0
        Minimum sum of instance weight needed in a child.

    gamma : float, default=0.0
        Minimum loss reduction required to make a split.

    reg_alpha : float, default=0.0
        L1 regularization term on weights.

    reg_lambda : float, default=1.0
        L2 regularization term on weights.

    objective : str, default="reg:squarederror"
        Regression learning objective.

    random_state : int, optional
        Seed for reproducibility.

    Attributes
    ----------
    model_ : xgb.XGBRegressor
        The underlying fitted XGBoost regressor object.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot d \\cdot D)` where :math:`T` = n_estimators,
      :math:`n` = n_samples, :math:`d` = n_features, :math:`D` = max_depth
    - Prediction: :math:`O(T \\cdot D)` per sample

    **When to use XGBoostRegressor:**

    - Structured / tabular regression tasks with moderate-to-large datasets
    - When the data contains **missing values** that should be handled natively
    - When L1 and L2 regularization are needed for controlling overfitting
    - Benchmarks where predictive accuracy on continuous targets is paramount

    References
    ----------
    .. [Chen2016] Chen, T. and Guestrin, C. (2016).
           **XGBoost: A Scalable Tree Boosting System.**
           *Proceedings of the 22nd ACM SIGKDD International Conference on
           Knowledge Discovery and Data Mining*, pp. 785-794.
           DOI: `10.1145/2939672.2939785 <https://doi.org/10.1145/2939672.2939785>`_

    .. [Friedman2001] Friedman, J.H. (2001).
           **Greedy Function Approximation: A Gradient Boosting Machine.**
           *The Annals of Statistics*, 29(5), pp. 1189-1232.
           DOI: `10.1214/aos/1013203451 <https://doi.org/10.1214/aos/1013203451>`_

    See Also
    --------
    :class:`~tuiml.algorithms.gradient_boosting.XGBoostClassifier` : XGBoost for classification tasks.
    :class:`~tuiml.algorithms.gradient_boosting.LightGBMRegressor` : LightGBM regressor with leaf-wise growth.
    :class:`~tuiml.algorithms.gradient_boosting.CatBoostRegressor` : CatBoost regressor with native categorical support.

    Examples
    --------
    Train an XGBoost regressor on a simple regression task:

    >>> from tuiml.algorithms.gradient_boosting import XGBoostRegressor
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y_train = np.array([1.5, 3.5, 5.5, 7.5])
    >>> reg = XGBoostRegressor(n_estimators=100, max_depth=5)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_train)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: str = 'reg:squarederror',
        random_state: Optional[int] = None
    ):
        """Initialize XGBoostRegressor.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting rounds.
        max_depth : int, default=6
            Maximum tree depth for base learners.
        learning_rate : float, default=0.3
            Step size shrinkage (eta).
        subsample : float, default=1.0
            Fraction of training samples per tree.
        colsample_bytree : float, default=1.0
            Fraction of columns per tree.
        min_child_weight : float, default=1.0
            Minimum sum of instance weight in a child.
        gamma : float, default=0.0
            Minimum loss reduction for a split.
        reg_alpha : float, default=0.0
            L1 regularization term.
        reg_lambda : float, default=1.0
            L2 regularization term.
        objective : str, default='reg:squarederror'
            Regression objective.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()

        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.random_state = random_state

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for algorithm parameters."""
        return {
            'n_estimators': {'type': int, 'default': 100, 'range': (1, 1000)},
            'max_depth': {'type': int, 'default': 6, 'range': (1, 20)},
            'learning_rate': {'type': float, 'default': 0.3, 'range': (0.01, 1.0)},
            'subsample': {'type': float, 'default': 1.0, 'range': (0.1, 1.0)},
            'colsample_bytree': {'type': float, 'default': 1.0, 'range': (0.1, 1.0)},
            'min_child_weight': {'type': float, 'default': 1.0, 'range': (0.0, 10.0)},
            'gamma': {'type': float, 'default': 0.0, 'range': (0.0, 10.0)},
            'reg_alpha': {'type': float, 'default': 0.0, 'range': (0.0, 10.0)},
            'reg_lambda': {'type': float, 'default': 1.0, 'range': (0.0, 10.0)},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "nominal", "missing_values"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "O(n_estimators * n_samples * n_features * max_depth)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        """Fit the XGBoost regressor to training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
            
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : XGBoostRegressor
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Initialize XGBoost regressor
        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            random_state=self.random_state
        )

        # Fit the model
        self.model_.fit(X, y)
        self.feature_importances_ = self.model_.feature_importances_

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
        return self.model_.predict(X)

    def __repr__(self) -> str:
        return (
            f"XGBoostRegressor(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, learning_rate={self.learning_rate})"
        )
