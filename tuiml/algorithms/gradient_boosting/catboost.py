"""CatBoost (Categorical Boosting) implementation."""

import numpy as np
from typing import Dict, List, Any, Optional
import warnings

from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except Exception:
    cb = None
    CATBOOST_AVAILABLE = False

@classifier(tags=["gradient-boosting", "catboost", "categorical"], version="1.0.0")
class CatBoostClassifier(Classifier):
    """CatBoost classifier with native support for **categorical features**.

    CatBoost is a **gradient boosting** algorithm that provides advanced handling
    of categorical features using **ordered target statistics**, robust
    regularization, and **ordered boosting** to reduce prediction shift.

    Overview
    --------
    The algorithm builds an ensemble of symmetric (oblivious) decision trees:

    1. Encode categorical features using **ordered target statistics** computed
       on a random permutation of the training data to avoid target leakage
    2. For each boosting iteration, compute the **negative gradient** of the
       loss function using the ordered boosting scheme
    3. Build a symmetric (**oblivious**) decision tree where all nodes at the
       same depth use the same split condition
    4. Compute optimal leaf values with **L2 regularization** on the leaf weights
    5. Add the new tree to the ensemble, scaled by the learning rate
    6. Repeat until the specified number of iterations is reached

    Theory
    ------
    CatBoost addresses **prediction shift** (a form of target leakage in
    gradient boosting) through ordered boosting. For each sample :math:`x_i`,
    the model :math:`M_i` used to compute gradients is trained only on
    samples appearing before :math:`x_i` in a random permutation :math:`\\sigma`.

    The ordered target statistic for a categorical feature value :math:`c` is:

    .. math::
        \\hat{x}_k^i = \\frac{\\sum_{j=1}^{p-1} [x_{\\sigma_j}^i = x_{\\sigma_p}^i]
        \\cdot y_{\\sigma_j} + a \\cdot P}{\\sum_{j=1}^{p-1}
        [x_{\\sigma_j}^i = x_{\\sigma_p}^i] + a}

    where :math:`a` is a prior weight and :math:`P` is the prior value.

    The regularized loss at iteration :math:`t` is:

    .. math::
        \\mathcal{L}^{(t)} = \\sum_{i=1}^{n} l(y_i, \\hat{y}_i^{(t-1)} + f_t(x_i))
        + \\frac{\\lambda}{2} \\sum_{j=1}^{T} w_j^2

    where :math:`\\lambda` is the L2 leaf regularization coefficient (``l2_leaf_reg``).

    Parameters
    ----------
    iterations : int, default=100
        Number of boosting iterations (trees to build).

    depth : int, default=6
        Depth of the decision trees.

    learning_rate : float, default=0.03
        Step size shrinkage to prevent overfitting.

    l2_leaf_reg : float, default=3.0
        L2 regularization coefficient for the leaves.

    border_count : int, default=128
        Number of discretization splits for numerical features.

    bagging_temperature : float, default=1.0
        Controls Bayesian bagging intensity. Higher values increase randomness.

    random_strength : float, default=1.0
        Randomness used for scoring splits.

    cat_features : list of int, optional
        Indices of categorical columns in the input data.

    verbose : bool, default=False
        Whether to print training progress and metrics.

    random_state : int, optional
        Seed for reproducibility.

    Attributes
    ----------
    model_ : cb.CatBoostClassifier
        The underlying fitted CatBoost model object.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels discovered during ``fit()``.

    n_classes_ : int
        Number of unique classes discovered during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot d \\cdot D)` where :math:`T` = iterations,
      :math:`n` = n_samples, :math:`d` = n_features, :math:`D` = depth
    - Prediction: :math:`O(T \\cdot D)` per sample (very fast due to oblivious trees)

    **When to use CatBoostClassifier:**

    - Datasets with **categorical features** that should not be one-hot encoded
    - When minimal hyperparameter tuning is desired (strong defaults)
    - When reducing **prediction shift** (target leakage) is important
    - Production systems where fast inference with oblivious trees is beneficial

    References
    ----------
    .. [Prokhorenkova2018] Prokhorenkova, L., Gusev, G., Vorobev, A.,
           Dorogush, A.V. and Gulin, A. (2018).
           **CatBoost: unbiased boosting with categorical features.**
           *Advances in Neural Information Processing Systems (NeurIPS)*, 31.

    .. [Dorogush2018] Dorogush, A.V., Ershov, V. and Gulin, A. (2018).
           **CatBoost: gradient boosting with categorical features support.**
           *arXiv preprint arXiv:1810.11363*.

    See Also
    --------
    :class:`~tuiml.algorithms.gradient_boosting.CatBoostRegressor` : CatBoost for regression tasks.
    :class:`~tuiml.algorithms.gradient_boosting.XGBoostClassifier` : XGBoost classifier with second-order optimization.
    :class:`~tuiml.algorithms.gradient_boosting.LightGBMClassifier` : LightGBM classifier with leaf-wise growth.

    Examples
    --------
    Train a CatBoost classifier with categorical feature support:

    >>> from tuiml.algorithms.gradient_boosting import CatBoostClassifier
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 0], [2, 1], [3, 0], [4, 1]])
    >>> y_train = np.array([0, 1, 0, 1])
    >>> clf = CatBoostClassifier(iterations=500, learning_rate=0.01)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_train)
    """

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 3.0,
        border_count: int = 128,
        bagging_temperature: float = 1.0,
        random_strength: float = 1.0,
        cat_features: Optional[List[int]] = None,
        verbose: bool = False,
        random_state: Optional[int] = None
    ):
        """Initialize CatBoostClassifier.

        Parameters
        ----------
        iterations : int, default=100
            Number of boosting iterations.
        depth : int, default=6
            Depth of the decision trees.
        learning_rate : float, default=0.03
            Step size shrinkage.
        l2_leaf_reg : float, default=3.0
            L2 regularization coefficient for leaves.
        border_count : int, default=128
            Number of discretization splits for numerical features.
        bagging_temperature : float, default=1.0
            Bayesian bagging intensity.
        random_strength : float, default=1.0
            Randomness for scoring splits.
        cat_features : list of int or None, default=None
            Indices of categorical columns.
        verbose : bool, default=False
            Whether to print training progress.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost is not installed. Install it with: pip install catboost"
            )

        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.bagging_temperature = bagging_temperature
        self.random_strength = random_strength
        self.cat_features = cat_features
        self.verbose = verbose
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None
        self.n_classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for algorithm parameters."""
        return {
            'iterations': {'type': int, 'default': 100, 'range': (1, 10000)},
            'depth': {'type': int, 'default': 6, 'range': (1, 16)},
            'learning_rate': {'type': float, 'default': 0.03, 'range': (0.001, 1.0)},
            'l2_leaf_reg': {'type': float, 'default': 3.0, 'range': (0.0, 10.0)},
            'border_count': {'type': int, 'default': 128, 'range': (1, 255)},
            'bagging_temperature': {'type': float, 'default': 1.0, 'range': (0.0, 10.0)},
            'random_strength': {'type': float, 'default': 1.0, 'range': (0.0, 10.0)},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "O(iterations * n_samples * n_features * depth)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Prokhorenkova et al. (2018). CatBoost: unbiased boosting with categorical features. NeurIPS."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostClassifier":
        """Fit the CatBoost classifier to training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.

        y : ndarray of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : CatBoostClassifier
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Initialize CatBoost classifier
        self.model_ = cb.CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            bagging_temperature=self.bagging_temperature,
            random_strength=self.random_strength,
            random_seed=self.random_state,
            verbose=self.verbose,
            allow_writing_files=False
        )

        # Fit the model
        self.model_.fit(X, y, cat_features=self.cat_features)
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
        return self.model_.predict(X).flatten()

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
            f"CatBoostClassifier(iterations={self.iterations}, "
            f"depth={self.depth}, learning_rate={self.learning_rate})"
        )

@regressor(tags=["gradient-boosting", "catboost", "categorical"], version="1.0.0")
class CatBoostRegressor(Regressor):
    """CatBoost regressor with native support for **categorical features**.

    Implementation of the CatBoost algorithm for regression tasks, using
    **ordered target statistics** and **oblivious decision trees** to handle
    categorical features efficiently without manual encoding.

    Overview
    --------
    The regression variant follows the same ordered boosting procedure:

    1. Encode categorical features using **ordered target statistics** to
       avoid target leakage during gradient computation
    2. For each iteration, compute the **negative gradient** of the loss
       (e.g., squared error) using ordered boosting
    3. Build a symmetric (**oblivious**) decision tree where all nodes at
       the same depth share the same split condition
    4. Compute optimal leaf values with **L2 regularization**
    5. Add the tree to the ensemble, scaled by the learning rate
    6. Repeat for the specified number of iterations

    Theory
    ------
    For the default RMSE objective, the loss for sample :math:`i` is:

    .. math::
        l(y_i, \\hat{y}_i) = \\frac{1}{2}(y_i - \\hat{y}_i)^2

    The ordered boosting scheme trains model :math:`M_i` on a prefix of a
    random permutation :math:`\\sigma` to compute the gradient for sample
    :math:`x_{\\sigma_i}`:

    .. math::
        g_i = \\frac{\\partial l(y_i, s)}{\\partial s}\\bigg|_{s=M_{\\sigma_i}(x_i)}

    The regularized leaf weight for leaf :math:`j` is:

    .. math::
        w_j^* = -\\frac{\\sum_{i \\in I_j} g_i}{|I_j| + \\lambda}

    where :math:`\\lambda` is the ``l2_leaf_reg`` parameter.

    Parameters
    ----------
    iterations : int, default=100
        Number of boosting iterations (trees to build).

    depth : int, default=6
        Depth of the decision trees.

    learning_rate : float, default=0.03
        Step size shrinkage to prevent overfitting.

    l2_leaf_reg : float, default=3.0
        L2 regularization coefficient for the leaves.

    border_count : int, default=128
        Number of discretization splits for numerical features.

    bagging_temperature : float, default=1.0
        Controls Bayesian bagging intensity.

    random_strength : float, default=1.0
        Randomness used for scoring splits.

    cat_features : list of int, optional
        Indices of categorical columns in the input data.

    verbose : bool, default=False
        Whether to print training progress and metrics.

    random_state : int, optional
        Seed for reproducibility.

    Attributes
    ----------
    model_ : cb.CatBoostRegressor
        The underlying fitted CatBoost regressor object.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T \\cdot n \\cdot d \\cdot D)` where :math:`T` = iterations,
      :math:`n` = n_samples, :math:`d` = n_features, :math:`D` = depth
    - Prediction: :math:`O(T \\cdot D)` per sample (very fast due to oblivious trees)

    **When to use CatBoostRegressor:**

    - Regression tasks with **categorical features** that should not be one-hot encoded
    - When minimal hyperparameter tuning is desired (strong defaults)
    - Datasets with mixed numerical and categorical features
    - When fast inference with oblivious trees is needed in production

    References
    ----------
    .. [Prokhorenkova2018] Prokhorenkova, L., Gusev, G., Vorobev, A.,
           Dorogush, A.V. and Gulin, A. (2018).
           **CatBoost: unbiased boosting with categorical features.**
           *Advances in Neural Information Processing Systems (NeurIPS)*, 31.

    .. [Dorogush2018] Dorogush, A.V., Ershov, V. and Gulin, A. (2018).
           **CatBoost: gradient boosting with categorical features support.**
           *arXiv preprint arXiv:1810.11363*.

    See Also
    --------
    :class:`~tuiml.algorithms.gradient_boosting.CatBoostClassifier` : CatBoost for classification tasks.
    :class:`~tuiml.algorithms.gradient_boosting.XGBoostRegressor` : XGBoost regressor with second-order optimization.
    :class:`~tuiml.algorithms.gradient_boosting.LightGBMRegressor` : LightGBM regressor with leaf-wise growth.

    Examples
    --------
    Train a CatBoost regressor with categorical feature support:

    >>> from tuiml.algorithms.gradient_boosting import CatBoostRegressor
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 0], [2, 1], [3, 0], [4, 1]])
    >>> y_train = np.array([1.5, 3.5, 2.5, 4.5])
    >>> reg = CatBoostRegressor(iterations=1000, depth=8)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_train)
    """

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 3.0,
        border_count: int = 128,
        bagging_temperature: float = 1.0,
        random_strength: float = 1.0,
        cat_features: Optional[List[int]] = None,
        verbose: bool = False,
        random_state: Optional[int] = None
    ):
        """Initialize CatBoostRegressor.

        Parameters
        ----------
        iterations : int, default=100
            Number of boosting iterations.
        depth : int, default=6
            Depth of the decision trees.
        learning_rate : float, default=0.03
            Step size shrinkage.
        l2_leaf_reg : float, default=3.0
            L2 regularization coefficient for leaves.
        border_count : int, default=128
            Number of discretization splits for numerical features.
        bagging_temperature : float, default=1.0
            Bayesian bagging intensity.
        random_strength : float, default=1.0
            Randomness for scoring splits.
        cat_features : list of int or None, default=None
            Indices of categorical columns.
        verbose : bool, default=False
            Whether to print training progress.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost is not installed. Install it with: pip install catboost"
            )

        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.bagging_temperature = bagging_temperature
        self.random_strength = random_strength
        self.cat_features = cat_features
        self.verbose = verbose
        self.random_state = random_state

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for algorithm parameters."""
        return {
            'iterations': {'type': int, 'default': 100, 'range': (1, 10000)},
            'depth': {'type': int, 'default': 6, 'range': (1, 16)},
            'learning_rate': {'type': float, 'default': 0.03, 'range': (0.001, 1.0)},
            'l2_leaf_reg': {'type': float, 'default': 3.0, 'range': (0.0, 10.0)},
            'border_count': {'type': int, 'default': 128, 'range': (1, 255)},
            'bagging_temperature': {'type': float, 'default': 1.0, 'range': (0.0, 10.0)},
            'random_strength': {'type': float, 'default': 1.0, 'range': (0.0, 10.0)},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "nominal", "missing_values"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "O(iterations * n_samples * n_features * depth)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Prokhorenkova et al. (2018). CatBoost: unbiased boosting with categorical features. NeurIPS."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostRegressor":
        """Fit the CatBoost regressor to training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.

        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : CatBoostRegressor
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Initialize CatBoost regressor
        self.model_ = cb.CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            bagging_temperature=self.bagging_temperature,
            random_strength=self.random_strength,
            random_seed=self.random_state,
            verbose=self.verbose,
            allow_writing_files=False
        )

        # Fit the model
        self.model_.fit(X, y, cat_features=self.cat_features)
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
            f"CatBoostRegressor(iterations={self.iterations}, "
            f"depth={self.depth}, learning_rate={self.learning_rate})"
        )
