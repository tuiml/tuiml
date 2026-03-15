"""AdditiveRegression implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Type
from copy import deepcopy

from tuiml.base.algorithms import Regressor, regressor, Algorithm

@regressor(tags=["ensemble", "boosting", "regression"], version="1.0.0")
class AdditiveRegression(Regressor):
    """Additive Regression (Gradient Boosting for regression).

    Enhances regression performance by iteratively fitting base regressors 
    to the residuals of previous predictions.

    Parameters
    ----------
    base_regressor : Regressor, optional
        The base regressor to use. If None, a simple decision stump is used.
    n_estimators : int, default=10
        The number of boosting iterations.
    shrinkage : float, default=1.0
        Learning rate/shrinkage factor.
    minimize_absolute_error : bool, default=False
        Whether to use L1 loss (MAE) instead of L2 loss (MSE).

    Attributes
    ----------
    estimators_ : list of Regressor
        The collection of fitted base regressors.
    initial_prediction_ : float
        The initial prediction (mean or median of target values).

    Examples
    --------
    >>> from tuiml.algorithms.ensemble import AdditiveRegression
    >>> reg = AdditiveRegression(n_estimators=50, shrinkage=0.1)
    >>> reg.fit(X_train, y_train)
    AdditiveRegression(...)
    >>> predictions = reg.predict(X_test)

    References
    ----------
    .. [1] Friedman, J. H. (2001). Greedy function approximation: A gradient 
           boosting machine. Annals of Statistics, 29(5), 1189-1232.
    """

    def __init__(
        self,
        base_regressor: Optional[Regressor] = None,
        n_estimators: int = 10,
        shrinkage: float = 1.0,
        minimize_absolute_error: bool = False,
    ):
        """
        Initialize AdditiveRegression.

        Args:
            base_regressor: Base regressor instance
            n_estimators: Number of boosting iterations
            shrinkage: Learning rate
            minimize_absolute_error: Use L1 loss
        """
        super().__init__()
        self.base_regressor = base_regressor
        self.n_estimators = n_estimators
        self.shrinkage = shrinkage
        self.minimize_absolute_error = minimize_absolute_error

        # Fitted attributes
        self.estimators_ = None
        self.initial_prediction_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_estimators": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "description": "Number of boosting iterations"
            },
            "shrinkage": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "maximum": 1,
                "description": "Learning rate (shrinkage factor)"
            },
            "minimize_absolute_error": {
                "type": "boolean",
                "default": False,
                "description": "Use L1 loss instead of L2"
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
        return "O(n_estimators * base_regressor_complexity)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Friedman, J. H. (2001). Greedy function approximation: A gradient "
            "boosting machine. Annals of Statistics, 29(5), 1189-1232."
        ]

    def _create_base_regressor(self) -> Regressor:
        """Create a new instance of the base regressor."""
        if self.base_regressor is not None:
            return deepcopy(self.base_regressor)

        # Default: Simple decision stump regressor
        return _SimpleStumpRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdditiveRegression":
        """Fit the AdditiveRegression model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : AdditiveRegression
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(y)

        # Initialize with mean (or median for L1)
        if self.minimize_absolute_error:
            self.initial_prediction_ = np.median(y)
        else:
            self.initial_prediction_ = np.mean(y)

        # Current predictions
        predictions = np.full(n_samples, self.initial_prediction_)

        # Fit boosting iterations
        self.estimators_ = []

        for _ in range(self.n_estimators):
            # Compute residuals (negative gradient of loss)
            if self.minimize_absolute_error:
                # L1 loss: gradient is sign of residual
                residuals = np.sign(y - predictions)
            else:
                # L2 loss: gradient is residual
                residuals = y - predictions

            # Fit base regressor to residuals
            regressor = self._create_base_regressor()
            regressor.fit(X, residuals)

            # Update predictions
            update = regressor.predict(X)
            predictions += self.shrinkage * update

            self.estimators_.append(regressor)

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

        # Start with initial prediction
        predictions = np.full(n_samples, self.initial_prediction_)

        # Add contributions from each estimator
        for estimator in self.estimators_:
            predictions += self.shrinkage * estimator.predict(X)

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

    def staged_predict(self, X: np.ndarray):
        """Predict at each boosting iteration.

        Yields predictions after each estimator is added.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Yields
        ------
        y_pred : np.ndarray of shape (n_samples,)
            Predictions at each stage.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.initial_prediction_)

        for estimator in self.estimators_:
            predictions = predictions + self.shrinkage * estimator.predict(X)
            yield predictions.copy()

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"AdditiveRegression(n_estimators={len(self.estimators_)}, "
                    f"shrinkage={self.shrinkage})")
        return f"AdditiveRegression(n_estimators={self.n_estimators})"

class _SimpleStumpRegressor:
    """
    Simple decision stump regressor for use as default base estimator.

    Finds the best single split and predicts mean in each region.
    """

    def __init__(self):
        self._is_fitted = False
        self.split_attr = -1
        self.split_value = 0.0
        self.left_pred = 0.0
        self.right_pred = 0.0
        self.default_pred = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SimpleStumpRegressor":
        """Fit the stump regressor."""
        n_samples, n_features = X.shape
        self.default_pred = np.mean(y)

        best_mse = float('inf')
        best_attr = -1
        best_value = 0.0
        best_left_pred = self.default_pred
        best_right_pred = self.default_pred

        for attr in range(n_features):
            values = np.unique(X[:, attr])
            values = values[~np.isnan(values)]

            if len(values) < 2:
                continue

            for i in range(len(values) - 1):
                split_value = (values[i] + values[i + 1]) / 2

                left_mask = X[:, attr] <= split_value
                right_mask = ~left_mask

                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue

                left_pred = np.mean(y[left_mask])
                right_pred = np.mean(y[right_mask])

                # Compute MSE
                mse = (np.sum((y[left_mask] - left_pred) ** 2) +
                       np.sum((y[right_mask] - right_pred) ** 2))

                if mse < best_mse:
                    best_mse = mse
                    best_attr = attr
                    best_value = split_value
                    best_left_pred = left_pred
                    best_right_pred = right_pred

        self.split_attr = best_attr
        self.split_value = best_value
        self.left_pred = best_left_pred
        self.right_pred = best_right_pred
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the stump."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.default_pred)

        if self.split_attr >= 0:
            left_mask = X[:, self.split_attr] <= self.split_value
            predictions[left_mask] = self.left_pred
            predictions[~left_mask] = self.right_pred

        return predictions
