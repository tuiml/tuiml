"""Ordinary Least Squares (OLS) linear regression with ridge regularization."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Regressor, regressor

@regressor(tags=["linear", "regression", "interpretable"], version="1.0.0")
class LinearRegression(Regressor):
    r"""Linear Regression for predicting **continuous values** using ordinary least squares.

    Implements **ordinary least squares (OLS)** linear regression with optional
    **ridge regularization** (L2) and automatic **feature selection** methods
    including M5 stepwise elimination and greedy backward elimination.

    Overview
    --------
    The algorithm fits a linear model through the following steps:

    1. Handle missing values by replacing with column means
    2. Standardize features to zero mean and unit variance
    3. Optionally eliminate highly colinear features (correlation > 0.99)
    4. Optionally perform feature selection (M5 or greedy backward elimination)
    5. Solve for coefficients using least squares with ridge regularization
    6. Transform coefficients back to the original feature scale

    Theory
    ------
    The model fits a linear equation relating features to the target:

    .. math::
        y = X\\beta + \epsilon

    where :math:`\\beta` is the coefficient vector and :math:`\epsilon` is the
    error term. The coefficients are estimated by minimizing the **regularized
    sum of squared residuals**:

    .. math::
        \hat{\\beta} = \\arg\min_{\\beta} \|y - X\\beta\|_2^2 + \lambda \|\\beta\|_2^2

    The closed-form solution is:

    .. math::
        \hat{\\beta} = (X^T X + \lambda I)^{-1} X^T y

    For numerical stability, the implementation uses ``np.linalg.lstsq`` with
    an augmented matrix rather than explicitly forming the Gram matrix.

    Parameters
    ----------
    ridge : float, default=1e-8
        Ridge regularization parameter (L2 penalty). A small value helps
        prevent numerical instability in matrix inversion.
    attribute_selection : {"none", "m5", "greedy"}, default="none"
        Feature selection method to use during fitting:

        - ``"none"`` -- Use all features.
        - ``"m5"`` -- M5 method (stepwise backward elimination based on AIC).
        - ``"greedy"`` -- Greedy backward elimination based on coefficient significance.
    eliminate_colinear : bool, default=True
        Whether to remove highly colinear attributes before fitting.
    fit_intercept : bool, default=True
        Whether to fit an intercept (bias) term.

    Attributes
    ----------
    coefficients_ : np.ndarray
        Regression coefficients of shape (n_features,).
    intercept_ : float
        Intercept term.
    selected_features_ : np.ndarray
        Indices of the features selected for the final model.
    std_devs_ : np.ndarray
        Standard deviations of features used for scaling.
    means_ : np.ndarray
        Means of features used for scaling.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot m^2 + m^3)` for :math:`n` samples and :math:`m` features.
    - Prediction: :math:`O(m)` per sample.

    **When to use LinearRegression:**

    - When the relationship between features and target is approximately linear
    - When you need an interpretable model with explicit feature coefficients
    - When the number of features is moderate relative to the number of samples
    - As a baseline for comparison with more complex regression methods

    References
    ----------
    .. [Akaike1974] Akaike, H. (1974).
           **A new look at the statistical model identification.**
           *IEEE Transactions on Automatic Control*, 19(6), 716-723.

    .. [HoerlKennard1970] Hoerl, A.E. and Kennard, R.W. (1970).
           **Ridge Regression: Biased Estimation for Nonorthogonal Problems.**
           *Technometrics*, 12(1), 55-67.

    See Also
    --------
    :class:`~tuiml.algorithms.linear.SimpleLinearRegression` : Univariate linear regression using the single best attribute.
    :class:`~tuiml.algorithms.linear.SGDRegressor` : SGD-trained linear regressor for large-scale datasets.

    Examples
    --------
    Basic regression with automatic feature handling:

    >>> import numpy as np
    >>> from tuiml.algorithms.linear import LinearRegression
    >>>
    >>> # Generating sample data
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([2, 4, 6, 8])
    >>>
    >>> # Fit the model
    >>> reg = LinearRegression()
    >>> reg.fit(X, y)
    >>>
    >>> # Predict
    >>> reg.predict([[5]])
    array([10.])
    """

    def __init__(
        self,
        ridge: float = 1e-8,
        attribute_selection: str = "none",
        eliminate_colinear: bool = True,
        fit_intercept: bool = True,
    ):
        """Initialize LinearRegression with configuration parameters.

        Parameters
        ----------
        ridge : float, default=1e-8
            Ridge parameter for regularization.
        attribute_selection : {"none", "m5", "greedy"}, default="none"
            Feature selection method.
        eliminate_colinear : bool, default=True
            Whether to eliminate colinear features.
        fit_intercept : bool, default=True
            Whether to fit intercept.
        """
        super().__init__()
        self.ridge = ridge
        self.attribute_selection = attribute_selection
        self.eliminate_colinear = eliminate_colinear
        self.fit_intercept = fit_intercept

        # Fitted attributes
        self.coefficients_ = None
        self.intercept_ = None
        self.selected_features_ = None
        self.std_devs_ = None
        self.means_ = None
        self.y_mean_ = None
        self.n_features_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "ridge": {
                "type": "number",
                "default": 1e-8,
                "minimum": 0,
                "description": "Ridge regularization parameter"
            },
            "attribute_selection": {
                "type": "string",
                "default": "none",
                "enum": ["none", "m5", "greedy"],
                "description": "Feature selection method"
            },
            "eliminate_colinear": {
                "type": "boolean",
                "default": True,
                "description": "Remove colinear attributes"
            },
            "fit_intercept": {
                "type": "boolean",
                "default": True,
                "description": "Whether to fit intercept term"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return [
            "numeric",
            "missing_values",
            "numeric_class"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m^2 + m^3) for n samples, m features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Akaike, H. (1974). A new look at the statistical model identification. "
            "IEEE Transactions on Automatic Control, 19(6), 716-723."
        ]

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Standardize features to zero mean and unit variance.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to standardize.
        fit : bool, default=False
            If True, compute and store mean and standard deviation.

        Returns
        -------
        X_std : np.ndarray of shape (n_samples, n_features)
            Standardized feature matrix.
        """
        if fit:
            self.means_ = np.nanmean(X, axis=0)
            self.std_devs_ = np.nanstd(X, axis=0)
            # Avoid division by zero
            self.std_devs_[self.std_devs_ == 0] = 1.0

        X_std = (X - self.means_) / self.std_devs_
        return X_std

    def _handle_missing(self, X: np.ndarray) -> np.ndarray:
        """Replace missing values with column means.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix potentially containing NaN values.

        Returns
        -------
        X_filled : np.ndarray of shape (n_samples, n_features)
            Feature matrix with NaN values replaced by column means.
        """
        X = X.copy()
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        return X

    def _compute_aic(self, X: np.ndarray, y: np.ndarray,
                     coeffs: np.ndarray, n_params: int) -> float:
        """Compute Akaike Information Criterion.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.
        coeffs : np.ndarray of shape (n_features,)
            Regression coefficients.
        n_params : int
            Number of model parameters (including intercept).

        Returns
        -------
        aic : float
            AIC value; lower is better.
        """
        n = len(y)
        y_pred = X @ coeffs
        residuals = y - y_pred
        sse = np.sum(residuals ** 2)

        if sse <= 0:
            return float('inf')

        # AIC = n * log(SSE/n) + 2 * k
        aic = n * np.log(sse / n) + 2 * n_params
        return aic

    def _eliminate_colinear_features(self, X: np.ndarray) -> np.ndarray:
        """Remove colinear features using correlation analysis.

        Uses vectorized correlation matrix computation instead of
        pairwise loops for O(n) speedup.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Standardized feature matrix.

        Returns
        -------
        keep_indices : np.ndarray
            Indices of non-colinear features to retain.
        """
        n_features = X.shape[1]

        # Single feature - nothing to eliminate
        if n_features <= 1:
            return np.arange(n_features)

        # Compute full correlation matrix once (vectorized)
        corr_matrix = np.corrcoef(X.T)

        # Handle NaN values (constant features)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Find highly correlated pairs
        keep = np.ones(n_features, dtype=bool)
        for i in range(n_features):
            if not keep[i]:
                continue
            # Check all j > i in one vectorized operation
            high_corr = np.abs(corr_matrix[i, i+1:]) > 0.99
            # Mark correlated features for removal
            indices_to_remove = np.where(high_corr)[0] + i + 1
            keep[indices_to_remove] = False

        return np.where(keep)[0]

    def _select_features_m5(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """M5 method for feature selection using AIC.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Standardized feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        selected : np.ndarray
            Indices of selected features after backward elimination.
        """
        n_samples, n_features = X.shape
        selected = list(range(n_features))

        while len(selected) > 1:
            # Fit model with current features
            X_sel = X[:, selected]
            coeffs = self._fit_ols(X_sel, y)
            current_aic = self._compute_aic(X_sel, y, coeffs, len(selected) + 1)

            best_aic = current_aic
            worst_feature = -1

            # Try removing each feature
            for i, feat_idx in enumerate(selected):
                test_selected = [s for j, s in enumerate(selected) if j != i]
                if len(test_selected) == 0:
                    continue

                X_test = X[:, test_selected]
                test_coeffs = self._fit_ols(X_test, y)
                test_aic = self._compute_aic(X_test, y, test_coeffs, len(test_selected) + 1)

                if test_aic < best_aic:
                    best_aic = test_aic
                    worst_feature = i

            if worst_feature == -1:
                break

            selected.pop(worst_feature)

        return np.array(selected)

    def _select_features_greedy(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Greedy backward elimination based on coefficient significance.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Standardized feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        selected : np.ndarray
            Indices of selected features after backward elimination.
        """
        n_samples, n_features = X.shape
        selected = list(range(n_features))

        while len(selected) > 1:
            X_sel = X[:, selected]
            coeffs = self._fit_ols(X_sel, y)

            # Find feature with smallest absolute coefficient
            min_idx = np.argmin(np.abs(coeffs))
            min_coeff = np.abs(coeffs[min_idx])

            # Stop if all coefficients are significant
            if min_coeff > 1e-6:
                break

            selected.pop(min_idx)

        return np.array(selected)

    def _fit_ols(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit ordinary least squares with ridge regularization.

        Uses ``np.linalg.lstsq`` directly for better numerical stability
        and performance (avoids forming the Gram matrix :math:`X^T X`).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        coeffs : np.ndarray of shape (n_features,)
            Fitted regression coefficients.
        """
        n_samples, n_features = X.shape

        if self.ridge > 0:
            # For ridge regression, augment X with sqrt(ridge) * I
            # This is equivalent to (X.T @ X + ridge * I)^-1 @ X.T @ y
            # but more numerically stable
            X_aug = np.vstack([X, np.sqrt(self.ridge) * np.eye(n_features)])
            y_aug = np.concatenate([y, np.zeros(n_features)])
            coeffs = np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]
        else:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

        return coeffs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit the Linear Regression model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training target values.

        Returns
        -------
        self : LinearRegression
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, self.n_features_ = X.shape

        # Handle missing values
        X = self._handle_missing(X)

        # Store target mean for intercept calculation
        self.y_mean_ = np.mean(y)

        # Standardize features
        X_std = self._standardize(X, fit=True)

        # Eliminate colinear features if requested
        if self.eliminate_colinear:
            non_colinear = self._eliminate_colinear_features(X_std)
            X_std = X_std[:, non_colinear]
            feature_map = non_colinear
        else:
            feature_map = np.arange(self.n_features_)

        # Feature selection
        if self.attribute_selection == "m5":
            selected = self._select_features_m5(X_std, y)
        elif self.attribute_selection == "greedy":
            selected = self._select_features_greedy(X_std, y)
        else:
            selected = np.arange(X_std.shape[1])

        # Map back to original feature indices
        self.selected_features_ = feature_map[selected]
        X_selected = X_std[:, selected]

        # Fit final model
        coeffs_std = self._fit_ols(X_selected, y - self.y_mean_)

        # Convert coefficients back to original scale
        self.coefficients_ = np.zeros(self.n_features_)
        for i, feat_idx in enumerate(self.selected_features_):
            self.coefficients_[feat_idx] = coeffs_std[i] / self.std_devs_[feat_idx]

        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = self.y_mean_ - np.sum(self.coefficients_ * self.means_)
        else:
            self.intercept_ = 0.0

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted continuous values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Handle missing values using training means
        X = X.copy()
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.means_, inds[1])

        # Predict
        predictions = X @ self.coefficients_ + self.intercept_

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the R-squared (coefficient of determination) score.

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
            n_selected = len(self.selected_features_)
            return (f"LinearRegression(ridge={self.ridge}, "
                    f"n_features={self.n_features_}, "
                    f"n_selected={n_selected})")
        return f"LinearRegression(ridge={self.ridge})"
