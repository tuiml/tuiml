"""Univariate linear regression using the best single attribute."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Regressor, regressor

@regressor(tags=["linear", "regression", "simple", "interpretable"], version="1.0.0")
class SimpleLinearRegression(Regressor):
    r"""Simple Linear Regression using a **single best attribute**.

    Fits a **univariate linear model** by selecting the single feature that
    minimizes the squared error, making it highly **interpretable** and
    fast to compute.

    Overview
    --------
    The algorithm fits a simple line to the data:

    1. For each feature, compute the least-squares slope and intercept
    2. Evaluate the sum of squared errors for each candidate feature
    3. Select the feature with the lowest squared error (or use the user-specified attribute)
    4. Store the slope, intercept, and correlation for the selected feature

    Theory
    ------
    The model fits a univariate linear equation:

    .. math::
        y = \\alpha x + \\beta

    where :math:`\\alpha` is the slope and :math:`\\beta` is the intercept.
    The least-squares estimates are:

    .. math::
        \\alpha = \\frac{\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}
                      {\sum_{i=1}^{n} (x_i - \\bar{x})^2}

    .. math::
        \\beta = \\bar{y} - \\alpha \\bar{x}

    The Pearson correlation coefficient :math:`r` measures the linear
    association between the selected feature and the target.

    Parameters
    ----------
    attribute_index : int, default=-1
        The index of the attribute to use for the regression line.
        If set to -1, the algorithm will automatically select the
        attribute that provides the best fit on the training data.

    Attributes
    ----------
    slope_ : float
        The slope (:math:`\\alpha`) of the regression line.
    intercept_ : float
        The intercept (:math:`\\beta`) of the regression line.
    selected_attribute_ : int
        The index of the attribute used for the final model.
    correlation_ : float
        The correlation coefficient between the selected attribute
        and the target.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot m)` for :math:`n` samples and :math:`m` features.
    - Prediction: :math:`O(1)` per sample.

    **When to use SimpleLinearRegression:**

    - When you need the most interpretable possible regression model
    - As a fast baseline or feature-importance diagnostic
    - When only one feature is expected to drive the target
    - In ensemble methods that require simple base learners (e.g., LogitBoost)

    References
    ----------
    .. [Weisberg2005] Weisberg, S. (2005).
           **Applied Linear Statistical Models.**
           *Wiley*, 3rd edition.

    See Also
    --------
    :class:`~tuiml.algorithms.linear.LinearRegression` : Multivariate OLS regression with ridge regularization and feature selection.
    :class:`~tuiml.algorithms.linear.SGDRegressor` : SGD-trained linear regressor for large-scale datasets.

    Examples
    --------
    Auto-selecting the best single feature for regression:

    >>> import numpy as np
    >>> from tuiml.algorithms.linear import SimpleLinearRegression
    >>>
    >>> # Generating sample data with two features
    >>> X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
    >>> y = np.array([2, 4, 6, 8])  # y = 2 * X[:, 0]
    >>>
    >>> # Fit the model (auto-selects attribute index 0)
    >>> reg = SimpleLinearRegression()
    >>> reg.fit(X, y)
    >>>
    >>> # Predict
    >>> reg.predict([[5, 50]])
    array([10.])
    >>>
    >>> print(reg.get_equation())
    y = 2.000000 * x[0] + 0.000000
    """

    def __init__(self, attribute_index: int = -1):
        """Initialize SimpleLinearRegression.

        Parameters
        ----------
        attribute_index : int, default=-1
            Index of attribute to use (-1 for auto-select best).
        """
        super().__init__()
        self.attribute_index = attribute_index

        # Fitted attributes
        self.slope_ = None
        self.intercept_ = None
        self.selected_attribute_ = None
        self.correlation_ = None
        self.n_features_ = None
        self.x_mean_ = None
        self.y_mean_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "attribute_index": {
                "type": "integer",
                "default": -1,
                "minimum": -1,
                "description": "Attribute index to use (-1 for auto-select best)"
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
        return "O(n * m) for n samples, m features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Standard univariate linear regression using least squares method."
        ]

    def _fit_single_attribute(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Fit regression for a single attribute.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples,)
            Single feature values.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        slope : float
            Least-squares slope.
        intercept : float
            Least-squares intercept.
        squared_error : float
            Sum of squared residuals.
        correlation : float
            Pearson correlation coefficient.
        """
        # Remove missing values
        valid = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[valid]
        y_valid = y[valid]

        if len(x_valid) < 2:
            return 0, np.mean(y), float('inf'), 0

        # Compute means
        x_mean = np.mean(x_valid)
        y_mean = np.mean(y_valid)

        # Compute slope using least squares
        x_centered = x_valid - x_mean
        y_centered = y_valid - y_mean

        numerator = np.sum(x_centered * y_centered)
        denominator = np.sum(x_centered ** 2)

        if denominator == 0:
            # Constant feature, predict mean
            return 0, y_mean, np.sum(y_centered ** 2), 0

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Compute squared error
        y_pred = slope * x_valid + intercept
        squared_error = np.sum((y_valid - y_pred) ** 2)

        # Compute correlation coefficient
        y_var = np.sum(y_centered ** 2)
        if y_var == 0:
            correlation = 0
        else:
            x_std = np.sqrt(denominator / len(x_valid))
            y_std = np.sqrt(y_var / len(y_valid))
            if x_std > 0 and y_std > 0:
                correlation = numerator / (len(x_valid) * x_std * y_std)
            else:
                correlation = 0

        return slope, intercept, squared_error, correlation

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleLinearRegression":
        """Fit the Simple Linear Regression model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training target values.

        Returns
        -------
        self : SimpleLinearRegression
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, self.n_features_ = X.shape
        self.y_mean_ = np.nanmean(y)

        if self.attribute_index >= 0:
            # Use specified attribute
            if self.attribute_index >= self.n_features_:
                raise ValueError(
                    f"attribute_index {self.attribute_index} out of range "
                    f"for {self.n_features_} features"
                )
            self.selected_attribute_ = self.attribute_index
            x = X[:, self.selected_attribute_]
            self.slope_, self.intercept_, _, self.correlation_ = \
                self._fit_single_attribute(x, y)
            self.x_mean_ = np.nanmean(x)
        else:
            # Auto-select best attribute
            best_error = float('inf')
            best_attr = 0
            best_slope = 0
            best_intercept = self.y_mean_
            best_corr = 0

            for i in range(self.n_features_):
                x = X[:, i]
                slope, intercept, error, corr = self._fit_single_attribute(x, y)

                if error < best_error:
                    best_error = error
                    best_attr = i
                    best_slope = slope
                    best_intercept = intercept
                    best_corr = corr

            self.selected_attribute_ = best_attr
            self.slope_ = best_slope
            self.intercept_ = best_intercept
            self.correlation_ = best_corr
            self.x_mean_ = np.nanmean(X[:, best_attr])

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
        predictions : np.ndarray of shape (n_samples,)
            Predicted continuous values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Extract selected attribute
        x = X[:, self.selected_attribute_].copy()

        # Replace missing values with training mean
        missing = np.isnan(x)
        x[missing] = self.x_mean_

        # Predict
        predictions = self.slope_ * x + self.intercept_

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

    def get_equation(self) -> str:
        """Get the regression equation as a string.

        Returns
        -------
        equation : str
            String representation of the equation.
        """
        self._check_is_fitted()
        if self.slope_ >= 0:
            return f"y = {self.slope_:.6f} * x[{self.selected_attribute_}] + {self.intercept_:.6f}"
        else:
            return f"y = {self.slope_:.6f} * x[{self.selected_attribute_}] - {abs(self.intercept_):.6f}"

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"SimpleLinearRegression(attribute={self.selected_attribute_}, "
                    f"slope={self.slope_:.4f}, intercept={self.intercept_:.4f})")
        return f"SimpleLinearRegression(attribute_index={self.attribute_index})"
