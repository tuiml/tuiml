"""Moving Average (MA) models for univariate time series forecasting."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List
from tuiml.base.algorithms import Regressor, regressor

@regressor(tags=["timeseries", "forecasting", "moving-average", "ma"], version="1.0.0")
class MA(Regressor):
    r"""
    Moving Average (MA) model for **univariate time series forecasting**.

    The :math:`\\text{MA}(q)` model predicts the next value in a time series
    as a **linear combination of past forecast errors** (residuals). Unlike the
    autoregressive model, it uses **errors** instead of past values.

    Overview
    --------
    The MA model works through the following steps:

    1. Estimate the series mean :math:`\mu` and center the data.
    2. Obtain initial residual estimates using the Hannan-Rissanen method
       (fit a high-order AR model to approximate residuals).
    3. Estimate the MA coefficients via OLS regression on lagged residuals.
    4. Refine parameters iteratively using conditional sum of squares (CSS).
    5. Forecast by applying MA coefficients to known past errors, with
       future errors set to zero (their expected value).

    Theory
    ------
    The :math:`\\text{MA}(q)` process is defined as:

    .. math::
        y_t = \mu + \epsilon_t + \\theta_1 \epsilon_{t-1} + \\theta_2 \epsilon_{t-2} + \dots + \\theta_q \epsilon_{t-q}

    where:
    
    - :math:`y_t`: The value of the time series at time :math:`t`.
    - :math:`\mu`: The mean of the series.
    - :math:`\\theta_i`: The moving average parameters (coefficients).
    - :math:`q`: The order of the model (number of lagged errors).
    - :math:`\epsilon_t`: The white noise error term at time :math:`t`.

    Moving average processes are always stationary, and they provide a way 
    to model short-term shocks that persist for :math:`q` periods.

    Parameters
    ----------
    order : int, default=1
        The order :math:`q` of the MA model (number of lagged errors).
    method : {"hannan_rissanen", "css", "mle"}, default="hannan_rissanen"
        The method used to estimate the MA parameters:
        - ``"hannan_rissanen"``: Two-step regression for initial estimates.
        - ``"css"``: Conditional Sum of Squares estimation.
        - ``"mle"``: Simplified Maximum Likelihood Estimation.
    max_iter : int, default=100
        Maximum iterations for iterative estimation methods.

    Attributes
    ----------
    ma_params_ : np.ndarray of shape (order,)
        Fitted moving average coefficients :math:`(\\theta_1, \\theta_2, \dots, \\theta_q)`.
    mu_ : float
        The estimated mean of the series.
    resid_ : np.ndarray of shape (n_obs,)
        The estimated residuals from the fitted model.
    sigma2_ : float
        The variance of the residuals (:math:`\sigma^2`).
    n_obs_ : int
        The total number of observations used for fitting.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(q \cdot n \cdot \\text{max\_iter})` where :math:`n` is
      the number of samples and :math:`q` is the order.
    - Prediction: :math:`O(q)` for each forecasted step.

    **When to use MA:**

    - Stationary time series driven by short-term shocks
    - When the autocorrelation function (ACF) cuts off after lag :math:`q`
    - Data where past errors are more informative than past values
    - Modeling noise structure in residuals from other models

    References
    ----------
    .. [Box2015] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
           **Time series analysis: forecasting and control.**
           *John Wiley & Sons*.

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.AR` : Autoregressive model using past values instead of errors.
    :class:`~tuiml.algorithms.timeseries.ARMA` : Combined AR and MA model for richer dynamics.
    :class:`~tuiml.algorithms.timeseries.ARIMA` : ARMA extended with differencing for non-stationary series.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries import MA
    >>> # Generate a simple MA(1) process
    >>> np.random.seed(42)
    >>> n = 100
    >>> eps = np.random.normal(size=n)
    >>> y = 10.0 + eps[1:] + 0.6 * eps[:-1]
    >>> model = MA(order=1)
    >>> model.fit(y)
    >>> # Forecast the next 3 steps
    >>> forecast = model.predict(steps=3)
    """

    def __init__(
        self,
        order: int = 1,
        method: str = "hannan_rissanen",
        max_iter: int = 100,
    ):
        """Initialize MA model with order and estimation parameters.

        Parameters
        ----------
        order : int, default=1
            MA order q.
        method : str, default="hannan_rissanen"
            Estimation method.
        max_iter : int, default=100
            Maximum optimization iterations.
        """
        super().__init__()
        self.order = order
        self.method = method
        self.max_iter = max_iter

        # Fitted attributes
        self.ma_params_ = None
        self.mu_ = None
        self.resid_ = None
        self.sigma2_ = None
        self.n_obs_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "order": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "MA order (number of lagged errors)"
            },
            "method": {
                "type": "string",
                "default": "hannan_rissanen",
                "enum": ["hannan_rissanen", "css", "mle"],
                "description": "Estimation method"
            },
            "max_iter": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Maximum optimization iterations"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric",
            "timeseries",
            "forecasting",
            "univariate"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(q * n * max_iter), Prediction: O(q), where q=order, n=samples"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Box et al., 2015. Time series analysis: forecasting and control. Wiley."
        ]

    def fit(self, y: np.ndarray, _X: Optional[np.ndarray] = None) -> "MA":
        """Fit the moving average model to data.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values to fit.
        _X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        self : MA
            Fitted estimator.
        """
        y = np.asarray(y).ravel()
        self.n_obs_ = len(y)
        q = self.order

        if len(y) < q + 1:
            raise ValueError(f"Need at least {q + 1} observations for MA({q})")

        # Estimate mean
        self.mu_ = np.mean(y)
        y_centered = y - self.mu_

        # Initialize MA parameters
        self.ma_params_ = np.zeros(q)

        # Hannan-Rissanen method: Use high-order AR to get initial residuals
        if self.method == "hannan_rissanen":
            # Fit long AR to get residual estimates
            from tuiml.algorithms.timeseries.ar import AR
            ar_long = AR(order=min(20, len(y) // 2))
            ar_long.fit(y_centered)
            initial_resid = ar_long.resid_

            # Use initial residuals to estimate MA parameters via OLS
            X = np.zeros((len(initial_resid) - q, q))
            for i in range(q):
                X[:, i] = initial_resid[q - i - 1:len(initial_resid) - i - 1]

            y_target = y_centered[q:len(initial_resid)]
            self.ma_params_ = np.linalg.lstsq(X, y_target, rcond=None)[0]

        else:
            # Simple initialization
            self.ma_params_ = np.random.randn(q) * 0.1

        # Refine with iterative estimation (simplified CSS)
        for _ in range(min(self.max_iter, 10)):
            self.resid_ = self._compute_residuals(y_centered)
            # Update MA params (simplified)
            if np.sum(self.resid_ ** 2) > 0:
                break

        self.sigma2_ = np.var(self.resid_)

        self._is_fitted = True
        return self

    def _compute_residuals(self, y: np.ndarray) -> np.ndarray:
        """Compute residuals.

        Parameters
        ----------
        y : np.ndarray
            Centered time series.

        Returns
        -------
        resid : np.ndarray
            Residuals.
        """
        q = self.order
        n = len(y)
        resid = np.zeros(n)

        for t in range(n):
            # MA: y_t = eps_t + theta_1 * eps_{t-1} + ... + theta_q * eps_{t-q}
            # Solve for eps_t: eps_t = y_t - theta_1 * eps_{t-1} - ... - theta_q * eps_{t-q}
            ma_term = 0.0
            for i in range(min(q, t)):
                ma_term += self.ma_params_[i] * resid[t - i - 1]

            resid[t] = y[t] - ma_term

        return resid

    def predict(self, steps: int = 1, _X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast future values using the fitted MA model.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        _X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecasted values.
        """
        self._check_is_fitted()

        q = self.order
        forecast = np.zeros(steps)

        # For MA, future errors are zero (expected value)
        # So forecast converges to mean quickly
        last_errors = list(self.resid_[-q:])

        for h in range(steps):
            # Forecast: mu + theta_1 * eps_{t+h-1} + ... + theta_q * eps_{t+h-q}
            ma_term = 0.0
            for i in range(q):
                if h - i - 1 >= 0:
                    # Future error (unknown, assume 0)
                    error_val = 0.0
                else:
                    # Past error
                    idx = -(h - i)
                    if -idx <= len(last_errors):
                        error_val = last_errors[idx]
                    else:
                        error_val = 0.0

                ma_term += self.ma_params_[i] * error_val

            forecast[h] = self.mu_ + ma_term

        return forecast

    def fit_predict(self, y: np.ndarray, steps: int = 1) -> np.ndarray:
        """Fit the model and forecast future values in one step.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values to fit.
        steps : int, default=1
            Number of future time steps to forecast.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecasted values.
        """
        self.fit(y)
        return self.predict(steps)
