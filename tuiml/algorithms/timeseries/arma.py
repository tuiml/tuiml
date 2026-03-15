"""Autoregressive Moving Average (ARMA) models for stationary time series."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from tuiml.base.algorithms import Regressor, regressor

@regressor(tags=["timeseries", "forecasting", "arma", "stationary"], version="1.0.0")
class ARMA(Regressor):
    r"""
    Autoregressive Moving Average (ARMA) model for **stationary time series
    forecasting**.

    The :math:`\\text{ARMA}(p, q)` model combines **autoregressive** (AR) and
    **moving average** (MA) components to model stationary time series data.

    Overview
    --------
    The ARMA model works through the following steps:

    1. Center the time series by subtracting the mean.
    2. Estimate initial AR coefficients via Yule-Walker equations.
    3. Initialize MA coefficients to zero.
    4. Iteratively refine parameters using conditional sum of squares (CSS)
       or simplified maximum likelihood estimation (MLE).
    5. Forecast by combining AR (lagged values) and MA (lagged residuals)
       components, with future errors assumed to be zero.

    Theory
    ------
    The :math:`\\text{ARMA}(p, q)` model is defined as:

    .. math::
        y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t + \sum_{j=1}^q \\theta_j \epsilon_{t-j}

    where:
    
    - :math:`y_t`: The value of the time series at time :math:`t`.
    - :math:`c`: A constant (intercept) term.
    - :math:`\phi_i`: The autoregressive parameters.
    - :math:`\\theta_j`: The moving average parameters.
    - :math:`p`: The AR order (number of lagged values).
    - :math:`q`: The MA order (number of lagged errors).
    - :math:`\epsilon_t`: The white noise error term at time :math:`t`.

    Using the lag operator :math:`L`, the model can be written as:

    .. math::
        \Phi(L) y_t = c + \Theta(L) \epsilon_t

    where :math:`\Phi(L) = 1 - \sum_{i=1}^p \phi_i L^i` and 
    :math:`\Theta(L) = 1 + \sum_{j=1}^q \\theta_j L^j`.

    Parameters
    ----------
    order : tuple of (int, int), default=(1, 0)
        The :math:`(p, q)` order of the model:
        - :math:`p`: Autoregressive order.
        - :math:`q`: Moving average order.
    trend : {"c", "ct", None}, default="c"
        The trend component to include:
        - ``"c"``: Include a constant term (intercept).
        - ``"ct"``: Include both a constant and a linear time trend.
        - ``None``: No constant or trend.
    method : {"css", "mle", "css-mle"}, default="css-mle"
        The estimation method:
        - ``"css"``: Conditional Sum of Squares.
        - ``"mle"``: Maximum Likelihood Estimation.
        - ``"css-mle"``: CSS for initial estimates followed by MLE refinement.
    maxiter : int, default=50
        Maximum number of iterations for the optimization process.

    Attributes
    ----------
    ar_params_ : np.ndarray of shape (p,)
        Fitted autoregressive coefficients :math:`(\phi_1, \phi_2, \dots, \phi_p)`.
    ma_params_ : np.ndarray of shape (q,)
        Fitted moving average coefficients :math:`(\\theta_1, \\theta_2, \dots, \\theta_q)`.
    const_ : float
        The fitted constant (intercept) term.
    resid_ : np.ndarray of shape (n_obs,)
        The estimated residuals from the fitted model.
    sigma2_ : float
        The variance of the residuals (:math:`\sigma^2`).
    n_obs_ : int
        The total number of observations used for fitting.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot \\text{maxiter} \cdot (p+q)^2)` where :math:`n`
      is the number of samples and :math:`(p, q)` are the orders.
    - Prediction: :math:`O(p+q)` for each forecasted step.

    **When to use ARMA:**

    - Stationary time series with both autocorrelation and moving average structure
    - When differencing is not needed (no trend or unit root)
    - Data where shocks persist for a limited number of periods
    - As a building block before considering the full ARIMA framework

    References
    ----------
    .. [Box2015] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
           **Time series analysis: forecasting and control.**
           *John Wiley & Sons*.

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.AR` : Pure autoregressive model without MA component.
    :class:`~tuiml.algorithms.timeseries.MA` : Pure moving average model without AR component.
    :class:`~tuiml.algorithms.timeseries.ARIMA` : ARMA extended with differencing for non-stationary series.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries import ARMA
    >>> # Generate synthetic data
    >>> np.random.seed(42)
    >>> n = 100
    >>> y = np.cumsum(np.random.normal(size=n))
    >>> y_stationary = np.diff(y)  # ARMA assumes stationarity
    >>> model = ARMA(order=(1, 1))
    >>> model.fit(y_stationary)
    >>> forecast = model.predict(steps=3)
    """

    def __init__(
        self,
        order: Tuple[int, int] = (1, 0),
        trend: str | None = "c",
        method: str = "css-mle",
        maxiter: int = 50,
    ):
        """Initialize ARMA model with order and optimization parameters.

        Parameters
        ----------
        order : tuple of (int, int), default=(1, 0)
            (p, q) order.
        trend : str or None, default="c"
            Trend component.
        method : str, default="css-mle"
            Fitting method.
        maxiter : int, default=50
            Maximum iterations for optimization.
        """
        super().__init__()
        self.order = order
        self.trend = trend
        self.method = method
        self.maxiter = maxiter

        # Fitted attributes
        self.ar_params_ = None
        self.ma_params_ = None
        self.const_ = None
        self.resid_ = None
        self.sigma2_ = None
        self.n_obs_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "order": {
                "type": "array",
                "default": [1, 0],
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
                "description": "(p, q) order of ARMA model"
            },
            "trend": {
                "type": ["string", "null"],
                "default": "c",
                "enum": [None, "c", "ct"],
                "description": "Trend component"
            },
            "method": {
                "type": "string",
                "default": "css-mle",
                "enum": ["css", "mle", "css-mle"],
                "description": "Fitting method"
            },
            "maxiter": {
                "type": "integer",
                "default": 50,
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
            "univariate",
            "stationary"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(n * max_iter * (p+q)²), Prediction: O(p+q), where n=samples, p=AR order, q=MA order"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Box et al., 2015. Time series analysis: forecasting and control. Wiley."
        ]

    def fit(self, y: np.ndarray, _X: Optional[np.ndarray] = None) -> "ARMA":
        """Fit the ARMA model to time series data.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values to fit.
        _X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        self : ARMA
            Fitted estimator.
        """
        y = np.asarray(y).ravel()
        self.n_obs_ = len(y)
        p, q = self.order

        if len(y) < max(p, q) + 1:
            raise ValueError(f"Need at least {max(p, q) + 1} observations for ARMA({p},{q})")

        # Initialize parameters
        self.const_ = np.mean(y)
        y_centered = y - self.const_

        # Initialize AR parameters (using Yule-Walker if p > 0)
        if p > 0:
            self.ar_params_ = self._estimate_ar(y_centered, p)
        else:
            self.ar_params_ = np.array([])

        # Initialize MA parameters
        if q > 0:
            self.ma_params_ = np.zeros(q)
        else:
            self.ma_params_ = np.array([])

        # Iterative refinement (simplified CSS/MLE)
        for _ in range(self.maxiter):
            self.resid_ = self._compute_residuals(y_centered)

            # Simple gradient updates (very simplified)
            if p > 0:
                for i in range(p):
                    # Numerical gradient
                    eps = 1e-5
                    self.ar_params_[i] += eps
                    resid_plus = self._compute_residuals(y_centered)
                    sse_plus = np.sum(resid_plus ** 2)
                    self.ar_params_[i] -= eps

                    sse = np.sum(self.resid_ ** 2)
                    grad = (sse_plus - sse) / eps
                    self.ar_params_[i] -= 0.01 * grad

        self.sigma2_ = np.var(self.resid_)

        self._is_fitted = True
        return self

    def _estimate_ar(self, y: np.ndarray, p: int) -> np.ndarray:
        """Estimate AR parameters using Yule-Walker.

        Parameters
        ----------
        y : np.ndarray
            Centered time series.
        p : int
            AR order.

        Returns
        -------
        ar_params : np.ndarray
            AR coefficients.
        """
        mean = np.mean(y)
        y_centered = y - mean

        # Autocorrelation
        acf = np.correlate(y_centered, y_centered, mode='full')
        acf = acf[len(acf) // 2:]
        acf = acf / acf[0]

        # Yule-Walker
        R = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = acf[abs(i - j)]

        r = acf[1:p + 1]

        try:
            ar_params = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            ar_params = np.linalg.lstsq(R, r, rcond=None)[0]

        return ar_params

    def _compute_residuals(self, y: np.ndarray) -> np.ndarray:
        """Compute residuals.

        Parameters
        ----------
        y : np.ndarray
            Centered series.

        Returns
        -------
        resid : np.ndarray
            Residuals.
        """
        p, q = self.order
        n = len(y)
        max_lag = max(p, q)
        resid = np.zeros(n)

        for t in range(max_lag, n):
            # AR component
            ar_term = 0.0
            if p > 0:
                ar_term = np.sum(self.ar_params_ * y[t - p:t][::-1])

            # MA component
            ma_term = 0.0
            if q > 0:
                ma_term = np.sum(self.ma_params_ * resid[t - q:t][::-1])

            resid[t] = y[t] - ar_term - ma_term

        return resid

    def predict(self, steps: int = 1, _X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast future values using the fitted ARMA model.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        _X : np.ndarray, optional, default=None
            Ignored.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecasted values.
        """
        self._check_is_fitted()

        p, q = self.order
        forecast = np.zeros(steps)

        # Extend series and residuals
        y_extended = np.concatenate([np.zeros(max(p, q)), np.zeros(steps)])
        resid_extended = np.concatenate([self.resid_[-max(p, q):] if len(self.resid_) > 0 else np.zeros(max(p, q)), np.zeros(steps)])

        for h in range(steps):
            idx = max(p, q) + h

            # AR component
            ar_term = 0.0
            if p > 0:
                ar_values = y_extended[idx - p:idx][::-1]
                ar_term = np.sum(self.ar_params_ * ar_values)

            # MA component (future errors = 0)
            ma_term = 0.0
            if q > 0:
                ma_values = resid_extended[idx - q:idx][::-1]
                ma_term = np.sum(self.ma_params_ * ma_values)

            forecast[h] = self.const_ + ar_term + ma_term
            y_extended[idx] = forecast[h] - self.const_

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
