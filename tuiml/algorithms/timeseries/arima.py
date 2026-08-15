"""Autoregressive Integrated Moving Average (ARIMA) models for time series."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from tuiml.base.algorithms import Regressor, regressor

@regressor(tags=["timeseries", "forecasting", "autoregressive"], version="1.0.0")
class ARIMA(Regressor):
    r"""
    Autoregressive Integrated Moving Average (ARIMA) model for **non-stationary
    time series forecasting**.

    ARIMA is a generalization of the ARMA model that includes **differencing**
    to handle non-stationary time series data. It is characterized by the
    triplet :math:`(p, d, q)`, representing the **autoregressive order**, the
    **degree of differencing**, and the **moving average order**, respectively.

    Overview
    --------
    The ARIMA modeling procedure follows these steps:

    1. Apply :math:`d` rounds of differencing to make the series stationary.
    2. Estimate the AR coefficients using Yule-Walker equations.
    3. Initialize the MA coefficients (set to zero initially).
    4. Compute the constant term based on the trend specification.
    5. Optionally refine parameters via simplified MLE (gradient descent).
    6. Forecast on the differenced scale and invert the differencing to
       recover predictions on the original scale.

    Theory
    ------
    The :math:`\\text{ARIMA}(p, d, q)` process is defined using the lag 
    operator :math:`L` as:

    .. math::
        (1 - \sum_{i=1}^p \phi_i L^i) (1 - L)^d y_t = (1 + \sum_{j=1}^q \\theta_j L^j) \epsilon_t

    where:
    
    - :math:`y_t`: The time series value at time :math:`t`.
    - :math:`L`: The lag operator, such that :math:`L y_t = y_{t-1}`.
    - :math:`d`: The degree of differencing required to make the series stationary.
    - :math:`p`: The number of autoregressive lags.
    - :math:`q`: The number of moving average lags.
    - :math:`\phi_i`: Autoregressive parameters.
    - :math:`\\theta_j`: Moving average parameters.
    - :math:`\epsilon_t`: White noise error term.

    The "integrated" part (I) refers to the differencing step: :math:`w_t = \Delta^d y_t`, 
    where :math:`\Delta = 1 - L`. After differencing, the resulting series 
    :math:`w_t` is modeled as an :math:`\\text{ARMA}(p, q)` process.

    Parameters
    ----------
    order : tuple of (int, int, int), default=(1, 0, 0)
        The :math:`(p, d, q)` order of the model:
        - :math:`p`: Autoregressive order.
        - :math:`d`: Degree of differencing.
        - :math:`q`: Moving average order.
    trend : {"c", "t", "ct", None}, default=None
        The trend component to include:
        - ``"c"``: Constant term.
        - ``"t"``: Linear trend.
        - ``"ct"``: Both constant and linear trend.
        - ``None``: No trend.
    method : {"css", "mle", "css-mle"}, default="css-mle"
        The estimation method used to fit the model.
    maxiter : int, default=50
        Maximum number of iterations for the optimization.

    Attributes
    ----------
    ar_params_ : np.ndarray of shape (p,)
        Fitted autoregressive coefficients :math:`(\phi_1, \phi_2, \dots, \phi_p)`.
    ma_params_ : np.ndarray of shape (q,)
        Fitted moving average coefficients :math:`(\\theta_1, \\theta_2, \dots, \\theta_q)`.
    const_ : float
        The fitted constant (intercept) term.
    resid_ : np.ndarray of shape (n_obs - d,)
        The estimated residuals from the fitted model.
    sigma2_ : float
        The variance of the residuals (:math:`\sigma^2`).
    n_obs_ : int
        The total number of observations used for fitting.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot \\text{maxiter} \cdot (p+q)^2)` where :math:`n`
      is the number of samples.
    - Prediction: :math:`O(p+q)` per step, plus :math:`O(d \cdot n)` for
      integrating back.

    **When to use ARIMA:**

    - Non-stationary time series that can be made stationary by differencing
    - Data with trends but without strong seasonal patterns
    - When both autoregressive and moving average components are needed
    - Short-to-medium term forecasting of univariate series

    **This model is non-seasonal, and has no exogenous regressors.** It once
    accepted a ``seasonal_order`` argument that was stored and never read, so
    passing ``(1, 1, 1, 12)`` quietly fitted a non-seasonal model and returned
    forecasts with no seasonal structure at all. The argument has been removed
    rather than left to mislead. For seasonal terms, exogenous regressors,
    exact Gaussian maximum likelihood via a Kalman filter, and forecast
    intervals, use :class:`~tuiml.algorithms.timeseries.SARIMAX`, which
    supersedes this class on every axis except speed.

    Parameters here are estimated by minimising the **conditional sum of
    squares**, conditioning on the first :math:`\\max(p, q)` observations,
    rather than the exact likelihood.

    References
    ----------
    .. [Box2015] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
           **Time series analysis: forecasting and control.**
           *John Wiley & Sons*.
    .. [Hyndman2018] Hyndman, R. J., & Athanasopoulos, G. (2018).
           **Forecasting: principles and practice.** *OTexts*.

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.SARIMAX` : Seasonal terms, exogenous regressors and exact maximum likelihood.
    :class:`~tuiml.algorithms.timeseries.AR` : Pure autoregressive model for stationary series.
    :class:`~tuiml.algorithms.timeseries.ARMA` : Combined AR and MA without differencing.
    :class:`~tuiml.algorithms.timeseries.ExponentialSmoothing` : State-space approach to forecasting with trend and seasonality.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries import ARIMA
    >>> # Generating a non-stationary random walk
    >>> np.random.seed(42)
    >>> y = np.cumsum(np.random.normal(size=100))
    >>> model = ARIMA(order=(1, 1, 1))
    >>> _ = model.fit(y)          # fit returns self; bind it to keep doctest quiet
    >>> model.predict(steps=5).shape
    (5,)

    The moving-average term is genuinely estimated, so ``q`` is not decorative:

    >>> rng = np.random.default_rng(0)
    >>> e = rng.normal(size=2001)
    >>> ma_series = e[1:] + 0.6 * e[:-1]        # MA(1) with theta = 0.6
    >>> fitted = ARIMA(order=(0, 0, 1)).fit(ma_series)
    >>> bool(abs(fitted.ma_params_[0] - 0.6) < 0.1)
    True
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 0),
        trend: str | None = None,
        method: str = "css-mle",
        maxiter: int = 50,
    ):
        """Initialize ARIMA model with order and optimization parameters.

        Parameters
        ----------
        order : tuple of (int, int, int), default=(1, 0, 0)
            (p, d, q) order.
        trend : str or None, default=None
            Trend component.
        method : str, default="css-mle"
            Fitting method.
        maxiter : int, default=50
            Maximum optimization iterations.
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
        self.trend_params_ = None
        self.resid_ = None
        self.y_train_ = None
        self.y_original_ = None
        self.n_obs_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "order": {
                "type": "array",
                "default": [1, 0, 0],
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "integer", "minimum": 0},
                "description": "(p, d, q) order of the ARIMA model"
            },
            "trend": {
                "type": ["string", "null"],
                "default": None,
                "enum": [None, "c", "t", "ct"],
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
            "Box et al., 2015. Time series analysis: forecasting and control. Wiley.",
            "Hyndman & Athanasopoulos, 2018. Forecasting: principles and practice."
        ]

    def _difference(self, y: np.ndarray, d: int) -> np.ndarray:
        """Apply differencing to make series stationary.

        Parameters
        ----------
        y : np.ndarray
            Time series.
        d : int
            Degree of differencing.

        Returns
        -------
        y_diff : np.ndarray
            Differenced series.
        """
        y_diff = y.copy()
        for _ in range(d):
            y_diff = np.diff(y_diff)
        return y_diff

    def _inverse_difference(self, y_diff: np.ndarray, y_base: np.ndarray, d: int) -> np.ndarray:
        """Invert differencing operation.

        Parameters
        ----------
        y_diff : np.ndarray
            Differenced values to invert.
        y_base : np.ndarray
            Base values for reconstruction.
        d : int
            Degree of differencing.

        Returns
        -------
        y : np.ndarray
            Reconstructed series.
        """
        y = np.asarray(y_diff).copy()

        # Get last d values from base for reconstruction
        for i in range(d):
            # Get appropriate base values
            base_level = self._difference(y_base, d - i - 1)
            last_val = base_level[-1]

            # Cumsum and add base
            y = np.cumsum(y) + last_val

        return y

    def fit(self, y: np.ndarray, _X: Optional[np.ndarray] = None) -> "ARIMA":
        """Fit the ARIMA model to time series data.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values to fit.
        _X : np.ndarray, optional, default=None
            Exogenous variables (not yet supported).

        Returns
        -------
        self : ARIMA
            Fitted estimator.
        """
        y = np.asarray(y).ravel()
        self.y_original_ = y.copy()
        self.n_obs_ = len(y)

        p, d, q = self.order

        # Apply differencing
        if d > 0:
            y_diff = self._difference(y, d)
        else:
            y_diff = y.copy()

        self.y_train_ = y_diff

        if len(y_diff) < max(p, q) + 1:
            raise ValueError("Not enough observations after differencing")

        # Initialize parameters with simple estimates
        if p > 0:
            # Use Yule-Walker equations for initial AR parameters
            self.ar_params_ = self._estimate_ar(y_diff, p)
        else:
            self.ar_params_ = np.array([])

        if q > 0:
            # Initialize MA parameters to small values
            self.ma_params_ = np.zeros(q)
        else:
            self.ma_params_ = np.array([])

        # Estimate constant
        if self.trend == "c" or self.trend == "ct":
            self.const_ = np.mean(y_diff)
        else:
            self.const_ = 0.0

        # Compute residuals
        self.resid_ = self._compute_residuals(y_diff)

        # Refinement with simple optimization (simplified MLE)
        if self.method in ["mle", "css-mle"]:
            self._refine_parameters(y_diff)

        self._is_fitted = True
        return self

    def _estimate_ar(self, y: np.ndarray, p: int) -> np.ndarray:
        """Estimate AR parameters using Yule-Walker equations.

        Parameters
        ----------
        y : np.ndarray
            Time series.
        p : int
            AR order.

        Returns
        -------
        ar_params : np.ndarray
            Estimated AR coefficients.
        """
        # Compute autocorrelations
        mean = np.mean(y)
        y_centered = y - mean

        # Autocorrelation function
        acf = np.correlate(y_centered, y_centered, mode='full')
        acf = acf[len(acf) // 2:]
        acf = acf / acf[0]  # Normalize

        # Yule-Walker: R * φ = r
        R = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = acf[abs(i - j)]

        r = acf[1:p + 1]

        # Solve for AR parameters
        try:
            ar_params = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            ar_params = np.linalg.lstsq(R, r, rcond=None)[0]

        return ar_params

    def _compute_residuals(self, y: np.ndarray) -> np.ndarray:
        """Compute residuals from current parameters.

        Parameters
        ----------
        y : np.ndarray
            Time series.

        Returns
        -------
        resid : np.ndarray
            Residuals.
        """
        p, _, q = self.order
        n = len(y)
        max_lag = max(p, q)

        resid = np.zeros(n)
        fitted = np.zeros(n)

        for t in range(max_lag, n):
            # AR component
            ar_term = 0.0
            if p > 0:
                ar_term = np.sum(self.ar_params_ * y[t - p:t][::-1])

            # MA component
            ma_term = 0.0
            if q > 0:
                ma_term = np.sum(self.ma_params_ * resid[t - q:t][::-1])

            fitted[t] = self.const_ + ar_term + ma_term
            resid[t] = y[t] - fitted[t]

        return resid

    def _pack_params(self) -> np.ndarray:
        """Flatten the free parameters into one vector for the optimiser."""
        return np.concatenate([[self.const_], self.ar_params_, self.ma_params_])

    def _unpack_params(self, theta: np.ndarray) -> None:
        """Write a flat parameter vector back onto the fitted attributes."""
        p, _, q = self.order
        self.const_ = float(theta[0])
        self.ar_params_ = np.asarray(theta[1:1 + p], dtype=float)
        self.ma_params_ = np.asarray(theta[1 + p:1 + p + q], dtype=float)

    def _refine_parameters(self, y: np.ndarray):
        """Minimise the conditional sum of squares over *all* parameters.

        This replaces a fixed-step numerical-gradient descent that looped over
        ``range(p)`` only. Because the loop never touched ``ma_params_``, the
        MA coefficients kept the zeros they were initialised with, and any
        model with ``q > 0`` silently returned :math:`\\theta = 0` -- an
        ``ARIMA(0, 0, 1)`` was a constant. The optimiser below varies the
        constant, the AR block and the MA block together, so ``q`` finally
        means something.

        Conditional sum of squares is the objective rather than the exact
        likelihood: residuals are computed by the recursion in
        :meth:`_compute_residuals`, conditioning on the first ``max(p, q)``
        observations. For the exact Gaussian likelihood through a Kalman
        filter, plus seasonal terms and exogenous regressors, use
        :class:`~tuiml.algorithms.timeseries.SARIMAX`.

        Parameters
        ----------
        y : np.ndarray
            The differenced series to fit.
        """
        from scipy.optimize import minimize

        p, _, q = self.order
        if p == 0 and q == 0 and self.trend not in ("c", "ct"):
            return

        max_lag = max(p, q)

        def objective(theta: np.ndarray) -> float:
            """Conditional sum of squares for a candidate parameter vector."""
            saved = (self.const_, self.ar_params_, self.ma_params_)
            try:
                self._unpack_params(theta)
                resid = self._compute_residuals(y)
                sse = float(np.sum(resid[max_lag:] ** 2))
            finally:
                self.const_, self.ar_params_, self.ma_params_ = saved
            # A diverging MA recursion produces inf/nan; report a large finite
            # value so the optimiser backs away instead of failing outright.
            return sse if np.isfinite(sse) else 1e300

        theta0 = self._pack_params()

        # Keep the search inside the invertible/stationary region. Individual
        # coefficients in (-1, 1) does not characterise it exactly for p or q
        # above 1, but it keeps the CSS recursion from exploding, which an
        # unbounded search does readily.
        bounds = [(None, None)] + [(-0.999, 0.999)] * (p + q)

        result = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": self.maxiter})

        # Only accept the result if it genuinely improved on the starting
        # point: a failed optimisation must not make the fit worse than the
        # Yule-Walker initialisation it began from.
        if np.all(np.isfinite(result.x)) and objective(result.x) <= objective(theta0):
            self._unpack_params(result.x)

        self.resid_ = self._compute_residuals(y)

    def predict(self, steps: int = 1, _X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast future values using the fitted ARIMA model.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        _X : np.ndarray, optional, default=None
            Exogenous variables (not yet supported).

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecasted values.
        """
        self._check_is_fitted()

        p, d, q = self.order
        forecast_diff = np.zeros(steps)

        # Extend series and residuals for forecasting
        y_extended = np.concatenate([self.y_train_, np.zeros(steps)])
        resid_extended = np.concatenate([self.resid_, np.zeros(steps)])

        n = len(self.y_train_)

        for t in range(steps):
            idx = n + t

            # AR component
            ar_term = 0.0
            if p > 0:
                ar_values = y_extended[idx - p:idx][::-1]
                ar_term = np.sum(self.ar_params_ * ar_values)

            # MA component (use zero for future errors)
            ma_term = 0.0
            if q > 0:
                ma_values = resid_extended[idx - q:idx][::-1]
                ma_term = np.sum(self.ma_params_ * ma_values)

            forecast_diff[t] = self.const_ + ar_term + ma_term
            y_extended[idx] = forecast_diff[t]

        # Inverse differencing
        if d > 0:
            forecast = self._inverse_difference(forecast_diff, self.y_original_, d)
        else:
            forecast = forecast_diff

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
