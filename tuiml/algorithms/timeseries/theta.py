"""Theta method for univariate time series forecasting."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List

from tuiml.base.algorithms import Regressor, regressor


@regressor(tags=["timeseries", "forecasting", "decomposition"], version="1.0.0")
class ThetaForecaster(Regressor):
    """
    Theta method for **univariate forecasting** by **curvature decomposition**.

    The Theta method decomposes a series into so-called **theta lines**,
    each obtained by modifying the local curvature (the second differences)
    of the original series by a coefficient :math:`\\theta`. The classic
    formulation uses two lines: :math:`\\theta = 0`, which is the ordinary
    least-squares straight line through the data, and :math:`\\theta = 2`,
    which doubles the curvature. Each line is extrapolated separately and
    the two forecasts are averaged. Despite its simplicity, the method won
    the M3 forecasting competition.

    Overview
    --------
    1. Optionally deseasonalise the series with a classical decomposition
       when ``season_length`` is given and a seasonality test fires.
    2. Fit an ordinary least-squares line :math:`a + b t` to the
       (deseasonalised) series.
    3. Build the theta line
       :math:`Z_{\\theta}(t) = \\theta y_t + (1 - \\theta)(a + b t)`.
    4. Extrapolate the :math:`\\theta = 0` line by simple linear
       extrapolation, and the :math:`\\theta` line by simple exponential
       smoothing (SES).
    5. Combine the two extrapolations with weights :math:`1/\\theta` and
       :math:`1 - 1/\\theta`, then reseasonalise.

    Theory
    ------
    A theta line rescales the second differences of the series,

    .. math::
        \\nabla^2 Z_{\\theta}(t) = \\theta \\, \\nabla^2 y_t ,

    so :math:`\\theta = 0` removes all curvature (a straight line) and
    :math:`\\theta > 1` amplifies it. The solution of that difference
    equation with the two boundary conditions that minimise the squared
    deviation from the data is

    .. math::
        Z_{\\theta}(t) = \\theta y_t + (1 - \\theta)(a + b t),

    with :math:`a, b` the OLS intercept and slope of :math:`y` on
    :math:`t = 1, \\dots, n`. The combined forecast is

    .. math::
        \\hat{y}_{n+h}
        = \\frac{1}{\\theta} \\, \\ell_n(Z_{\\theta})
          + \\left(1 - \\frac{1}{\\theta}\\right) \\bigl(a + b (n + h)\\bigr),

    where :math:`\\ell_n(Z_{\\theta})` is the SES level of the theta line.

    **Equivalence with SES plus drift.** Hyndman and Billah (2003) showed
    that for :math:`\\theta = 2` and equal weights the method is *exactly*
    simple exponential smoothing with a drift of :math:`b / 2`:

    .. math::
        \\hat{y}_{n+h}
        = \\ell_n + \\frac{b}{2}
          \\left[ h - 1 + \\frac{1}{\\alpha}
                  - \\frac{(1 - \\alpha)^n}{\\alpha} \\right],

    where :math:`\\ell_n` is the SES level of the *original* series
    initialised at :math:`\\ell_0 = y_1`. This implementation reproduces
    that identity to machine precision.

    Parameters
    ----------
    theta : float, default=2.0
        Curvature coefficient of the second theta line. Must be strictly
        positive. ``theta=2`` gives the classic Theta method.
    alpha : float, optional, default=None
        SES smoothing parameter for the theta line. When ``None`` it is
        chosen on a deterministic grid by minimising the in-sample sum of
        squared one-step errors of the theta line.
    season_length : int, optional, default=None
        Number of periods in a season. When ``None`` (or 1) no seasonal
        adjustment is attempted.
    seasonal : {"mul", "add"}, default="mul"
        Type of seasonality used by the classical decomposition.
        ``"mul"`` falls back to ``"add"`` when the series is not strictly
        positive.
    seasonality_test : bool, default=True
        When ``True``, seasonal adjustment is applied only if the
        autocorrelation at lag ``season_length`` is significant at the
        90% level.

    Attributes
    ----------
    alpha_ : float
        SES smoothing parameter actually used.
    intercept_ : float
        OLS intercept :math:`a` of the deseasonalised series.
    slope_ : float
        OLS slope :math:`b` of the deseasonalised series.
    level_ : float
        Final SES level of the theta line.
    drift_ : float
        Slope of the combined forecast function with respect to the horizon,
        :math:`b (1 - 1/\\theta)`. For the classic :math:`\\theta = 2` this
        is :math:`b / 2`, the drift of the equivalent SES-with-drift model.
    seasonal_indices_ : np.ndarray or None
        Estimated seasonal indices of length ``season_length``.
    is_seasonal_ : bool
        Whether seasonal adjustment was applied.
    seasonal_mode_ : str or None
        The decomposition actually used, ``"mul"`` or ``"add"``.
    fitted_values_ : np.ndarray
        In-sample one-step-ahead forecasts.
    resid_ : np.ndarray
        In-sample residuals.
    n_obs_ : int
        Number of training observations.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n)` for a fixed ``alpha``, :math:`O(gn)` when
      ``alpha`` is optimised over a grid of :math:`g` values.
    - Prediction: :math:`O(h)`.

    **When to use ThetaForecaster:**

    - Short and medium series where a robust, low-variance benchmark is
      wanted; it is very hard to beat on the M-competition data.
    - Series with a clear local trend that should be damped rather than
      extrapolated at full strength.
    - Monthly or quarterly business data, combined with ``season_length``.
    - As a baseline against which ARIMA or exponential smoothing is judged.

    References
    ----------
    .. [Assimakopoulos2000] Assimakopoulos, V., & Nikolopoulos, K. (2000).
           **The theta model: a decomposition approach to forecasting.**
           *International Journal of Forecasting*, 16(4), 521-530.
           :doi:`10.1016/S0169-2070(00)00066-2`
    .. [Hyndman2003] Hyndman, R. J., & Billah, B. (2003). **Unmasking the
           Theta method.** *International Journal of Forecasting*, 19(2),
           287-290. :doi:`10.1016/S0169-2070(01)00143-1`
    .. [Fiorucci2016] Fiorucci, J. A., Pellegrini, T. R., Louzada, F.,
           Petropoulos, F., & Koehler, A. B. (2016). **Models for optimising
           the theta method and their relationship to state space models.**
           *International Journal of Forecasting*, 32(4), 1151-1161.
           :doi:`10.1016/j.ijforecast.2016.02.005`

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.ExponentialSmoothing` : SES, Holt and Holt-Winters smoothing.
    :class:`~tuiml.algorithms.timeseries.ARIMA` : Box-Jenkins forecasting with differencing.
    :class:`~tuiml.algorithms.timeseries.croston.CrostonForecaster` : Forecasting for intermittent demand.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.theta import ThetaForecaster
    >>> y = np.arange(1.0, 21.0)
    >>> model = ThetaForecaster(theta=2.0, alpha=0.3).fit(y)
    >>> forecast = model.predict(steps=3)
    >>> forecast.shape
    (3,)
    >>> bool(np.all(np.diff(forecast) > 0))
    True
    """

    def __init__(
        self,
        theta: float = 2.0,
        alpha: float | None = None,
        season_length: int | None = None,
        seasonal: str = "mul",
        seasonality_test: bool = True,
    ):
        """Initialize the Theta forecaster.

        Parameters
        ----------
        theta : float, default=2.0
            Curvature coefficient of the second theta line.
        alpha : float, optional, default=None
            SES smoothing parameter; optimised on a grid when ``None``.
        season_length : int, optional, default=None
            Number of periods in a season.
        seasonal : {"mul", "add"}, default="mul"
            Type of seasonality for the classical decomposition.
        seasonality_test : bool, default=True
            Whether to gate seasonal adjustment on an autocorrelation test.
        """
        super().__init__()
        self.theta = theta
        self.alpha = alpha
        self.season_length = season_length
        self.seasonal = seasonal
        self.seasonality_test = seasonality_test

        # Fitted attributes
        self.alpha_ = None
        self.intercept_ = None
        self.slope_ = None
        self.level_ = None
        self.drift_ = None
        self.seasonal_indices_ = None
        self.is_seasonal_ = False
        self.seasonal_mode_ = None
        self.fitted_values_ = None
        self.resid_ = None
        self.n_obs_ = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "theta": {
                "type": "number",
                "default": 2.0,
                "exclusiveMinimum": 0.0,
                "description": "Curvature coefficient of the second theta line",
            },
            "alpha": {
                "type": ["number", "null"],
                "default": None,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "SES smoothing parameter; optimised when null",
            },
            "season_length": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 1,
                "description": "Number of periods in a season",
            },
            "seasonal": {
                "type": "string",
                "default": "mul",
                "enum": ["mul", "add"],
                "description": "Seasonal decomposition type",
            },
            "seasonality_test": {
                "type": "boolean",
                "default": True,
                "description": "Gate seasonal adjustment on an autocorrelation test",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric",
            "timeseries",
            "forecasting",
            "univariate",
            "trend",
            "seasonality",
            "interpretable",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(n) for fixed alpha, Prediction: O(h)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Assimakopoulos & Nikolopoulos, 2000. The theta model. IJF 16(4), 521-530.",
            "Hyndman & Billah, 2003. Unmasking the Theta method. IJF 19(2), 287-290.",
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ols_line(y: np.ndarray) -> tuple:
        """Fit ``a + b t`` by least squares with ``t = 1, ..., n``.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Series values.

        Returns
        -------
        intercept : float
            The OLS intercept :math:`a`.
        slope : float
            The OLS slope :math:`b`.
        """
        n = len(y)
        t = np.arange(1, n + 1, dtype=float)
        t_mean = t.mean()
        y_mean = y.mean()
        denom = np.sum((t - t_mean) ** 2)
        slope = 0.0 if denom == 0.0 else float(np.sum((t - t_mean) * (y - y_mean)) / denom)
        intercept = float(y_mean - slope * t_mean)
        return intercept, slope

    @staticmethod
    def _ses_levels(z: np.ndarray, alpha: float) -> np.ndarray:
        """Run simple exponential smoothing initialised at the first value.

        Parameters
        ----------
        z : np.ndarray of shape (n_samples,)
            Series to smooth.
        alpha : float
            Smoothing parameter in ``(0, 1]``.

        Returns
        -------
        levels : np.ndarray of shape (n_samples + 1,)
            ``levels[t]`` is the level after observing ``z[:t]``, with
            ``levels[0] = z[0]``.
        """
        n = len(z)
        levels = np.empty(n + 1, dtype=float)
        levels[0] = z[0]
        for t in range(n):
            levels[t + 1] = alpha * z[t] + (1.0 - alpha) * levels[t]
        return levels

    @classmethod
    def _optimise_alpha(cls, z: np.ndarray) -> float:
        """Pick the SES parameter minimising in-sample squared error.

        Parameters
        ----------
        z : np.ndarray of shape (n_samples,)
            Theta line to smooth.

        Returns
        -------
        alpha : float
            The grid value minimising the sum of squared one-step errors.
        """
        grid = np.round(np.arange(0.01, 1.0 + 1e-9, 0.01), 2)
        best_alpha, best_sse = 0.5, np.inf
        for a in grid:
            levels = cls._ses_levels(z, float(a))
            errors = z - levels[:-1]
            sse = float(np.sum(errors ** 2))
            if sse < best_sse - 1e-12:
                best_sse, best_alpha = sse, float(a)
        return best_alpha

    @staticmethod
    def _acf(y: np.ndarray, max_lag: int) -> np.ndarray:
        """Return the sample autocorrelation function up to ``max_lag``.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Series values.
        max_lag : int
            Largest lag to compute.

        Returns
        -------
        acf : np.ndarray of shape (max_lag + 1,)
            Autocorrelations, with ``acf[0] == 1``.
        """
        n = len(y)
        centred = y - y.mean()
        denom = float(np.sum(centred ** 2))
        out = np.zeros(max_lag + 1, dtype=float)
        out[0] = 1.0
        if denom == 0.0:
            return out
        for lag in range(1, max_lag + 1):
            out[lag] = float(np.sum(centred[lag:] * centred[:-lag]) / denom)
        return out

    def _is_seasonal(self, y: np.ndarray, m: int) -> bool:
        """Test whether the lag-``m`` autocorrelation is significant.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Series values.
        m : int
            Candidate season length.

        Returns
        -------
        seasonal : bool
            ``True`` when the series is judged seasonal at the 90% level.
        """
        n = len(y)
        if n < 3 * m:
            return False
        acf = self._acf(y, m)
        # Bartlett-style critical value used by the standard Theta / M4 code.
        limit = 1.645 * np.sqrt((1.0 + 2.0 * np.sum(acf[1:m] ** 2)) / n)
        return bool(abs(acf[m]) > limit)

    @staticmethod
    def _centred_moving_average(y: np.ndarray, m: int) -> np.ndarray:
        """Compute the centred moving average of order ``m``.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Series values.
        m : int
            Window length. Even windows use the 2xm average.

        Returns
        -------
        trend : np.ndarray of shape (n_samples,)
            Trend estimate with ``np.nan`` at the un-computable ends.
        """
        n = len(y)
        trend = np.full(n, np.nan, dtype=float)
        if m % 2 == 1:
            half = m // 2
            for t in range(half, n - half):
                trend[t] = float(np.mean(y[t - half:t + half + 1]))
        else:
            half = m // 2
            weights = np.ones(m + 1, dtype=float)
            weights[0] = 0.5
            weights[-1] = 0.5
            weights /= m
            for t in range(half, n - half):
                trend[t] = float(np.dot(weights, y[t - half:t + half + 1]))
        return trend

    def _seasonal_indices(self, y: np.ndarray, m: int, mode: str) -> np.ndarray:
        """Estimate seasonal indices by classical decomposition.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Series values.
        m : int
            Season length.
        mode : {"mul", "add"}
            Decomposition type.

        Returns
        -------
        indices : np.ndarray of shape (m,)
            Seasonal indices aligned to ``t % m``, normalised to mean one
            (multiplicative) or mean zero (additive).
        """
        trend = self._centred_moving_average(y, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            if mode == "mul":
                detrended = np.where(trend == 0.0, np.nan, y / trend)
            else:
                detrended = y - trend

        indices = np.empty(m, dtype=float)
        neutral = 1.0 if mode == "mul" else 0.0
        for s in range(m):
            season_values = detrended[s::m]
            season_values = season_values[np.isfinite(season_values)]
            indices[s] = float(np.mean(season_values)) if season_values.size else neutral

        if mode == "mul":
            mean = float(np.mean(indices))
            indices = indices / mean if mean != 0.0 else np.ones(m, dtype=float)
            indices[indices <= 0.0] = 1e-8
        else:
            indices = indices - float(np.mean(indices))
        return indices

    def _season_factor(self, positions: np.ndarray) -> np.ndarray:
        """Return the seasonal factor for a set of time positions.

        Parameters
        ----------
        positions : np.ndarray of shape (k,)
            Zero-based time indices.

        Returns
        -------
        factors : np.ndarray of shape (k,)
            Seasonal factors, neutral when the model is non-seasonal.
        """
        if not self.is_seasonal_:
            return np.zeros(len(positions), dtype=float) if self.seasonal_mode_ == "add" \
                else np.ones(len(positions), dtype=float)
        m = len(self.seasonal_indices_)
        return self.seasonal_indices_[positions % m]

    # ------------------------------------------------------------------
    # Fitting and prediction
    # ------------------------------------------------------------------

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> "ThetaForecaster":
        """Fit the Theta model to a univariate series.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        self : ThetaForecaster
            Fitted estimator.
        """
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        if n < 3:
            raise ValueError("ThetaForecaster requires at least 3 observations")
        if not np.isfinite(y).all():
            raise ValueError("y must not contain NaN or infinite values")
        if self.theta is None or float(self.theta) <= 0.0:
            raise ValueError("theta must be strictly positive")
        if self.seasonal not in ("mul", "add"):
            raise ValueError("seasonal must be 'mul' or 'add'")
        if self.alpha is not None and not (0.0 < float(self.alpha) <= 1.0):
            raise ValueError("alpha must lie in (0, 1]")

        theta = float(self.theta)
        self.n_obs_ = n

        # ---- seasonal adjustment -------------------------------------
        m = self.season_length
        self.is_seasonal_ = False
        self.seasonal_indices_ = None
        self.seasonal_mode_ = None
        y_adj = y
        if m is not None and int(m) > 1 and n >= 2 * int(m):
            m = int(m)
            if (not self.seasonality_test) or self._is_seasonal(y, m):
                mode = self.seasonal
                if mode == "mul" and np.any(y <= 0.0):
                    mode = "add"
                indices = self._seasonal_indices(y, m, mode)
                self.seasonal_mode_ = mode
                self.seasonal_indices_ = indices
                self.is_seasonal_ = True
                factors = indices[np.arange(n) % m]
                y_adj = y / factors if mode == "mul" else y - factors

        # ---- theta lines ---------------------------------------------
        a, b = self._ols_line(y_adj)
        self.intercept_, self.slope_ = a, b
        t = np.arange(1, n + 1, dtype=float)
        line = a + b * t
        z = theta * y_adj + (1.0 - theta) * line

        alpha = float(self.alpha) if self.alpha is not None else self._optimise_alpha(z)
        self.alpha_ = alpha
        levels = self._ses_levels(z, alpha)
        self.level_ = float(levels[-1])
        # Slope of the combined forecast function with respect to h.
        self.drift_ = float(b * (1.0 - 1.0 / theta))

        # ---- in-sample one-step forecasts ----------------------------
        w_theta = 1.0 / theta
        w_line = 1.0 - w_theta
        fitted_adj = w_theta * levels[:-1] + w_line * line
        if self.is_seasonal_:
            factors = self.seasonal_indices_[np.arange(n) % len(self.seasonal_indices_)]
            fitted = fitted_adj * factors if self.seasonal_mode_ == "mul" else fitted_adj + factors
        else:
            fitted = fitted_adj
        self.fitted_values_ = fitted
        self.resid_ = y - fitted

        self._is_fitted = True
        return self

    def predict(self, steps: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast future values of the series.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecasted values.
        """
        self._check_is_fitted()
        if steps < 1:
            raise ValueError("steps must be >= 1")

        theta = float(self.theta)
        h = np.arange(1, steps + 1, dtype=float)
        line_future = self.intercept_ + self.slope_ * (self.n_obs_ + h)
        forecast = (1.0 / theta) * self.level_ + (1.0 - 1.0 / theta) * line_future

        if self.is_seasonal_:
            m = len(self.seasonal_indices_)
            positions = (self.n_obs_ + np.arange(steps)) % m
            factors = self.seasonal_indices_[positions]
            forecast = forecast * factors if self.seasonal_mode_ == "mul" else forecast + factors
        return forecast

    def fit_predict(self, y: np.ndarray, steps: int = 1) -> np.ndarray:
        """Fit the model and forecast in one call.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values.
        steps : int, default=1
            Number of future time steps to forecast.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecasted values.
        """
        self.fit(y)
        return self.predict(steps)
