"""Exponential Smoothing models for time series forecasting."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List
from tuiml.base.algorithms import Regressor, regressor

@regressor(tags=["timeseries", "forecasting", "smoothing"], version="1.0.0")
class ExponentialSmoothing(Regressor):
    r"""
    Exponential Smoothing model for **time series forecasting** with
    **trend and seasonal components**.

    Exponential smoothing is a family of forecasting methods that use
    **weighted averages** of past observations, with weights decreasing
    **exponentially** as the observations get older. This implementation
    supports Simple (SES), Double (Holt's), and Triple (Holt-Winters)
    exponential smoothing.

    Overview
    --------
    The Exponential Smoothing procedure works as follows:

    1. Select the model type: Simple (no trend/season), Double (trend),
       or Triple (trend + seasonality).
    2. Initialize the level, trend, and seasonal components from the data.
    3. Apply the recursive smoothing equations to update each component
       at every time step using smoothing parameters :math:`\\alpha`,
       :math:`\\beta`, and :math:`\gamma`.
    4. Compute fitted values and residuals from the one-step-ahead
       forecasts during training.
    5. Generate multi-step forecasts by extrapolating the final level,
       trend, and seasonal components.

    Theory
    ------
    **Simple Exponential Smoothing (SES):**
    Suitable for data with no clear trend or seasonal pattern.
    
    .. math::
        \hat{y}_{t+1} = \\alpha y_t + (1 - \\alpha) \hat{y}_t

    **Double Exponential Smoothing (Holt's Linear Trend):**
    Adds a trend component to SES.
    
    .. math::
        \\begin{aligned}
        \ell_t &= \\alpha y_t + (1 - \\alpha)(\ell_{t-1} + b_{t-1}) \\
        b_t &= \\beta (\ell_t - \ell_{t-1}) + (1 - \\beta) b_{t-1} \\
        \hat{y}_{t+h} &= \ell_t + h b_t
        \end{aligned}

    **Triple Exponential Smoothing (Holt-Winters):**
    Adds a seasonal component to Holt's method. Supports both additive 
    and multiplicative seasonality.

    *Additive Seasonality:*
    
    .. math::
        \\begin{aligned}
        \ell_t &= \\alpha (y_t - s_{t-m}) + (1 - \\alpha)(\ell_{t-1} + b_{t-1}) \\
        b_t &= \\beta (\ell_t - \ell_{t-1}) + (1 - \\beta) b_{t-1} \\
        s_t &= \gamma (y_t - \ell_{t-1} - b_{t-1}) + (1 - \gamma) s_{t-m} \\
        \hat{y}_{t+h} &= \ell_t + h b_t + s_{t-m+h_m}
        \end{aligned}

    Parameters
    ----------
    trend : {"add", "mul", None}, default=None
        Type of trend component.
    seasonal : {"add", "mul", None}, default=None
        Type of seasonal component.
    seasonal_periods : int, optional, default=None
        Number of periods in a season (e.g., 12 for monthly data).
    smoothing_level : float, optional, default=None
        The alpha (:math:`\\alpha`) parameter for the level.
    smoothing_trend : float, optional, default=None
        The beta (:math:`\\beta`) parameter for the trend.
    smoothing_seasonal : float, optional, default=None
        The gamma (:math:`\gamma`) parameter for the seasonal component.
    damped_trend : bool, default=False
        Whether to dampen the trend over time.

    Attributes
    ----------
    level_ : float
        The final level component.
    trend_ : float or None
        The final trend component.
    seasonal_ : np.ndarray or None
        The final seasonal components.
    params_ : dict
        The smoothing parameters used.
    fitted_values_ : np.ndarray
        The values fitted to the training data.
    resid_ : np.ndarray
        The residuals from the fitted model.
    n_obs_ : int
        The number of observations in the training data.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n)` where :math:`n` is the number of samples.
    - Prediction: :math:`O(h)` where :math:`h` is the forecast horizon.

    **When to use ExponentialSmoothing:**

    - Time series with trend and/or seasonal patterns
    - When recent observations should carry more weight than older ones
    - Short-to-medium term forecasting with limited data
    - Business and demand forecasting applications
    - When a simple, fast, and interpretable model is desired

    References
    ----------
    .. [Hyndman2008] Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008).
           **Forecasting with exponential smoothing: the state space approach.**
           *Springer Science & Business Media*.
    .. [Gardner2006] Gardner Jr, E. S. (2006). **Exponential smoothing: The state of the art.**
           *Journal of Forecasting*, 25(4), 637-666.

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.ARIMA` : Box-Jenkins approach to forecasting with differencing.
    :class:`~tuiml.algorithms.timeseries.Prophet` : Additive model with trend, seasonality, and holiday effects.
    :class:`~tuiml.algorithms.timeseries.STLDecomposition` : Decomposition of trend and seasonal components using LOESS.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries import ExponentialSmoothing
    >>> # Generating data with trend
    >>> y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=float)
    >>> model = ExponentialSmoothing(trend="add")
    >>> model.fit(y)
    >>> forecast = model.predict(steps=3)
    """

    def __init__(
        self,
        trend: str | None = None,
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        smoothing_level: float | None = None,
        smoothing_trend: float | None = None,
        smoothing_seasonal: float | None = None,
        damped_trend: bool = False,
    ):
        """Initialize Exponential Smoothing with components and parameters.

        Parameters
        ----------
        trend : {"add", "mul", None}, default=None
            Trend component type.
        seasonal : {"add", "mul", None}, default=None
            Seasonal component type.
        seasonal_periods : int, optional, default=None
            Season length in number of periods.
        smoothing_level : float, optional, default=None
            Level smoothing parameter (alpha).
        smoothing_trend : float, optional, default=None
            Trend smoothing parameter (beta).
        smoothing_seasonal : float, optional, default=None
            Seasonal smoothing parameter (gamma).
        damped_trend : bool, default=False
            Whether to apply damping to the trend.
        """
        super().__init__()
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal
        self.damped_trend = damped_trend

        # Fitted attributes
        self.level_ = None
        self.trend_ = None
        self.seasonal_ = None
        self.params_ = {}
        self.fitted_values_ = None
        self.resid_ = None
        self.n_obs_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "trend": {
                "type": ["string", "null"],
                "default": None,
                "enum": [None, "add", "mul"],
                "description": "Trend component type"
            },
            "seasonal": {
                "type": ["string", "null"],
                "default": None,
                "enum": [None, "add", "mul"],
                "description": "Seasonal component type"
            },
            "seasonal_periods": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 2,
                "description": "Number of periods in a season"
            },
            "smoothing_level": {
                "type": ["number", "null"],
                "default": None,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Smoothing parameter for level (alpha)"
            },
            "smoothing_trend": {
                "type": ["number", "null"],
                "default": None,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Smoothing parameter for trend (beta)"
            },
            "smoothing_seasonal": {
                "type": ["number", "null"],
                "default": None,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Smoothing parameter for seasonal (gamma)"
            },
            "damped_trend": {
                "type": "boolean",
                "default": False,
                "description": "Apply damping to trend"
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
            "trend",
            "seasonality"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(n), Prediction: O(h), where n=samples, h=forecast horizon"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Hyndman et al., 2008. Forecasting with exponential smoothing. Springer.",
            "Gardner Jr, 2006. Exponential smoothing: The state of the art. Journal of Forecasting."
        ]

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> "ExponentialSmoothing":
        """Fit the exponential smoothing model to time series data.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values to fit.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        self : ExponentialSmoothing
            Fitted estimator.
        """
        y = np.asarray(y).ravel()
        self.n_obs_ = len(y)

        if self.seasonal is not None and self.seasonal_periods is None:
            raise ValueError("seasonal_periods must be specified when seasonal is not None")

        if self.seasonal_periods is not None and self.seasonal_periods >= self.n_obs_:
            raise ValueError("seasonal_periods must be < number of observations")

        # Initialize smoothing parameters
        alpha = self.smoothing_level if self.smoothing_level is not None else 0.2
        beta = self.smoothing_trend if self.smoothing_trend is not None else 0.1
        gamma = self.smoothing_seasonal if self.smoothing_seasonal is not None else 0.1

        self.params_ = {"alpha": alpha, "beta": beta, "gamma": gamma}

        # Initialize components
        if self.seasonal is not None:
            m = self.seasonal_periods
            # Initial level: average of first season
            self.level_ = np.mean(y[:m])

            # Initial trend: average of differences between consecutive seasons
            if self.trend is not None:
                if len(y) >= 2 * m:
                    self.trend_ = np.mean([(y[i + m] - y[i]) / m for i in range(m)])
                else:
                    self.trend_ = 0.0
            else:
                self.trend_ = None

            # Initial seasonal components
            self.seasonal_ = np.zeros(m)
            if self.seasonal == "add":
                for i in range(m):
                    self.seasonal_[i] = y[i] - self.level_
            else:  # multiplicative
                for i in range(m):
                    self.seasonal_[i] = y[i] / self.level_ if self.level_ != 0 else 1.0
        else:
            # No seasonality
            self.level_ = y[0]
            if self.trend is not None:
                self.trend_ = y[1] - y[0] if len(y) > 1 else 0.0
            else:
                self.trend_ = None
            self.seasonal_ = None

        # Apply smoothing equations
        self.fitted_values_ = np.zeros(self.n_obs_)

        for t in range(self.n_obs_):
            # Forecast for time t
            forecast = self._forecast_one_step(t)
            self.fitted_values_[t] = forecast

            # Update components
            self._update_components(t, y[t], alpha, beta, gamma)

        # Compute residuals
        self.resid_ = y - self.fitted_values_

        self._is_fitted = True
        return self

    def _forecast_one_step(self, t: int) -> float:
        """Forecast one step ahead from current state.

        Parameters
        ----------
        t : int
            Current time index.

        Returns
        -------
        forecast : float
            One-step-ahead forecast.
        """
        forecast = self.level_

        # Add trend
        if self.trend_ is not None:
            forecast += self.trend_

        # Add or multiply seasonal
        if self.seasonal_ is not None:
            m = self.seasonal_periods
            season_idx = t % m
            if self.seasonal == "add":
                forecast += self.seasonal_[season_idx]
            else:  # multiplicative
                forecast *= self.seasonal_[season_idx]

        return forecast

    def _update_components(self, t: int, y_t: float, alpha: float, beta: float, gamma: float):
        """Update level, trend, and seasonal components.

        Parameters
        ----------
        t : int
            Current time index.
        y_t : float
            Observed value at time t.
        alpha : float
            Level smoothing parameter.
        beta : float
            Trend smoothing parameter.
        gamma : float
            Seasonal smoothing parameter.
        """
        if self.seasonal is not None:
            m = self.seasonal_periods
            season_idx = t % m

            # Update level
            if self.seasonal == "add":
                level_new = alpha * (y_t - self.seasonal_[season_idx]) + (1 - alpha) * (
                    self.level_ + (self.trend_ if self.trend_ is not None else 0)
                )
            else:  # multiplicative
                level_new = alpha * (y_t / self.seasonal_[season_idx] if self.seasonal_[season_idx] != 0 else y_t) + (
                    1 - alpha
                ) * (self.level_ + (self.trend_ if self.trend_ is not None else 0))

            # Update trend
            if self.trend_ is not None:
                trend_new = beta * (level_new - self.level_) + (1 - beta) * self.trend_
            else:
                trend_new = None

            # Update seasonal
            if self.seasonal == "add":
                seasonal_new = gamma * (y_t - level_new) + (1 - gamma) * self.seasonal_[season_idx]
            else:  # multiplicative
                seasonal_new = gamma * (y_t / level_new if level_new != 0 else 1.0) + (
                    1 - gamma
                ) * self.seasonal_[season_idx]

            self.seasonal_[season_idx] = seasonal_new
            self.level_ = level_new
            self.trend_ = trend_new

        else:
            # No seasonality
            # Update level
            level_new = alpha * y_t + (1 - alpha) * (
                self.level_ + (self.trend_ if self.trend_ is not None else 0)
            )

            # Update trend
            if self.trend_ is not None:
                trend_new = beta * (level_new - self.level_) + (1 - beta) * self.trend_
            else:
                trend_new = None

            self.level_ = level_new
            self.trend_ = trend_new

    def predict(self, steps: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast future values using the fitted smoothing model.

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

        forecast = np.zeros(steps)

        for h in range(1, steps + 1):
            pred = self.level_

            # Add trend
            if self.trend_ is not None:
                pred += h * self.trend_

            # Add or multiply seasonal
            if self.seasonal_ is not None:
                m = self.seasonal_periods
                season_idx = (self.n_obs_ + h - 1) % m
                if self.seasonal == "add":
                    pred += self.seasonal_[season_idx]
                else:  # multiplicative
                    pred *= self.seasonal_[season_idx]

            forecast[h - 1] = pred

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
