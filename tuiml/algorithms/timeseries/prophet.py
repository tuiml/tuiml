"""Prophet forecasting model for business time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from tuiml.base.algorithms import Regressor, regressor

@regressor(tags=["timeseries", "forecasting", "prophet", "additive"], version="1.0.0")
class Prophet(Regressor):
    r"""
    Prophet forecasting model for **business time series** with **trend,
    seasonality, and holiday effects**.

    Prophet is a procedure for forecasting time series data based on an
    **additive decomposition model** where non-linear trends are fit with
    yearly, weekly, and daily seasonality, plus **holiday effects**. It works
    best with time series that have strong seasonal effects and several
    seasons of historical data.

    Overview
    --------
    The Prophet modeling procedure follows these steps:

    1. Decompose the time series into trend, seasonality, and holiday
       components using an additive (or multiplicative) model.
    2. Fit a piecewise linear (or logistic) growth model for the trend,
       automatically detecting changepoints.
    3. Model seasonal patterns using Fourier series with configurable
       order for yearly, weekly, and daily periodicities.
    4. Incorporate holiday effects as indicator variables with prior
       regularization.
    5. Generate forecasts by summing the extrapolated components and
       optionally produce uncertainty intervals via simulation.

    Theory
    ------
    Prophet uses a decomposable time series model with three main 
    structural components: trend, seasonality, and holidays.

    .. math::
        y(t) = g(t) + s(t) + h(t) + \epsilon_t

    where:
    
    - :math:`g(t)`: The trend function which models non-periodic changes 
      in the value of the time series.
    - :math:`s(t)`: Periodic changes (e.g., weekly and yearly seasonality).
    - :math:`h(t)`: The effects of holidays which occur on potentially 
      irregular schedules over one or more days.
    - :math:`\epsilon_t`: The error term represents any idiosyncratic 
      changes which are not accommodated by the model (assumed to be 
      normally distributed).

    **Trend Component** :math:`g(t)`:
    Prophet implements a piecewise linear growth model:
    
    .. math::
        g(t) = (k + a(t)^T \delta)t + (m + a(t)^T \gamma)

    **Seasonality Component** :math:`s(t)`:
    The seasonal component is modeled using Fourier series:
    
    .. math::
        s(t) = \sum_{n=1}^N \left( a_n \cos\left(\\frac{2\pi n t}{P}\\right) + b_n \sin\left(\\frac{2\pi n t}{P}\\right) \\right)

    Parameters
    ----------
    growth : {"linear", "logistic"}, default="linear"
        The trend growth model.
    changepoints : list of str, optional, default=None
        List of dates at which to include potential changepoints.
    n_changepoints : int, default=25
        Number of potential changepoints to automatically detect.
    changepoint_range : float, default=0.8
        Proportion of history in which trend changepoints are allowed.
    yearly_seasonality : bool, int, or "auto", default="auto"
        Fit yearly seasonality.
    weekly_seasonality : bool, int, or "auto", default="auto"
        Fit weekly seasonality.
    daily_seasonality : bool, int, or "auto", default="auto"
        Fit daily seasonality.
    seasonality_mode : {"additive", "multiplicative"}, default="additive"
        How seasonality components are integrated into the forecast.
    seasonality_prior_scale : float, default=10.0
        Parameter modulating the strength of the seasonality model.
    changepoint_prior_scale : float, default=0.05
        Parameter modulating the flexibility of the automatic changepoint 
        selection.
    holidays_prior_scale : float, default=10.0
        Parameter modulating the strength of the holiday effects.
    mcmc_samples : int, default=0
        If > 0, will perform full Bayesian sampling with the specified 
        number of MCMC samples.
    interval_width : float, default=0.80
        Width of the uncertainty intervals provided for the forecast.
    uncertainty_samples : int, default=1000
        Number of simulated draws used to estimate uncertainty intervals.

    Attributes
    ----------
    trend_ : np.ndarray
        The fitted trend component.
    seasonal_ : np.ndarray
        The fitted seasonal component.
    params_ : dict
        The fitted model parameters.
    changepoints_ : np.ndarray
        The detected changepoint locations.
    n_features_in_ : int
        The number of input features (always 1 for univariate).

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot k)` where :math:`n` is the number of
      samples and :math:`k` is the number of features/holiday effects.
    - Prediction: :math:`O(h)` where :math:`h` is the forecast horizon.

    **When to use Prophet:**

    - Business time series with strong seasonal patterns (yearly, weekly, daily)
    - Data with holiday effects or known special events
    - Time series with missing values or outliers
    - When an analyst-friendly, easily tunable model is desired
    - Long-horizon forecasting where trend changepoints are expected

    References
    ----------
    .. [Taylor2018] Taylor, S. J., & Letham, B. (2018). **Forecasting at scale.**
           *The American Statistician*, 72(1), 37-45.
    .. [Prophet] Facebook Prophet Documentation: https://facebook.github.io/prophet/

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.ExponentialSmoothing` : Classical smoothing methods with trend and seasonality.
    :class:`~tuiml.algorithms.timeseries.STLDecomposition` : Decomposition of trend and seasonal components using LOESS.
    :class:`~tuiml.algorithms.timeseries.ARIMA` : Box-Jenkins approach to forecasting non-stationary series.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tuiml.algorithms.timeseries import Prophet
    >>> # Generate synthetic data
    >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
    >>> y = np.arange(100) * 0.1 + np.random.normal(size=100)
    >>> model = Prophet()
    >>> model.fit(y, dates=dates)
    >>> forecast = model.predict(steps=10)
    """

    def __init__(
        self,
        growth: str = "linear",
        changepoints: List[str] | None = None,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        yearly_seasonality: bool | int | str = "auto",
        weekly_seasonality: bool | int | str = "auto",
        daily_seasonality: bool | int | str = "auto",
        seasonality_mode: str = "additive",
        seasonality_prior_scale: float = 10.0,
        changepoint_prior_scale: float = 0.05,
        holidays_prior_scale: float = 10.0,
        mcmc_samples: int = 0,
        interval_width: float = 0.8,
        uncertainty_samples: int = 1000,
    ):
        """Initialize Prophet model with specified components and priors.

        Parameters
        ----------
        growth : {"linear", "logistic"}, default="linear"
            Trend model.
        changepoints : list of str, optional, default=None
            Dates for potential changepoints.
        n_changepoints : int, default=25
            Number of potential changepoints to automatically detect.
        changepoint_range : float, default=0.8
            Proportion of history for detecting changepoints.
        yearly_seasonality : bool, int, or "auto", default="auto"
            Fit yearly seasonality.
        weekly_seasonality : bool, int, or "auto", default="auto"
            Fit weekly seasonality.
        daily_seasonality : bool, int, or "auto", default="auto"
            Fit daily seasonality.
        seasonality_mode : {"additive", "multiplicative"}, default="additive"
            Integration of seasonality.
        seasonality_prior_scale : float, default=10.0
            Seasonality model strength.
        changepoint_prior_scale : float, default=0.05
            Trend flexibility.
        holidays_prior_scale : float, default=10.0
            Holiday effect strength.
        mcmc_samples : int, default=0
            Number of MCMC samples.
        interval_width : float, default=0.8
            Width of uncertainty intervals.
        uncertainty_samples : int, default=1000
            Number of draws for uncertainty estimating.
        """
        super().__init__()
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples

        # Fitted attributes
        self.trend_ = None
        self.seasonal_ = None
        self.params_ = None
        self.changepoints_ = None
        self.n_features_in_ = None
        self._model = None
        self._use_prophet = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "growth": {
                "type": "string",
                "default": "linear",
                "enum": ["linear", "logistic"],
                "description": "Type of trend component"
            },
            "n_changepoints": {
                "type": "integer",
                "default": 25,
                "minimum": 0,
                "description": "Number of potential changepoints"
            },
            "changepoint_range": {
                "type": "number",
                "default": 0.8,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Proportion of history for changepoints"
            },
            "yearly_seasonality": {
                "oneOf": [
                    {"type": "boolean"},
                    {"type": "integer", "minimum": 1},
                    {"type": "string", "enum": ["auto"]}
                ],
                "default": "auto",
                "description": "Yearly seasonality setting"
            },
            "weekly_seasonality": {
                "oneOf": [
                    {"type": "boolean"},
                    {"type": "integer", "minimum": 1},
                    {"type": "string", "enum": ["auto"]}
                ],
                "default": "auto",
                "description": "Weekly seasonality setting"
            },
            "daily_seasonality": {
                "oneOf": [
                    {"type": "boolean"},
                    {"type": "integer", "minimum": 1},
                    {"type": "string", "enum": ["auto"]}
                ],
                "default": "auto",
                "description": "Daily seasonality setting"
            },
            "seasonality_mode": {
                "type": "string",
                "default": "additive",
                "enum": ["additive", "multiplicative"],
                "description": "Seasonality mode"
            },
            "seasonality_prior_scale": {
                "type": "number",
                "default": 10.0,
                "minimum": 0.0,
                "description": "Seasonality strength"
            },
            "changepoint_prior_scale": {
                "type": "number",
                "default": 0.05,
                "minimum": 0.0,
                "description": "Trend flexibility"
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
            "seasonality",
            "holidays",
            "uncertainty",
            "missing_values"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(n * k), Prediction: O(h), where n=samples, k=features, h=forecast horizon"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Taylor & Letham, 2018. Forecasting at scale. The American Statistician.",
            "Facebook Prophet: https://facebook.github.io/prophet/"
        ]

    def fit(
        self,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        _X: Optional[np.ndarray] = None
    ) -> "Prophet":
        """Fit the Prophet model to time series data.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values to fit.
        dates : pd.DatetimeIndex, optional, default=None
            Datetime index for the series. Recommended for best results.
        _X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        self : Prophet
            Fitted estimator.
        """
        y = np.asarray(y).ravel()
        self.n_features_in_ = 1

        # Create dates if not provided
        if dates is None:
            dates = pd.date_range(start='2020-01-01', periods=len(y), freq='D')
        elif not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)

        # Try to import prophet
        try:
            from prophet import Prophet as FBProphet
            self._use_prophet = True
        except ImportError:
            self._use_prophet = False
            print("Warning: prophet not installed. Using simplified implementation.")

        if self._use_prophet:
            # Use Facebook Prophet
            df = pd.DataFrame({
                'ds': dates,
                'y': y
            })

            # Create model
            self._model = FBProphet(
                growth=self.growth,
                changepoints=self.changepoints,
                n_changepoints=self.n_changepoints,
                changepoint_range=self.changepoint_range,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                seasonality_mode=self.seasonality_mode,
                seasonality_prior_scale=self.seasonality_prior_scale,
                changepoint_prior_scale=self.changepoint_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                mcmc_samples=self.mcmc_samples,
                interval_width=self.interval_width,
                uncertainty_samples=self.uncertainty_samples
            )

            # Fit model
            self._model.fit(df, show_progress=False)

            # Extract components
            forecast = self._model.predict(df)
            self.trend_ = forecast['trend'].values

            # Extract seasonal component
            seasonal_cols = [col for col in forecast.columns if 'seasonal' in col or col == 'weekly' or col == 'yearly']
            if seasonal_cols:
                self.seasonal_ = forecast[seasonal_cols].sum(axis=1).values
            else:
                self.seasonal_ = np.zeros_like(self.trend_)

            self.changepoints_ = self._model.changepoints.values if hasattr(self._model, 'changepoints') else None
            self.params_ = self._model.params if hasattr(self._model, 'params') else {}

        else:
            # Simplified implementation without prophet library
            # Use linear trend + simple seasonality
            n = len(y)
            t = np.arange(n)

            # Fit linear trend
            A = np.column_stack([np.ones(n), t])
            trend_params = np.linalg.lstsq(A, y, rcond=None)[0]
            self.trend_ = A @ trend_params

            # Simple yearly seasonality (if enough data)
            detrended = y - self.trend_
            if len(y) >= 365:
                period = 365
                seasonal_pattern = np.zeros(period)
                for i in range(period):
                    seasonal_pattern[i] = np.mean(detrended[i::period])
                self.seasonal_ = np.tile(seasonal_pattern, int(np.ceil(n / period)))[:n]
            else:
                self.seasonal_ = np.zeros(n)

            self.changepoints_ = None
            self.params_ = {'intercept': trend_params[0], 'slope': trend_params[1]}

        self._is_fitted = True
        return self

    def predict(
        self,
        steps: int = 1,
        freq: str = "D",
        include_history: bool = False
    ) -> np.ndarray | pd.DataFrame:
        """Forecast future values using the fitted Prophet model.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        freq : str, default="D"
            Frequency of predictions ('D' for daily, 'W' for weekly, etc.).
        include_history : bool, default=False
            If True, return a DataFrame containing the forecast and its 
            components (trend, seasonal, etc.). If False, return only 
            the forecasted values as an array.

        Returns
        -------
        forecast : np.ndarray or pd.DataFrame
            The forecasted values or a detailed components DataFrame.
        """
        self._check_is_fitted()

        if self._use_prophet and self._model is not None:
            # Use Prophet's predict
            future = self._model.make_future_dataframe(periods=steps, freq=freq)
            forecast_df = self._model.predict(future)

            if include_history:
                return forecast_df
            else:
                # Return only the forecasted values (last 'steps' rows)
                return forecast_df['yhat'].values[-steps:]

        else:
            # Simplified prediction
            n_history = len(self.trend_)
            forecast = np.zeros(steps)

            for i in range(steps):
                # Linear trend extrapolation
                if self.params_ is not None:
                    trend_val = self.params_['intercept'] + self.params_['slope'] * (n_history + i)
                else:
                    trend_val = self.trend_[-1]

                # Seasonal component (repeat pattern)
                if len(self.seasonal_) > 0:
                    seasonal_val = self.seasonal_[(n_history + i) % len(self.seasonal_)]
                else:
                    seasonal_val = 0

                forecast[i] = trend_val + seasonal_val

            if include_history:
                # Create simple DataFrame
                dates = pd.date_range(start='2020-01-01', periods=n_history + steps, freq=freq)
                df = pd.DataFrame({
                    'ds': dates,
                    'yhat': np.concatenate([self.trend_ + self.seasonal_, forecast])
                })
                return df
            else:
                return forecast

    def fit_predict(self, y: np.ndarray, steps: int = 1, dates: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """Fit the model and forecast future values in one step.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values to fit.
        steps : int, default=1
            Number of future time steps to forecast.
        dates : pd.DatetimeIndex, optional
            Datetime index for the series.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecasted values.
        """
        self.fit(y, dates=dates)
        return self.predict(steps=steps, include_history=False)

    def plot_components(self):
        """Plot forecast components (trend, seasonality).

        Requires matplotlib and prophet to be installed.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure with component plots.
        """
        self._check_is_fitted()

        if self._use_prophet and self._model is not None:
            return self._model.plot_components(self._model.predict(self._model.history))
        else:
            raise NotImplementedError("plot_components requires prophet library")
