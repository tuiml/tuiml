"""Seasonal-Trend decomposition using LOESS (STL) for time series."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List
from tuiml.base.algorithms import Regressor, regressor

@regressor(tags=["timeseries", "decomposition", "seasonality"], version="1.0.0")
class STLDecomposition(Regressor):
    r"""
    Seasonal-Trend decomposition using LOESS (STL) for **time series
    decomposition**.

    STL is a **robust decomposition** method for time series that extracts
    **trend**, **seasonal**, and **remainder** components. It uses locally
    weighted regression (**LOESS**) for smoothing, which allows it to handle
    non-linear trends and changing seasonality.

    Overview
    --------
    The STL decomposition procedure works as follows:

    1. Detrend the series by subtracting the current trend estimate.
    2. Extract the seasonal component by smoothing cycle-subseries
       (one subseries per seasonal period position) using LOESS.
    3. Apply a low-pass filter to the seasonal component.
    4. Deseasonalize the series by subtracting the seasonal component.
    5. Extract the trend by LOESS smoothing of the deseasonalized series.
    6. Repeat steps 1-5 (inner loop) to refine components.
    7. Optionally apply robustness weights (outer loop) to reduce the
       influence of outliers, then re-run the inner loop.

    Theory
    ------
    The STL algorithm assumes an additive decomposition model:

    .. math::
        Y_t = T_t + S_t + R_t

    where:
    
    - :math:`Y_t`: The observed time series value at time :math:`t`.
    - :math:`T_t`: The trend component, representing long-term progression.
    - :math:`S_t`: The seasonal component, representing repeating patterns.
    - :math:`R_t`: The remainder (residual) component, representing noise 
      or irregular variations.

    The algorithm consists of two recursive loops:
    
    1. **Inner Loop**: Iteratively updates the trend and seasonal 
       components using LOESS smoothing and filtering.
    2. **Outer Loop**: Computes robustness weights to reduce the 
       impact of outliers in the subsequent inner loop iterations.

    Parameters
    ----------
    period : int
        The period of the seasonality (e.g., 12 for monthly data).
    seasonal : int, default=7
        The length of the seasonal smoother window. Must be an odd integer.
    trend : int, optional, default=None
        The length of the trend smoother window. Must be an odd integer. 
        If None, a default value is calculated based on the period and 
        seasonal parameters.
    low_pass : int, optional, default=None
        The length of the low-pass filter window. Must be an odd integer.
    seasonal_deg : {0, 1}, default=1
        The degree of locally-fitted polynomial for seasonal smoothing.
    trend_deg : {0, 1}, default=1
        The degree of locally-fitted polynomial for trend smoothing.
    low_pass_deg : {0, 1}, default=1
        The degree of locally-fitted polynomial for low-pass smoothing.
    robust : bool, default=False
        If True, use robust fitting with bisquare weights in the outer loop 
         to reduce the influence of outliers.
    seasonal_jump : int, default=1
        Linear interpolation step size for seasonal smoothing to 
        increase computation speed.
    trend_jump : int, default=1
        Linear interpolation step size for trend smoothing to 
        increase computation speed.

    Attributes
    ----------
    trend_ : np.ndarray of shape (n_obs,)
        The extracted trend component.
    seasonal_ : np.ndarray of shape (n_obs,)
        The extracted seasonal component.
    resid_ : np.ndarray of shape (n_obs,)
        The extracted remainder component.
    weights_ : np.ndarray of shape (n_obs,) or None
        The robustness weights used during the fitting process (only 
        available if ``robust=True``).
    n_obs_ : int
        The total number of observations in the input series.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot k)` where :math:`n` is the number of
      samples and :math:`k` is the number of loop iterations.
    - Prediction: Not applicable (decomposition only).

    **When to use STLDecomposition:**

    - Exploratory analysis of seasonal time series
    - When the seasonal pattern may change over time
    - Data with outliers requiring robust decomposition
    - As a preprocessing step before forecasting with other models
    - When you need separate trend, seasonal, and residual components

    References
    ----------
    .. [Cleveland1990] Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990).
           **STL: A seasonal-trend decomposition.**
           *Journal of Official Statistics*, 6(1), 3-73.

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.ExponentialSmoothing` : Holt-Winters method with built-in trend and seasonal decomposition.
    :class:`~tuiml.algorithms.timeseries.Prophet` : Additive decomposition model with trend changepoints and holiday effects.
    :class:`~tuiml.algorithms.timeseries.ARIMA` : Box-Jenkins forecasting after removing trend via differencing.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries import STLDecomposition
    >>> t = np.linspace(0, 10, 100)
    >>> y = 0.5 * t + np.sin(t) + np.random.normal(size=100)
    >>> stl = STLDecomposition(period=10)
    >>> stl.fit(y)
    >>> trend, seasonal, resid = stl.trend_, stl.seasonal_, stl.resid_
    """

    def __init__(
        self,
        period: int,
        seasonal: int = 7,
        trend: int | None = None,
        low_pass: int | None = None,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        low_pass_deg: int = 1,
        robust: bool = False,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
    ):
        """Initialize STL Decomposition with specified smoothing parameters.

        Parameters
        ----------
        period : int
            Seasonal period length.
        seasonal : int, default=7
            Seasonal smoother length.
        trend : int, optional, default=None
            Trend smoother length.
        low_pass : int, optional, default=None
            Low-pass filter length.
        seasonal_deg : {0, 1}, default=1
            Seasonal smoother polynomial degree.
        trend_deg : {0, 1}, default=1
            Trend smoother polynomial degree.
        low_pass_deg : {0, 1}, default=1
            Low-pass smoother polynomial degree.
        robust : bool, default=False
            Enable robust fitting.
        seasonal_jump : int, default=1
            Seasonal smoother step size.
        trend_jump : int, default=1
            Trend smoother step size.
        """
        super().__init__()
        self.period = period
        self.seasonal = seasonal
        self.trend = trend
        self.low_pass = low_pass
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump

        # Fitted attributes
        self.trend_ = None
        self.seasonal_ = None
        self.resid_ = None
        self.weights_ = None
        self.n_obs_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "period": {
                "type": "integer",
                "minimum": 2,
                "description": "Length of the seasonal period"
            },
            "seasonal": {
                "type": "integer",
                "default": 7,
                "minimum": 3,
                "description": "Seasonal smoother length (must be odd)"
            },
            "trend": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 3,
                "description": "Trend smoother length (must be odd)"
            },
            "seasonal_deg": {
                "type": "integer",
                "default": 1,
                "minimum": 0,
                "maximum": 1,
                "description": "Seasonal smoother polynomial degree"
            },
            "trend_deg": {
                "type": "integer",
                "default": 1,
                "minimum": 0,
                "maximum": 1,
                "description": "Trend smoother polynomial degree"
            },
            "robust": {
                "type": "boolean",
                "default": False,
                "description": "Enable robust fitting"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric",
            "timeseries",
            "decomposition",
            "univariate",
            "seasonality",
            "trend_extraction"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(n * k), Prediction: N/A, where n=samples, k=smoother length"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Cleveland et al., 1990. STL: A seasonal-trend decomposition. Journal of Official Statistics."
        ]

    def _loess_smooth(
        self,
        y: np.ndarray,
        span: int,
        degree: int = 1,
        jump: int = 1,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply LOESS (locally weighted regression) smoothing.

        Parameters
        ----------
        y : np.ndarray
            Values to smooth.
        span : int
            Smoothing window size.
        degree : int, default=1
            Polynomial degree (0 or 1).
        jump : int, default=1
            Step size for computation.
        weights : np.ndarray or None
            Robustness weights.

        Returns
        -------
        smoothed : np.ndarray
            Smoothed values.
        """
        n = len(y)
        smoothed = np.zeros(n)

        if weights is None:
            weights = np.ones(n)

        # Half-window size
        half_span = span // 2

        for i in range(0, n, jump):
            # Define window
            left = max(0, i - half_span)
            right = min(n - 1, i + half_span)

            # Get data in window
            idx = np.arange(left, right + 1)
            y_window = y[idx]
            w_window = weights[idx]

            # Distances from center point
            distances = np.abs(idx - i)
            max_dist = np.max(distances) if len(distances) > 0 else 1

            # Tricube weight function
            if max_dist > 0:
                tricube_weights = (1 - (distances / max_dist) ** 3) ** 3
            else:
                tricube_weights = np.ones_like(distances)

            # Combine with robustness weights
            combined_weights = tricube_weights * w_window[: len(tricube_weights)]

            # Fit weighted polynomial
            if degree == 0:
                # Weighted mean
                smoothed[i] = np.average(y_window, weights=combined_weights)
            else:
                # Weighted linear regression
                if np.sum(combined_weights) > 0:
                    x_window = idx - i
                    try:
                        # Weighted least squares
                        W = np.diag(combined_weights)
                        X = np.column_stack([np.ones_like(x_window), x_window])
                        coeffs = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y_window, rcond=None)[0]
                        smoothed[i] = coeffs[0]  # Intercept at x=0 (center point)
                    except np.linalg.LinAlgError:
                        smoothed[i] = np.average(y_window, weights=combined_weights)
                else:
                    smoothed[i] = y_window[0] if len(y_window) > 0 else 0

        # Fill in gaps if jump > 1
        if jump > 1:
            for i in range(n):
                if i % jump != 0:
                    # Linear interpolation
                    left_idx = (i // jump) * jump
                    right_idx = min(left_idx + jump, n - 1)
                    if right_idx > left_idx:
                        alpha = (i - left_idx) / (right_idx - left_idx)
                        smoothed[i] = (1 - alpha) * smoothed[left_idx] + alpha * smoothed[right_idx]
                    else:
                        smoothed[i] = smoothed[left_idx]

        return smoothed

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> "STLDecomposition":
        """Fit the STL decomposition to time series data.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values to decompose.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with estimators.

        Returns
        -------
        self : STLDecomposition
            Fitted estimator.
        """
        y = np.asarray(y).ravel()
        self.n_obs_ = len(y)

        # Set default parameters
        if self.trend is None:
            # Formula from Cleveland et al. (1990)
            self.trend = int(np.ceil(1.5 * self.period / (1 - 1.5 / self.seasonal)))
            if self.trend % 2 == 0:
                self.trend += 1  # Make odd

        if self.low_pass is None:
            self.low_pass = self.period
            if self.low_pass % 2 == 0:
                self.low_pass += 1  # Make odd

        # Make sure parameters are odd
        if self.seasonal % 2 == 0:
            self.seasonal += 1
        if self.trend % 2 == 0:
            self.trend += 1

        # Initialize
        seasonal = np.zeros(self.n_obs_)
        trend = np.zeros(self.n_obs_)
        weights = np.ones(self.n_obs_)

        # Inner loop iterations
        n_inner = 2

        # Outer loop iterations (robustness)
        n_outer = 1 if not self.robust else 15

        for outer in range(n_outer):
            for inner in range(n_inner):
                # Step 1: Detrend
                detrended = y - trend

                # Step 2: Cycle-subseries smoothing
                cycle_subseries = np.zeros((self.period, int(np.ceil(self.n_obs_ / self.period))))

                for i in range(self.n_obs_):
                    cycle_idx = i % self.period
                    subseries_idx = i // self.period
                    cycle_subseries[cycle_idx, subseries_idx] = detrended[i]

                # Smooth each subseries
                for cycle_idx in range(self.period):
                    subseries = cycle_subseries[cycle_idx, :]
                    # Remove zeros (padding)
                    n_subseries = int(np.ceil((self.n_obs_ - cycle_idx) / self.period))
                    subseries = subseries[:n_subseries]

                    # Smooth
                    if len(subseries) > 1:
                        smoothed_subseries = self._loess_smooth(
                            subseries, self.seasonal, self.seasonal_deg, jump=self.seasonal_jump
                        )
                        # Put back
                        for j, val in enumerate(smoothed_subseries):
                            idx = cycle_idx + j * self.period
                            if idx < self.n_obs_:
                                seasonal[idx] = val
                    else:
                        seasonal[cycle_idx] = subseries[0] if len(subseries) > 0 else 0

                # Step 3: Low-pass filtering of seasonal
                seasonal = self._loess_smooth(seasonal, self.low_pass, self.low_pass_deg)

                # Step 4: Deseasonalize
                deseasonalized = y - seasonal

                # Step 5: Trend smoothing
                trend = self._loess_smooth(
                    deseasonalized, self.trend, self.trend_deg, jump=self.trend_jump, weights=weights
                )

            # Robustness weights
            if self.robust and outer < n_outer - 1:
                resid = y - seasonal - trend
                abs_resid = np.abs(resid)
                mad = np.median(abs_resid)

                # Bisquare weights
                if mad > 0:
                    u = abs_resid / (6 * mad)
                    weights = np.where(u < 1, (1 - u ** 2) ** 2, 0)
                else:
                    weights = np.ones_like(resid)

        # Final components
        self.trend_ = trend
        self.seasonal_ = seasonal
        self.resid_ = y - trend - seasonal

        if self.robust:
            self.weights_ = weights

        self._is_fitted = True
        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the reconstructed series from trend and seasonal components.

        Parameters
        ----------
        X : np.ndarray, optional, default=None
            Ignored.

        Returns
        -------
        reconstructed : np.ndarray of shape (n_obs,)
            Reconstructed series :math:`T_t + S_t`.
        """
        self._check_is_fitted()
        return self.trend_ + self.seasonal_

    def fit_predict(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the decomposition and return the reconstructed series.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values to fit.
        X : np.ndarray, optional, default=None
            Ignored.

        Returns
        -------
        reconstructed : np.ndarray of shape (n_samples,)
            Reconstructed series :math:`T_t + S_t`.
        """
        self.fit(y, X)
        return self.predict(None)

    def get_components(self) -> Dict[str, np.ndarray]:
        """Get all decomposed components.

        Returns
        -------
        components : dict
            Dictionary with keys 'trend', 'seasonal', 'resid'.
        """
        self._check_is_fitted()
        return {
            "trend": self.trend_,
            "seasonal": self.seasonal_,
            "resid": self.resid_
        }
