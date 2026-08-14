"""Vector Autoregression (VAR) for multivariate time series forecasting."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Union

from tuiml.base.algorithms import Regressor, regressor


@regressor(tags=["timeseries", "forecasting", "multivariate"], version="1.0.0")
class VAR(Regressor):
    """
    Vector Autoregression for **multivariate time series forecasting**.

    A VAR treats every series in a panel as a linear function of the
    **past values of all the series**, so cross-series feedback is modelled
    explicitly rather than assumed away. It is the multivariate
    generalisation of the univariate AR model and the standard workhorse
    for macroeconomic and sensor-panel forecasting.

    Overview
    --------
    1. Stack the ``p`` lagged observations of every series into a single
       design matrix :math:`Z`, one row per usable time point.
    2. Estimate the coefficient matrices by ordinary least squares.
    3. Optionally choose the lag order over ``1..maxlags`` by AIC or BIC,
       comparing all candidates on an identical sample.
    4. Forecast recursively by feeding each prediction back in as the most
       recent lag (iterating the companion form).

    Theory
    ------
    A VAR of order :math:`p` on :math:`k` series is

    .. math::
        y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \\dots + A_p y_{t-p}
              + \\varepsilon_t ,
        \\qquad \\varepsilon_t \\sim (0, \\Sigma),

    with :math:`y_t \\in \\mathbb{R}^k`, :math:`A_i \\in
    \\mathbb{R}^{k \\times k}` and :math:`c \\in \\mathbb{R}^k`. Writing
    :math:`z_t = (1, y_{t-1}^\\top, \\dots, y_{t-p}^\\top)^\\top` and
    stacking the rows gives :math:`Y = Z B + E`, whose least-squares
    solution is

    .. math::
        \\hat{B} = (Z^\\top Z)^{-1} Z^\\top Y .

    Because **every equation shares exactly the same regressors**, the
    seemingly-unrelated-regressions correction collapses and
    **equation-by-equation OLS is the exact conditional maximum-likelihood
    estimator** for a Gaussian VAR with an identical lag structure across
    equations -- there is nothing to gain from a joint GLS step.

    Lag order is chosen by minimising an information criterion built from
    the ML residual covariance :math:`\\hat{\\Sigma}_p = E^\\top E / T`:

    .. math::
        \\mathrm{AIC}(p) = \\ln|\\hat{\\Sigma}_p| + \\frac{2 p k^2}{T},
        \\qquad
        \\mathrm{BIC}(p) = \\ln|\\hat{\\Sigma}_p| + \\frac{\\ln(T) p k^2}{T}.

    Multi-step forecasts iterate the companion form: the :math:`h`-step
    forecast uses previous forecasts in place of the unobserved future
    values, which is the conditional mean :math:`E[y_{n+h} \\mid y_{1:n}]`.

    Parameters
    ----------
    lags : int or "auto", default=1
        Number of lags :math:`p`. ``"auto"`` selects the order over
        ``1..maxlags`` by the criterion given in ``ic``.
    maxlags : int, optional, default=None
        Upper bound for automatic order selection. When ``None`` it is
        ``min(10, (n_timepoints - 1) // (n_series + 1))``, floored at 1.
    ic : {"aic", "bic"}, default="aic"
        Information criterion used when ``lags="auto"``.
    trend : {"c", "n"}, default="c"
        ``"c"`` includes an intercept, ``"n"`` omits it.

    Attributes
    ----------
    coefs_ : np.ndarray of shape (lags, n_series, n_series)
        Coefficient matrices, ``coefs_[i - 1]`` is :math:`A_i`.
    intercept_ : np.ndarray of shape (n_series,)
        Estimated intercept :math:`c`; zeros when ``trend="n"``.
    lags_ : int
        Lag order actually used.
    n_series_ : int
        Number of series in the panel.
    n_obs_ : int
        Number of time points in the training data.
    sigma_ : np.ndarray of shape (n_series, n_series)
        Maximum-likelihood residual covariance :math:`E^\\top E / T`.
    ic_values_ : dict or None
        Criterion value per candidate lag when ``lags="auto"``.
    endog_ : np.ndarray of shape (n_obs, n_series)
        The training panel, retained to seed recursive forecasts.
    input_was_1d_ : bool
        Whether ``fit`` received a 1-D series, in which case ``predict``
        returns a 1-D forecast.
    fitted_values_ : np.ndarray of shape (n_obs - lags, n_series)
        In-sample one-step-ahead forecasts.
    resid_ : np.ndarray of shape (n_obs - lags, n_series)
        In-sample residuals.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(T d^2 + d^3)` with :math:`d = pk + 1` regressors
      and :math:`T` usable time points; order selection multiplies this by
      the number of candidate lags.
    - Prediction: :math:`O(h p k^2)`.

    **When to use VAR:**

    - Several series that plausibly drive one another (demand and price,
      several sensors on one machine, macroeconomic aggregates).
    - When you want forecasts for the whole panel jointly rather than one
      model per series.
    - Requires roughly stationary series -- difference or detrend first if
      the panel trends; a unit root makes the OLS estimates unreliable.
    - Parameters grow as :math:`p k^2`, so keep :math:`p` small on wide
      panels or the fit overfits.

    References
    ----------
    .. [Sims1980] Sims, C. A. (1980). **Macroeconomics and reality.**
           *Econometrica*, 48(1), 1-48. :doi:`10.2307/1912017`
    .. [Lutkepohl2005] Lutkepohl, H. (2005). **New Introduction to Multiple
           Time Series Analysis.** *Springer*.
           :doi:`10.1007/978-3-540-27752-1`
    .. [Hamilton1994] Hamilton, J. D. (1994). **Time Series Analysis.**
           *Princeton University Press*.

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.ARIMA` : Univariate Box-Jenkins forecasting.
    :class:`~tuiml.algorithms.timeseries.AR` : Univariate autoregression.
    :class:`~tuiml.algorithms.timeseries.theta.ThetaForecaster` : Univariate Theta method.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.var import VAR
    >>> rng = np.random.default_rng(0)
    >>> n, y = 300, np.zeros((300, 2))
    >>> A = np.array([[0.5, 0.1], [-0.2, 0.3]])
    >>> for t in range(1, n):
    ...     y[t] = A @ y[t - 1] + rng.normal(size=2)
    >>> model = VAR(lags=1).fit(y)
    >>> model.coefs_.shape
    (1, 2, 2)
    >>> model.predict(steps=4).shape
    (4, 2)
    """

    def __init__(
        self,
        lags: Union[int, str] = 1,
        maxlags: int | None = None,
        ic: str = "aic",
        trend: str = "c",
    ):
        """Initialize the VAR model.

        Parameters
        ----------
        lags : int or "auto", default=1
            Lag order, or ``"auto"`` to select it by information criterion.
        maxlags : int, optional, default=None
            Upper bound for automatic order selection.
        ic : {"aic", "bic"}, default="aic"
            Information criterion used when ``lags="auto"``.
        trend : {"c", "n"}, default="c"
            Whether to include an intercept.
        """
        super().__init__()
        self.lags = lags
        self.maxlags = maxlags
        self.ic = ic
        self.trend = trend

        # Fitted attributes
        self.coefs_ = None
        self.intercept_ = None
        self.lags_ = None
        self.n_series_ = None
        self.n_obs_ = None
        self.sigma_ = None
        self.ic_values_ = None
        self.endog_ = None
        self.input_was_1d_ = False
        self.fitted_values_ = None
        self.resid_ = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "lags": {
                "type": ["integer", "string"],
                "default": 1,
                "minimum": 1,
                "description": "Lag order, or 'auto' for information-criterion selection",
            },
            "maxlags": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 1,
                "description": "Upper bound for automatic lag-order selection",
            },
            "ic": {
                "type": "string",
                "default": "aic",
                "enum": ["aic", "bic"],
                "description": "Information criterion for lag selection",
            },
            "trend": {
                "type": "string",
                "default": "c",
                "enum": ["c", "n"],
                "description": "Include an intercept ('c') or not ('n')",
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
            "multivariate",
            "multivariate_timeseries",
            "stationary",
            "interpretable",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(T*d^2 + d^3) with d = lags*n_series + 1, Prediction: O(h*p*k^2)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Sims, 1980. Macroeconomics and reality. Econometrica 48(1), 1-48.",
            "Lutkepohl, 2005. New Introduction to Multiple Time Series Analysis. Springer.",
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_design(self, y: np.ndarray, p: int, start: int) -> tuple:
        """Stack lagged observations into an OLS design matrix.

        Parameters
        ----------
        y : np.ndarray of shape (n_timepoints, n_series)
            The panel.
        p : int
            Lag order.
        start : int
            First time index to use as a target row. Must be ``>= p``.
            Holding it fixed across candidate lags keeps information
            criteria comparable.

        Returns
        -------
        Z : np.ndarray of shape (n_timepoints - start, n_regressors)
            Design matrix, intercept first when ``trend="c"``.
        Y : np.ndarray of shape (n_timepoints - start, n_series)
            Target rows.
        """
        n, k = y.shape
        rows = n - start
        blocks = []
        if self.trend == "c":
            blocks.append(np.ones((rows, 1), dtype=float))
        for lag in range(1, p + 1):
            blocks.append(y[start - lag:n - lag, :])
        Z = np.hstack(blocks) if blocks else np.empty((rows, 0), dtype=float)
        Y = y[start:, :]
        return Z, Y

    @staticmethod
    def _ols(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Solve ``Y = Z B`` by least squares, rejecting a singular design.

        Parameters
        ----------
        Z : np.ndarray of shape (n_rows, n_regressors)
            Design matrix.
        Y : np.ndarray of shape (n_rows, n_series)
            Targets.

        Returns
        -------
        B : np.ndarray of shape (n_regressors, n_series)
            Coefficient estimates.

        Raises
        ------
        ValueError
            If the design has fewer rows than regressors, or is rank
            deficient. Returning ``pinv`` estimates here would silently
            hide a collinear or constant series.
        """
        n_rows, n_cols = Z.shape
        if n_rows < n_cols:
            raise ValueError(
                f"VAR design matrix has {n_rows} rows for {n_cols} regressors: "
                "the series is too short for this lag order"
            )
        B, _, rank, _ = np.linalg.lstsq(Z, Y, rcond=None)
        if rank < n_cols:
            raise ValueError(
                f"VAR design matrix is rank deficient (rank {rank} < {n_cols} "
                "regressors): the series are collinear or constant. Drop a "
                "duplicated series, lower `lags`, or remove `trend='c'`."
            )
        return B

    def _log_det_sigma(self, Z: np.ndarray, Y: np.ndarray) -> float:
        """Return ``ln|Sigma_hat|`` for one candidate fit.

        Parameters
        ----------
        Z : np.ndarray of shape (n_rows, n_regressors)
            Design matrix.
        Y : np.ndarray of shape (n_rows, n_series)
            Targets.

        Returns
        -------
        log_det : float
            Log determinant of the ML residual covariance, or ``inf`` when
            the covariance is singular.
        """
        B = self._ols(Z, Y)
        resid = Y - Z @ B
        sigma = resid.T @ resid / Y.shape[0]
        sign, log_det = np.linalg.slogdet(sigma)
        if sign <= 0:
            return float("inf")
        return float(log_det)

    def _select_order(self, y: np.ndarray) -> int:
        """Choose the lag order by AIC or BIC on a fixed sample.

        Parameters
        ----------
        y : np.ndarray of shape (n_timepoints, n_series)
            The panel.

        Returns
        -------
        p : int
            Selected lag order.
        """
        n, k = y.shape
        if self.ic not in ("aic", "bic"):
            raise ValueError(f"ic must be 'aic' or 'bic', got {self.ic!r}")

        if self.maxlags is not None:
            maxlags = int(self.maxlags)
        else:
            maxlags = min(10, (n - 1) // (k + 1))
        maxlags = max(1, maxlags)
        # Every candidate must leave enough rows to be estimable.
        while maxlags > 1 and (n - maxlags) < (maxlags * k + 1):
            maxlags -= 1

        T = n - maxlags
        values: Dict[int, float] = {}
        for p in range(1, maxlags + 1):
            Z, Y = self._build_design(y, p, maxlags)
            try:
                log_det = self._log_det_sigma(Z, Y)
            except ValueError:
                continue
            penalty = 2.0 if self.ic == "aic" else float(np.log(T))
            values[p] = log_det + penalty * p * k * k / T

        if not values:
            raise ValueError(
                "VAR could not estimate any candidate lag order: the series "
                "is too short or rank deficient"
            )
        self.ic_values_ = values
        return min(values, key=lambda p: (values[p], p))

    # ------------------------------------------------------------------
    # Fitting and prediction
    # ------------------------------------------------------------------

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> "VAR":
        """Fit the VAR by equation-by-equation OLS on stacked lags.

        Parameters
        ----------
        y : np.ndarray of shape (n_timepoints, n_series) or (n_timepoints,)
            The panel of series. A 1-D array is treated as a single-series
            panel and ``predict`` then returns a 1-D forecast.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        self : VAR
            Fitted estimator.
        """
        y = np.asarray(y, dtype=float)
        self.input_was_1d_ = y.ndim == 1
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim != 2:
            raise ValueError("y must be 1-D or 2-D (n_timepoints, n_series)")
        if not np.isfinite(y).all():
            raise ValueError("y must not contain NaN or infinite values")
        if self.trend not in ("c", "n"):
            raise ValueError(f"trend must be 'c' or 'n', got {self.trend!r}")

        n, k = y.shape
        self.n_obs_, self.n_series_ = n, k
        self.endog_ = y.copy()
        self.ic_values_ = None

        if isinstance(self.lags, str):
            if self.lags != "auto":
                raise ValueError(f"lags must be an int or 'auto', got {self.lags!r}")
            p = self._select_order(y)
        else:
            p = int(self.lags)
            if p < 1:
                raise ValueError("lags must be >= 1")
        if n <= p:
            raise ValueError(
                f"VAR needs more than {p} time points to fit {p} lags, got {n}"
            )
        self.lags_ = p

        Z, Y = self._build_design(y, p, p)
        B = self._ols(Z, Y)

        offset = 1 if self.trend == "c" else 0
        self.intercept_ = B[0, :].copy() if offset else np.zeros(k, dtype=float)
        coefs = np.empty((p, k, k), dtype=float)
        for i in range(p):
            coefs[i] = B[offset + i * k: offset + (i + 1) * k, :].T
        self.coefs_ = coefs

        self.fitted_values_ = Z @ B
        self.resid_ = Y - self.fitted_values_
        self.sigma_ = self.resid_.T @ self.resid_ / Y.shape[0]

        self._is_fitted = True
        return self

    def predict(self, steps: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast the panel forward by iterating the companion form.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        forecast : np.ndarray of shape (steps, n_series), or (steps,)
            Forecasted values. The 1-D shape is returned when ``fit``
            received a 1-D series.
        """
        self._check_is_fitted()
        if steps < 1:
            raise ValueError("steps must be >= 1")

        p, k = self.lags_, self.n_series_
        # history[-1] is the most recent observation.
        history = list(self.endog_[-p:])
        out = np.empty((steps, k), dtype=float)
        for h in range(steps):
            pred = self.intercept_.copy()
            for i in range(p):
                pred = pred + self.coefs_[i] @ history[-(i + 1)]
            out[h] = pred
            history.append(pred)
            history = history[-p:]

        if self.input_was_1d_:
            return out[:, 0]
        return out

    def fit_predict(self, y: np.ndarray, steps: int = 1) -> np.ndarray:
        """Fit the model and forecast in one call.

        Parameters
        ----------
        y : np.ndarray of shape (n_timepoints, n_series) or (n_timepoints,)
            The panel of series.
        steps : int, default=1
            Number of future time steps to forecast.

        Returns
        -------
        forecast : np.ndarray of shape (steps, n_series), or (steps,)
            Forecasted values.
        """
        self.fit(y)
        return self.predict(steps)
