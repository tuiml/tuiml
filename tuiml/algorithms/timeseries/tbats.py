"""TBATS: trigonometric seasonal exponential smoothing with Box-Cox and ARMA errors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from tuiml.base.algorithms import Regressor, regressor

__all__ = ["TBATS"]


# ---------------------------------------------------------------------------
# Box-Cox helpers
# ---------------------------------------------------------------------------

def _box_cox(y: np.ndarray, lam: float) -> np.ndarray:
    """Apply the Box-Cox transform.

    Parameters
    ----------
    y : np.ndarray
        Strictly positive series when ``lam`` implies a log or root.
    lam : float
        Transform parameter :math:`\\lambda`.

    Returns
    -------
    z : np.ndarray
        Transformed series.
    """
    if abs(lam) < 1e-8:
        return np.log(y)
    return (np.power(y, lam) - 1.0) / lam


def _inv_box_cox(z: np.ndarray, lam: float) -> np.ndarray:
    """Invert the Box-Cox transform.

    Parameters
    ----------
    z : np.ndarray
        Transformed values.
    lam : float
        Transform parameter :math:`\\lambda`.

    Returns
    -------
    y : np.ndarray
        Values on the original scale.
    """
    if abs(lam) < 1e-8:
        return np.exp(z)
    base = lam * z + 1.0
    base = np.maximum(base, 1e-12)
    return np.power(base, 1.0 / lam)


def _hannan_rissanen(resid: np.ndarray, p: int, q: int) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate ARMA(p, q) coefficients for a residual series.

    Uses the two-stage Hannan-Rissanen procedure: a long autoregression
    supplies proxy innovations, which then enter a linear regression
    alongside the lagged residuals.

    Parameters
    ----------
    resid : np.ndarray of shape (n,)
        Residual series.
    p : int
        Autoregressive order.
    q : int
        Moving-average order.

    Returns
    -------
    ar : np.ndarray of shape (p,)
        Autoregressive coefficients.
    ma : np.ndarray of shape (q,)
        Moving-average coefficients.
    """
    n = len(resid)
    if p == 0 and q == 0:
        return np.zeros(0), np.zeros(0)
    order_long = int(min(max(p, q) + 3, max(1, n // 5), 12))
    if n <= 2 * order_long + p + q + 2:
        return np.zeros(p), np.zeros(q)

    def _lagmat(x, k, start):
        return np.column_stack([x[start - j - 1: len(x) - j - 1] for j in range(k)])

    Zl = _lagmat(resid, order_long, order_long)
    yl = resid[order_long:]
    coef, *_ = np.linalg.lstsq(Zl, yl, rcond=None)
    e = np.zeros(n)
    e[order_long:] = yl - Zl @ coef

    start = max(p, q) + order_long
    if n - start < p + q + 2:
        return np.zeros(p), np.zeros(q)
    cols = []
    if p:
        cols.append(_lagmat(resid, p, start))
    if q:
        cols.append(_lagmat(e, q, start))
    Z = np.hstack(cols)
    beta, *_ = np.linalg.lstsq(Z, resid[start:], rcond=None)
    ar, ma = beta[:p], beta[p:]
    # Shrink toward zero if the fitted AR part looks explosive.
    if p and np.sum(np.abs(ar)) >= 0.99:
        ar = ar * (0.95 / np.sum(np.abs(ar)))
    if q and np.sum(np.abs(ma)) >= 0.99:
        ma = ma * (0.95 / np.sum(np.abs(ma)))
    return ar, ma


@regressor(tags=["timeseries", "forecasting", "seasonal", "smoothing",
                 "trigonometric"], version="1.0.0")
class TBATS(Regressor):
    r"""
    **T**\ rigonometric seasonality, **B**\ ox-Cox transform, **A**\ RMA
    errors, **T**\ rend and **S**\ easonal components -- an exponential
    smoothing state-space model for series with **complex, multiple, high
    frequency or non-integer seasonal periods**.

    Classical seasonal models carry one state per seasonal index, so a daily
    series with a yearly cycle needs 365 states and a period of 365.25 cannot
    be expressed at all. TBATS instead represents each seasonal pattern by a
    handful of **trigonometric (Fourier) terms**, so the state count depends
    on how many harmonics the pattern needs rather than on the length of the
    period. That single change is what makes ``seasonal_periods=[7, 365.25]``
    both representable and cheap.

    Overview
    --------
    Fitting proceeds as follows:

    1. Optionally Box-Cox transform the series to stabilise the variance,
       either at a fixed :math:`\\lambda` or by a small grid search.
    2. Seed the level, trend and seasonal states by ordinary least squares
       on a design of ``[1, t, cos/sin harmonics]``, which gives the
       recursion a starting point already close to the data.
    3. Run the state-space smoothing recursion, updating level, damped trend
       and every trigonometric seasonal pair from the one-step error.
    4. Minimise the sum of squared one-step errors over the smoothing
       parameters :math:`(\\alpha, \\beta, \phi, \gamma_1, \gamma_2)` with a
       bounded optimiser.
    5. Fit ARMA errors to the residual series and add their forecast.
    6. Forecast by iterating the recursion with zero errors and inverting
       the Box-Cox transform.

    Theory
    ------
    With :math:`y_t^{(\\lambda)}` the Box-Cox transformed observation, the
    model is

    .. math::
        \\begin{aligned}
        y_t^{(\\lambda)} &= \ell_{t-1} + \phi b_{t-1}
            + \sum_{i=1}^{M} s^{(i)}_{t-1} + d_t \\\\
        \ell_t &= \ell_{t-1} + \phi b_{t-1} + \\alpha d_t \\\\
        b_t &= \phi b_{t-1} + \\beta d_t
        \end{aligned}

    where :math:`\phi` is the damping parameter and :math:`d_t` is an ARMA
    error process. Each seasonal component is the sum of :math:`k_i`
    harmonic pairs that rotate at the seasonal frequencies
    :math:`\\lambda^{(i)}_j = 2\pi j / m_i`:

    .. math::
        \\begin{aligned}
        s^{(i)}_{j,t} &= s^{(i)}_{j,t-1}\cos\\lambda^{(i)}_j
            + s^{*(i)}_{j,t-1}\sin\\lambda^{(i)}_j + \gamma^{(i)}_1 d_t \\\\
        s^{*(i)}_{j,t} &= -s^{(i)}_{j,t-1}\sin\\lambda^{(i)}_j
            + s^{*(i)}_{j,t-1}\cos\\lambda^{(i)}_j + \gamma^{(i)}_2 d_t
        \end{aligned}

    with :math:`s^{(i)}_t = \sum_{j=1}^{k_i} s^{(i)}_{j,t}`. Because
    :math:`m_i` enters only through the angle
    :math:`2\pi j/m_i`, it need not be an integer -- ``365.25`` is as valid
    as ``12``. The Box-Cox transform is

    .. math::
        y^{(\\lambda)} = \\begin{cases}
            (y^{\\lambda} - 1)/\\lambda, & \\lambda \\neq 0 \\\\
            \log y, & \\lambda = 0 .
        \end{cases}

    Parameters
    ----------
    seasonal_periods : sequence of float or float or None, default=None
        Seasonal period lengths. May be non-integer (``365.25``) and there
        may be several (``[7, 365.25]``). None disables seasonality.
    n_harmonics : sequence of int or int or None, default=None
        Number of harmonic pairs per seasonal period. None picks
        ``min(floor(m / 2), 5)`` for each period, which keeps the state
        small for very long periods.
    use_trend : bool, default=True
        Include a local linear trend component.
    damped_trend : bool, default=True
        Damp the trend with a fitted :math:`\phi \in [0.8, 1]`. Ignored
        when ``use_trend`` is False.
    box_cox : bool, default=False
        Apply a Box-Cox transform. Requires strictly positive data.
    box_cox_lambda : float or None, default=None
        Fixed :math:`\\lambda`. When None and ``box_cox`` is True,
        :math:`\\lambda` is chosen from a small grid by profile likelihood.
    use_arma_errors : bool, default=True
        Fit ARMA errors to the smoothing residuals and add their forecast.
    arma_order : tuple of (int, int), default=(1, 0)
        The ``(p, q)`` order of the residual ARMA model.
    maxiter : int, default=40
        Maximum optimiser iterations for the smoothing parameters.
    tol : float, default=1e-6
        Optimiser convergence tolerance.

    Attributes
    ----------
    params_ : dict
        Fitted smoothing parameters: ``alpha``, ``beta``, ``phi`` and the
        per-period ``gamma1``/``gamma2``.
    lambda_ : float or None
        Box-Cox parameter actually used, None when ``box_cox`` is False.
    harmonics_ : list of int
        Number of harmonic pairs used for each seasonal period.
    level_ : float
        Final level state.
    trend_ : float
        Final trend state (0.0 when ``use_trend`` is False).
    seasonal_ : np.ndarray
        Final trigonometric seasonal states, stacked as
        :math:`(s_1, \dots, s_K, s^{*}_1, \dots, s^{*}_K)`.
    ar_params_ : np.ndarray
        Fitted AR coefficients of the residual ARMA model.
    ma_params_ : np.ndarray
        Fitted MA coefficients of the residual ARMA model.
    fitted_values_ : np.ndarray of shape (n_samples,)
        One-step-ahead in-sample predictions on the original scale.
    resid_ : np.ndarray of shape (n_samples,)
        One-step-ahead errors on the transformed scale.
    sse_ : float
        Minimised sum of squared one-step errors.
    aic_ : float
        Akaike information criterion computed from ``sse_``.
    n_obs_ : int
        Number of observations supplied to :meth:`fit`.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(\\text{maxiter} \cdot k \cdot n K)` where
      :math:`K = \sum_i k_i` is the total number of harmonic pairs and
      :math:`k` the number of free smoothing parameters. Crucially it does
      **not** grow with the seasonal period lengths.
    - Prediction: :math:`O(h K)` for :math:`h` steps.

    **When to use TBATS:**

    - Multiple simultaneous seasonalities -- daily plus weekly plus yearly.
    - **Non-integer** periods such as 365.25 (leap years) or 52.18 (weeks
      per year), which seasonal ARIMA and Holt-Winters cannot represent.
    - **High-frequency** seasonality where one state per seasonal index
      would be prohibitive.
    - Multiplicative-looking variance that a Box-Cox transform can tame.

    **Simplifications relative to De Livera et al. (2011).** This
    implementation is deliberately a well-tested subset rather than a
    partial version of the whole paper:

    - The ARMA error stage is estimated in a **second pass** (Hannan-Rissanen
      on the smoothing residuals) rather than jointly with the smoothing
      parameters inside a single likelihood. Point forecasts are almost
      unaffected; standard errors from a joint fit would be tighter.
    - Model selection over the discrete choices (trend on/off, damping
      on/off, ARMA order, number of harmonics) is **not** automated by AIC
      as in the paper; those are constructor parameters.
    - Estimation minimises the sum of squared errors rather than the exact
      Gaussian likelihood, and the Box-Cox :math:`\\lambda` is chosen over a
      small grid rather than jointly optimised.

    Seasonality here is additive on the (optionally Box-Cox transformed)
    scale, which is the standard TBATS formulation -- multiplicative
    behaviour is obtained through the transform, not through a separate
    multiplicative seasonal form.

    References
    ----------
    .. [DeLivera2011] De Livera, A. M., Hyndman, R. J., & Snyder, R. D.
           (2011). **Forecasting time series with complex seasonal patterns
           using exponential smoothing.** *Journal of the American
           Statistical Association*, 106(496), 1513-1527.
           :doi:`10.1198/jasa.2011.tm09771`
    .. [Hyndman2008] Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder,
           R. D. (2008). **Forecasting with Exponential Smoothing: The State
           Space Approach.** *Springer*. :doi:`10.1007/978-3-540-71918-2`
    .. [BoxCox1964] Box, G. E. P., & Cox, D. R. (1964). **An analysis of
           transformations.** *Journal of the Royal Statistical Society:
           Series B*, 26(2), 211-243.
           :doi:`10.1111/j.2517-6161.1964.tb00553.x`

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.ExponentialSmoothing` : Holt-Winters smoothing, one state per seasonal index, integer periods only.
    :class:`~tuiml.algorithms.timeseries.SARIMAX` : Seasonal ARIMA with exogenous regressors, for a single integer seasonal period.
    :class:`~tuiml.algorithms.timeseries.Prophet` : Additive regression with Fourier seasonality and holiday effects.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.tbats import TBATS
    >>> t = np.arange(120)
    >>> y = 10.0 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12)
    >>> model = TBATS(seasonal_periods=[12]).fit(y)
    >>> forecast = model.predict(steps=12)
    >>> truth = 10.0 + 0.05 * np.arange(120, 132) + 3 * np.sin(
    ...     2 * np.pi * np.arange(120, 132) / 12)
    >>> bool(np.mean(np.abs(forecast - truth)) < 0.2)
    True

    A non-integer period is handled exactly like an integer one:

    >>> t = np.arange(200)
    >>> y = 5.0 + 2 * np.sin(2 * np.pi * t / 52.18)
    >>> model = TBATS(seasonal_periods=52.18).fit(y)
    >>> model.predict(steps=4).shape
    (4,)
    >>> model.harmonics_
    [5]
    """

    def __init__(
        self,
        seasonal_periods: Optional[Sequence[float] | float] = None,
        n_harmonics: Optional[Sequence[int] | int] = None,
        use_trend: bool = True,
        damped_trend: bool = True,
        box_cox: bool = False,
        box_cox_lambda: Optional[float] = None,
        use_arma_errors: bool = True,
        arma_order: Tuple[int, int] = (1, 0),
        maxiter: int = 40,
        tol: float = 1e-6,
    ):
        """Initialise a TBATS specification.

        Parameters
        ----------
        seasonal_periods : sequence of float, float or None, default=None
            Seasonal period lengths, possibly non-integer.
        n_harmonics : sequence of int, int or None, default=None
            Harmonic pairs per period; None picks a default per period.
        use_trend : bool, default=True
            Include a local linear trend.
        damped_trend : bool, default=True
            Damp the trend.
        box_cox : bool, default=False
            Apply a Box-Cox transform.
        box_cox_lambda : float or None, default=None
            Fixed Box-Cox lambda, or None to select over a grid.
        use_arma_errors : bool, default=True
            Model the residuals with an ARMA process.
        arma_order : tuple of (int, int), default=(1, 0)
            Order of the residual ARMA model.
        maxiter : int, default=40
            Maximum optimiser iterations.
        tol : float, default=1e-6
            Optimiser convergence tolerance.
        """
        super().__init__()
        self.seasonal_periods = seasonal_periods
        self.n_harmonics = n_harmonics
        self.use_trend = use_trend
        self.damped_trend = damped_trend
        self.box_cox = box_cox
        self.box_cox_lambda = box_cox_lambda
        self.use_arma_errors = use_arma_errors
        self.arma_order = arma_order
        self.maxiter = maxiter
        self.tol = tol

        self.params_ = None
        self.lambda_ = None
        self.harmonics_ = None
        self.level_ = None
        self.trend_ = None
        self.seasonal_ = None
        self.ar_params_ = None
        self.ma_params_ = None
        self.fitted_values_ = None
        self.resid_ = None
        self.sse_ = None
        self.aic_ = None
        self.n_obs_ = None

    # -- metadata ----------------------------------------------------------

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "seasonal_periods": {
                "type": ["array", "number", "null"],
                "default": None,
                "items": {"type": "number", "exclusiveMinimum": 1},
                "description": "Seasonal period lengths, possibly non-integer",
            },
            "n_harmonics": {
                "type": ["array", "integer", "null"],
                "default": None,
                "items": {"type": "integer", "minimum": 1},
                "description": "Harmonic pairs per seasonal period",
            },
            "use_trend": {
                "type": "boolean",
                "default": True,
                "description": "Include a local linear trend component",
            },
            "damped_trend": {
                "type": "boolean",
                "default": True,
                "description": "Damp the trend component",
            },
            "box_cox": {
                "type": "boolean",
                "default": False,
                "description": "Apply a Box-Cox variance-stabilising transform",
            },
            "box_cox_lambda": {
                "type": ["number", "null"],
                "default": None,
                "description": "Fixed Box-Cox lambda, or None to select it",
            },
            "use_arma_errors": {
                "type": "boolean",
                "default": True,
                "description": "Model the smoothing residuals with an ARMA process",
            },
            "arma_order": {
                "type": "array",
                "default": [1, 0],
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
                "description": "(p, q) order of the residual ARMA model",
            },
            "maxiter": {
                "type": "integer",
                "default": 40,
                "minimum": 1,
                "description": "Maximum optimiser iterations",
            },
            "tol": {
                "type": "number",
                "default": 1e-6,
                "exclusiveMinimum": 0,
                "description": "Optimiser convergence tolerance",
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
            "seasonal",
            "seasonality",
            "trend",
            "noise_tolerant",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return (
            "Training: O(maxiter * k * n * K), Prediction: O(h * K), where "
            "K is the total number of harmonic pairs (independent of the "
            "seasonal period lengths)"
        )

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "De Livera, Hyndman & Snyder, 2011. Forecasting time series with "
            "complex seasonal patterns using exponential smoothing. JASA "
            "106(496), 1513-1527.",
            "Hyndman, Koehler, Ord & Snyder, 2008. Forecasting with "
            "Exponential Smoothing: The State Space Approach. Springer.",
            "Box & Cox, 1964. An analysis of transformations. JRSS-B 26(2).",
        ]

    # -- specification helpers --------------------------------------------

    def _periods(self) -> List[float]:
        """Return the seasonal periods as a validated list of floats."""
        sp = self.seasonal_periods
        if sp is None:
            return []
        if np.isscalar(sp):
            sp = [sp]
        out = [float(v) for v in sp]
        for m in out:
            if not np.isfinite(m) or m <= 1.0:
                raise ValueError(
                    f"seasonal_periods must all be > 1; got {self.seasonal_periods!r}"
                )
        return out

    def _harmonics(self, periods: List[float], n: int) -> List[int]:
        """Choose the number of harmonic pairs for each seasonal period.

        Parameters
        ----------
        periods : list of float
            Seasonal period lengths.
        n : int
            Number of observations, used to keep the state identifiable.

        Returns
        -------
        ks : list of int
            Harmonic pairs per period.
        """
        nh = self.n_harmonics
        if nh is None:
            ks = [max(1, min(int(m // 2), 5)) for m in periods]
        else:
            if np.isscalar(nh):
                nh = [nh] * len(periods)
            ks = [int(v) for v in nh]
            if len(ks) != len(periods):
                raise ValueError(
                    "n_harmonics must have one entry per seasonal period; "
                    f"got {len(ks)} for {len(periods)} periods"
                )
            if any(k < 1 for k in ks):
                raise ValueError("n_harmonics entries must be >= 1")
        # Keep the OLS seeding design well conditioned.
        budget = max(1, (n - 4) // 2)
        while sum(ks) > budget and max(ks) > 1:
            ks[int(np.argmax(ks))] -= 1
        return ks

    def _frequencies(self, periods: List[float], ks: List[int]) -> np.ndarray:
        """Return the angular frequency of every harmonic pair.

        Parameters
        ----------
        periods : list of float
            Seasonal period lengths.
        ks : list of int
            Harmonic pairs per period.

        Returns
        -------
        omega : np.ndarray of shape (K,)
            Angular frequencies :math:`2\\pi j / m_i`.
        """
        out = []
        for m, k in zip(periods, ks):
            for j in range(1, k + 1):
                out.append(2.0 * np.pi * j / m)
        return np.asarray(out, dtype=float)

    def _gamma_index(self, ks: List[int]) -> np.ndarray:
        """Map each harmonic pair to the seasonal component it belongs to.

        Parameters
        ----------
        ks : list of int
            Harmonic pairs per period.

        Returns
        -------
        idx : np.ndarray of shape (K,)
            Index of the owning seasonal period, as integers.
        """
        return np.concatenate(
            [np.full(k, i, dtype=int) for i, k in enumerate(ks)]
        ) if ks else np.zeros(0, dtype=int)

    def _seed_states(self, z: np.ndarray, omega: np.ndarray):
        """Seed level, trend and seasonal states by least squares.

        Parameters
        ----------
        z : np.ndarray of shape (n,)
            Transformed series.
        omega : np.ndarray of shape (K,)
            Angular frequencies.

        Returns
        -------
        level : float
            Initial level.
        trend : float
            Initial slope.
        seas : np.ndarray of shape (2K,)
            Initial trigonometric states.
        """
        n = len(z)
        t = np.arange(n, dtype=float)
        cols = [np.ones(n)]
        if self.use_trend:
            cols.append(t)
        for w in omega:
            cols.append(np.cos(w * t))
            cols.append(np.sin(w * t))
        A = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        level = float(coef[0])
        i = 1
        trend = 0.0
        if self.use_trend:
            trend = float(coef[1])
            i = 2
        K = len(omega)
        seas = np.zeros(2 * K)
        for j in range(K):
            seas[j] = coef[i + 2 * j]
            seas[K + j] = coef[i + 2 * j + 1]
        return level, trend, seas

    def _recursion(self, z: np.ndarray, theta: np.ndarray, omega: np.ndarray,
                   gidx: np.ndarray, seed):
        """Run the TBATS smoothing recursion over the transformed series.

        Parameters
        ----------
        z : np.ndarray of shape (n,)
            Transformed series.
        theta : np.ndarray
            Smoothing parameters ``[alpha, beta, phi, gamma1..., gamma2...]``
            with the trend/damping entries present only when enabled.
        omega : np.ndarray of shape (K,)
            Angular frequencies.
        gidx : np.ndarray of shape (K,)
            Owning seasonal period of each harmonic pair.
        seed : tuple
            ``(level, trend, seas)`` initial states.

        Returns
        -------
        sse : float
            Sum of squared one-step errors.
        fitted : np.ndarray of shape (n,)
            One-step-ahead predictions on the transformed scale.
        resid : np.ndarray of shape (n,)
            One-step-ahead errors.
        state : tuple
            Final ``(level, trend, seas)`` states.
        """
        alpha, beta, phi, g1, g2 = self._unpack(theta, gidx)
        level, trend, seas = seed
        level = float(level)
        trend = float(trend)
        seas = np.asarray(seas, dtype=float).copy()
        K = len(omega)
        cos_w, sin_w = np.cos(omega), np.sin(omega)
        n = len(z)
        fitted = np.empty(n)
        resid = np.empty(n)
        for t in range(n):
            s_sum = seas[:K].sum() if K else 0.0
            pred = level + phi * trend + s_sum
            e = z[t] - pred
            fitted[t] = pred
            resid[t] = e
            new_level = level + phi * trend + alpha * e
            new_trend = phi * trend + beta * e
            if K:
                s, sstar = seas[:K], seas[K:]
                new_s = s * cos_w + sstar * sin_w + g1 * e
                new_sstar = -s * sin_w + sstar * cos_w + g2 * e
                seas = np.concatenate([new_s, new_sstar])
            level, trend = new_level, new_trend
        sse = float(np.dot(resid, resid))
        if not np.isfinite(sse):
            sse = np.inf
        return sse, fitted, resid, (level, trend, seas)

    def _unpack(self, theta: np.ndarray, gidx: np.ndarray):
        """Split the optimiser vector into named smoothing parameters.

        Parameters
        ----------
        theta : np.ndarray
            Flat optimiser vector.
        gidx : np.ndarray of shape (K,)
            Owning seasonal period of each harmonic pair.

        Returns
        -------
        alpha, beta, phi : float
            Level, trend and damping parameters.
        g1, g2 : np.ndarray of shape (K,)
            Per-harmonic seasonal smoothing parameters, broadcast from the
            per-period values.
        """
        i = 0
        alpha = float(theta[i]); i += 1
        if self.use_trend:
            beta = float(theta[i]); i += 1
            if self.damped_trend:
                phi = float(theta[i]); i += 1
            else:
                phi = 1.0
        else:
            beta, phi = 0.0, 0.0
        n_seas = self._n_seasonal
        if n_seas:
            g1_per = theta[i:i + n_seas]; i += n_seas
            g2_per = theta[i:i + n_seas]; i += n_seas
            g1 = np.asarray(g1_per, float)[gidx]
            g2 = np.asarray(g2_per, float)[gidx]
        else:
            g1 = g2 = np.zeros(0)
        return alpha, beta, phi, g1, g2

    def _bounds_and_start(self) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Return the optimiser start vector and its box constraints.

        Returns
        -------
        x0 : np.ndarray
            Initial smoothing parameters.
        bounds : list of tuple
            Per-parameter ``(low, high)`` bounds.
        """
        x0, bounds = [0.1], [(1e-4, 0.95)]
        if self.use_trend:
            x0.append(0.01)
            bounds.append((1e-5, 0.5))
            if self.damped_trend:
                x0.append(0.98)
                bounds.append((0.80, 1.0))
        n_seas = self._n_seasonal
        for _ in range(n_seas):
            x0.append(0.001)
            bounds.append((0.0, 0.3))
        for _ in range(n_seas):
            x0.append(0.001)
            bounds.append((0.0, 0.3))
        return np.asarray(x0, dtype=float), bounds

    def _fit_transformed(self, z: np.ndarray, omega, gidx):
        """Optimise the smoothing parameters for one transformed series.

        Parameters
        ----------
        z : np.ndarray of shape (n,)
            Transformed series.
        omega : np.ndarray of shape (K,)
            Angular frequencies.
        gidx : np.ndarray of shape (K,)
            Owning seasonal period of each harmonic pair.

        Returns
        -------
        theta : np.ndarray
            Fitted smoothing parameters.
        sse : float
            Minimised sum of squared errors.
        seed : tuple
            Seed states used by the recursion.
        """
        seed = self._seed_states(z, omega)
        x0, bounds = self._bounds_and_start()

        def obj(th):
            return self._recursion(z, th, omega, gidx, seed)[0]

        res = minimize(
            obj, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": int(self.maxiter), "ftol": self.tol,
                     "gtol": self.tol, "maxls": 20},
        )
        theta = res.x if np.isfinite(res.fun) and res.fun <= obj(x0) else x0
        return theta, float(obj(theta)), seed

    # -- fitting -----------------------------------------------------------

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> "TBATS":
        """Fit the TBATS model to a time series.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            The time series to model.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with other regressors;
            use :class:`~tuiml.algorithms.timeseries.SARIMAX` when
            exogenous regressors are needed.

        Returns
        -------
        self : TBATS
            Fitted estimator.
        """
        y = np.asarray(y, dtype=float).ravel()
        if y.size == 0:
            raise ValueError("y must be a non-empty 1-D array")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values")
        n = len(y)
        self.n_obs_ = n

        periods = self._periods()
        if periods and max(periods) > n:
            periods = [m for m in periods if m <= n]
        ks = self._harmonics(periods, n)
        self.harmonics_ = list(ks)
        self._periods_used = periods
        self._n_seasonal = len(periods)
        omega = self._frequencies(periods, ks)
        gidx = self._gamma_index(ks)
        self._omega, self._gidx = omega, gidx

        if len(y) < 3:
            raise ValueError("TBATS needs at least 3 observations")

        # -- Box-Cox ------------------------------------------------------
        if self.box_cox:
            if np.min(y) <= 0:
                raise ValueError(
                    "box_cox=True requires strictly positive data; the series "
                    f"has a minimum of {np.min(y):.6g}"
                )
            if self.box_cox_lambda is not None:
                lambdas = [float(self.box_cox_lambda)]
            else:
                lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
        else:
            lambdas = [None]

        best = None
        log_y_sum = float(np.sum(np.log(y))) if self.box_cox else 0.0
        for lam in lambdas:
            z = y if lam is None else _box_cox(y, lam)
            theta, sse, seed = self._fit_transformed(z, omega, gidx)
            if sse <= 0:
                sse = 1e-300
            # Box-Cox-adjusted profile criterion; comparable across lambdas.
            crit = n * np.log(sse / n)
            if lam is not None:
                crit -= 2.0 * (lam - 1.0) * log_y_sum
            if best is None or crit < best[0]:
                best = (crit, lam, theta, sse, seed, z)

        _, lam, theta, sse, seed, z = best
        self.lambda_ = lam

        sse, fitted_z, resid_z, state = self._recursion(z, theta, omega, gidx, seed)
        self.level_, self.trend_, self.seasonal_ = state
        self.resid_ = resid_z
        self.sse_ = float(sse)
        self._theta = theta
        self._seed = seed

        alpha, beta, phi, g1, g2 = self._unpack(theta, gidx)
        self.params_ = {
            "alpha": alpha,
            "beta": beta if self.use_trend else 0.0,
            "phi": phi if self.use_trend else 0.0,
            "gamma1": (np.asarray(theta, float)[self._n_fixed():
                                                self._n_fixed() + self._n_seasonal]
                       if self._n_seasonal else np.zeros(0)),
            "gamma2": (np.asarray(theta, float)[self._n_fixed() + self._n_seasonal:
                                                self._n_fixed() + 2 * self._n_seasonal]
                       if self._n_seasonal else np.zeros(0)),
        }

        # -- ARMA errors ---------------------------------------------------
        p_arma, q_arma = (int(v) for v in self.arma_order)
        if self.use_arma_errors and (p_arma or q_arma):
            self.ar_params_, self.ma_params_ = _hannan_rissanen(
                resid_z, p_arma, q_arma
            )
        else:
            self.ar_params_, self.ma_params_ = np.zeros(0), np.zeros(0)

        self.fitted_values_ = (fitted_z if lam is None
                               else _inv_box_cox(fitted_z, lam))
        k = len(theta) + 1 + len(self.ar_params_) + len(self.ma_params_)
        self.aic_ = float(n * np.log(max(sse, 1e-300) / n) + 2 * k)

        self._is_fitted = True
        return self

    def _n_fixed(self) -> int:
        """Return the number of non-seasonal entries in the parameter vector."""
        k = 1
        if self.use_trend:
            k += 1
            if self.damped_trend:
                k += 1
        return k

    # -- forecasting -------------------------------------------------------

    def _arma_forecast(self, steps: int) -> np.ndarray:
        """Forecast the residual ARMA process forward.

        Parameters
        ----------
        steps : int
            Forecast horizon.

        Returns
        -------
        d : np.ndarray of shape (steps,)
            Residual forecasts on the transformed scale.
        """
        p, q = len(self.ar_params_), len(self.ma_params_)
        if p == 0 and q == 0:
            return np.zeros(steps)
        hist = list(self.resid_[-max(p, 1):]) if p else []
        # Innovations beyond the sample are zero in expectation, so the MA
        # part only contributes while the last observed shocks are in range.
        eps = list(self.resid_[-max(q, 1):]) if q else []
        out = np.zeros(steps)
        for h in range(steps):
            val = 0.0
            for i in range(p):
                val += self.ar_params_[i] * hist[-1 - i]
            for j in range(q):
                idx = len(eps) - 1 - j + h
                if 0 <= idx < len(eps):
                    val += self.ma_params_[j] * eps[idx]
            out[h] = val
            if p:
                hist.append(val)
        return out

    def predict(self, steps: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast future values.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with other regressors.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Point forecasts on the scale of the original series.
        """
        self._check_is_fitted()
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be >= 1")

        omega = self._omega
        K = len(omega)
        cos_w, sin_w = np.cos(omega), np.sin(omega)
        _, _, phi, _, _ = self._unpack(self._theta, self._gidx)

        level = float(self.level_)
        trend = float(self.trend_)
        seas = np.asarray(self.seasonal_, dtype=float).copy()
        d = self._arma_forecast(steps)

        out = np.empty(steps)
        for h in range(steps):
            s_sum = seas[:K].sum() if K else 0.0
            out[h] = level + phi * trend + s_sum + d[h]
            new_level = level + phi * trend
            new_trend = phi * trend
            if K:
                s, sstar = seas[:K], seas[K:]
                seas = np.concatenate([
                    s * cos_w + sstar * sin_w,
                    -s * sin_w + sstar * cos_w,
                ])
            level, trend = new_level, new_trend

        if self.lambda_ is not None:
            out = _inv_box_cox(out, self.lambda_)
        return out

    def fit_predict(self, y: np.ndarray, steps: int = 1) -> np.ndarray:
        """Fit the model and forecast in one call.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series to fit.
        steps : int, default=1
            Number of future time steps to forecast.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecasted values.
        """
        self.fit(y)
        return self.predict(steps)
