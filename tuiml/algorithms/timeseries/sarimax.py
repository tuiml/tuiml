"""Seasonal ARIMA with eXogenous regressors, estimated by Kalman-filter MLE."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from tuiml.base.algorithms import Regressor, regressor

__all__ = ["SARIMAX"]


# ---------------------------------------------------------------------------
# Polynomial helpers
# ---------------------------------------------------------------------------

def _pacf_to_ar(pacf: np.ndarray) -> np.ndarray:
    """Map partial autocorrelations to stationary AR coefficients.

    Implements the Levinson-Durbin recursion of the Monahan/Jones
    reparameterisation: any vector of partial autocorrelations in
    :math:`(-1, 1)` maps to an AR polynomial whose roots lie strictly
    outside the unit circle.

    Parameters
    ----------
    pacf : np.ndarray of shape (p,)
        Partial autocorrelations, each strictly inside ``(-1, 1)``.

    Returns
    -------
    phi : np.ndarray of shape (p,)
        Stationary autoregressive coefficients.
    """
    p = len(pacf)
    if p == 0:
        return np.zeros(0)
    phi = np.zeros(p)
    for k in range(p):
        new = np.empty(k + 1)
        new[k] = pacf[k]
        for i in range(k):
            new[i] = phi[i] - pacf[k] * phi[k - 1 - i]
        phi[: k + 1] = new
    return phi


def _unconstrained_to_pacf(u: np.ndarray) -> np.ndarray:
    """Squash unconstrained reals into partial autocorrelations.

    Parameters
    ----------
    u : np.ndarray of shape (p,)
        Unconstrained optimiser coordinates.

    Returns
    -------
    pacf : np.ndarray of shape (p,)
        Values in ``(-1, 1)``, obtained with ``tanh``.
    """
    return np.tanh(np.clip(u, -8.0, 8.0))


def _poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two lag polynomials given by their coefficient arrays.

    Parameters
    ----------
    a : np.ndarray
        Coefficients of the first polynomial, lowest lag first.
    b : np.ndarray
        Coefficients of the second polynomial, lowest lag first.

    Returns
    -------
    c : np.ndarray
        Coefficients of the product.
    """
    return np.convolve(a, b)


def _expand_seasonal(coefs: np.ndarray, seasonal: np.ndarray, period: int,
                     sign: float) -> np.ndarray:
    """Expand a multiplicative seasonal lag polynomial into a flat one.

    Builds :math:`(1 + s\\,\\phi(L))(1 + s\\,\\Phi(L^{m}))` in "1 first"
    coefficient form, where ``sign`` is ``-1`` for autoregressive
    polynomials and ``+1`` for moving-average polynomials.

    Parameters
    ----------
    coefs : np.ndarray of shape (p,)
        Non-seasonal coefficients.
    seasonal : np.ndarray of shape (P,)
        Seasonal coefficients.
    period : int
        Seasonal period :math:`m`.
    sign : float
        ``-1.0`` for AR polynomials, ``+1.0`` for MA polynomials.

    Returns
    -------
    poly : np.ndarray
        Product polynomial, leading coefficient 1.
    """
    a = np.concatenate([[1.0], sign * np.asarray(coefs, dtype=float)])
    if len(seasonal) == 0 or period <= 0:
        return a
    b = np.zeros(period * len(seasonal) + 1)
    b[0] = 1.0
    for i, c in enumerate(seasonal):
        b[(i + 1) * period] = sign * c
    return _poly_mul(a, b)


def _diff(x: np.ndarray, lag: int, times: int) -> np.ndarray:
    """Apply ``times`` rounds of lag-``lag`` differencing.

    Parameters
    ----------
    x : np.ndarray of shape (n,) or (n, k)
        Series or design matrix to difference along axis 0.
    lag : int
        Differencing lag (1 for regular, ``m`` for seasonal).
    times : int
        Number of rounds.

    Returns
    -------
    out : np.ndarray
        Differenced array, shorter by ``lag * times`` rows.
    """
    out = np.asarray(x, dtype=float)
    for _ in range(times):
        out = out[lag:] - out[:-lag]
    return out


def _psi_weights(ar: np.ndarray, ma: np.ndarray, n: int) -> np.ndarray:
    """Compute the first ``n`` MA(:math:`\\infty`) weights of an ARMA process.

    Parameters
    ----------
    ar : np.ndarray of shape (p,)
        Autoregressive coefficients in :math:`1 - \\phi_1 L - \\dots` form.
    ma : np.ndarray of shape (q,)
        Moving-average coefficients in :math:`1 + \\theta_1 L + \\dots` form.
    n : int
        Number of weights to return, starting at :math:`\\psi_0 = 1`.

    Returns
    -------
    psi : np.ndarray of shape (n,)
        Impulse-response weights.
    """
    psi = np.zeros(n)
    psi[0] = 1.0
    p, q = len(ar), len(ma)
    for j in range(1, n):
        val = ma[j - 1] if j <= q else 0.0
        for i in range(1, min(j, p) + 1):
            val += ar[i - 1] * psi[j - i]
        psi[j] = val
    return psi


@regressor(tags=["timeseries", "forecasting", "autoregressive", "seasonal",
                 "exogenous"], version="1.0.0")
class SARIMAX(Regressor):
    r"""
    Seasonal AutoRegressive Integrated Moving Average model with
    **eXogenous regressors**, estimated by **exact Gaussian maximum
    likelihood** through a **Kalman filter**.

    SARIMAX is the full :math:`(p, d, q) \times (P, D, Q, m)` specification
    plus a regression on external covariates. Unlike a hand-rolled
    conditional-least-squares ARIMA, the model is written in **state-space
    form** and the *exact* likelihood is evaluated by the Kalman filter, so
    the first observations are used rather than discarded, and the
    optimiser is kept inside the stationary/invertible region by
    reparameterising every AR and MA block through **partial
    autocorrelations** (Monahan/Jones transform).

    Overview
    --------
    The estimation procedure runs as follows:

    1. Difference the series :math:`d` times at lag 1 and :math:`D` times at
       lag :math:`m`, applying the same operators to the exogenous design.
    2. Expand the multiplicative seasonal polynomials
       :math:`\phi(L)\Phi(L^m)` and :math:`\\theta(L)\Theta(L^m)` into flat
       lag polynomials.
    3. Place the resulting ARMA process in Harvey's companion state-space
       form with a single innovation.
    4. Initialise the state covariance from the stationary solution of the
       discrete Lyapunov equation :math:`P = T P T' + R R'`.
    5. Run the Kalman filter, concentrating :math:`\sigma^2` out of the
       likelihood, and maximise the resulting profile log-likelihood over
       the transformed ARMA parameters and the regression coefficients.
    6. Forecast by iterating the state transition forward, then invert the
       differencing operators to return to the original scale.

    Theory
    ------
    The model is

    .. math::
        \phi(L)\, \Phi(L^{m})\, (1 - L)^{d} (1 - L^{m})^{D}
        \left( y_t - \\beta^{\top} x_t \\right)
        = \\theta(L)\, \Theta(L^{m})\, \\varepsilon_t ,
        \qquad \\varepsilon_t \sim N(0, \sigma^2).

    Writing the differenced, regression-adjusted series as :math:`w_t`, the
    state-space representation is

    .. math::
        \\begin{aligned}
        \\alpha_{t+1} &= T \\alpha_t + R \\varepsilon_t \\\\
        w_t &= Z \\alpha_t
        \end{aligned}

    with :math:`T` the companion matrix of the expanded AR polynomial,
    :math:`R = (1, \\theta_1, \dots, \\theta_{r-1})^{\top}` and
    :math:`Z = (1, 0, \dots, 0)`. The Kalman recursions give the
    one-step-ahead innovations :math:`v_t` and their variances
    :math:`\sigma^2 F_t`, and the exact log-likelihood is

    .. math::
        \log L = -\\frac{n}{2}\log(2\pi\sigma^2)
                 -\\frac{1}{2}\sum_{t=1}^{n}\log F_t
                 -\\frac{1}{2\sigma^2}\sum_{t=1}^{n} \\frac{v_t^2}{F_t}.

    Concentrating out :math:`\sigma^2` yields
    :math:`\hat{\sigma}^2 = n^{-1}\sum_t v_t^2 / F_t`, so only the ARMA and
    regression parameters remain to be optimised numerically.

    Stationarity is enforced structurally. Each AR block is parameterised by
    partial autocorrelations :math:`r_k = \\tanh(u_k) \in (-1, 1)` which the
    Levinson-Durbin recursion maps to coefficients whose polynomial has all
    roots outside the unit circle, so the optimiser physically cannot reach
    an explosive region.

    Parameters
    ----------
    order : tuple of (int, int, int), default=(1, 0, 0)
        The non-seasonal :math:`(p, d, q)` order.
    seasonal_order : tuple of (int, int, int, int), default=(0, 0, 0, 0)
        The seasonal :math:`(P, D, Q, m)` order. A period ``m`` of 0 or 1
        disables the seasonal component.
    trend : {"c", "t", "ct", None}, default=None
        Deterministic terms added to the *differenced* series: ``"c"`` a
        constant, ``"t"`` a linear time index, ``"ct"`` both.
    maxiter : int, default=50
        Maximum number of optimiser iterations. Kept small by default so
        that a fit on a short series stays well under a second.
    tol : float, default=1e-6
        Convergence tolerance passed to the optimiser.
    enforce_stationarity : bool, default=True
        If True, autoregressive blocks are reparameterised through partial
        autocorrelations. If False, the raw coefficients are optimised
        directly (faster, but the optimiser may wander).
    enforce_invertibility : bool, default=True
        Same as above for the moving-average blocks.

    Attributes
    ----------
    ar_params_ : np.ndarray of shape (p,)
        Fitted non-seasonal autoregressive coefficients.
    ma_params_ : np.ndarray of shape (q,)
        Fitted non-seasonal moving-average coefficients.
    seasonal_ar_params_ : np.ndarray of shape (P,)
        Fitted seasonal autoregressive coefficients.
    seasonal_ma_params_ : np.ndarray of shape (Q,)
        Fitted seasonal moving-average coefficients.
    exog_params_ : np.ndarray of shape (k_exog,)
        Regression coefficients on the exogenous columns, in input order.
    trend_params_ : np.ndarray
        Coefficients of the deterministic terms selected by ``trend``.
    sigma2_ : float
        Concentrated innovation variance :math:`\hat{\sigma}^2`.
    loglik_ : float
        Maximised exact Gaussian log-likelihood.
    aic_ : float
        Akaike information criterion.
    bic_ : float
        Bayesian information criterion.
    resid_ : np.ndarray
        Standardised one-step-ahead prediction errors :math:`v_t/\sqrt{F_t}`.
    state_ : np.ndarray of shape (r,)
        Predicted state at the first out-of-sample time point.
    state_cov_ : np.ndarray of shape (r, r)
        Covariance of ``state_``, in units of :math:`\sigma^2`.
    n_obs_ : int
        Number of observations supplied to :meth:`fit`.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(\\text{maxiter} \cdot k \cdot n r^3)` where
      :math:`r = \max(p^*, q^* + 1)` is the state dimension of the expanded
      polynomials, :math:`n` the sample size and :math:`k` the number of
      free parameters (numerical gradients cost one filter pass each).
    - Prediction: :math:`O(h r^2)` for :math:`h` steps.

    **When to use SARIMAX:**

    - You have **exogenous regressors** (price, temperature, a promotion
      dummy) that should drive the series alongside its own dynamics.
    - You need a **genuine seasonal** :math:`(P, D, Q, m)` component rather
      than a plain ARIMA on a de-seasonalised series.
    - You want **exact** maximum likelihood and calibrated forecast
      intervals from the Kalman variance rather than point forecasts alone.

    **Relationship to** :class:`~tuiml.algorithms.timeseries.ARIMA`:
    the two are deliberately not interchangeable.
    :class:`~tuiml.algorithms.timeseries.ARIMA` is the light, fast option:
    it estimates a non-seasonal :math:`(p, d, q)` model by Yule-Walker plus
    conditional-sum-of-squares refinement, it **ignores** exogenous input
    (its ``fit`` signature spells the argument ``_X``), and its
    ``seasonal_order`` argument is accepted but not acted upon. Reach for
    ``SARIMAX`` whenever you actually need exogenous regressors, a working
    seasonal specification, exact-likelihood estimates, or forecast
    intervals; reach for ``ARIMA`` when you want a cheap non-seasonal point
    forecast and nothing more.

    **Interval caveat:** :meth:`predict_interval` builds the forecast
    variance from the MA(:math:`\infty`) weights of the *expanded* model,
    which includes the differencing operators, so intervals widen correctly
    with the horizon under differencing. Parameter-estimation uncertainty is
    not included -- the intervals condition on the fitted parameters.

    References
    ----------
    .. [Box2015] Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung,
           G. M. (2015). **Time Series Analysis: Forecasting and Control**,
           5th ed. *Wiley*. :doi:`10.1111/jtsa.12194`
    .. [Harvey1990] Harvey, A. C. (1990). **Forecasting, Structural Time
           Series Models and the Kalman Filter.** *Cambridge University
           Press*. :doi:`10.1017/CBO9781107049994`
    .. [Monahan1984] Monahan, J. F. (1984). **A note on enforcing
           stationarity in autoregressive-moving average models.**
           *Biometrika*, 71(2), 403-404. :doi:`10.1093/biomet/71.2.403`
    .. [Jones1980] Jones, R. H. (1980). **Maximum likelihood fitting of
           ARMA models to time series with missing observations.**
           *Technometrics*, 22(3), 389-395. :doi:`10.2307/1268324`

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.ARIMA` : Lighter non-seasonal model with no exogenous support; see the Notes above for how to choose.
    :class:`~tuiml.algorithms.timeseries.TBATS` : Trigonometric seasonal state-space model for high-frequency or non-integer periods.
    :class:`~tuiml.algorithms.timeseries.ExponentialSmoothing` : Holt-Winters smoothing with integer seasonality.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.sarimax import SARIMAX
    >>> rng = np.random.default_rng(0)
    >>> eps = rng.normal(scale=0.5, size=400)
    >>> y = np.zeros(400)
    >>> for t in range(1, 400):
    ...     y[t] = 0.7 * y[t - 1] + eps[t]
    >>> model = SARIMAX(order=(1, 0, 0)).fit(y)
    >>> bool(abs(model.ar_params_[0] - 0.7) < 0.05)
    True
    >>> model.predict(steps=3).shape
    (3,)

    An exogenous regression recovers the true coefficient:

    >>> x = rng.normal(size=(300, 1))
    >>> y2 = 3.0 * x[:, 0] + 0.01 * rng.normal(size=300)
    >>> m2 = SARIMAX(order=(0, 0, 0)).fit(y2, x)
    >>> bool(abs(m2.exog_params_[0] - 3.0) < 0.01)
    True
    >>> bool(np.all(np.abs(m2.predict(steps=2, X=np.zeros((2, 1)))) < 0.01))
    True

    Calling ``predict`` without the future exogenous values is an error:

    >>> m2.predict(steps=2)
    Traceback (most recent call last):
        ...
    ValueError: This SARIMAX was fitted with 1 exogenous regressor(s); predict() requires X with the future values for those regressors.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
        maxiter: int = 50,
        tol: float = 1e-6,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
    ):
        """Initialise a SARIMAX specification.

        Parameters
        ----------
        order : tuple of (int, int, int), default=(1, 0, 0)
            Non-seasonal ``(p, d, q)`` order.
        seasonal_order : tuple of (int, int, int, int), default=(0, 0, 0, 0)
            Seasonal ``(P, D, Q, m)`` order.
        trend : str or None, default=None
            Deterministic terms: ``"c"``, ``"t"``, ``"ct"`` or None.
        maxiter : int, default=50
            Maximum optimiser iterations.
        tol : float, default=1e-6
            Optimiser convergence tolerance.
        enforce_stationarity : bool, default=True
            Constrain AR blocks to the stationary region.
        enforce_invertibility : bool, default=True
            Constrain MA blocks to the invertible region.
        """
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.maxiter = maxiter
        self.tol = tol
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self.ar_params_ = None
        self.ma_params_ = None
        self.seasonal_ar_params_ = None
        self.seasonal_ma_params_ = None
        self.exog_params_ = None
        self.trend_params_ = None
        self.sigma2_ = None
        self.loglik_ = None
        self.aic_ = None
        self.bic_ = None
        self.resid_ = None
        self.state_ = None
        self.state_cov_ = None
        self.n_obs_ = None

    # -- metadata ----------------------------------------------------------

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "order": {
                "type": "array",
                "default": [1, 0, 0],
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "integer", "minimum": 0},
                "description": "Non-seasonal (p, d, q) order",
            },
            "seasonal_order": {
                "type": "array",
                "default": [0, 0, 0, 0],
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "integer", "minimum": 0},
                "description": "Seasonal (P, D, Q, m) order",
            },
            "trend": {
                "type": ["string", "null"],
                "default": None,
                "enum": [None, "c", "t", "ct"],
                "description": "Deterministic terms on the differenced series",
            },
            "maxiter": {
                "type": "integer",
                "default": 50,
                "minimum": 1,
                "description": "Maximum optimiser iterations",
            },
            "tol": {
                "type": "number",
                "default": 1e-6,
                "exclusiveMinimum": 0,
                "description": "Optimiser convergence tolerance",
            },
            "enforce_stationarity": {
                "type": "boolean",
                "default": True,
                "description": "Constrain AR blocks to the stationary region",
            },
            "enforce_invertibility": {
                "type": "boolean",
                "default": True,
                "description": "Constrain MA blocks to the invertible region",
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
            "trend",
            "stationary",
            "uncertainty",
            "gaussian_assumption",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return (
            "Training: O(maxiter * k * n * r^3) Kalman passes, "
            "Prediction: O(h * r^2), where r=max(p*, q*+1) is the state size"
        )

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Box, Jenkins, Reinsel & Ljung, 2015. Time Series Analysis: "
            "Forecasting and Control, 5th ed. Wiley.",
            "Harvey, 1990. Forecasting, Structural Time Series Models and "
            "the Kalman Filter. Cambridge University Press.",
            "Monahan, 1984. A note on enforcing stationarity in "
            "autoregressive-moving average models. Biometrika 71(2), 403-404.",
        ]

    # -- specification helpers --------------------------------------------

    def _spec(self) -> Tuple[int, int, int, int, int, int, int]:
        """Unpack and validate the order tuples.

        Returns
        -------
        spec : tuple of int
            ``(p, d, q, P, D, Q, m)`` with ``m`` set to 0 when the seasonal
            component is disabled.
        """
        p, d, q = (int(v) for v in self.order)
        so = self.seasonal_order
        if so is None:
            P = D = Q = m = 0
        else:
            P, D, Q, m = (int(v) for v in so)
        if m <= 1:
            if P or D or Q:
                raise ValueError(
                    "seasonal_order needs a period m > 1 to use (P, D, Q); "
                    f"got {self.seasonal_order!r}"
                )
            P = D = Q = m = 0
        if min(p, d, q, P, D, Q) < 0:
            raise ValueError("orders must be non-negative")
        return p, d, q, P, D, Q, m

    def _n_trend(self) -> int:
        """Return the number of deterministic trend columns."""
        return {None: 0, "c": 1, "t": 1, "ct": 2}[self.trend]

    def _trend_design(self, start: int, n: int) -> np.ndarray:
        """Build the deterministic design block for ``n`` time points.

        Parameters
        ----------
        start : int
            One-based index of the first time point.
        n : int
            Number of rows to build.

        Returns
        -------
        D : np.ndarray of shape (n, n_trend)
            Deterministic design matrix.
        """
        k = self._n_trend()
        out = np.empty((n, k))
        t = np.arange(start, start + n, dtype=float)
        col = 0
        if self.trend in ("c", "ct"):
            out[:, col] = 1.0
            col += 1
        if self.trend in ("t", "ct"):
            out[:, col] = t
        return out

    def _unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split the optimiser vector into model coefficient blocks.

        Parameters
        ----------
        theta : np.ndarray
            Flat parameter vector in optimiser coordinates.

        Returns
        -------
        blocks : tuple of np.ndarray
            ``(ar, ma, sar, sma, beta)`` in natural (untransformed) units.
        """
        p, _, q, P, _, Q, _ = self._spec()
        raw_ar = theta[:p]
        i = p
        raw_ma = theta[i:i + q]
        i += q
        raw_sar = theta[i:i + P]
        i += P
        raw_sma = theta[i:i + Q]
        i += Q
        beta = theta[i:]

        if self.enforce_stationarity:
            ar = _pacf_to_ar(_unconstrained_to_pacf(raw_ar))
            sar = _pacf_to_ar(_unconstrained_to_pacf(raw_sar))
        else:
            ar, sar = np.asarray(raw_ar, float), np.asarray(raw_sar, float)
        if self.enforce_invertibility:
            ma = _pacf_to_ar(_unconstrained_to_pacf(raw_ma))
            sma = _pacf_to_ar(_unconstrained_to_pacf(raw_sma))
        else:
            ma, sma = np.asarray(raw_ma, float), np.asarray(raw_sma, float)
        return ar, ma, sar, sma, np.asarray(beta, float)

    def _expanded_poly(self, ar, ma, sar, sma) -> Tuple[np.ndarray, np.ndarray]:
        """Expand the multiplicative seasonal polynomials into flat ones.

        Parameters
        ----------
        ar, ma, sar, sma : np.ndarray
            Non-seasonal and seasonal AR/MA coefficient blocks.

        Returns
        -------
        phi : np.ndarray
            Flat AR coefficients (``1 - phi_1 L - ...`` convention).
        theta : np.ndarray
            Flat MA coefficients (``1 + theta_1 L + ...`` convention).
        """
        _, _, _, _, _, _, m = self._spec()
        ar_poly = _expand_seasonal(ar, sar, m, -1.0)
        ma_poly = _expand_seasonal(ma, sma, m, +1.0)
        return -ar_poly[1:], ma_poly[1:]

    @staticmethod
    def _state_space(phi: np.ndarray, theta: np.ndarray):
        """Build Harvey's companion state-space matrices.

        Parameters
        ----------
        phi : np.ndarray of shape (p*,)
            Flat autoregressive coefficients.
        theta : np.ndarray of shape (q*,)
            Flat moving-average coefficients.

        Returns
        -------
        T : np.ndarray of shape (r, r)
            Transition matrix.
        R : np.ndarray of shape (r,)
            Innovation loading vector.
        """
        p, q = len(phi), len(theta)
        r = max(p, q + 1)
        T = np.zeros((r, r))
        if p:
            T[:p, 0] = phi
        if r > 1:
            T[:r - 1, 1:] = np.eye(r - 1)
        R = np.zeros(r)
        R[0] = 1.0
        if q:
            R[1:q + 1] = theta
        return T, R

    @staticmethod
    def _lyapunov(T: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Solve :math:`P = T P T' + R R'` by squaring.

        Parameters
        ----------
        T : np.ndarray of shape (r, r)
            Transition matrix, assumed stable.
        R : np.ndarray of shape (r,)
            Innovation loading vector.

        Returns
        -------
        P : np.ndarray of shape (r, r)
            Stationary state covariance in units of ``sigma2``.
        """
        Q = np.outer(R, R)
        P = Q.copy()
        A = T.copy()
        for _ in range(60):
            add = A @ P @ A.T
            P = P + add
            if np.max(np.abs(add)) < 1e-14 * max(1.0, np.max(np.abs(P))):
                break
            A = A @ A
            if not np.all(np.isfinite(A)) or np.max(np.abs(A)) < 1e-15:
                break
        if not np.all(np.isfinite(P)):
            P = np.eye(len(R)) * 1e6
        return P

    @staticmethod
    def _kalman(w: np.ndarray, T: np.ndarray, R: np.ndarray, P0: np.ndarray):
        """Run the Kalman filter with ``sigma2`` concentrated out.

        Parameters
        ----------
        w : np.ndarray of shape (n,)
            Differenced, regression-adjusted series.
        T : np.ndarray of shape (r, r)
            Transition matrix.
        R : np.ndarray of shape (r,)
            Innovation loading vector.
        P0 : np.ndarray of shape (r, r)
            Initial state covariance.

        Returns
        -------
        loglik : float
            Profile (concentrated) log-likelihood.
        sigma2 : float
            Concentrated innovation variance estimate.
        resid : np.ndarray of shape (n,)
            Standardised prediction errors.
        a : np.ndarray of shape (r,)
            Predicted state for the first out-of-sample point.
        P : np.ndarray of shape (r, r)
            Covariance of ``a``.
        """
        n = len(w)
        r = T.shape[0]
        a = np.zeros(r)
        P = P0.copy()
        RRt = np.outer(R, R)
        ssr = 0.0
        logdet = 0.0
        resid = np.zeros(n)
        for t in range(n):
            F = P[0, 0]
            if not np.isfinite(F) or F <= 1e-12:
                F = 1e-12
            v = w[t] - a[0]
            resid[t] = v / np.sqrt(F)
            ssr += v * v / F
            logdet += np.log(F)
            PZ = P[:, 0]
            K = T @ PZ / F
            a = T @ a + K * v
            P = T @ P @ T.T - np.outer(K, K) * F + RRt
            P = 0.5 * (P + P.T)
        sigma2 = ssr / n
        if not np.isfinite(sigma2) or sigma2 <= 0:
            return -np.inf, np.nan, resid, a, P
        loglik = -0.5 * n * (np.log(2 * np.pi) + 1.0 + np.log(sigma2)) - 0.5 * logdet
        return loglik, sigma2, resid, a, P

    def _neg_loglik(self, theta: np.ndarray, w_raw: np.ndarray,
                    design: np.ndarray) -> float:
        """Return the negative profile log-likelihood at ``theta``.

        Parameters
        ----------
        theta : np.ndarray
            Optimiser coordinates.
        w_raw : np.ndarray of shape (n,)
            Differenced series.
        design : np.ndarray of shape (n, k)
            Differenced deterministic + exogenous design.

        Returns
        -------
        nll : float
            Negative log-likelihood (``np.inf`` on numerical failure).
        """
        ar, ma, sar, sma, beta = self._unpack(theta)
        w = w_raw - design @ beta if design.shape[1] else w_raw
        phi, th = self._expanded_poly(ar, ma, sar, sma)
        T, R = self._state_space(phi, th)
        P0 = self._lyapunov(T, R)
        ll, _, _, _, _ = self._kalman(w, T, R, P0)
        return -ll if np.isfinite(ll) else np.inf

    # -- fitting -----------------------------------------------------------

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> "SARIMAX":
        """Fit the model by exact maximum likelihood.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            The time series to model.
        X : np.ndarray of shape (n_samples, k_exog), optional, default=None
            Exogenous regressors aligned row-wise with ``y``.

        Returns
        -------
        self : SARIMAX
            Fitted estimator.
        """
        y = np.asarray(y, dtype=float).ravel()
        if y.ndim != 1 or y.size == 0:
            raise ValueError("y must be a non-empty 1-D array")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values")

        p, d, q, P, D, Q, m = self._spec()

        if X is not None:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X has {X.shape[0]} rows but y has {y.shape[0]}; "
                    "exogenous regressors must be aligned with the series"
                )
            if not np.all(np.isfinite(X)):
                raise ValueError("X contains non-finite values")
        self._k_exog = 0 if X is None else X.shape[1]
        self._exog_train = None if X is None else X.copy()
        self.y_original_ = y.copy()
        self.n_obs_ = len(y)

        # Difference the series (and the exogenous design identically).
        w = _diff(_diff(y, m, D) if m else y, 1, d)
        n_lost = d + D * m
        if len(w) < max(p + P * m, q + Q * m) + 2:
            raise ValueError(
                f"Not enough observations: {len(w)} left after differencing "
                f"for the requested order {self.order}/{self.seasonal_order}"
            )

        # Deterministic terms live on the differenced scale (differencing a
        # column of ones would annihilate it); exogenous regressors are
        # differenced alongside the series.
        self._n_w = len(w)
        blocks = []
        if self._n_trend():
            blocks.append(self._trend_design(1, len(w)))
        if X is not None:
            blocks.append(_diff(_diff(X, m, D) if m else X, 1, d))
        design = np.hstack(blocks) if blocks else np.zeros((len(w), 0))

        n_free = p + q + P + Q + design.shape[1]
        theta0 = np.zeros(n_free)
        if design.shape[1]:
            beta0, *_ = np.linalg.lstsq(design, w, rcond=None)
            theta0[p + q + P + Q:] = beta0

        if n_free:
            res = minimize(
                self._neg_loglik,
                theta0,
                args=(w, design),
                method="L-BFGS-B",
                options={"maxiter": int(self.maxiter), "ftol": self.tol,
                         "gtol": self.tol, "maxls": 20},
            )
            theta_hat = res.x if np.isfinite(res.fun) else theta0
        else:
            theta_hat = theta0

        ar, ma, sar, sma, beta = self._unpack(theta_hat)
        w_adj = w - design @ beta if design.shape[1] else w
        phi, th = self._expanded_poly(ar, ma, sar, sma)
        T, R = self._state_space(phi, th)
        P0 = self._lyapunov(T, R)
        ll, sigma2, resid, a_last, P_last = self._kalman(w_adj, T, R, P0)

        self.ar_params_ = ar
        self.ma_params_ = ma
        self.seasonal_ar_params_ = sar
        self.seasonal_ma_params_ = sma
        n_tr = self._n_trend()
        self.trend_params_ = beta[:n_tr]
        self.exog_params_ = beta[n_tr:]
        self._beta_all = beta
        self._phi_full = phi
        self._theta_full = th
        self._T, self._R = T, R
        self._n_diff_lost = n_lost
        self._w_last = w.copy()
        self.sigma2_ = float(sigma2)
        self.loglik_ = float(ll)
        k = n_free + 1
        n_eff = len(w_adj)
        self.aic_ = float(2 * k - 2 * ll)
        self.bic_ = float(k * np.log(n_eff) - 2 * ll)
        self.resid_ = resid
        self.state_ = a_last
        self.state_cov_ = P_last

        self._is_fitted = True
        return self

    # -- forecasting -------------------------------------------------------

    def _future_design(self, steps: int, X: Optional[np.ndarray]) -> np.ndarray:
        """Build the differenced design block for the forecast horizon.

        Parameters
        ----------
        steps : int
            Forecast horizon.
        X : np.ndarray or None
            Future exogenous rows, or None if the model has no exogenous part.

        Returns
        -------
        design : np.ndarray of shape (steps, k)
            Differenced deterministic + exogenous design for the horizon.
        """
        _, d, _, _, D, _, m = self._spec()
        if self._k_exog and X is None:
            raise ValueError(
                f"This SARIMAX was fitted with {self._k_exog} exogenous "
                "regressor(s); predict() requires X with the future values "
                "for those regressors."
            )
        if self._k_exog == 0 and X is not None:
            raise ValueError(
                "This SARIMAX was fitted without exogenous regressors, so "
                "predict() must be called without X."
            )
        if self._k_exog:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape != (steps, self._k_exog):
                raise ValueError(
                    f"X must have shape ({steps}, {self._k_exog}) for a "
                    f"{steps}-step forecast; got {X.shape}"
                )

        blocks = []
        if self._n_trend():
            blocks.append(self._trend_design(self._n_w + 1, steps))
        if self._k_exog:
            full = np.vstack([self._exog_train, X])
            dif = _diff(_diff(full, m, D) if m else full, 1, d)
            blocks.append(dif[-steps:])
        if not blocks:
            return np.zeros((steps, 0))
        return np.hstack(blocks)

    def _undo_diff(self, f: np.ndarray) -> np.ndarray:
        """Integrate forecasts back onto the original scale.

        Parameters
        ----------
        f : np.ndarray of shape (steps,)
            Forecasts of the fully differenced series.

        Returns
        -------
        out : np.ndarray of shape (steps,)
            Forecasts on the scale of the original series.
        """
        _, d, _, _, D, _, m = self._spec()
        y = self.y_original_
        # Rebuild the chain of intermediate series so each operator can be
        # inverted against the right history.
        seasonal_stages = [np.asarray(y, float)]
        for _ in range(D):
            seasonal_stages.append(_diff(seasonal_stages[-1], m, 1))
        regular_stages = [seasonal_stages[-1]]
        for _ in range(d):
            regular_stages.append(_diff(regular_stages[-1], 1, 1))

        cur = np.asarray(f, float)
        for level in range(d - 1, -1, -1):
            cur = self._integrate(cur, regular_stages[level], 1)
        for level in range(D - 1, -1, -1):
            cur = self._integrate(cur, seasonal_stages[level], m)
        return cur

    @staticmethod
    def _integrate(f: np.ndarray, hist: np.ndarray, lag: int) -> np.ndarray:
        """Invert one lag-``lag`` differencing operator.

        Parameters
        ----------
        f : np.ndarray of shape (steps,)
            Forecasts of the differenced series.
        hist : np.ndarray
            History of the *undifferenced* series.
        lag : int
            Differencing lag.

        Returns
        -------
        out : np.ndarray of shape (steps,)
            Forecasts of the undifferenced series.
        """
        n, h = len(hist), len(f)
        ext = np.concatenate([hist, np.zeros(h)])
        for i in range(h):
            ext[n + i] = f[i] + ext[n + i - lag]
        return ext[n:]

    def predict(self, steps: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast future values.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        X : np.ndarray of shape (steps, k_exog), optional, default=None
            Future values of the exogenous regressors. Required if and only
            if the model was fitted with exogenous regressors.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Point forecasts on the scale of the original series.
        """
        self._check_is_fitted()
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be >= 1")

        design = self._future_design(steps, X)
        T, R = self._T, self._R
        a = self.state_.copy()
        core = np.empty(steps)
        for h in range(steps):
            core[h] = a[0]
            a = T @ a
        if design.shape[1]:
            core = core + design @ self._beta_all
        return self._undo_diff(core)

    def forecast_variance(self, steps: int = 1) -> np.ndarray:
        """Return the forecast error variance for each horizon.

        Variances come from the MA(:math:`\\infty`) weights of the expanded
        model, which folds the differencing operators into the
        autoregressive polynomial, so they grow correctly with the horizon.

        Parameters
        ----------
        steps : int, default=1
            Forecast horizon.

        Returns
        -------
        var : np.ndarray of shape (steps,)
            Forecast error variances.
        """
        self._check_is_fitted()
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be >= 1")
        _, d, _, _, D, _, m = self._spec()
        ar_poly = np.concatenate([[1.0], -self._phi_full])
        for _ in range(d):
            ar_poly = _poly_mul(ar_poly, np.array([1.0, -1.0]))
        for _ in range(D):
            seas = np.zeros(m + 1)
            seas[0], seas[m] = 1.0, -1.0
            ar_poly = _poly_mul(ar_poly, seas)
        phi_star = -ar_poly[1:]
        psi = _psi_weights(phi_star, self._theta_full, steps)
        return self.sigma2_ * np.cumsum(psi ** 2)

    def predict_interval(self, steps: int = 1, alpha: float = 0.05,
                         X: Optional[np.ndarray] = None):
        """Forecast with a Gaussian prediction interval.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        alpha : float, default=0.05
            Significance level; ``0.05`` gives a 95% interval.
        X : np.ndarray of shape (steps, k_exog), optional, default=None
            Future exogenous regressors, required if the model uses them.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Point forecasts.
        lower : np.ndarray of shape (steps,)
            Lower interval bounds.
        upper : np.ndarray of shape (steps,)
            Upper interval bounds.
        """
        self._check_is_fitted()
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        forecast = self.predict(steps=steps, X=X)
        se = np.sqrt(self.forecast_variance(steps))
        z = float(norm.ppf(1.0 - alpha / 2.0))
        return forecast, forecast - z * se, forecast + z * se

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
