"""Croston's method and its variants for intermittent demand forecasting."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List

from tuiml.base.algorithms import Regressor, regressor


@regressor(tags=["timeseries", "forecasting", "intermittent-demand"], version="1.0.0")
class CrostonForecaster(Regressor):
    """
    Croston's method for **intermittent demand** forecasting.

    Intermittent demand series are mostly zero with occasional non-zero
    demands. Simple exponential smoothing applied directly to such a series
    is biased and updates at the wrong moments, because a long run of zeros
    drags the level down between demands. Croston's insight is to smooth
    **two separate series**: the sizes of the non-zero demands and the
    **inter-arrival intervals** between them. The per-period forecast is
    the ratio of the two.

    Overview
    --------
    1. Walk through the series and record the non-zero demands and the
       number of periods since the previous non-zero demand.
    2. Exponentially smooth the demand sizes into :math:`\\hat{z}`.
    3. Exponentially smooth the inter-arrival intervals into
       :math:`\\hat{p}` (both updated only when a demand occurs).
    4. Form the per-period forecast :math:`\\hat{y} = \\hat{z} / \\hat{p}`,
       optionally multiplied by a bias-correction factor.
    5. The forecast is flat over the horizon.

    Theory
    ------
    Let :math:`z_j` be the size of the :math:`j`-th non-zero demand and
    :math:`q_j` the number of periods since the previous one. On a period
    with a demand,

    .. math::
        \\hat{z}_j = \\alpha z_j + (1 - \\alpha) \\hat{z}_{j-1},
        \\qquad
        \\hat{p}_j = \\alpha q_j + (1 - \\alpha) \\hat{p}_{j-1},

    and on a period without demand both estimates are carried forward.
    The forecast is

    .. math::
        \\hat{y}_{n+h} = c \\, \\frac{\\hat{z}}{\\hat{p}},

    with a variant-dependent correction factor :math:`c`:

    - ``"classic"``: :math:`c = 1` (Croston, 1972).
    - ``"sba"``: :math:`c = 1 - \\alpha/2`, the Syntetos-Boylan
      approximation, which removes the leading term of the inversion bias
      in :math:`E[\\hat{z}/\\hat{p}]`.
    - ``"sbj"``: :math:`c = 1 - \\alpha/(2 - \\alpha)`, the
      Shale-Boylan-Johnston correction derived under Poisson arrivals.

    The ``"tsb"`` variant (Teunter, Syntetos and Babai, 2011) replaces the
    interval by the **demand probability** :math:`\\hat{d}`, updated
    *every* period,

    .. math::
        \\hat{d}_t = \\alpha_p \\mathbb{1}[y_t > 0]
                     + (1 - \\alpha_p) \\hat{d}_{t-1},
        \\qquad
        \\hat{y}_{n+h} = \\hat{d}_n \\, \\hat{z}_n ,

    which lets the forecast decay when demand stops -- Croston's estimate
    never does, making it obsolescence-blind.

    Parameters
    ----------
    alpha : float, default=0.1
        Smoothing parameter for demand sizes, and for intervals in the
        Croston-family variants. Must lie in ``(0, 1]``.
    variant : {"classic", "sba", "sbj", "tsb"}, default="classic"
        Which estimator to use. See the Theory section.
    alpha_prob : float, optional, default=None
        Smoothing parameter for the demand probability in the ``"tsb"``
        variant. Defaults to ``alpha`` when ``None``. Ignored by the other
        variants.

    Attributes
    ----------
    demand_ : float
        Final smoothed non-zero demand size :math:`\\hat{z}`.
    interval_ : float
        Final smoothed inter-arrival interval :math:`\\hat{p}`. Set to
        ``nan`` for the ``"tsb"`` variant, which does not estimate it.
    probability_ : float
        Final smoothed demand probability :math:`\\hat{d}`. Set to ``nan``
        for the Croston-family variants.
    correction_ : float
        Bias-correction factor :math:`c` applied to the ratio.
    forecast_ : float
        The flat per-period forecast.
    n_nonzero_ : int
        Number of non-zero demands in the training series.
    fitted_values_ : np.ndarray
        In-sample one-step-ahead forecasts.
    resid_ : np.ndarray
        In-sample residuals.
    n_obs_ : int
        Number of training observations.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n)`.
    - Prediction: :math:`O(h)`.

    **When to use CrostonForecaster:**

    - Spare parts, slow-moving inventory and any series where most periods
      are zero and the mean inter-demand interval exceeds about 1.3.
    - When the quantity of interest is the expected demand *per period*
      for a safety-stock or reorder-point calculation.
    - Prefer ``"sba"`` over ``"classic"`` in practice: the ratio estimator
      is biased upward, and the correction is essentially free.
    - Prefer ``"tsb"`` when items go obsolete and the forecast must decay
      after demand stops.

    References
    ----------
    .. [Croston1972] Croston, J. D. (1972). **Forecasting and stock control
           for intermittent demands.** *Operational Research Quarterly*,
           23(3), 289-303. :doi:`10.1057/jors.1972.50`
    .. [Syntetos2005] Syntetos, A. A., & Boylan, J. E. (2005). **The accuracy
           of intermittent demand estimates.** *International Journal of
           Forecasting*, 21(2), 303-314.
           :doi:`10.1016/j.ijforecast.2004.10.001`
    .. [Shale2006] Shale, E. A., Boylan, J. E., & Johnston, F. R. (2006).
           **Forecasting for intermittent demand: the estimation of an
           unbiased average.** *Journal of the Operational Research
           Society*, 57(5), 588-592. :doi:`10.1057/palgrave.jors.2602031`
    .. [Teunter2011] Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011).
           **Intermittent demand: linking forecasting to inventory
           obsolescence.** *European Journal of Operational Research*,
           214(3), 606-615. :doi:`10.1016/j.ejor.2011.05.018`

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.ExponentialSmoothing` : Smoothing for continuous demand.
    :class:`~tuiml.algorithms.timeseries.theta.ThetaForecaster` : Theta method for trended series.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.croston import CrostonForecaster
    >>> y = np.array([0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10], dtype=float)
    >>> model = CrostonForecaster(alpha=0.1, variant="classic").fit(y)
    >>> float(model.predict(steps=1)[0])
    2.5
    >>> sba = CrostonForecaster(alpha=0.1, variant="sba").fit(y)
    >>> float(sba.predict(steps=1)[0])
    2.375
    """

    _CORRECTIONS = ("classic", "sba", "sbj", "tsb")

    def __init__(
        self,
        alpha: float = 0.1,
        variant: str = "classic",
        alpha_prob: float | None = None,
    ):
        """Initialize the Croston forecaster.

        Parameters
        ----------
        alpha : float, default=0.1
            Smoothing parameter for demand sizes and intervals.
        variant : {"classic", "sba", "sbj", "tsb"}, default="classic"
            Which estimator to use.
        alpha_prob : float, optional, default=None
            Probability smoothing parameter for the ``"tsb"`` variant.
        """
        super().__init__()
        self.alpha = alpha
        self.variant = variant
        self.alpha_prob = alpha_prob

        # Fitted attributes
        self.demand_ = None
        self.interval_ = None
        self.probability_ = None
        self.correction_ = None
        self.forecast_ = None
        self.n_nonzero_ = None
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
            "alpha": {
                "type": "number",
                "default": 0.1,
                "exclusiveMinimum": 0.0,
                "maximum": 1.0,
                "description": "Smoothing parameter for demand sizes and intervals",
            },
            "variant": {
                "type": "string",
                "default": "classic",
                "enum": ["classic", "sba", "sbj", "tsb"],
                "description": "Croston variant / bias correction to apply",
            },
            "alpha_prob": {
                "type": ["number", "null"],
                "default": None,
                "exclusiveMinimum": 0.0,
                "maximum": 1.0,
                "description": "Probability smoothing parameter for the TSB variant",
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
            "interpretable",
            "noise_tolerant",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(n), Prediction: O(h)"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Croston, 1972. Forecasting and stock control for intermittent demands. ORQ 23(3).",
            "Syntetos & Boylan, 2005. The accuracy of intermittent demand estimates. IJF 21(2).",
            "Shale, Boylan & Johnston, 2006. Estimation of an unbiased average. JORS 57(5).",
            "Teunter, Syntetos & Babai, 2011. Intermittent demand and obsolescence. EJOR 214(3).",
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _correction_factor(self, alpha: float) -> float:
        """Return the variant's bias-correction factor.

        Parameters
        ----------
        alpha : float
            Smoothing parameter in use.

        Returns
        -------
        factor : float
            Multiplier applied to the size/interval ratio.
        """
        if self.variant == "sba":
            return 1.0 - alpha / 2.0
        if self.variant == "sbj":
            return 1.0 - alpha / (2.0 - alpha)
        return 1.0

    # ------------------------------------------------------------------
    # Fitting and prediction
    # ------------------------------------------------------------------

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> "CrostonForecaster":
        """Fit the Croston model to an intermittent demand series.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Demand per period. Zeros denote periods without demand.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        self : CrostonForecaster
            Fitted estimator.
        """
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        if n < 1:
            raise ValueError("CrostonForecaster requires at least 1 observation")
        if not np.isfinite(y).all():
            raise ValueError("y must not contain NaN or infinite values")
        if self.variant not in self._CORRECTIONS:
            raise ValueError(
                f"variant must be one of {self._CORRECTIONS}, got {self.variant!r}"
            )
        alpha = float(self.alpha)
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must lie in (0, 1]")
        alpha_p = alpha if self.alpha_prob is None else float(self.alpha_prob)
        if not (0.0 < alpha_p <= 1.0):
            raise ValueError("alpha_prob must lie in (0, 1]")

        self.n_obs_ = n
        nonzero = np.flatnonzero(y != 0.0)
        self.n_nonzero_ = int(nonzero.size)
        self.correction_ = self._correction_factor(alpha)
        fitted = np.zeros(n, dtype=float)

        if self.n_nonzero_ == 0:
            # An all-zero history forecasts zero. No ratio is ever formed,
            # so there is nothing to divide by.
            self.demand_ = 0.0
            self.interval_ = float(n) if self.variant != "tsb" else np.nan
            self.probability_ = 0.0 if self.variant == "tsb" else np.nan
            self.forecast_ = 0.0
            self.fitted_values_ = fitted
            self.resid_ = y - fitted
            self._is_fitted = True
            return self

        first = int(nonzero[0])

        if self.variant == "tsb":
            demand = float(y[first])
            prob = 1.0 / float(first + 1)
            for t in range(n):
                fitted[t] = prob * demand
                if t < first:
                    continue
                if y[t] != 0.0:
                    demand = alpha * float(y[t]) + (1.0 - alpha) * demand
                    prob = alpha_p * 1.0 + (1.0 - alpha_p) * prob
                else:
                    prob = (1.0 - alpha_p) * prob
            self.demand_ = float(demand)
            self.probability_ = float(prob)
            self.interval_ = np.nan
            self.forecast_ = float(prob * demand)
        else:
            demand = float(y[first])
            # The first interval counts the periods from the start of the
            # series up to and including the first demand.
            interval = float(first + 1)
            since = 1
            for t in range(n):
                fitted[t] = self.correction_ * demand / interval
                if t < first:
                    continue
                if t > first:
                    if y[t] != 0.0:
                        demand = alpha * float(y[t]) + (1.0 - alpha) * demand
                        interval = alpha * float(since) + (1.0 - alpha) * interval
                        since = 1
                    else:
                        since += 1
            self.demand_ = float(demand)
            self.interval_ = float(interval)
            self.probability_ = np.nan
            # ``interval`` is a convex combination of counts >= 1, so it can
            # never reach zero; the guard documents that invariant.
            self.forecast_ = float(self.correction_ * demand / interval) if interval > 0.0 else 0.0

        self.fitted_values_ = fitted
        self.resid_ = y - fitted
        self._is_fitted = True
        return self

    def predict(self, steps: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast the expected demand per period.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Flat per-period demand forecast.
        """
        self._check_is_fitted()
        if steps < 1:
            raise ValueError("steps must be >= 1")
        return np.full(steps, self.forecast_, dtype=float)

    def fit_predict(self, y: np.ndarray, steps: int = 1) -> np.ndarray:
        """Fit the model and forecast in one call.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Demand per period.
        steps : int, default=1
            Number of future time steps to forecast.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Flat per-period demand forecast.
        """
        self.fit(y)
        return self.predict(steps)
