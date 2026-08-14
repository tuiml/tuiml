"""Cox proportional_hazards model with partial-likelihood estimation."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List

from tuiml.base.algorithms import Survival, survival


def _step_function_eval(timeline, values, query, initial=1.0):
    """Evaluate a right-continuous step function defined on ``timeline``."""
    query = np.asarray(query, dtype=np.float64)
    scalar = query.ndim == 0
    q = np.atleast_1d(query)
    idx = np.searchsorted(timeline, q, side="right") - 1
    out = np.full(q.shape, initial, dtype=np.float64)
    mask = idx >= 0
    out[mask] = values[idx[mask]]
    return out[()] if scalar else out


@survival(tags=["semiparametric", "proportional_hazards", "linear"], version="1.0.0")
class CoxPHSurvival(Survival):
    """Cox proportional_hazards model.

    The **semiparametric** workhorse of survival analysis. It models the hazard
    of subject :math:`i` as

    .. math::
        h(t \\mid x_i) = h_0(t) \\exp(x_i^T \\beta)

    where :math:`h_0(t)` is an unspecified baseline hazard and :math:`\\beta`
    are covariate effects. A positive coefficient means the covariate raises
    the hazard and therefore shortens expected survival.

    Overview
    --------
    1. Order subjects by observed time, descending.
    2. Maximise the **partial likelihood** in :math:`\\beta` with Newton-Raphson
       (optional L2 penalty).
    3. Estimate the baseline hazard via the **Breslow** estimator and expose the
       corresponding baseline survival function.

    Theory
    ------
    For right-censored data the partial log-likelihood is

    .. math::
        \\ell(\\beta) = \\sum_{i: \\delta_i = 1}
        \\left[ x_i^T \\beta
        - \\log \\sum_{j: t_j \\geq t_i} \\exp(x_j^T \\beta) \\right]

    whose gradient and Hessian are accumulated over risk sets
    :math:`R(t_i) = \\{j : t_j \\geq t_i\\}`:

    .. math::
        \\nabla \\ell = \\sum_{i: \\delta_i = 1}
        \\left( x_i - \\frac{\\sum_{j \\in R(t_i)} x_j e^{x_j^T \\beta}}
        {\\sum_{j \\in R(t_i)} e^{x_j^T \\beta}} \\right)

    With L2 regularisation, :math:`\\ell(\\beta) - \\frac{\\alpha}{2}\\|\\beta\\|^2`
    is maximised, which shrinks coefficients toward zero. The baseline hazard is

    .. math::
        \\hat{h}_0(t_j) = \\frac{d_j}{\\sum_{i \\in R(t_j)} \\exp(x_i^T \\beta)}

    and :math:`\\hat{S}_0(t) = \\exp\\!\\left(-\\sum_{t_j \\leq t} \\hat{h}_0(t_j)\\right)`.

    Parameters
    ----------
    penalty : str, default="l2"
        Regularisation type. ``"l2"`` adds a ridge penalty; ``None`` fits the
        unpenalised partial likelihood.
    alpha : float, default=0.0
        Regularisation strength (ignored unless ``penalty="l2"``).
    tol : float, default=1e-6
        Newton-Raphson convergence tolerance on the max coefficient change.
    max_iter : int, default=100
        Maximum Newton-Raphson iterations.

    Attributes
    ----------
    coefficients_ : np.ndarray of shape (n_features,)
        Fitted covariate effects (log hazard ratios).
    baseline_times_ : np.ndarray of shape (n_events,)
        Sorted unique event times.
    baseline_hazard_ : np.ndarray of shape (n_events,)
        Breslow baseline hazard increments.
    baseline_cumulative_hazard_ : np.ndarray of shape (n_events,)
        Cumulative baseline hazard.
    baseline_survival_ : np.ndarray of shape (n_events,)
        Baseline survival :math:`\\exp(-H_0(t))`.
    n_features_in_ : int
        Number of features seen during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Fitting: :math:`O(I \\cdot (n p + n p^2))` for :math:`I` iterations,
      dominated by the per-risk-set Hessian accumulation.
    - Prediction: :math:`O(p)` per sample.

    **When to use CoxPHSurvival:**

    - When the proportional_hazards assumption is reasonable.
    - When you need interpretable coefficients rather than pure predictive power.

    References
    ----------
    .. [Cox1972] Cox, D.R. (1972). **Regression Models and Life-Tables.**
           *Journal of the Royal Statistical Society, Series B*, 34(2), 187-220.
           DOI: `10.1111/j.2517-6161.1972.tb00899.x <https://doi.org/10.1111/j.2517-6161.1972.tb00899.x>`_
    .. [Breslow1972] Breslow, N.E. (1972). **Discussion of Cox (1972).**
           *Journal of the Royal Statistical Society, Series B*, 34(2), 216-217.

    See Also
    --------
    :class:`~tuiml.algorithms.survival.KaplanMeierEstimator` : Non-parametric baseline.
    :class:`~tuiml.algorithms.survival.RandomSurvivalForest` : Non-linear ensemble.

    Examples
    --------
    >>> from tuiml.algorithms.survival import CoxPHSurvival
    >>> import numpy as np
    >>> # A single binary covariate: group 1 always fails first.
    >>> X = np.array([[1.], [1.], [1.], [0.], [0.], [0.]])
    >>> time = np.array([1., 2., 3., 4., 5., 6.])
    >>> event = np.ones(6)
    >>> cox = CoxPHSurvival().fit(X, time, event)
    >>> bool(cox.coefficients_[0] > 0)
    True
    """

    def __init__(
        self,
        penalty: Optional[str] = "l2",
        alpha: float = 0.0,
        tol: float = 1e-6,
        max_iter: int = 100,
    ):
        """Initialize the Cox proportional_hazards model.

        Parameters
        ----------
        penalty : str or None, default="l2"
            Regularisation type (``"l2"`` or ``None``).
        alpha : float, default=0.0
            Regularisation strength.
        tol : float, default=1e-6
            Convergence tolerance.
        max_iter : int, default=100
            Maximum Newton-Raphson iterations.
        """
        super().__init__()
        self.penalty = penalty
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

        self.coefficients_ = None
        self.baseline_times_ = None
        self.baseline_hazard_ = None
        self.baseline_cumulative_hazard_ = None
        self.baseline_survival_ = None
        self.n_features_in_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "penalty": {
                "type": ["string", "null"],
                "default": "l2",
                "enum": ["l2", None],
                "description": "Regularisation type (l2 or none)",
            },
            "alpha": {
                "type": "number",
                "default": 0.0,
                "minimum": 0.0,
                "description": "L2 regularisation strength",
            },
            "tol": {
                "type": "number",
                "default": 1e-6,
                "minimum": 0.0,
                "description": "Newton-Raphson convergence tolerance",
            },
            "max_iter": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Maximum Newton-Raphson iterations",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "censored", "regression", "interpretable", "proportional_hazards"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Fit: O(I * (n*p + n*p^2)); predict: O(p) per sample"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Cox, D.R., 1972. Regression models and life-tables. JRSS-B, 34(2), 187-220.",
            "Breslow, N.E., 1972. Discussion of Cox (1972). JRSS-B, 34(2), 216-217."
        ]

    def fit(self, X, time, event) -> "CoxPHSurvival":
        """Fit the model by maximising the (penalised) partial likelihood.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariate matrix.
        time : array-like of shape (n_samples,)
            Observed time (event or censoring).
        event : array-like of shape (n_samples,)
            Event indicator (1 = event observed, 0 = right-censored).

        Returns
        -------
        self : CoxPHSurvival
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        time = np.asarray(time, dtype=np.float64).ravel()
        event = np.asarray(event, dtype=np.float64).ravel()
        n, p = X.shape

        if len(time) != n or len(event) != n:
            raise ValueError("X, time and event must describe the same number of samples")
        if np.any(time < 0):
            raise ValueError("time must be non-negative")
        if not np.all((event == 0) | (event == 1)):
            raise ValueError("event must contain only 0 and 1")
        if np.sum(event) == 0:
            raise ValueError("at least one observed event is required to fit a Cox model")
        if self.penalty not in (None, "l2"):
            raise ValueError(f"Unknown penalty '{self.penalty}'; use 'l2' or None")

        # Descending time order; risk sets are built by walking times high to
        # low so that tied times share one risk set (Breslow's convention).
        order = np.argsort(-time, kind="stable")
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]

        alpha = self.alpha if self.penalty == "l2" else 0.0
        beta = np.zeros(p, dtype=np.float64)

        for _ in range(self.max_iter):
            eta = np.clip(X_sorted @ beta, -50.0, 50.0)
            r = np.exp(eta)

            rsum = 0.0
            rsum1 = np.zeros(p, dtype=np.float64)
            rsum2 = np.zeros((p, p), dtype=np.float64)
            grad = np.zeros(p, dtype=np.float64)
            info = np.zeros((p, p), dtype=np.float64)

            i = 0
            while i < n:
                tj = time_sorted[i]
                j = i
                while j < n and time_sorted[j] == tj:
                    j += 1
                block = slice(i, j)
                rsum += r[block].sum()
                rsum1 += (r[block, None] * X_sorted[block]).sum(axis=0)
                for k in range(i, j):
                    rsum2 += r[k] * np.outer(X_sorted[k], X_sorted[k])

                d_events = event_sorted[block].sum()
                if d_events > 0:
                    mean = rsum1 / rsum
                    grad += (
                        (X_sorted[block] * event_sorted[block, None]).sum(axis=0)
                        - d_events * mean
                    )
                    info += d_events * (rsum2 / rsum - np.outer(mean, mean))
                i = j

            if alpha:
                grad = grad - alpha * beta
                info += alpha * np.eye(p)

            try:
                delta = np.linalg.solve(info, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.solve(info + 1e-6 * np.eye(p), grad)

            beta = beta + delta
            if np.max(np.abs(delta)) < self.tol:
                break

        self.coefficients_ = beta
        self.n_features_in_ = p

        # Breslow baseline hazard on the original (unsorted) data.
        r_full = np.exp(np.clip(X @ beta, -50.0, 50.0))
        event_times = np.unique(time[event == 1])
        h0 = np.zeros_like(event_times, dtype=np.float64)
        for k, tj in enumerate(event_times):
            d_j = np.sum(event[time == tj])
            h0[k] = d_j / np.sum(r_full[time >= tj])

        self.baseline_times_ = event_times
        self.baseline_hazard_ = h0
        self.baseline_cumulative_hazard_ = np.cumsum(h0)
        self.baseline_survival_ = np.exp(-self.baseline_cumulative_hazard_)
        self._is_fitted = True
        return self

    def predict_risk(self, X) -> np.ndarray:
        """Return the linear predictor :math:`X \\beta` for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.

        Returns
        -------
        risk : np.ndarray of shape (n_samples,)
            Linear predictor. Higher values mean higher hazard and an earlier
            expected event.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coefficients_

    def _baseline_cumulative_hazard_at(self, times):
        """Evaluate the baseline cumulative hazard step function at ``times``."""
        return _step_function_eval(
            self.baseline_times_, self.baseline_cumulative_hazard_, times, initial=0.0
        )

    def predict_cumulative_hazard(self, X, times=None) -> np.ndarray:
        """Return the predicted cumulative hazard for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.
        times : array-like, optional
            Time point(s) at which to evaluate the hazard. Defaults to the
            fitted ``baseline_times_``.

        Returns
        -------
        H : np.ndarray of shape (n_samples, n_times)
            :math:`H_i(t) = H_0(t) \\exp(x_i^T \\beta)`.
        """
        self._check_is_fitted()
        if times is None:
            times = self.baseline_times_
        H0 = self._baseline_cumulative_hazard_at(times)
        exp_eta = np.exp(np.clip(self.predict_risk(X), -50.0, 50.0))
        return np.outer(exp_eta, H0)

    def predict_survival_function(self, X, times=None) -> np.ndarray:
        """Return the predicted survival function for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.
        times : array-like, optional
            Time point(s) at which to evaluate the survival function. Defaults
            to the fitted ``baseline_times_``.

        Returns
        -------
        S : np.ndarray of shape (n_samples, n_times)
            :math:`S_i(t) = \\exp(-H_i(t))`.
        """
        self._check_is_fitted()
        return np.exp(-self.predict_cumulative_hazard(X, times))
