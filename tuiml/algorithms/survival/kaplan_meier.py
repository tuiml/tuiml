"""Non-parametric survival_curve estimators (Kaplan-Meier and Nelson-Aalen)."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List

from tuiml.base.algorithms import Survival, survival


def _step_function_eval(timeline, values, query, initial=1.0):
    """Evaluate a right-continuous step function defined on ``timeline``.

    Parameters
    ----------
    timeline : np.ndarray of shape (n_times,)
        Sorted breakpoints (ascending) where the step function changes.
    values : np.ndarray of shape (n_times,)
        Function value on ``[timeline[k], timeline[k + 1])``.
    query : float or array-like
        Point(s) at which to evaluate the step function.
    initial : float, default=1.0
        Value for ``query < timeline[0]``.

    Returns
    -------
    out : float or np.ndarray
        The step-function value at each query point.
    """
    query = np.asarray(query, dtype=np.float64)
    scalar = query.ndim == 0
    q = np.atleast_1d(query)
    idx = np.searchsorted(timeline, q, side="right") - 1
    out = np.full(q.shape, initial, dtype=np.float64)
    mask = idx >= 0
    out[mask] = values[idx[mask]]
    return out[()] if scalar else out


def _kaplan_meier(time, event):
    """Compute the product-limit and Nelson-Aalen estimators.

    Parameters
    ----------
    time : array-like of shape (n_samples,)
        Observed time (event or censoring).
    event : array-like of shape (n_samples,)
        Event indicator (1 = event, 0 = censored).

    Returns
    -------
    timeline : np.ndarray
        Sorted unique event times.
    survival : np.ndarray
        Product-limit survival at each event time.
    cumulative_hazard : np.ndarray
        Nelson-Aalen cumulative hazard at each event time.
    n_risk : np.ndarray
        Number at risk just before each event time.
    n_events : np.ndarray
        Number of events at each event time.
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.float64).ravel()

    event_times = np.unique(time[event == 1])
    survival = np.zeros_like(event_times, dtype=np.float64)
    cumulative_hazard = np.zeros_like(event_times, dtype=np.float64)
    n_risk = np.zeros_like(event_times, dtype=np.int64)
    n_events = np.zeros_like(event_times, dtype=np.int64)

    product = 1.0
    hazard = 0.0
    for k, tj in enumerate(event_times):
        d = np.sum(event[time == tj])
        r = np.sum(time >= tj)
        n_events[k] = d
        n_risk[k] = r
        hazard += d / r
        product *= (1.0 - d / r)
        cumulative_hazard[k] = hazard
        survival[k] = product

    return event_times, survival, cumulative_hazard, n_risk, n_events


@survival(tags=["non_parametric", "baseline", "survival_curve"], version="1.0.0")
class KaplanMeierEstimator(Survival):
    """Kaplan-Meier product-limit estimator of the survival function.

    The **non_parametric maximum-likelihood estimate** of :math:`S(t) = P(T > t)`
    under right-censoring. No covariates: it describes the population as a
    whole, so it is a baseline every covariate model must beat rather than a
    per-sample risk model.

    Overview
    --------
    1. Sort the observed times and keep only the times at which an event
       occurred (``timeline_``).
    2. At each event time :math:`t_j`, compute :math:`d_j` (events at
       :math:`t_j`) and :math:`n_j` (subjects at risk just before :math:`t_j`).
    3. Multiply the survival factors :math:`(1 - d_j / n_j)` to form the
       step-function estimate of :math:`S(t)`.

    Theory
    ------
    The product-limit estimator is

    .. math::
        \\hat{S}(t) = \\prod_{j: t_j \\leq t}
        \\left(1 - \\frac{d_j}{n_j}\\right)

    and the corresponding Nelson-Aalen cumulative hazard is

    .. math::
        \\hat{H}(t) = \\sum_{j: t_j \\leq t} \\frac{d_j}{n_j}.

    Censored observations contribute to :math:`n_j` while they are still at
    risk, but never contribute an event, which is how the estimator respects
    partial information.

    Parameters
    ----------
    This estimator takes no hyperparameters.

    Attributes
    ----------
    timeline_ : np.ndarray of shape (n_events,)
        Sorted unique event times.
    survival_ : np.ndarray of shape (n_events,)
        Product-limit survival at each ``timeline_`` entry.
    cumulative_hazard_ : np.ndarray of shape (n_events,)
        Nelson-Aalen cumulative hazard at each ``timeline_`` entry.
    n_risk_ : np.ndarray of shape (n_events,)
        Number of subjects at risk just before each event time.
    n_events_ : np.ndarray of shape (n_events,)
        Number of events at each event time.
    total_cumulative_hazard_ : float
        Cumulative hazard at the final event time; the constant risk score.

    Notes
    -----
    **Complexity:**

    - Fitting: :math:`O(n \\log n)` (sorting dominates).
    - Prediction: :math:`O(\\log n)` per query via binary search.

    **When to use KaplanMeierEstimator:**

    - As a descriptive summary of a single population's survival curve.
    - As the baseline against which covariate models are compared.
    - The survival function is only defined up to the last event time; beyond
      the largest observed time the estimate is not identifiable.

    References
    ----------
    .. [Kaplan1958] Kaplan, E.L. and Meier, P. (1958).
           **Nonparametric Estimation from Incomplete Observations.**
           *Journal of the American Statistical Association*, 53(282), 457-481.
           DOI: `10.1080/01621459.1958.10501452 <https://doi.org/10.1080/01621459.1958.10501452>`_

    See Also
    --------
    :class:`~tuiml.algorithms.survival.NelsonAalenEstimator` : Cumulative-hazard analogue.
    :class:`~tuiml.algorithms.survival.CoxPHSurvival` : Covariate-adjusted model.

    Examples
    --------
    >>> from tuiml.algorithms.survival import KaplanMeierEstimator
    >>> import numpy as np
    >>> km = KaplanMeierEstimator().fit([2, 3, 4, 5], [1, 1, 0, 1])
    >>> km.timeline_.tolist()
    [2.0, 3.0, 5.0]
    >>> np.round(km.survival_, 4).tolist()
    [0.75, 0.5, 0.0]
    >>> np.round(km.predict_survival_function([1, 2, 3, 4, 5, 6]), 6).tolist()
    [1.0, 0.75, 0.5, 0.5, 0.0, 0.0]
    """

    def __init__(self):
        """Initialize the Kaplan-Meier estimator."""
        super().__init__()
        self.timeline_ = None
        self.survival_ = None
        self.cumulative_hazard_ = None
        self.n_risk_ = None
        self.n_events_ = None
        self.total_cumulative_hazard_ = 0.0

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["survival_curve", "non_parametric", "censored", "numeric"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Fit: O(n log n); predict: O(log n) per query via binary search"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Kaplan, E.L. and Meier, P., 1958. Nonparametric estimation from "
            "incomplete observations. JASA, 53(282), 457-481."
        ]

    def fit(self, time, event) -> "KaplanMeierEstimator":
        """Fit the product-limit estimator.

        Parameters
        ----------
        time : array-like of shape (n_samples,)
            Observed time (event or censoring).
        event : array-like of shape (n_samples,)
            Event indicator (1 = event observed, 0 = right-censored).

        Returns
        -------
        self : KaplanMeierEstimator
            Fitted estimator.
        """
        time = np.asarray(time, dtype=np.float64).ravel()
        event = np.asarray(event, dtype=np.float64).ravel()
        if len(time) != len(event):
            raise ValueError("time and event must have the same length")
        if np.any(time < 0):
            raise ValueError("time must be non-negative")
        if not np.all((event == 0) | (event == 1)):
            raise ValueError("event must contain only 0 and 1")

        (
            self.timeline_,
            self.survival_,
            self.cumulative_hazard_,
            self.n_risk_,
            self.n_events_,
        ) = _kaplan_meier(time, event)
        self.total_cumulative_hazard_ = (
            float(self.cumulative_hazard_[-1]) if len(self.cumulative_hazard_) else 0.0
        )
        self._is_fitted = True
        return self

    def predict_survival_function(self, times):
        """Return the estimated survival probability at ``times``.

        Parameters
        ----------
        times : float or array-like
            Time point(s) at which to evaluate :math:`\\hat{S}(t)`.

        Returns
        -------
        S : float or np.ndarray
            Product-limit survival estimate at each query time.
        """
        self._check_is_fitted()
        return _step_function_eval(self.timeline_, self.survival_, times, initial=1.0)

    def predict_cumulative_hazard(self, times):
        """Return the Nelson-Aalen cumulative hazard at ``times``.

        Parameters
        ----------
        times : float or array-like
            Time point(s) at which to evaluate :math:`\\hat{H}(t)`.

        Returns
        -------
        H : float or np.ndarray
            Cumulative hazard estimate at each query time.
        """
        self._check_is_fitted()
        return _step_function_eval(
            self.timeline_, self.cumulative_hazard_, times, initial=0.0
        )

    def predict_risk(self, X) -> np.ndarray:
        """Return a constant risk score (a non_parametric baseline).

        Kaplan-Meier models the population, not individual risk, so every
        sample receives the same score: the total Nelson-Aalen cumulative
        hazard over the observation window.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates. Only the number of rows is used.

        Returns
        -------
        risk : np.ndarray of shape (n_samples,)
            Constant baseline risk.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        n = X.shape[0] if X.ndim == 2 else len(X)
        return np.full(n, self.total_cumulative_hazard_, dtype=np.float64)


@survival(tags=["non_parametric", "baseline", "cumulative-hazard"], version="1.0.0")
class NelsonAalenEstimator(Survival):
    """Nelson-Aalen estimator of the cumulative hazard function.

    The **non_parametric estimate** of the cumulative hazard
    :math:`H(t) = -\\log S(t)`, from which a survival curve
    :math:`\\hat{S}(t) = e^{-\\hat{H}(t)}` follows. Like Kaplan-Meier it is a
    population-level baseline with no covariates.

    Overview
    --------
    1. Keep only the event times (``timeline_``).
    2. Accumulate the increments :math:`d_j / n_j` into the cumulative hazard.
    3. Expose the survival function as :math:`\\exp(-\\hat{H}(t))`.

    Theory
    ------
    .. math::
        \\hat{H}(t) = \\sum_{j: t_j \\leq t} \\frac{d_j}{n_j},
        \\qquad
        \\hat{S}(t) = \\exp\\!\\left(-\\hat{H}(t)\\right).

    The Nelson-Aalen estimator has slightly lower finite-sample bias than the
    Kaplan-Meier estimate of the same quantity and is the canonical input for
    leaf-hazard conventions in survival ensembles.

    Parameters
    ----------
    This estimator takes no hyperparameters.

    Attributes
    ----------
    timeline_ : np.ndarray of shape (n_events,)
        Sorted unique event times.
    cumulative_hazard_ : np.ndarray of shape (n_events,)
        Nelson-Aalen cumulative hazard at each ``timeline_`` entry.
    survival_ : np.ndarray of shape (n_events,)
        Exponential survival :math:`\\exp(-\\hat{H}(t))` at each entry.
    n_risk_ : np.ndarray of shape (n_events,)
        Number of subjects at risk just before each event time.
    n_events_ : np.ndarray of shape (n_events,)
        Number of events at each event time.
    total_cumulative_hazard_ : float
        Cumulative hazard at the final event time; the constant risk score.

    Notes
    -----
    **Complexity:**

    - Fitting: :math:`O(n \\log n)` (sorting dominates).
    - Prediction: :math:`O(\\log n)` per query via binary search.

    References
    ----------
    .. [Nelson1969] Nelson, W. (1969). **Hazard Plotting for Incomplete Failure
           Data.** *Journal of Quality Technology*, 1(1), 27-52.
    .. [Aalen1978] Aalen, O. (1978). **Nonparametric Inference for a Family of
           Counting Processes.** *The Annals of Statistics*, 6(4), 701-726.

    See Also
    --------
    :class:`~tuiml.algorithms.survival.KaplanMeierEstimator` : Product-limit analogue.

    Examples
    --------
    >>> from tuiml.algorithms.survival import NelsonAalenEstimator
    >>> import numpy as np
    >>> na = NelsonAalenEstimator().fit([2, 3, 4, 5], [1, 1, 0, 1])
    >>> np.round(na.cumulative_hazard_, 4).tolist()
    [0.25, 0.5833, 1.5833]
    >>> np.round(na.predict_cumulative_hazard([1, 2, 3, 5]), 4).tolist()
    [0.0, 0.25, 0.5833, 1.5833]
    """

    def __init__(self):
        """Initialize the Nelson-Aalen estimator."""
        super().__init__()
        self.timeline_ = None
        self.cumulative_hazard_ = None
        self.survival_ = None
        self.n_risk_ = None
        self.n_events_ = None
        self.total_cumulative_hazard_ = 0.0

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["survival_curve", "non_parametric", "censored", "numeric"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Fit: O(n log n); predict: O(log n) per query via binary search"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Nelson, W., 1969. Hazard plotting for incomplete failure data. "
            "Journal of Quality Technology, 1(1), 27-52.",
            "Aalen, O., 1978. Nonparametric inference for a family of counting "
            "processes. The Annals of Statistics, 6(4), 701-726."
        ]

    def fit(self, time, event) -> "NelsonAalenEstimator":
        """Fit the cumulative-hazard estimator.

        Parameters
        ----------
        time : array-like of shape (n_samples,)
            Observed time (event or censoring).
        event : array-like of shape (n_samples,)
            Event indicator (1 = event observed, 0 = right-censored).

        Returns
        -------
        self : NelsonAalenEstimator
            Fitted estimator.
        """
        time = np.asarray(time, dtype=np.float64).ravel()
        event = np.asarray(event, dtype=np.float64).ravel()
        if len(time) != len(event):
            raise ValueError("time and event must have the same length")
        if np.any(time < 0):
            raise ValueError("time must be non-negative")
        if not np.all((event == 0) | (event == 1)):
            raise ValueError("event must contain only 0 and 1")

        (
            self.timeline_,
            _,
            self.cumulative_hazard_,
            self.n_risk_,
            self.n_events_,
        ) = _kaplan_meier(time, event)
        self.survival_ = np.exp(-self.cumulative_hazard_)
        self.total_cumulative_hazard_ = (
            float(self.cumulative_hazard_[-1]) if len(self.cumulative_hazard_) else 0.0
        )
        self._is_fitted = True
        return self

    def predict_survival_function(self, times):
        """Return the estimated survival probability :math:`e^{-\\hat{H}(t)}`.

        Parameters
        ----------
        times : float or array-like
            Time point(s) at which to evaluate :math:`\\hat{S}(t)`.

        Returns
        -------
        S : float or np.ndarray
            Exponential survival estimate at each query time.
        """
        self._check_is_fitted()
        H = _step_function_eval(
            self.timeline_, self.cumulative_hazard_, times, initial=0.0
        )
        return np.exp(-H)

    def predict_cumulative_hazard(self, times):
        """Return the Nelson-Aalen cumulative hazard at ``times``.

        Parameters
        ----------
        times : float or array-like
            Time point(s) at which to evaluate :math:`\\hat{H}(t)`.

        Returns
        -------
        H : float or np.ndarray
            Cumulative hazard estimate at each query time.
        """
        self._check_is_fitted()
        return _step_function_eval(
            self.timeline_, self.cumulative_hazard_, times, initial=0.0
        )

    def predict_risk(self, X) -> np.ndarray:
        """Return a constant risk score (a non_parametric baseline).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates. Only the number of rows is used.

        Returns
        -------
        risk : np.ndarray of shape (n_samples,)
            Constant baseline risk.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        n = X.shape[0] if X.ndim == 2 else len(X)
        return np.full(n, self.total_cumulative_hazard_, dtype=np.float64)
