"""Evaluation metrics for survival models (hand-rolled, NumPy only)."""

from __future__ import annotations

import numpy as np
from math import erfc, sqrt


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


def concordance_index(risk, time, event):
    """Harrell's concordance index (C-index).

    The C-index is the probability that, for a randomly chosen comparable pair,
    the subject with the shorter observed event time is assigned the higher
    risk. A pair is comparable when the ordering of its event times is known
    despite censoring: the earlier time must be an observed event.

    Parameters
    ----------
    risk : array-like of shape (n_samples,)
        Predicted risk scores (higher = earlier expected event).
    time : array-like of shape (n_samples,)
        Observed times (event or censoring).
    event : array-like of shape (n_samples,)
        Event indicators (1 = event observed, 0 = right-censored).

    Returns
    -------
    c_index : float
        Concordance index in ``[0, 1]``. Returns ``nan`` when no pair is
        comparable.

    Notes
    -----
    Ties in time are comparable only when both are events, in which case a tied
    risk counts as :math:`\\tfrac{1}{2}` and a difference counts as
    :math:`\\tfrac{1}{2}` (the standard Harrell convention).

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.survival.metrics import concordance_index
    >>> time = np.array([1., 2., 3., 4.])
    >>> event = np.ones(4)
    >>> concordance_index([4, 3, 2, 1], time, event)
    1.0
    >>> concordance_index([1, 2, 3, 4], time, event)
    0.0
    """
    risk = np.asarray(risk, dtype=np.float64).ravel()
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.float64).ravel()
    n = len(risk)

    if len(time) != n or len(event) != n:
        raise ValueError("risk, time and event must have the same length")

    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] < time[j]:
                if event[i] == 1:
                    den += 1.0
                    if risk[i] > risk[j]:
                        num += 1.0
                    elif risk[i] == risk[j]:
                        num += 0.5
            elif time[j] < time[i]:
                if event[j] == 1:
                    den += 1.0
                    if risk[j] > risk[i]:
                        num += 1.0
                    elif risk[i] == risk[j]:
                        num += 0.5
            else:  # tied times: comparable only if both are events
                if event[i] == 1 and event[j] == 1:
                    den += 1.0
                    num += 0.5

    if den == 0:
        return float("nan")
    return num / den


def _breslow_baseline(eta, time, event):
    """Breslow baseline cumulative hazard for a fixed linear predictor ``eta``.

    Parameters
    ----------
    eta : np.ndarray of shape (n,)
        Linear predictor values.
    time : np.ndarray of shape (n,)
        Observed times.
    event : np.ndarray of shape (n,)
        Event indicators.

    Returns
    -------
    event_times : np.ndarray
        Sorted unique event times.
    cumulative_hazard : np.ndarray
        Cumulative baseline hazard at each event time.
    """
    eta = np.clip(np.asarray(eta, dtype=np.float64), -50.0, 50.0)
    exp_eta = np.exp(eta)
    event_times = np.unique(time[event == 1])
    h0 = np.zeros(len(event_times), dtype=np.float64)
    for k, tj in enumerate(event_times):
        d = np.sum(event[time == tj])
        h0[k] = d / np.sum(exp_eta[time >= tj])
    return event_times, np.cumsum(h0)


def _censoring_survival(time, event, query):
    """Kaplan-Meier survival of the *censoring* process, evaluated at ``query``.

    Treats censoring (``1 - event``) as the event of interest, giving the
    probability :math:`\\hat{G}(t)` of remaining uncensored past ``t``.
    """
    cens = (1.0 - event).astype(bool)
    ctimes = np.unique(time[cens])
    query = np.asarray(query, dtype=np.float64)
    if len(ctimes) == 0:
        return np.ones(np.atleast_1d(query).shape, dtype=np.float64)

    values = np.empty(len(ctimes), dtype=np.float64)
    product = 1.0
    for k, t in enumerate(ctimes):
        d = np.sum(cens & (time == t))
        r = np.sum(time >= t)
        product *= (1.0 - d / r)
        values[k] = product
    return _step_function_eval(ctimes, values, query, initial=1.0)


def _risk_to_survival(risk, time, event, times):
    """Convert risk scores to survival curves via a Breslow baseline.

    The centered risk is treated as a linear predictor with coefficient one;
    the baseline hazard is estimated by Breslow's method and each subject's
    survival is :math:`\\exp(-H_0(t) e^{\\eta_i})`.

    Parameters
    ----------
    risk : np.ndarray of shape (n,)
        Risk scores.
    time, event : np.ndarray of shape (n,)
        Observed times and event indicators.
    times : np.ndarray of shape (n_times,)
        Query times.

    Returns
    -------
    S : np.ndarray of shape (n, n_times)
        Predicted survival probabilities.
    """
    risk = np.asarray(risk, dtype=np.float64).ravel()
    eta = risk - risk.mean()
    exp_eta = np.exp(np.clip(eta, -50.0, 50.0))
    event_times, H0 = _breslow_baseline(eta, time, event)
    H0_at_t = _step_function_eval(event_times, H0, times, initial=0.0)
    return np.exp(-np.outer(exp_eta, H0_at_t))


def integrated_brier_score(risk_or_model, X, time, event, times):
    """Integrated Brier score with inverse-probability-of-censoring weights.

    The Brier score at time :math:`t` measures the squared error between the
    predicted survival :math:`\\hat{S}_i(t)` and the observed survival status
    :math:`1\\{t_i > t\\}`, reweighted to correct for censoring (Graf et al.,
    1999):

    .. math::
        BS(t) = \\frac{1}{n} \\sum_{i=1}^{n}
        \\hat{w}_i(t) \\left( \\hat{S}_i(t) - 1\\{t_i > t\\} \\right)^2

    where the inverse-probability-of-censoring weight is
    :math:`\\hat{w}_i(t) = 1\\{t_i \\leq t, \\delta_i = 1\\} / \\hat{G}(t_i)
    + 1\\{t_i > t\\} / \\hat{G}(t)`. The integrated Brier score is the mean of
    :math:`BS(t)` over the requested ``times``.

    Parameters
    ----------
    risk_or_model : array-like or object
        Either a fitted survival model exposing
        ``predict_survival_function(X, times)`` (returning an
        ``(n_samples, n_times)`` array), or a 1-D array of risk scores, which
        are converted to survival curves via a Breslow baseline.
    X : np.ndarray of shape (n_samples, n_features)
        Covariates. Used only when ``risk_or_model`` is a model.
    time : array-like of shape (n_samples,)
        Observed times (event or censoring).
    event : array-like of shape (n_samples,)
        Event indicators (1 = event observed, 0 = right-censored).
    times : array-like of shape (n_times,)
        Time grid over which to integrate the Brier score.

    Returns
    -------
    ibs : float
        Integrated Brier score (lower is better).

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.survival.metrics import integrated_brier_score
    >>> rng = np.random.RandomState(0)
    >>> time = rng.uniform(1, 10, size=30)
    >>> event = np.ones(30)
    >>> risk = -time  # perfectly anti-correlated risk: high risk at short time
    >>> ibs = integrated_brier_score(risk, np.ones((30, 1)), time, event, [2., 5., 8.])
    >>> 0.0 <= ibs <= 0.5
    True
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.float64).ravel()
    times = np.asarray(times, dtype=np.float64).ravel()
    n = len(time)

    if hasattr(risk_or_model, "predict_survival_function"):
        S = np.asarray(
            risk_or_model.predict_survival_function(X, times), dtype=np.float64
        )
    else:
        S = _risk_to_survival(risk_or_model, time, event, times)

    if S.ndim != 2 or S.shape != (n, len(times)):
        raise ValueError(
            "predict_survival_function(X, times) must return shape (n_samples, n_times)"
        )

    G_at_time = _censoring_survival(time, event, time)  # G(T_i), length n
    G_at_times = _censoring_survival(time, event, times)  # G(t), length n_times

    bs = np.zeros(len(times), dtype=np.float64)
    for k in range(len(times)):
        event_before = (time <= times[k]) & (event == 1)
        survived = time > times[k]
        w = np.zeros(n, dtype=np.float64)
        w[event_before] = 1.0 / np.maximum(G_at_time[event_before], 1e-12)
        w[survived] = 1.0 / np.maximum(G_at_times[k], 1e-12)
        y_true = survived.astype(np.float64)  # observed survival status I(T > t)
        diff = (y_true - S[:, k]) ** 2
        bs[k] = np.sum(w * diff) / n

    return float(np.mean(bs))


def logrank_test(time_a, event_a, time_b, event_b):
    """Two-sample log-rank test for equality of survival curves.

    Compares the observed number of events in group A against the number
    expected under the null hypothesis that both groups share the same hazard.
    The test statistic is asymptotically chi-square with one degree of freedom.

    Parameters
    ----------
    time_a : array-like of shape (n_a,)
        Observed times for group A.
    event_a : array-like of shape (n_a,)
        Event indicators for group A.
    time_b : array-like of shape (n_b,)
        Observed times for group B.
    event_b : array-like of shape (n_b,)
        Event indicators for group B.

    Returns
    -------
    statistic : float
        The chi-square log-rank statistic.
    p_value : float
        Two-sided p-value from the chi-square distribution with 1 degree of
        freedom.

    Examples
    --------
    >>> from tuiml.algorithms.survival.metrics import logrank_test
    >>> stat, p = logrank_test([1, 2, 3, 4], [1, 1, 1, 1],
    ...                        [5, 6, 7, 8], [1, 1, 1, 1])
    >>> stat > 0 and p < 0.05
    True
    """
    time_a = np.asarray(time_a, dtype=np.float64).ravel()
    event_a = np.asarray(event_a, dtype=np.float64).ravel()
    time_b = np.asarray(time_b, dtype=np.float64).ravel()
    event_b = np.asarray(event_b, dtype=np.float64).ravel()

    time = np.concatenate([time_a, time_b])
    event = np.concatenate([event_a, event_b])
    group = np.concatenate(
        [np.zeros(len(time_a)), np.ones(len(time_b))]
    ).astype(bool)  # False = group A, True = group B

    event_times = np.unique(time[event == 1])
    O = 0.0
    E = 0.0
    V = 0.0
    for tj in event_times:
        at_risk = time >= tj
        d = np.sum(event[time == tj])
        n = np.sum(at_risk)
        d1 = np.sum((~group) & (time == tj) & (event == 1))
        n1 = np.sum((~group) & at_risk)
        n0 = n - n1
        O += d1
        E += d * n1 / n
        if n > 1:
            V += n1 * n0 * d * (n - d) / (n * n * (n - 1.0))

    statistic = (O - E) ** 2 / V if V > 0 else 0.0
    # Survival function of chi-square(1): P(Z^2 > x) = erfc(sqrt(x / 2)).
    p_value = erfc(sqrt(statistic / 2.0))
    return float(statistic), float(p_value)
