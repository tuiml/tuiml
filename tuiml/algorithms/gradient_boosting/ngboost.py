"""NGBoost - Natural Gradient Boosting for probabilistic prediction.

A native, pure-NumPy implementation of Duan et al. (ICML 2020). Unlike the
other members of :mod:`tuiml.algorithms.gradient_boosting`, nothing here wraps
an external boosting library: the base learners are TuiML's own
:class:`~tuiml.algorithms.trees.DecisionTreeRegressor`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor
from tuiml.algorithms.trees.decision_tree import DecisionTreeRegressor

__all__ = ["NGBoostRegressor", "NGBoostClassifier"]


_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)

#: How far the log-scale parameter may drift from its marginal value. Wider
#: than any sane fit needs, tight enough that ``exp`` never overflows.
_LOG_SCALE_SPAN = 15.0

#: Probabilities are clipped away from 0/1 before any division or logarithm.
_PROBA_EPS = 1e-12

_ERF = np.frompyfunc(math.erf, 1, 1)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, evaluated with the exact libm ``erf``.

    Parameters
    ----------
    z : np.ndarray
        Points at which to evaluate :math:`\\Phi`.

    Returns
    -------
    cdf : np.ndarray
        :math:`\\Phi(z) = \\frac{1}{2}(1 + \\mathrm{erf}(z / \\sqrt{2}))`.
    """
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 + np.asarray(_ERF(z / _SQRT2), dtype=np.float64))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal density.

    Parameters
    ----------
    z : np.ndarray
        Points at which to evaluate :math:`\\varphi`.

    Returns
    -------
    pdf : np.ndarray
        :math:`\\varphi(z) = e^{-z^2/2} / \\sqrt{2\\pi}`.
    """
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-0.5 * z * z) / _SQRT_2PI


def _norm_ppf(p: float) -> float:
    """Standard normal quantile function (scalar).

    Uses Acklam's rational approximation followed by one Halley refinement,
    which brings the result to roughly machine precision.

    Parameters
    ----------
    p : float
        Probability in the open interval ``(0, 1)``.

    Returns
    -------
    z : float
        The value :math:`z` with :math:`\\Phi(z) = p`.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie strictly between 0 and 1")

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # One Halley step against the exact CDF.
    e = float(_norm_cdf(np.array(x))) - p
    u = e * _SQRT_2PI * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


def _safe_std(y: np.ndarray) -> float:
    """Return a strictly positive standard deviation, computed about the mean.

    Centring first avoids the catastrophic cancellation of the
    :math:`E[x^2] - \\bar{x}^2` shortcut on targets with a large offset.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)
        Values whose spread is wanted.

    Returns
    -------
    sigma : float
        Sample standard deviation, floored away from zero.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    centred = y - y.mean()
    sigma = float(np.sqrt(np.mean(centred * centred)))
    if not np.isfinite(sigma) or sigma <= 0.0:
        return 1.0
    return sigma


# ---------------------------------------------------------------------------
# Distributions
#
# Each distribution owns its parameterisation, its proper scoring rule, the
# gradient of that rule, the Riemannian metric induced by it, and the natural
# gradient (metric-inverse times gradient). The boosting loop below knows
# nothing else about them.
# ---------------------------------------------------------------------------

class _NormalDist:
    """Normal distribution parameterised by :math:`(\\mu, \\log \\sigma)`."""

    n_params = 2
    param_names = ("loc", "log_scale")

    def __init__(self, scoring: str = "log", log_scale_min: float = -30.0,
                 log_scale_max: float = 30.0):
        """Initialize the Normal distribution helper.

        Parameters
        ----------
        scoring : str, default="log"
            Proper scoring rule, ``"log"`` or ``"crps"``.
        log_scale_min : float, default=-30.0
            Lower clamp for :math:`\\log \\sigma`.
        log_scale_max : float, default=30.0
            Upper clamp for :math:`\\log \\sigma`.
        """
        self.scoring = scoring
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

    # -- target handling ---------------------------------------------------
    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Return the target on the scale the parameters describe."""
        return np.asarray(y, dtype=np.float64).ravel()

    def validate_y(self, y: np.ndarray) -> None:
        """Raise if the target is outside the distribution's support."""
        return None

    # -- initialisation ----------------------------------------------------
    def fit_marginal(self, y: np.ndarray) -> np.ndarray:
        """Return the marginal (intercept-only) parameter vector.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Targets already put through :meth:`transform_y`.

        Returns
        -------
        params : np.ndarray of shape (2,)
            ``[mean(y), log(std(y))]``.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        return np.array([float(y.mean()), math.log(_safe_std(y))], dtype=np.float64)

    def set_bounds(self, params0: np.ndarray) -> None:
        """Centre the log-scale clamp on the marginal fit.

        Parameters
        ----------
        params0 : np.ndarray of shape (2,)
            Marginal parameter vector from :meth:`fit_marginal`.
        """
        self.log_scale_min = float(params0[1]) - _LOG_SCALE_SPAN
        self.log_scale_max = float(params0[1]) + _LOG_SCALE_SPAN

    def clamp(self, params: np.ndarray) -> np.ndarray:
        """Clip the log-scale into its admissible range."""
        params = np.array(params, dtype=np.float64, copy=True)
        params[:, 1] = np.clip(params[:, 1], self.log_scale_min, self.log_scale_max)
        return params

    # -- scoring rule ------------------------------------------------------
    def _mu_sigma(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split a parameter matrix into ``(mu, sigma)``."""
        mu = params[:, 0]
        sigma = np.exp(np.clip(params[:, 1], self.log_scale_min, self.log_scale_max))
        return mu, sigma

    def score(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-sample value of the scoring rule.

        Parameters
        ----------
        params : np.ndarray of shape (n_samples, 2)
            Distribution parameters.
        y : np.ndarray of shape (n_samples,)
            Transformed targets.

        Returns
        -------
        score : np.ndarray of shape (n_samples,)
            Negative log-likelihood, or CRPS when ``scoring="crps"``.
        """
        mu, sigma = self._mu_sigma(params)
        z = (y - mu) / sigma
        if self.scoring == "crps":
            return sigma * (z * (2.0 * _norm_cdf(z) - 1.0)
                            + 2.0 * _norm_pdf(z) - _INV_SQRT_PI)
        return 0.5 * z * z + np.log(sigma) + 0.5 * math.log(2.0 * math.pi)

    def grad(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ordinary gradient of the scoring rule w.r.t. :math:`(\\mu, \\log\\sigma)`.

        Parameters
        ----------
        params : np.ndarray of shape (n_samples, 2)
            Distribution parameters.
        y : np.ndarray of shape (n_samples,)
            Transformed targets.

        Returns
        -------
        grad : np.ndarray of shape (n_samples, 2)
            Column 0 is :math:`\\partial S / \\partial \\mu`, column 1 is
            :math:`\\partial S / \\partial \\log \\sigma`.
        """
        mu, sigma = self._mu_sigma(params)
        z = (y - mu) / sigma
        out = np.empty((len(y), 2), dtype=np.float64)
        if self.scoring == "crps":
            out[:, 0] = -(2.0 * _norm_cdf(z) - 1.0)
            out[:, 1] = sigma * (2.0 * _norm_pdf(z) - _INV_SQRT_PI)
        else:
            out[:, 0] = -z / sigma
            out[:, 1] = 1.0 - z * z
        return out

    def metric(self, params: np.ndarray) -> np.ndarray:
        """Riemannian metric induced by the scoring rule.

        For the log score this is the Fisher information

        .. math::
            I(\\theta) = \\begin{pmatrix} \\sigma^{-2} & 0 \\\\
                                          0 & 2 \\end{pmatrix}

        and for CRPS it is
        :math:`\\frac{1}{2\\sqrt{\\pi}}\\,\\mathrm{diag}(\\sigma^{-1}, \\sigma/2)`.

        Parameters
        ----------
        params : np.ndarray of shape (n_samples, 2)
            Distribution parameters.

        Returns
        -------
        metric : np.ndarray of shape (n_samples, 2, 2)
            One positive-definite matrix per sample.
        """
        _, sigma = self._mu_sigma(params)
        n = params.shape[0]
        out = np.zeros((n, 2, 2), dtype=np.float64)
        if self.scoring == "crps":
            out[:, 0, 0] = 1.0 / (2.0 * math.sqrt(math.pi) * sigma)
            out[:, 1, 1] = sigma / (4.0 * math.sqrt(math.pi))
        else:
            out[:, 0, 0] = 1.0 / (sigma * sigma)
            out[:, 1, 1] = 2.0
        return out

    def natural_grad(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Natural gradient :math:`I(\\theta)^{-1} \\nabla S`.

        Both metrics are diagonal here, so the inverse is applied in closed
        form rather than through a linear solve.

        Parameters
        ----------
        params : np.ndarray of shape (n_samples, 2)
            Distribution parameters.
        y : np.ndarray of shape (n_samples,)
            Transformed targets.

        Returns
        -------
        nat_grad : np.ndarray of shape (n_samples, 2)
            Metric-preconditioned gradient.
        """
        _, sigma = self._mu_sigma(params)
        g = self.grad(params, y)
        out = np.empty_like(g)
        if self.scoring == "crps":
            k = 2.0 * math.sqrt(math.pi)
            out[:, 0] = g[:, 0] * k * sigma
            out[:, 1] = g[:, 1] * 4.0 * math.sqrt(math.pi) / sigma
        else:
            out[:, 0] = g[:, 0] * sigma * sigma
            out[:, 1] = g[:, 1] * 0.5
        return out

    # -- summaries ---------------------------------------------------------
    def params_dict(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Return the fitted parameters as a name-keyed dictionary."""
        mu, sigma = self._mu_sigma(params)
        return {"loc": mu, "scale": sigma}

    def mean(self, params: np.ndarray) -> np.ndarray:
        """Return the distribution mean for each sample."""
        return params[:, 0].copy()

    def interval(self, params: np.ndarray, alpha: float) -> np.ndarray:
        """Return the equal-tailed :math:`1 - \\alpha` interval."""
        mu, sigma = self._mu_sigma(params)
        z = _norm_ppf(1.0 - alpha / 2.0)
        return np.column_stack([mu - z * sigma, mu + z * sigma])


class _LogNormalDist(_NormalDist):
    """Log-normal distribution: :math:`\\log y \\sim \\mathcal{N}(\\mu, \\sigma^2)`."""

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Return ``log(y)``, the scale on which the parameters live."""
        return np.log(np.asarray(y, dtype=np.float64).ravel())

    def validate_y(self, y: np.ndarray) -> None:
        """Raise if any target is non-positive."""
        y = np.asarray(y, dtype=np.float64)
        if np.any(y <= 0.0):
            raise ValueError("dist='lognormal' requires strictly positive targets")

    def params_dict(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Return the underlying normal's parameters, on the log scale."""
        mu, sigma = self._mu_sigma(params)
        return {"loc": mu, "scale": sigma}

    def mean(self, params: np.ndarray) -> np.ndarray:
        """Return :math:`\\exp(\\mu + \\sigma^2 / 2)`."""
        mu, sigma = self._mu_sigma(params)
        return np.exp(mu + 0.5 * sigma * sigma)

    def interval(self, params: np.ndarray, alpha: float) -> np.ndarray:
        """Return the interval on the original (positive) scale."""
        return np.exp(super().interval(params, alpha))


class _ExponentialDist:
    """Exponential distribution parameterised by :math:`\\log \\beta`."""

    n_params = 1
    param_names = ("log_scale",)

    def __init__(self, scoring: str = "log", log_scale_min: float = -30.0,
                 log_scale_max: float = 30.0):
        """Initialize the Exponential distribution helper.

        Parameters
        ----------
        scoring : str, default="log"
            Only ``"log"`` is supported.
        log_scale_min : float, default=-30.0
            Lower clamp for :math:`\\log \\beta`.
        log_scale_max : float, default=30.0
            Upper clamp for :math:`\\log \\beta`.
        """
        if scoring != "log":
            raise ValueError("dist='exponential' supports scoring='log' only")
        self.scoring = scoring
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Return the target unchanged."""
        return np.asarray(y, dtype=np.float64).ravel()

    def validate_y(self, y: np.ndarray) -> None:
        """Raise if any target is negative."""
        y = np.asarray(y, dtype=np.float64)
        if np.any(y < 0.0):
            raise ValueError("dist='exponential' requires non-negative targets")

    def fit_marginal(self, y: np.ndarray) -> np.ndarray:
        """Return ``[log(mean(y))]``, the marginal MLE.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Targets already put through :meth:`transform_y`.

        Returns
        -------
        params : np.ndarray of shape (1,)
            The marginal log-scale.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        m = float(y.mean())
        if not np.isfinite(m) or m <= 0.0:
            m = 1.0
        return np.array([math.log(m)], dtype=np.float64)

    def set_bounds(self, params0: np.ndarray) -> None:
        """Centre the log-scale clamp on the marginal fit."""
        self.log_scale_min = float(params0[0]) - _LOG_SCALE_SPAN
        self.log_scale_max = float(params0[0]) + _LOG_SCALE_SPAN

    def clamp(self, params: np.ndarray) -> np.ndarray:
        """Clip the log-scale into its admissible range."""
        params = np.array(params, dtype=np.float64, copy=True)
        params[:, 0] = np.clip(params[:, 0], self.log_scale_min, self.log_scale_max)
        return params

    def _beta(self, params: np.ndarray) -> np.ndarray:
        """Return the scale :math:`\\beta = e^{\\log \\beta}`."""
        return np.exp(np.clip(params[:, 0], self.log_scale_min, self.log_scale_max))

    def score(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the negative log-likelihood :math:`\\log \\beta + y / \\beta`."""
        beta = self._beta(params)
        return np.log(beta) + y / beta

    def grad(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return :math:`\\partial S / \\partial \\log \\beta = 1 - y / \\beta`."""
        beta = self._beta(params)
        return (1.0 - y / beta).reshape(-1, 1)

    def metric(self, params: np.ndarray) -> np.ndarray:
        """Return the Fisher information, which is the constant ``1``."""
        return np.ones((params.shape[0], 1, 1), dtype=np.float64)

    def natural_grad(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the natural gradient, equal to the ordinary one here."""
        return self.grad(params, y)

    def params_dict(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Return the fitted scale."""
        return {"scale": self._beta(params)}

    def mean(self, params: np.ndarray) -> np.ndarray:
        """Return the distribution mean, :math:`\\beta`."""
        return self._beta(params)

    def interval(self, params: np.ndarray, alpha: float) -> np.ndarray:
        """Return the equal-tailed :math:`1 - \\alpha` interval."""
        beta = self._beta(params)
        lo = -beta * math.log(1.0 - alpha / 2.0)
        hi = -beta * math.log(alpha / 2.0)
        return np.column_stack([lo, hi])


class _CategoricalDist:
    """Categorical distribution over ``K`` classes, parameterised by ``K-1`` logits.

    Class ``0`` is the reference category and carries an implicit logit of
    zero, which makes the Fisher information non-singular.
    """

    param_names = ("logits",)

    def __init__(self, n_classes: int, scoring: str = "log", logit_clip: float = 25.0):
        """Initialize the categorical helper.

        Parameters
        ----------
        n_classes : int
            Number of classes :math:`K \\geq 2`.
        scoring : str, default="log"
            Only ``"log"`` is supported.
        logit_clip : float, default=25.0
            Absolute clamp on each logit.
        """
        if scoring != "log":
            raise ValueError("NGBoostClassifier supports scoring='log' only")
        self.n_classes = int(n_classes)
        self.n_params = self.n_classes - 1
        self.scoring = scoring
        self.logit_clip = logit_clip

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Return the class indices unchanged."""
        return np.asarray(y, dtype=np.int64).ravel()

    def validate_y(self, y: np.ndarray) -> None:
        """No extra support constraints beyond valid class indices."""
        return None

    def fit_marginal(self, y: np.ndarray) -> np.ndarray:
        """Return the marginal log-odds against the reference class."""
        y = self.transform_y(y)
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64) + 1.0
        p = counts / counts.sum()
        return np.log(p[1:] / p[0])

    def set_bounds(self, params0: np.ndarray) -> None:
        """No data-dependent bounds: logits are clamped absolutely."""
        return None

    def clamp(self, params: np.ndarray) -> np.ndarray:
        """Clip every logit into ``[-logit_clip, logit_clip]``."""
        return np.clip(np.asarray(params, dtype=np.float64), -self.logit_clip,
                       self.logit_clip)

    def proba(self, params: np.ndarray) -> np.ndarray:
        """Return the class probabilities implied by the logits.

        Parameters
        ----------
        params : np.ndarray of shape (n_samples, n_classes - 1)
            Logits relative to the reference class.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Row-stochastic probability matrix.
        """
        params = np.clip(params, -self.logit_clip, self.logit_clip)
        full = np.hstack([np.zeros((params.shape[0], 1)), params])
        full = full - full.max(axis=1, keepdims=True)
        e = np.exp(full)
        p = e / e.sum(axis=1, keepdims=True)
        return np.clip(p, _PROBA_EPS, 1.0)

    def score(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the per-sample negative log-likelihood."""
        p = self.proba(params)
        return -np.log(p[np.arange(len(y)), y])

    def grad(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return :math:`p_j - \\mathbb{1}\\{y = j\\}` for ``j = 1..K-1``."""
        p = self.proba(params)
        onehot = np.zeros_like(p)
        onehot[np.arange(len(y)), y] = 1.0
        return (p - onehot)[:, 1:]

    def metric(self, params: np.ndarray) -> np.ndarray:
        """Return :math:`\\mathrm{diag}(p) - p p^T` over the non-reference classes."""
        p = self.proba(params)[:, 1:]
        m = -p[:, :, None] * p[:, None, :]
        idx = np.arange(self.n_params)
        m[:, idx, idx] += p
        return m

    def natural_grad(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the natural gradient in closed form.

        The inverse of :math:`\\mathrm{diag}(p) - p p^T` on the reduced
        parameterisation is :math:`\\mathrm{diag}(1/p_j) + p_0^{-1} \\mathbf{1}
        \\mathbf{1}^T`, so no linear solve is needed.

        Parameters
        ----------
        params : np.ndarray of shape (n_samples, n_classes - 1)
            Logits.
        y : np.ndarray of shape (n_samples,)
            Class indices.

        Returns
        -------
        nat_grad : np.ndarray of shape (n_samples, n_classes - 1)
            Metric-preconditioned gradient.
        """
        p = self.proba(params)
        g = self.grad(params, y)
        p_rest = p[:, 1:]
        p0 = p[:, 0:1]
        return g / p_rest + g.sum(axis=1, keepdims=True) / p0

    def params_dict(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Return the class probabilities."""
        return {"proba": self.proba(params)}


def _make_distribution(dist: str, scoring: str) -> Any:
    """Build the distribution helper named by ``dist``.

    Parameters
    ----------
    dist : str
        One of ``"normal"``, ``"lognormal"``, ``"exponential"``.
    scoring : str
        ``"log"`` or ``"crps"``.

    Returns
    -------
    distribution : object
        The distribution helper instance.
    """
    if scoring not in ("log", "crps"):
        raise ValueError(f"Unknown scoring rule '{scoring}'; use 'log' or 'crps'")
    if dist == "normal":
        return _NormalDist(scoring=scoring)
    if dist == "lognormal":
        return _LogNormalDist(scoring=scoring)
    if dist == "exponential":
        return _ExponentialDist(scoring=scoring)
    raise ValueError(
        f"Unknown dist '{dist}'; use 'normal', 'lognormal' or 'exponential'"
    )


# ---------------------------------------------------------------------------
# Shared boosting machinery
# ---------------------------------------------------------------------------

#: Geometric ladder of candidate stage scalings for the line search.
_LINE_SEARCH_SCALES = 2.0 ** np.arange(1.0, -21.0, -1.0)


class _NGBoostBase:
    """Boosting loop shared by the NGBoost regressor and classifier."""

    def _tree(self, seed: int) -> DecisionTreeRegressor:
        """Return a fresh base learner configured from the estimator's params."""
        return DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=seed,
        )

    def _line_search(self, dist, params, direction, y) -> float:
        """Return the stage scaling that minimises the mean score.

        The candidate scalings form a geometric ladder; the best finite,
        strictly improving one wins. Zero is returned when nothing improves,
        which stops boosting.

        Parameters
        ----------
        dist : object
            Distribution helper.
        params : np.ndarray of shape (n_samples, n_params)
            Current parameters.
        direction : np.ndarray of shape (n_samples, n_params)
            Fitted natural-gradient direction (descent is ``-direction``).
        y : np.ndarray of shape (n_samples,)
            Transformed targets.

        Returns
        -------
        scale : float
            Non-negative stage scaling.
        """
        base = float(np.mean(dist.score(params, y)))
        if not np.isfinite(base):
            return 0.0
        best_scale, best_loss = 0.0, base
        for scale in _LINE_SEARCH_SCALES:
            candidate = dist.clamp(params - scale * direction)
            s = dist.score(candidate, y)
            if not np.all(np.isfinite(s)):
                continue
            loss = float(np.mean(s))
            if np.isfinite(loss) and loss < best_loss:
                best_scale, best_loss = float(scale), loss
        return best_scale

    def _boost(self, X, y_t, dist, rng):
        """Run the boosting loop and record the fitted stages.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix.
        y_t : np.ndarray of shape (n_samples,)
            Targets on the distribution's own scale.
        dist : object
            Distribution helper.
        rng : np.random.Generator
            Source of randomness for minibatch sampling and tree seeds.

        Returns
        -------
        None
        """
        n = X.shape[0]
        n_params = dist.n_params

        init = dist.fit_marginal(y_t)
        dist.set_bounds(init)
        params = dist.clamp(np.tile(init, (n, 1)))

        self.init_params_ = init
        self.estimators_ = []
        self.scalings_ = []
        self.train_score_ = [float(np.mean(dist.score(params, y_t)))]

        n_sub = max(1, int(round(self.minibatch_frac * n)))
        for stage in range(self.n_estimators):
            nat_grad = (dist.natural_grad(params, y_t) if self.natural_gradient
                        else dist.grad(params, y_t))
            if not np.all(np.isfinite(nat_grad)):
                break

            if n_sub < n:
                idx = rng.choice(n, size=n_sub, replace=False)
            else:
                idx = np.arange(n)

            trees = []
            direction = np.empty((n, n_params), dtype=np.float64)
            for k in range(n_params):
                seed = int(rng.integers(0, 2 ** 31 - 1))
                tree = self._tree(seed).fit(X[idx], nat_grad[idx, k])
                trees.append(tree)
                direction[:, k] = tree.predict(X)

            if not np.all(np.isfinite(direction)):
                break

            scale = self._line_search(dist, params[idx], direction[idx], y_t[idx])
            if scale <= 0.0:
                break

            step = self.learning_rate * scale
            params = dist.clamp(params - step * direction)
            self.estimators_.append(trees)
            self.scalings_.append(step)

            new_score = float(np.mean(dist.score(params, y_t)))
            improvement = self.train_score_[-1] - new_score
            self.train_score_.append(new_score)
            if self.verbose:
                print(f"[NGBoost] stage {stage:4d}  score={new_score:.6f}")
            if improvement < self.tol:
                break

        self.n_estimators_ = len(self.estimators_)

    def _raw_params(self, X: np.ndarray) -> np.ndarray:
        """Replay the fitted stages to obtain per-sample distribution parameters.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix.

        Returns
        -------
        params : np.ndarray of shape (n_samples, n_params)
            Predicted parameters, clamped exactly as during ``fit``.
        """
        X = self._check_X(X)
        dist = self.dist_
        n = X.shape[0]
        params = dist.clamp(np.tile(self.init_params_, (n, 1)))
        for trees, step in zip(self.estimators_, self.scalings_):
            direction = np.column_stack([t.predict(X) for t in trees])
            params = dist.clamp(params - step * direction)
        return params

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        """Coerce ``X`` to a 2-D float array and check its width."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 1- or 2-dimensional")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self.n_features_in_}"
            )
        return X

    def _validate_common(self) -> None:
        """Validate the hyperparameters shared by both estimators."""
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must lie in (0, 1]")
        if not (0.0 < self.minibatch_frac <= 1.0):
            raise ValueError("minibatch_frac must lie in (0, 1]")
        if self.tol < 0.0:
            raise ValueError("tol must be non-negative")


@regressor(tags=["ensemble", "boosting", "probabilistic", "uncertainty", "tree"],
           version="1.0.0")
class NGBoostRegressor(_NGBoostBase, Regressor):
    """NGBoost fits a **whole predictive distribution**, not just a mean.

    Ordinary gradient boosting drives one number per sample towards the truth.
    NGBoost drives *all* the parameters of a probability distribution — for the
    Normal, both :math:`\\mu` and :math:`\\log \\sigma` — using the **natural
    gradient** of a proper scoring rule. The natural gradient is the ordinary
    gradient premultiplied by the inverse Riemannian metric of the score, which
    makes each step invariant to how the distribution happens to be
    parameterised. Boosting on the raw gradient instead is badly conditioned:
    the :math:`\\mu` and :math:`\\log \\sigma` directions live on different
    scales, and the fit drifts towards whichever one the parameterisation
    happens to favour.

    Overview
    --------
    1. Initialise every sample with the marginal MLE of the distribution.
    2. At each stage compute the natural gradient
       :math:`I(\\theta)^{-1} \\nabla_\\theta S(\\theta, y)` per sample.
    3. Fit one :class:`~tuiml.algorithms.trees.DecisionTreeRegressor` per
       distribution parameter against that natural gradient.
    4. Line-search the stage scaling that minimises the mean score, shrink it
       by ``learning_rate``, and take the step.

    Theory
    ------
    For a proper scoring rule :math:`S` the induced Riemannian metric is

    .. math::
        I(\\theta) = \\mathbb{E}_{y \\sim P_\\theta}
        \\left[ \\nabla_\\theta S(\\theta, y)\\,
                \\nabla_\\theta S(\\theta, y)^T \\right]

    which for the log score is the Fisher information. With the Normal
    parameterised as :math:`\\theta = (\\mu, \\log \\sigma)` and
    :math:`z = (y - \\mu)/\\sigma`, the negative log-likelihood has gradient

    .. math::
        \\nabla_\\theta S = \\left( -\\frac{z}{\\sigma},\\; 1 - z^2 \\right)

    and Fisher information

    .. math::
        I(\\theta) = \\begin{pmatrix} \\sigma^{-2} & 0 \\\\
                                      0 & 2 \\end{pmatrix}

    so the natural gradient collapses to the strikingly simple, scale-free

    .. math::
        \\tilde{\\nabla}_\\theta S =
        \\left( \\mu - y,\\; \\tfrac{1}{2}(1 - z^2) \\right)

    The CRPS alternative,

    .. math::
        \\mathrm{CRPS}(\\theta, y) = \\sigma \\left[ z(2\\Phi(z) - 1)
        + 2\\varphi(z) - \\pi^{-1/2} \\right]

    is also proper, and its own metric is used when ``scoring="crps"``.

    Parameters
    ----------
    dist : str, default="normal"
        Predictive distribution: ``"normal"``, ``"lognormal"`` (requires
        strictly positive targets) or ``"exponential"`` (requires non-negative
        targets).
    scoring : str, default="log"
        Proper scoring rule. ``"log"`` is the negative log-likelihood;
        ``"crps"`` is the continuous ranked probability score and is available
        for ``dist="normal"`` and ``"lognormal"``.
    n_estimators : int, default=100
        Maximum number of boosting stages.
    learning_rate : float, default=0.1
        Shrinkage applied to each stage on top of the line-searched scaling.
    max_depth : int, default=3
        Maximum depth of each base learner.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node of a base learner.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf of a base learner.
    natural_gradient : bool, default=True
        Boost on the natural gradient. Set to ``False`` to recover ordinary
        (non-invariant) gradient boosting of the same score.
    minibatch_frac : float, default=1.0
        Fraction of rows sampled without replacement per stage.
    tol : float, default=1e-5
        Stop once a stage improves the mean training score by less than this.
    random_state : int, optional
        Seed for the minibatch sampler and the base-learner tie-breaking.
    verbose : bool, default=False
        Print the training score after each stage.

    Attributes
    ----------
    dist_ : object
        The fitted distribution helper.
    init_params_ : np.ndarray of shape (n_params,)
        Marginal parameter vector the boosting started from.
    estimators_ : list of list of DecisionTreeRegressor
        One inner list per stage, one tree per distribution parameter.
    scalings_ : list of float
        Per-stage step size, ``learning_rate`` times the line-searched scaling.
    train_score_ : list of float
        Mean training score after each stage, including the initialisation.
    n_estimators_ : int
        Number of stages actually fitted (may be below ``n_estimators`` when
        the ``tol`` early stop triggers).
    n_features_in_ : int
        Number of features seen during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Fitting: :math:`O(M \\cdot k \\cdot n p \\log n)` for :math:`M` stages and
      :math:`k` distribution parameters, plus a constant-size line search per
      stage.
    - Prediction: :math:`O(M \\cdot k \\cdot d)` per sample for depth
      :math:`d`.

    **When to use NGBoostRegressor:**

    - When a calibrated predictive *interval* matters as much as the point
      estimate — risk pricing, forecasting, anything downstream of a decision
      threshold.
    - When the noise is heteroscedastic, so a single global error bar would
      misstate the uncertainty for most samples.
    - Prefer a plain boosted regressor when only the conditional mean is
      wanted: NGBoost pays for its second head in accuracy and runtime.

    References
    ----------
    .. [Duan2020] Duan, T., Avati, A., Ding, D.Y., Thai, K.K., Basu, S., Ng,
           A.Y., Schuler, A. (2020). **NGBoost: Natural Gradient Boosting for
           Probabilistic Prediction.** *Proceedings of the 37th International
           Conference on Machine Learning (ICML)*, PMLR 119, 2690-2700.
           DOI: `10.48550/arXiv.1910.03225 <https://doi.org/10.48550/arXiv.1910.03225>`_
    .. [Amari1998] Amari, S. (1998). **Natural Gradient Works Efficiently in
           Learning.** *Neural Computation*, 10(2), 251-276.
           DOI: `10.1162/089976698300017746 <https://doi.org/10.1162/089976698300017746>`_
    .. [Gneiting2007] Gneiting, T., Raftery, A.E. (2007). **Strictly Proper
           Scoring Rules, Prediction, and Estimation.** *Journal of the
           American Statistical Association*, 102(477), 359-378.
           DOI: `10.1198/016214506000001437 <https://doi.org/10.1198/016214506000001437>`_

    See Also
    --------
    :class:`~tuiml.algorithms.gradient_boosting.NGBoostClassifier` : Categorical counterpart.
    :class:`~tuiml.algorithms.gradient_boosting.XGBoostRegressor` : Point-estimate boosting.
    :class:`~tuiml.algorithms.trees.DecisionTreeRegressor` : The base learner.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.gradient_boosting import NGBoostRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-3, 3, size=(300, 1))
    >>> # Noise grows with x: a single global error bar cannot describe this.
    >>> y = X[:, 0] ** 2 + rng.normal(0, 0.2 + 0.5 * np.abs(X[:, 0]))
    >>> model = NGBoostRegressor(n_estimators=60, random_state=0).fit(X, y)
    >>> params = model.predict_dist(X)
    >>> sorted(params)
    ['loc', 'scale']
    >>> # The fitted scale tracks the true |x|-driven noise.
    >>> bool(np.corrcoef(params["scale"], np.abs(X[:, 0]))[0, 1] > 0.7)
    True
    >>> lower, upper = model.predict_interval(X, alpha=0.05).T
    >>> bool(np.mean((y >= lower) & (y <= upper)) > 0.85)
    True
    """

    def __init__(
        self,
        dist: str = "normal",
        scoring: str = "log",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        natural_gradient: bool = True,
        minibatch_frac: float = 1.0,
        tol: float = 1e-5,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        """Initialize the NGBoost regressor.

        Parameters
        ----------
        dist : str, default="normal"
            Predictive distribution.
        scoring : str, default="log"
            Proper scoring rule (``"log"`` or ``"crps"``).
        n_estimators : int, default=100
            Maximum number of boosting stages.
        learning_rate : float, default=0.1
            Stage shrinkage.
        max_depth : int, default=3
            Base-learner depth.
        min_samples_split : int, default=2
            Base-learner split threshold.
        min_samples_leaf : int, default=1
            Base-learner leaf threshold.
        natural_gradient : bool, default=True
            Use the natural gradient rather than the ordinary one.
        minibatch_frac : float, default=1.0
            Row subsampling fraction per stage.
        tol : float, default=1e-5
            Minimum per-stage score improvement before stopping.
        random_state : int, optional
            Random seed.
        verbose : bool, default=False
            Print progress.
        """
        super().__init__()
        self.dist = dist
        self.scoring = scoring
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.natural_gradient = natural_gradient
        self.minibatch_frac = minibatch_frac
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        self.dist_ = None
        self.init_params_ = None
        self.estimators_ = None
        self.scalings_ = None
        self.train_score_ = None
        self.n_estimators_ = None
        self.n_features_in_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "dist": {
                "type": "string",
                "default": "normal",
                "enum": ["normal", "lognormal", "exponential"],
                "description": "Predictive distribution family",
            },
            "scoring": {
                "type": "string",
                "default": "log",
                "enum": ["log", "crps"],
                "description": "Proper scoring rule to boost on",
            },
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Maximum number of boosting stages",
            },
            "learning_rate": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Shrinkage applied to each boosting stage",
            },
            "max_depth": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Maximum depth of each base learner",
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "minimum": 2,
                "description": "Minimum samples required to split a node",
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Minimum samples required at a leaf",
            },
            "natural_gradient": {
                "type": "boolean",
                "default": True,
                "description": "Boost on the natural gradient rather than the ordinary one",
            },
            "minibatch_frac": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Fraction of rows sampled per stage",
            },
            "tol": {
                "type": "number",
                "default": 1e-5,
                "minimum": 0.0,
                "description": "Minimum per-stage score improvement before stopping",
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility",
            },
            "verbose": {
                "type": "boolean",
                "default": False,
                "description": "Print the training score after each stage",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric", "numeric_class", "regression", "ensemble", "tree",
            "non_linear", "probabilistic", "uncertainty",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Fit: O(M*k*n*p*log n); predict: O(M*k*d) per sample"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Duan, T. et al., 2020. NGBoost: Natural Gradient Boosting for "
            "Probabilistic Prediction. ICML, PMLR 119, 2690-2700.",
            "Amari, S., 1998. Natural Gradient Works Efficiently in Learning. "
            "Neural Computation, 10(2), 251-276.",
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NGBoostRegressor":
        """Fit the boosted distributional model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Continuous targets.

        Returns
        -------
        self : NGBoostRegressor
            Fitted estimator.
        """
        self._validate_common()
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if X.shape[0] < 2:
            raise ValueError("at least 2 samples are required to fit NGBoost")

        dist = _make_distribution(self.dist, self.scoring)
        dist.validate_y(y)
        self.dist_ = dist
        self.n_features_in_ = X.shape[1]

        rng = np.random.default_rng(self.random_state)
        self._boost(X, dist.transform_y(y), dist, rng)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the conditional mean of the fitted distribution.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted mean for each sample.
        """
        self._check_is_fitted()
        return self.dist_.mean(self._raw_params(X))

    def predict_dist(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return the predicted distribution parameters.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        params : dict of str to np.ndarray
            ``{"loc", "scale"}`` for ``normal`` and ``lognormal`` (the
            log-normal's parameters describe :math:`\\log y`), ``{"scale"}``
            for ``exponential``. Each value has shape ``(n_samples,)``.
        """
        self._check_is_fitted()
        return self.dist_.params_dict(self._raw_params(X))

    def predict_interval(self, X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Return equal-tailed prediction intervals.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.
        alpha : float, default=0.05
            Miscoverage level; ``alpha=0.05`` gives a nominal 95% interval.

        Returns
        -------
        interval : np.ndarray of shape (n_samples, 2)
            Column 0 is the lower bound, column 1 the upper bound.
        """
        self._check_is_fitted()
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie strictly between 0 and 1")
        return self.dist_.interval(self._raw_params(X), alpha)

    def score_samples(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the per-sample value of the fitted scoring rule.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.
        y : np.ndarray of shape (n_samples,)
            True targets.

        Returns
        -------
        score : np.ndarray of shape (n_samples,)
            Lower is better.
        """
        self._check_is_fitted()
        y = np.asarray(y, dtype=np.float64).ravel()
        self.dist_.validate_y(y)
        return self.dist_.score(self._raw_params(X), self.dist_.transform_y(y))

    def __repr__(self) -> str:
        """Return a concise representation of the estimator."""
        return (f"NGBoostRegressor(dist='{self.dist}', scoring='{self.scoring}', "
                f"n_estimators={self.n_estimators}, "
                f"learning_rate={self.learning_rate})")


@classifier(tags=["ensemble", "boosting", "probabilistic", "uncertainty", "tree"],
            version="1.0.0")
class NGBoostClassifier(_NGBoostBase, Classifier):
    """NGBoost for categorical targets: boosting on natural-gradient logits.

    The classification counterpart of
    :class:`~tuiml.algorithms.gradient_boosting.NGBoostRegressor`. The
    predictive distribution is **categorical** over :math:`K` classes,
    parameterised by :math:`K - 1` logits against a reference class, and the
    boosting stages follow the natural gradient of the log score — the
    ordinary gradient premultiplied by the inverse Fisher information of the
    multinomial. Binary problems are the :math:`K = 2` case and reduce to a
    Bernoulli with a single logit.

    Overview
    --------
    1. Initialise every sample with the marginal class log-odds.
    2. Compute the natural gradient of the negative log-likelihood per sample.
    3. Fit one :class:`~tuiml.algorithms.trees.DecisionTreeRegressor` per logit.
    4. Line-search the stage scaling, shrink by ``learning_rate``, step.

    Theory
    ------
    With class ``0`` as reference, :math:`p = \\mathrm{softmax}(0, \\eta)` and

    .. math::
        \\nabla_\\eta S = p_{1:K} - \\mathbb{1}\\{y\\},
        \\qquad
        I(\\eta) = \\mathrm{diag}(p_{1:K}) - p_{1:K} p_{1:K}^T

    The reduced Fisher information is non-singular and its inverse is
    available in closed form,

    .. math::
        I(\\eta)^{-1} = \\mathrm{diag}(p_j^{-1})
        + p_0^{-1} \\mathbf{1} \\mathbf{1}^T

    so the natural gradient needs no linear solve. For :math:`K = 2` this
    collapses to :math:`(p - y) / (p(1 - p))`, the Bernoulli case.

    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of boosting stages.
    learning_rate : float, default=0.1
        Shrinkage applied to each stage on top of the line-searched scaling.
    max_depth : int, default=3
        Maximum depth of each base learner.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node of a base learner.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf of a base learner.
    natural_gradient : bool, default=True
        Boost on the natural gradient rather than the ordinary one.
    minibatch_frac : float, default=1.0
        Fraction of rows sampled without replacement per stage.
    tol : float, default=1e-5
        Stop once a stage improves the mean training score by less than this.
    random_state : int, optional
        Seed for the minibatch sampler and the base-learner tie-breaking.
    verbose : bool, default=False
        Print the training score after each stage.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted class labels seen during ``fit()``.
    dist_ : object
        The fitted categorical distribution helper.
    init_params_ : np.ndarray of shape (n_classes - 1,)
        Marginal log-odds the boosting started from.
    estimators_ : list of list of DecisionTreeRegressor
        One inner list per stage, one tree per logit.
    scalings_ : list of float
        Per-stage step size.
    train_score_ : list of float
        Mean training negative log-likelihood after each stage.
    n_estimators_ : int
        Number of stages actually fitted.
    n_features_in_ : int
        Number of features seen during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Fitting: :math:`O(M (K-1) n p \\log n)` for :math:`M` stages.
    - Prediction: :math:`O(M (K-1) d)` per sample.

    **When to use NGBoostClassifier:**

    - When well-behaved class probabilities matter more than raw accuracy.
    - When the natural-gradient step's parameterisation invariance is wanted
      on a multiclass problem with very unbalanced classes.
    - Prefer a plain boosted classifier when only the argmax is consumed.

    References
    ----------
    .. [Duan2020] Duan, T., Avati, A., Ding, D.Y., Thai, K.K., Basu, S., Ng,
           A.Y., Schuler, A. (2020). **NGBoost: Natural Gradient Boosting for
           Probabilistic Prediction.** *Proceedings of the 37th International
           Conference on Machine Learning (ICML)*, PMLR 119, 2690-2700.
           DOI: `10.48550/arXiv.1910.03225 <https://doi.org/10.48550/arXiv.1910.03225>`_
    .. [Amari1998] Amari, S. (1998). **Natural Gradient Works Efficiently in
           Learning.** *Neural Computation*, 10(2), 251-276.
           DOI: `10.1162/089976698300017746 <https://doi.org/10.1162/089976698300017746>`_

    See Also
    --------
    :class:`~tuiml.algorithms.gradient_boosting.NGBoostRegressor` : Continuous counterpart.
    :class:`~tuiml.algorithms.gradient_boosting.XGBoostClassifier` : Point-estimate boosting.
    :class:`~tuiml.algorithms.trees.DecisionTreeRegressor` : The base learner.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.gradient_boosting import NGBoostClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    >>> model = NGBoostClassifier(n_estimators=40, random_state=0).fit(X, y)
    >>> model.classes_.tolist()
    [0, 1]
    >>> proba = model.predict_proba(X)
    >>> proba.shape
    (200, 2)
    >>> bool(np.allclose(proba.sum(axis=1), 1.0))
    True
    >>> float((model.predict(X) == y).mean()) > 0.9
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        natural_gradient: bool = True,
        minibatch_frac: float = 1.0,
        tol: float = 1e-5,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        """Initialize the NGBoost classifier.

        Parameters
        ----------
        n_estimators : int, default=100
            Maximum number of boosting stages.
        learning_rate : float, default=0.1
            Stage shrinkage.
        max_depth : int, default=3
            Base-learner depth.
        min_samples_split : int, default=2
            Base-learner split threshold.
        min_samples_leaf : int, default=1
            Base-learner leaf threshold.
        natural_gradient : bool, default=True
            Use the natural gradient rather than the ordinary one.
        minibatch_frac : float, default=1.0
            Row subsampling fraction per stage.
        tol : float, default=1e-5
            Minimum per-stage score improvement before stopping.
        random_state : int, optional
            Random seed.
        verbose : bool, default=False
            Print progress.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.natural_gradient = natural_gradient
        self.minibatch_frac = minibatch_frac
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        self.classes_ = None
        self.dist_ = None
        self.init_params_ = None
        self.estimators_ = None
        self.scalings_ = None
        self.train_score_ = None
        self.n_estimators_ = None
        self.n_features_in_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Maximum number of boosting stages",
            },
            "learning_rate": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Shrinkage applied to each boosting stage",
            },
            "max_depth": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Maximum depth of each base learner",
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "minimum": 2,
                "description": "Minimum samples required to split a node",
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Minimum samples required at a leaf",
            },
            "natural_gradient": {
                "type": "boolean",
                "default": True,
                "description": "Boost on the natural gradient rather than the ordinary one",
            },
            "minibatch_frac": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Fraction of rows sampled per stage",
            },
            "tol": {
                "type": "number",
                "default": 1e-5,
                "minimum": 0.0,
                "description": "Minimum per-stage score improvement before stopping",
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility",
            },
            "verbose": {
                "type": "boolean",
                "default": False,
                "description": "Print the training score after each stage",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric", "binary_class", "multiclass", "ensemble", "tree",
            "non_linear", "probabilistic", "uncertainty",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Fit: O(M*(K-1)*n*p*log n); predict: O(M*(K-1)*d) per sample"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Duan, T. et al., 2020. NGBoost: Natural Gradient Boosting for "
            "Probabilistic Prediction. ICML, PMLR 119, 2690-2700.",
            "Amari, S., 1998. Natural Gradient Works Efficiently in Learning. "
            "Neural Computation, 10(2), 251-276.",
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NGBoostClassifier":
        """Fit the boosted categorical model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : NGBoostClassifier
            Fitted estimator.
        """
        self._validate_common()
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("NGBoostClassifier requires at least 2 classes")

        dist = _CategoricalDist(len(self.classes_))
        self.dist_ = dist
        self.n_features_in_ = X.shape[1]

        rng = np.random.default_rng(self.random_state)
        self._boost(X, y_idx.astype(np.int64), dist, rng)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Row-stochastic class probabilities, ordered as ``classes_``.
        """
        self._check_is_fitted()
        p = self.dist_.proba(self._raw_params(X))
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels drawn from ``classes_``.
        """
        self._check_is_fitted()
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_dist(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return the predicted distribution parameters.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        params : dict of str to np.ndarray
            ``{"proba": array of shape (n_samples, n_classes)}`` — the
            parameters of the predictive categorical distribution.
        """
        self._check_is_fitted()
        return {"proba": self.predict_proba(X)}

    def predict_interval(self, X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Return the categorical analogue of a prediction interval.

        A nominal target has no ordering, so an interval is replaced by the
        **highest-probability credible set**: the smallest set of classes whose
        probability mass reaches :math:`1 - \\alpha`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.
        alpha : float, default=0.05
            Miscoverage level.

        Returns
        -------
        mask : np.ndarray of shape (n_samples, n_classes), dtype=bool
            ``mask[i, k]`` is ``True`` when class ``classes_[k]`` belongs to
            the credible set of sample ``i``. Every row contains at least the
            most probable class.
        """
        self._check_is_fitted()
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie strictly between 0 and 1")
        proba = self.predict_proba(X)
        order = np.argsort(-proba, axis=1)
        sorted_p = np.take_along_axis(proba, order, axis=1)
        cumulative = np.cumsum(sorted_p, axis=1)
        # Keep every class up to and including the one that crosses 1 - alpha.
        keep = np.zeros_like(cumulative, dtype=bool)
        keep[:, 0] = True
        keep[:, 1:] = cumulative[:, :-1] < (1.0 - alpha)
        mask = np.zeros_like(keep)
        np.put_along_axis(mask, order, keep, axis=1)
        return mask

    def score_samples(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the per-sample negative log-likelihood.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.
        y : np.ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        score : np.ndarray of shape (n_samples,)
            Lower is better.
        """
        self._check_is_fitted()
        y = np.asarray(y).ravel()
        idx = np.searchsorted(self.classes_, y)
        if np.any(idx >= len(self.classes_)) or np.any(self.classes_[idx] != y):
            raise ValueError("y contains labels unseen during fit")
        proba = self.predict_proba(X)
        return -np.log(np.clip(proba[np.arange(len(y)), idx], _PROBA_EPS, 1.0))

    def __repr__(self) -> str:
        """Return a concise representation of the estimator."""
        return (f"NGBoostClassifier(n_estimators={self.n_estimators}, "
                f"learning_rate={self.learning_rate})")
