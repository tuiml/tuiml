"""
Parametric significance tests for comparing learning algorithms.

A *parametric* test assumes an explicit distributional form for the data. The
tests here assume the quantity being tested (the per-fold or per-dataset score
difference, or the within-group residual) is drawn from a **normal**
distribution. When that assumption is reasonable these tests are the most
powerful option available; when it fails -- heavy tails, a handful of datasets,
accuracies saturating near 1.0, or an outlier dataset -- they become
anti-conservative and report "significant" far too often. In that regime use
the rank-based tests in :mod:`tuiml.evaluation.statistics.nonparametric`
instead.

Contents::

    paired_t_test            Student's paired t-test on matched scores
    corrected_paired_t_test  Paired t-test with the Nadeau & Bengio variance
                             correction for resampled / cross-validated scores
    one_way_anova            Omnibus F-test across k independent groups
    PairedStats              Result container returned by the paired tests
    SignificanceLevel        WIN / LOSS / TIE verdict enum

Notes
-----
**Choosing a test.** Two algorithms scored on the *same* folds or the *same*
datasets are *paired* -- use :func:`paired_t_test`. If those scores come from
resampling that reuses training data (k-fold CV, repeated random splits), the
folds are not independent, the ordinary t-test's variance estimate is too small
and its Type I error rate is badly inflated; use
:func:`corrected_paired_t_test`. For more than two algorithms an omnibus test
comes first: :func:`one_way_anova` for independent groups, or -- much more
commonly in machine learning, where the same datasets are reused across
algorithms -- :func:`~tuiml.evaluation.statistics.friedman_test`.

**Multiplicity.** Running every pairwise t-test over k algorithms performs
:math:`k(k-1)/2` tests, so the probability of at least one false positive grows
quickly. Always feed the resulting p-values through one of the procedures in
:mod:`tuiml.evaluation.statistics.corrections`.

References
----------
.. [Student1908] Student (W. S. Gosset) (1908). "The Probable Error of a Mean".
   Biometrika, 6(1), 1-25.
.. [Dietterich1998] Dietterich, T. G. (1998). "Approximate Statistical Tests for
   Comparing Supervised Classification Learning Algorithms". Neural
   Computation, 10(7), 1895-1923.
.. [Demsar2006] Demsar, J. (2006). "Statistical Comparisons of Classifiers over
   Multiple Data Sets". Journal of Machine Learning Research, 7, 1-30.

See Also
--------
:mod:`tuiml.evaluation.statistics.nonparametric` : Rank-based counterparts that
    make no normality assumption.
:mod:`tuiml.evaluation.statistics.corrections` : Family-wise error rate and
    false discovery rate control for many pairwise tests.

Examples
--------
>>> import numpy as np
>>> from tuiml.evaluation.statistics import paired_t_test
>>> model_a = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
>>> model_b = np.array([0.82, 0.84, 0.81, 0.83, 0.82])
>>> stats = paired_t_test(model_a, model_b, significance_level=0.05)
>>> round(float(stats.p_value), 4)
0.0004
>>> stats.x_better()
True
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

class SignificanceLevel(Enum):
    """Verdict of a two-sided paired comparison between two algorithms.

    A test does not simply return "different"/"not different": once the null
    hypothesis of equal performance is rejected, the *sign* of the mean
    difference says which algorithm won. This enum encodes both facts in one
    value, always from the point of view of the **first** argument (``x``) of
    the test.

    Attributes
    ----------
    TIE : int
        Value ``0``. The null hypothesis was **not** rejected
        (:math:`p \\geq \\alpha`). This is *not* evidence that the two
        algorithms are equal, only that the data are too few or too noisy to
        distinguish them.
    WIN : int
        Value ``1``. The null hypothesis was rejected and ``x`` is the better
        algorithm. "Better" is decided by the ``higher_better`` flag of the
        test: with ``higher_better=True`` (accuracy, F1) this means
        ``mean(x) > mean(y)``; with ``higher_better=False`` (error rate, RMSE)
        it means ``mean(x) < mean(y)``.
    LOSS : int
        Value ``-1``. The null hypothesis was rejected and ``y`` is the better
        algorithm.

    Notes
    -----
    The comparison performed is always **two-sided**, so ``WIN`` and ``LOSS``
    partition the same rejection region; the enum records which tail the
    observed difference fell into. Do not read a ``WIN`` at
    :math:`\\alpha = 0.05` as a one-sided claim at :math:`\\alpha = 0.025`.

    See Also
    --------
    :class:`~tuiml.evaluation.statistics.PairedStats` : Container that carries
        this verdict alongside the statistic and p-value.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.statistics import SignificanceLevel, paired_t_test
    >>> a = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
    >>> b = np.array([0.82, 0.84, 0.81, 0.83, 0.82])
    >>> paired_t_test(a, b).significance is SignificanceLevel.WIN
    True
    >>> paired_t_test(b, a).significance is SignificanceLevel.LOSS
    True
    >>> SignificanceLevel.TIE.value
    0
    """
    TIE = 0
    WIN = 1
    LOSS = -1

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """
        Return JSON Schema for the SignificanceLevel enum.

        Returns
        -------
        dict
            JSON Schema describing the enum values.
        """
        return {
            "type": "object",
            "title": "SignificanceLevel",
            "description": "Significance level indicators for statistical tests.",
            "properties": {
                "value": {
                    "type": "integer",
                    "enum": [-1, 0, 1],
                    "enumNames": ["LOSS", "TIE", "WIN"],
                    "description": "Significance indicator: -1 (LOSS), 0 (TIE), or 1 (WIN)."
                }
            },
            "required": ["value"]
        }

@dataclass
class PairedStats:
    """Complete result of a paired comparison between two algorithms.

    Returned by every paired test in this package -- :func:`paired_t_test`,
    :func:`corrected_paired_t_test` and
    :func:`~tuiml.evaluation.statistics.wilcoxon_signed_rank_test` -- so that
    the three are drop-in interchangeable. It bundles the descriptive
    statistics of both samples, the test statistic, the two-sided p-value and
    the WIN/LOSS/TIE verdict.

    Attributes
    ----------
    x_mean : float
        Sample mean of the first algorithm's scores.
    y_mean : float
        Sample mean of the second algorithm's scores.
    x_std : float
        Sample standard deviation of the first algorithm's scores, computed
        with ``ddof=1`` (unbiased, divides by :math:`n - 1`).
    y_std : float
        Sample standard deviation of the second algorithm's scores
        (``ddof=1``).
    diff_mean : float
        Mean of the paired differences :math:`d_i = x_i - y_i`. This is the
        effect size in the original units of the score; its **sign** decides
        WIN vs LOSS. Note it is ``x_mean - y_mean``, not the other way round.
    diff_std : float
        Spread of the paired differences. For :func:`paired_t_test` and
        :func:`~tuiml.evaluation.statistics.wilcoxon_signed_rank_test` this is
        the ordinary ``ddof=1`` standard deviation of :math:`d_i`. For
        :func:`corrected_paired_t_test` it is instead the **corrected standard
        error** :math:`\\sqrt{(1/n + n_{test}/n_{train})\\, s_d^2}`, which is
        why the two are not comparable across tests.
    t_statistic : float
        The test statistic. For the two t-tests this is Student's
        :math:`t`, distributed with :math:`n - 1` degrees of freedom under the
        null. For the Wilcoxon test the field instead carries the **normal
        approximation z-score** of the signed-rank statistic (the name is kept
        only so the container stays uniform).
    p_value : float
        **Two-sided** p-value: the probability, *assuming the null hypothesis
        of no difference is true*, of observing a statistic at least as extreme
        as this one in either direction. It is not the probability that the
        null is true, and ``1 - p_value`` is not the probability that the
        difference is real. Lies in :math:`[0, 1]`.
    correlation : float
        Pearson correlation between the two score vectors, in
        :math:`[-1, 1]`. High positive correlation (the usual case when both
        algorithms are scored on the same folds) is precisely what makes the
        *paired* design more powerful than an unpaired one, because the
        dataset-to-dataset variance cancels in :math:`d_i`. Reported as ``0.0``
        when either sample is constant.
    significance : SignificanceLevel
        Verdict at the ``significance_level`` supplied to the test:
        ``WIN`` (``x`` better), ``LOSS`` (``y`` better) or ``TIE`` (null not
        rejected). See :class:`SignificanceLevel`.
    n : int
        Number of paired observations actually used, **after** dropping pairs
        containing NaN. For the Wilcoxon test, pairs with a zero difference are
        dropped as well, so ``n`` can be smaller than the input length.

    Notes
    -----
    The verdict is computed with a strict comparison, ``p_value <
    significance_level``; a p-value exactly equal to :math:`\\alpha` is
    reported as ``TIE``.

    Because the whole object is derived from one ``significance_level``,
    re-reading ``significance`` under a different :math:`\\alpha` is invalid --
    re-run the test, or threshold ``p_value`` yourself.

    See Also
    --------
    :func:`~tuiml.evaluation.statistics.paired_t_test` : Parametric producer of
        this result.
    :func:`~tuiml.evaluation.statistics.wilcoxon_signed_rank_test` :
        Non-parametric producer of this result.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.statistics import paired_t_test
    >>> a = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
    >>> b = np.array([0.82, 0.84, 0.81, 0.83, 0.82])
    >>> stats = paired_t_test(a, b)
    >>> stats.n
    5
    >>> round(float(stats.diff_mean), 4)
    0.026
    >>> round(float(stats.t_statistic), 4)
    10.6145
    >>> round(float(stats.p_value), 4)
    0.0004
    >>> stats.is_significant(), stats.x_better(), stats.y_better()
    (True, True, False)
    """
    x_mean: float
    y_mean: float
    x_std: float
    y_std: float
    diff_mean: float
    diff_std: float
    t_statistic: float
    p_value: float
    correlation: float
    significance: SignificanceLevel
    n: int

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """
        Return JSON Schema for PairedStats dataclass fields.

        Returns
        -------
        dict
            JSON Schema describing all dataclass fields.
        """
        return {
            "type": "object",
            "title": "PairedStats",
            "description": "Statistics for paired comparison between two sets of results.",
            "properties": {
                "x_mean": {
                    "type": "number",
                    "description": "Mean of first sample."
                },
                "y_mean": {
                    "type": "number",
                    "description": "Mean of second sample."
                },
                "x_std": {
                    "type": "number",
                    "description": "Standard deviation of first sample."
                },
                "y_std": {
                    "type": "number",
                    "description": "Standard deviation of second sample."
                },
                "diff_mean": {
                    "type": "number",
                    "description": "Mean of differences between samples."
                },
                "diff_std": {
                    "type": "number",
                    "description": "Standard deviation of differences."
                },
                "t_statistic": {
                    "type": "number",
                    "description": "T-test statistic value."
                },
                "p_value": {
                    "type": "number",
                    "description": "P-value from the t-test.",
                    "minimum": 0,
                    "maximum": 1
                },
                "correlation": {
                    "type": "number",
                    "description": "Correlation coefficient between samples.",
                    "minimum": -1,
                    "maximum": 1
                },
                "significance": {
                    "type": "integer",
                    "enum": [-1, 0, 1],
                    "enumNames": ["LOSS", "TIE", "WIN"],
                    "description": "Significance indicator: -1 (LOSS), 0 (TIE), or 1 (WIN)."
                },
                "n": {
                    "type": "integer",
                    "description": "Number of paired observations.",
                    "minimum": 1
                }
            },
            "required": [
                "x_mean", "y_mean", "x_std", "y_std",
                "diff_mean", "diff_std", "t_statistic",
                "p_value", "correlation", "significance", "n"
            ]
        }

    def is_significant(self) -> bool:
        """Report whether the null hypothesis of no difference was rejected.

        Returns
        -------
        significant : bool
            ``True`` if ``p_value`` fell below the significance level supplied
            to the test (equivalently, ``significance`` is not ``TIE``).
            ``False`` means the evidence was insufficient -- **not** that the
            two algorithms perform equally.
        """
        return self.significance != SignificanceLevel.TIE

    def x_better(self) -> bool:
        """Report whether the first algorithm won significantly.

        Returns
        -------
        better : bool
            ``True`` only if the difference was significant *and* pointed in
            ``x``'s favour, as judged by the ``higher_better`` flag of the
            test. ``False`` covers both "``y`` won" and "tie".
        """
        return self.significance == SignificanceLevel.WIN

    def y_better(self) -> bool:
        """Report whether the second algorithm won significantly.

        Returns
        -------
        better : bool
            ``True`` only if the difference was significant *and* pointed in
            ``y``'s favour. ``False`` covers both "``x`` won" and "tie".
        """
        return self.significance == SignificanceLevel.LOSS

def _t_distribution_cdf(t: float, df: int) -> float:
    """Evaluate the CDF of Student's t-distribution.

    Uses the standard normal as a limit approximation for ``df > 100`` and the
    regularised incomplete beta identity
    :math:`P(T \\leq t) = 1 - \\tfrac{1}{2} I_{x}(df/2, 1/2)` with
    :math:`x = df / (df + t^2)` otherwise.

    Parameters
    ----------
    t : float
        Value at which to evaluate the CDF.
    df : int
        Degrees of freedom. Values ``<= 0`` return ``0.5``.

    Returns
    -------
    cdf : float
        :math:`P(T \\leq t)`, in :math:`[0, 1]`.
    """
    if df <= 0:
        return 0.5

    # For large df, use normal approximation
    if df > 100:
        from math import erf, sqrt
        return 0.5 * (1 + erf(t / sqrt(2)))

    # For smaller df, use beta function approximation
    x = df / (df + t * t)
    a = df / 2
    b = 0.5

    if t < 0:
        return 0.5 * _incomplete_beta(x, a, b)
    else:
        return 1 - 0.5 * _incomplete_beta(x, a, b)

def _incomplete_beta(x: float, a: float, b: float, max_iter: int = 100) -> float:
    """Evaluate the regularised incomplete beta function :math:`I_x(a, b)`.

    Uses the hypergeometric series, applying the reflection identity
    :math:`I_x(a, b) = 1 - I_{1-x}(b, a)` when ``x`` is large enough that the
    series would converge slowly.

    Parameters
    ----------
    x : float
        Upper limit of integration, expected in :math:`[0, 1]`. Values outside
        that range return ``0.0``.
    a : float
        First shape parameter, must be positive.
    b : float
        Second shape parameter, must be positive.
    max_iter : int, default=100
        Maximum number of series terms; the loop also stops early once a term
        falls below ``1e-10``.

    Returns
    -------
    value : float
        :math:`I_x(a, b)`, in :math:`[0, 1]`.
    """
    if x < 0 or x > 1:
        return 0.0

    # Handle edge cases to avoid log(0)
    if x == 0 or x < 1e-15:
        return 0.0
    if x == 1 or x > 1 - 1e-15:
        return 1.0

    if x > (a + 1) / (a + b + 2):
        return 1 - _incomplete_beta(1 - x, b, a, max_iter)

    from math import lgamma, exp, log

    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    # Use math.log instead of np.log for scalar values (avoids numpy warnings)
    front = exp(a * log(x) + b * log(1 - x) - lbeta) / a

    result = 1.0
    term = 1.0
    for n in range(1, max_iter):
        term *= (a + b + n - 1) * x / (a + n)
        result += term
        if abs(term) < 1e-10:
            break

    return front * result

def paired_t_test(
    x: np.ndarray,
    y: np.ndarray,
    significance_level: float = 0.05,
    higher_better: bool = True
) -> PairedStats:
    """Student's **paired** t-test comparing two algorithms on matched scores.

    Tests whether the mean of the per-pair differences
    :math:`d_i = x_i - y_i` differs from zero. Pairing means observation
    :math:`i` of ``x`` and observation :math:`i` of ``y`` must refer to the
    *same* fold, split or dataset; the shared difficulty of that fold then
    cancels out of :math:`d_i`, which is what makes the paired design far more
    sensitive than comparing two independent samples.

    Hypotheses
    ----------
    The test is **two-sided**:

    - :math:`H_0: \\mu_d = 0` -- the two algorithms have the same expected
      score, any observed gap is sampling noise.
    - :math:`H_1: \\mu_d \\neq 0` -- their expected scores differ, in either
      direction.

    Theory
    ------
    With :math:`\\bar{d}` the mean and :math:`s_d` the ``ddof=1`` standard
    deviation of the :math:`n` differences, the statistic is

    .. math::
        t = \\frac{\\bar{d}}{s_d / \\sqrt{n}}

    which follows a Student t-distribution with :math:`\\nu = n - 1` degrees
    of freedom under :math:`H_0`. The two-sided p-value is
    :math:`p = 2\\,[1 - F_{\\nu}(|t|)]`.

    Parameters
    ----------
    x : ndarray of shape (n,)
        Scores of the first algorithm, one entry per fold or dataset (e.g.
        accuracies of model A).
    y : ndarray of shape (n,)
        Scores of the second algorithm, **aligned element-wise** with ``x``.
        Must have the same length as ``x``.
    significance_level : float, default=0.05
        The :math:`\\alpha` against which the p-value is thresholded to produce
        the WIN/LOSS/TIE verdict. It does not change the p-value itself.
    higher_better : bool, default=True
        Orientation of the score. ``True`` for accuracy, F1, AUC; ``False`` for
        error rate, RMSE, log-loss. Only affects which of ``WIN``/``LOSS`` is
        reported, never the statistic or the p-value.

    Returns
    -------
    stats : PairedStats
        Full result: ``t_statistic``, the two-sided ``p_value``, the means and
        standard deviations of both samples, ``diff_mean``, ``correlation``,
        the ``significance`` verdict and the effective sample size ``n``. Read
        ``stats.p_value < alpha`` or ``stats.is_significant()`` for the
        decision and ``stats.x_better()`` for its direction.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different lengths, if fewer than 2
        observations are supplied, or if fewer than 2 complete pairs survive
        NaN removal.

    Notes
    -----
    **Assumptions.**

    1. *Paired* data -- ``x[i]`` and ``y[i]`` measured on the same fold or
       dataset. Comparing unrelated runs violates this and inflates
       significance.
    2. The differences :math:`d_i` are approximately **normally distributed**.
       The test is fairly robust for :math:`n \\gtrsim 30`, but with the 5-10
       datasets typical of a machine-learning study a single outlier dominates
       both :math:`\\bar{d}` and :math:`s_d`.
    3. The differences are **independent** across :math:`i`. This is the
       assumption that k-fold cross-validation breaks, because the training
       sets overlap; see :func:`corrected_paired_t_test`.
    4. Scores are on an interval scale and commensurable across datasets.
       Averaging accuracy differences over heterogeneous datasets is exactly
       the practice [Demsar2006]_ argues against.

    **Complexity.** :math:`O(n)` time and memory.

    **Handling of degenerate input.** Pairs containing NaN in either vector are
    dropped. If every difference is identical (zero standard error), the test
    returns ``t = 0`` and ``p = 1`` rather than dividing by zero -- note this
    makes a perfect, perfectly consistent win look like a tie.

    **When to prefer something else.** Use
    :func:`~tuiml.evaluation.statistics.wilcoxon_signed_rank_test` when
    normality is doubtful or :math:`n` is small -- [Demsar2006]_ recommends it
    as the default for comparing two classifiers over multiple datasets. Use
    :func:`corrected_paired_t_test` for cross-validated or repeated-resampling
    scores. For three or more algorithms run an omnibus test first
    (:func:`~tuiml.evaluation.statistics.friedman_test`) rather than all
    pairwise t-tests.

    References
    ----------
    .. [Student1908] Student (W. S. Gosset) (1908). "The Probable Error of a
       Mean". Biometrika, 6(1), 1-25.
    .. [Dietterich1998] Dietterich, T. G. (1998). "Approximate Statistical
       Tests for Comparing Supervised Classification Learning Algorithms".
       Neural Computation, 10(7), 1895-1923.
    .. [Demsar2006] Demsar, J. (2006). "Statistical Comparisons of Classifiers
       over Multiple Data Sets". Journal of Machine Learning Research, 7, 1-30.

    See Also
    --------
    :func:`~tuiml.evaluation.statistics.corrected_paired_t_test` : Same test
        with the Nadeau & Bengio variance correction for overlapping training
        sets.
    :func:`~tuiml.evaluation.statistics.wilcoxon_signed_rank_test` :
        Non-parametric counterpart; drops the normality assumption.
    :func:`~tuiml.evaluation.statistics.one_way_anova` : Extension to more than
        two *independent* groups.
    :func:`~tuiml.evaluation.statistics.holm_correction` : Adjust the p-values
        when many pairs are tested.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.statistics import paired_t_test
    >>> model_a_acc = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
    >>> model_b_acc = np.array([0.82, 0.84, 0.81, 0.83, 0.82])
    >>> stats = paired_t_test(model_a_acc, model_b_acc, significance_level=0.05)
    >>> round(float(stats.t_statistic), 4)
    10.6145
    >>> round(float(stats.p_value), 4)
    0.0004
    >>> stats.is_significant()
    True
    >>> stats.x_better()
    True

    With an error-style metric, flip ``higher_better`` so the verdict points at
    the algorithm with the *lower* score:

    >>> err_a = np.array([0.15, 0.13, 0.17, 0.14, 0.16])
    >>> err_b = np.array([0.18, 0.16, 0.19, 0.17, 0.18])
    >>> paired_t_test(err_a, err_b, higher_better=False).x_better()
    True
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations for paired t-test")

    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    n = len(x)

    if n < 2:
        raise ValueError("Not enough valid observations after removing NaN")

    # Calculate statistics
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)

    # Differences
    diff = x - y
    diff_mean = np.mean(diff)
    diff_std = np.std(diff, ddof=1)

    # T-statistic
    se = diff_std / np.sqrt(n)
    if se == 0:
        t_statistic = 0.0
        p_value = 1.0
    else:
        t_statistic = diff_mean / se
        df = n - 1
        p_value = 2 * (1 - _t_distribution_cdf(abs(t_statistic), df))

    # Correlation
    if x_std > 0 and y_std > 0:
        correlation = np.corrcoef(x, y)[0, 1]
    else:
        correlation = 0.0

    # Determine significance
    if p_value < significance_level:
        if higher_better:
            significance = SignificanceLevel.WIN if diff_mean > 0 else SignificanceLevel.LOSS
        else:
            significance = SignificanceLevel.WIN if diff_mean < 0 else SignificanceLevel.LOSS
    else:
        significance = SignificanceLevel.TIE

    return PairedStats(
        x_mean=x_mean,
        y_mean=y_mean,
        x_std=x_std,
        y_std=y_std,
        diff_mean=diff_mean,
        diff_std=diff_std,
        t_statistic=t_statistic,
        p_value=p_value,
        correlation=correlation,
        significance=significance,
        n=n
    )

def corrected_paired_t_test(
    x: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_test: int,
    significance_level: float = 0.05,
    higher_better: bool = True
) -> PairedStats:
    """Corrected resampled paired t-test for cross-validated scores.

    The ordinary paired t-test assumes the :math:`n` score differences are
    independent. Scores produced by k-fold cross-validation or repeated random
    subsampling are **not**: the training sets overlap, so the differences are
    positively correlated. The usual variance estimator is therefore too small,
    :math:`|t|` is too large, and the test declares differences significant far
    more often than :math:`\\alpha` allows -- [Dietterich1998]_ measured Type I
    error rates several times the nominal level. This function applies the
    [NadeauBengio2003]_ correction, which inflates the variance estimate to
    account for that overlap.

    Hypotheses
    ----------
    Identical to :func:`paired_t_test` and still **two-sided**:

    - :math:`H_0: \\mu_d = 0` -- both algorithms have the same expected
      generalization performance.
    - :math:`H_1: \\mu_d \\neq 0`.

    Only the standard error in the denominator changes.

    Theory
    ------
    With :math:`s_d^2` the ordinary ``ddof=1`` variance of the differences, the
    corrected statistic replaces :math:`s_d^2/n` by

    .. math::
        \\widehat{\\sigma}^2 = \\left(\\frac{1}{n}
            + \\frac{n_{test}}{n_{train}}\\right) s_d^2

    giving

    .. math::
        t = \\frac{\\bar{d}}{\\sqrt{\\widehat{\\sigma}^2}}

    compared against a t-distribution with :math:`n - 1` degrees of freedom.
    The extra :math:`n_{test}/n_{train}` term is the price of reusing training
    data; it never vanishes with more resampling rounds, which is why simply
    repeating cross-validation cannot buy unlimited significance.

    Parameters
    ----------
    x : ndarray of shape (n,)
        Per-fold scores of the first algorithm.
    y : ndarray of shape (n,)
        Per-fold scores of the second algorithm, aligned element-wise with
        ``x`` (same folds, same order).
    n_train : int
        Number of training examples in **one** resampling round. For 10-fold CV
        on 1000 examples this is 900.
    n_test : int
        Number of test examples in one resampling round. For 10-fold CV on 1000
        examples this is 100. The ratio :math:`n_{test}/n_{train}` -- not the
        absolute sizes -- drives the correction; for k-fold CV it equals
        :math:`1/(k-1)`.
    significance_level : float, default=0.05
        :math:`\\alpha` used to turn the p-value into the WIN/LOSS/TIE verdict.
    higher_better : bool, default=True
        ``True`` when larger scores are better (accuracy), ``False`` for error
        metrics. Affects only the direction of the verdict.

    Returns
    -------
    stats : PairedStats
        Same container as :func:`paired_t_test`, with two differences worth
        noting: ``t_statistic`` and ``p_value`` are the *corrected* ones, and
        ``diff_std`` holds the corrected **standard error**
        :math:`\\sqrt{\\widehat{\\sigma}^2}` rather than a standard deviation,
        so it is not comparable to the ``diff_std`` of the uncorrected test.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different lengths, or if fewer than 2 complete
        pairs survive NaN removal.

    Notes
    -----
    **Assumptions.**

    1. Paired, element-wise aligned scores from a resampling scheme with a
       constant train/test split ratio.
    2. Approximately normal differences, as for the uncorrected test.
    3. ``n_train`` and ``n_test`` describe a *single* round, not the totals
       accumulated over all rounds. Passing the totals silently shrinks the
       correction toward nothing.

    **Complexity.** :math:`O(n)` time and memory.

    **When to prefer it.** Use this instead of :func:`paired_t_test` whenever
    the scores come from k-fold CV, repeated k-fold CV, or 5x2 CV -- i.e.
    almost every model comparison run on a single dataset. If the :math:`n`
    scores are one-per-dataset over :math:`n` genuinely distinct datasets there
    is no training-set overlap and the plain paired t-test is appropriate.
    [BouckaertFrank2004]_ found 10x10-fold CV with this correction to give the
    best replicability among the schemes they compared.

    **Degenerate input.** If the corrected variance is zero the function
    returns ``t = 0`` and ``p = 1``.

    References
    ----------
    .. [NadeauBengio2003] Nadeau, C., & Bengio, Y. (2003). "Inference for the
       Generalization Error". Machine Learning, 52(3), 239-281.
    .. [Dietterich1998] Dietterich, T. G. (1998). "Approximate Statistical
       Tests for Comparing Supervised Classification Learning Algorithms".
       Neural Computation, 10(7), 1895-1923.
    .. [BouckaertFrank2004] Bouckaert, R. R., & Frank, E. (2004). "Evaluating
       the Replicability of Significance Tests for Comparing Learning
       Algorithms". Advances in Knowledge Discovery and Data Mining (PAKDD),
       LNCS 3056, 3-12.

    See Also
    --------
    :func:`~tuiml.evaluation.statistics.paired_t_test` : Uncorrected version;
        valid only when the folds are independent.
    :func:`~tuiml.evaluation.statistics.wilcoxon_signed_rank_test` :
        Distribution-free alternative when normality is doubtful.

    Examples
    --------
    10-fold cross-validation on 1000 examples, so ``n_train=900`` and
    ``n_test=100`` per round:

    >>> import numpy as np
    >>> from tuiml.evaluation.statistics import (
    ...     corrected_paired_t_test, paired_t_test)
    >>> fold_a = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
    >>> fold_b = np.array([0.82, 0.84, 0.81, 0.83, 0.82])
    >>> stats = corrected_paired_t_test(fold_a, fold_b, n_train=900, n_test=100)
    >>> round(float(stats.t_statistic), 4)
    8.5105
    >>> round(float(stats.p_value), 4)
    0.001

    The correction always shrinks the statistic relative to the uncorrected
    test, so it can only ever make a result *less* significant:

    >>> plain = paired_t_test(fold_a, fold_b)
    >>> bool(abs(stats.t_statistic) < abs(plain.t_statistic))
    True
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)

    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    n = len(x)

    if n < 2:
        raise ValueError("Not enough valid observations")

    # Calculate statistics
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)

    diff = x - y
    diff_mean = np.mean(diff)
    diff_var = np.var(diff, ddof=1)

    # Corrected variance (Nadeau & Bengio)
    correction_factor = 1/n + n_test/n_train
    corrected_var = correction_factor * diff_var

    diff_std = np.sqrt(corrected_var) if corrected_var > 0 else 0

    # T-statistic with correction
    if diff_std == 0:
        t_statistic = 0.0
        p_value = 1.0
    else:
        t_statistic = diff_mean / diff_std
        df = n - 1
        p_value = 2 * (1 - _t_distribution_cdf(abs(t_statistic), df))

    # Correlation
    if x_std > 0 and y_std > 0:
        correlation = np.corrcoef(x, y)[0, 1]
    else:
        correlation = 0.0

    # Significance
    if p_value < significance_level:
        if higher_better:
            significance = SignificanceLevel.WIN if diff_mean > 0 else SignificanceLevel.LOSS
        else:
            significance = SignificanceLevel.WIN if diff_mean < 0 else SignificanceLevel.LOSS
    else:
        significance = SignificanceLevel.TIE

    return PairedStats(
        x_mean=x_mean,
        y_mean=y_mean,
        x_std=x_std,
        y_std=y_std,
        diff_mean=diff_mean,
        diff_std=diff_std,
        t_statistic=t_statistic,
        p_value=p_value,
        correlation=correlation,
        significance=significance,
        n=n
    )

def one_way_anova(
    *groups,
    significance_level: float = 0.05
) -> Tuple[float, float, bool]:
    """One-way ANOVA: omnibus F-test across :math:`k` **independent** groups.

    Answers a single question -- "is *any* of these groups different from the
    others?" -- without saying which. It is the multi-group generalisation of
    the unpaired t-test, and the parametric analogue of
    :func:`~tuiml.evaluation.statistics.friedman_test`.

    Hypotheses
    ----------
    - :math:`H_0: \\mu_1 = \\mu_2 = \\dots = \\mu_k` -- every group has the
      same population mean.
    - :math:`H_1`: at least one :math:`\\mu_i` differs. Rejecting says nothing
      about *which* group, or about how many; that needs a post-hoc test.

    Theory
    ------
    The total variability is split into a between-group and a within-group
    part. With :math:`\\bar{X}` the grand mean, :math:`n_i` and
    :math:`\\bar{X}_i` the size and mean of group :math:`i`, and
    :math:`N = \\sum_i n_i`:

    .. math::
        SS_B = \\sum_{i=1}^{k} n_i (\\bar{X}_i - \\bar{X})^2, \\quad
        SS_W = \\sum_{i=1}^{k} \\sum_{j=1}^{n_i} (X_{ij} - \\bar{X}_i)^2

    .. math::
        F = \\frac{SS_B / (k - 1)}{SS_W / (N - k)}

    Under :math:`H_0`, :math:`F` follows an F-distribution with
    :math:`(k-1, N-k)` degrees of freedom. Large :math:`F` means the group
    means are spread out relative to the noise within groups. The p-value is
    the **upper tail** :math:`P(F_{k-1,\\,N-k} > F)` -- one-sided by
    construction, because only large :math:`F` contradicts :math:`H_0`, even
    though the underlying alternative is two-sided in each mean.

    Parameters
    ----------
    *groups : array-like
        Two or more 1-D arrays of scores, passed as separate positional
        arguments (``one_way_anova(a, b, c)``). Groups may have **different
        lengths** and are treated as mutually **independent** samples --
        entries are not paired across groups.
    significance_level : float, default=0.05
        :math:`\\alpha` used only to compute the returned boolean; it does not
        affect ``f_statistic`` or ``p_value``. Keyword-only.

    Returns
    -------
    f_statistic : float
        The F-ratio :math:`MS_B / MS_W`. Always non-negative; ``0.0`` when the
        within-group mean square is zero. Values near 1 are what
        :math:`H_0` predicts.
    p_value : float
        Upper-tail probability :math:`P(F_{k-1,\\,N-k} > F)` under
        :math:`H_0`, in :math:`[0, 1]`.
    significant : bool
        ``p_value < significance_level``. ``True`` means "the groups are not
        all equal"; it does **not** identify the winner.

    Raises
    ------
    ValueError
        If fewer than two groups are supplied.

    Notes
    -----
    **Assumptions.**

    1. **Independence** between and within groups. This is the assumption that
       rules ANOVA out for the usual machine-learning layout, where the same
       datasets or folds are reused by every algorithm -- those samples are
       matched, not independent. Use
       :func:`~tuiml.evaluation.statistics.friedman_test` (or a repeated
       measures design) there.
    2. **Normality** of the residuals within each group.
    3. **Homoscedasticity** -- equal variance across groups. ANOVA tolerates
       moderate violations when group sizes are balanced and degrades quickly
       when they are not.

    **Complexity.** :math:`O(N)` in the total number of observations.

    **After a rejection.** A significant omnibus result licenses post-hoc
    pairwise comparisons, but each of those is a fresh test, so their p-values
    must be adjusted -- see
    :func:`~tuiml.evaluation.statistics.holm_correction`.

    References
    ----------
    .. [Fisher1925] Fisher, R. A. (1925). "Statistical Methods for Research
       Workers". Oliver and Boyd, Edinburgh.
    .. [Demsar2006] Demsar, J. (2006). "Statistical Comparisons of Classifiers
       over Multiple Data Sets". Journal of Machine Learning Research, 7, 1-30.

    See Also
    --------
    :func:`~tuiml.evaluation.statistics.friedman_test` : Non-parametric
        omnibus test for *matched* samples; the right choice for comparing
        several algorithms over shared datasets.
    :func:`~tuiml.evaluation.statistics.paired_t_test` : Two-algorithm case.
    :func:`~tuiml.evaluation.statistics.bonferroni_correction` : Adjust
        post-hoc pairwise p-values.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.statistics import one_way_anova
    >>> group_a = np.array([0.85, 0.87, 0.83])
    >>> group_b = np.array([0.82, 0.84, 0.81])
    >>> group_c = np.array([0.75, 0.78, 0.74])
    >>> f_stat, p_value, significant = one_way_anova(group_a, group_b, group_c)
    >>> round(float(f_stat), 4)
    19.5
    >>> round(float(p_value), 4)
    0.0024
    >>> bool(significant)
    True
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups)  # Number of groups

    if k < 2:
        raise ValueError("Need at least 2 groups")

    # Total number of observations
    n_total = sum(len(g) for g in groups)

    # Grand mean
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    # Between-group sum of squares (SSB)
    ssb = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)

    # Within-group sum of squares (SSW)
    ssw = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    # Degrees of freedom
    df_between = k - 1
    df_within = n_total - k

    # Mean squares
    msb = ssb / df_between
    msw = ssw / df_within if df_within > 0 else 1

    # F-statistic
    f_statistic = msb / msw if msw > 0 else 0

    # P-value (F-distribution)
    p_value = 1 - _f_distribution_cdf(f_statistic, df_between, df_within)

    return f_statistic, p_value, p_value < significance_level

def _f_distribution_cdf(f: float, d1: int, d2: int) -> float:
    """Evaluate the CDF of the F-distribution.

    Uses the identity :math:`F(f; d_1, d_2) = I_{x}(d_1/2, d_2/2)` with
    :math:`x = d_1 f / (d_1 f + d_2)`, evaluated through the reflected form
    :math:`1 - I_{1-x}(d_2/2, d_1/2)` for numerical stability.

    Parameters
    ----------
    f : float
        Value at which to evaluate the CDF. Non-positive values return ``0.0``.
    d1 : int
        Numerator degrees of freedom.
    d2 : int
        Denominator degrees of freedom.

    Returns
    -------
    cdf : float
        :math:`P(F \\leq f)`, in :math:`[0, 1]`.
    """
    if f <= 0 or d1 <= 0 or d2 <= 0:
        return 0.0

    x = d1 * f / (d1 * f + d2)
    return 1 - _incomplete_beta(1 - x, d2 / 2, d1 / 2)
