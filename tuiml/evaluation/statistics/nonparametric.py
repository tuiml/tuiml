"""
Non-parametric statistical tests.

Tests that don't assume normal distribution.
"""

import numpy as np
from typing import Dict, List, Tuple
from .parametric import PairedStats, SignificanceLevel

def wilcoxon_signed_rank_test(
    x: np.ndarray,
    y: np.ndarray,
    significance_level: float = 0.05,
    higher_better: bool = True
) -> PairedStats:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Parameters
    ----------
    x, y : ndarray
        Results from two models.
    significance_level : float
        Significance level.
    higher_better : bool
        If True, higher is better.

    Returns
    -------
    stats : PairedStats
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    # Remove NaN and zero differences
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    diff = x - y
    nonzero = diff != 0
    diff = diff[nonzero]
    n = len(diff)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x, ddof=1) if len(x) > 1 else 0
    y_std = np.std(y, ddof=1) if len(y) > 1 else 0
    diff_mean = np.mean(diff) if n > 0 else 0
    diff_std = np.std(diff, ddof=1) if n > 1 else 0

    if n < 2:
        return PairedStats(
            x_mean=x_mean, y_mean=y_mean,
            x_std=x_std, y_std=y_std,
            diff_mean=diff_mean, diff_std=diff_std,
            t_statistic=0.0, p_value=1.0,
            correlation=0.0,
            significance=SignificanceLevel.TIE,
            n=n
        )

    # Rank absolute differences
    abs_diff = np.abs(diff)
    ranks = np.argsort(np.argsort(abs_diff)) + 1

    # Calculate W+ and W-
    w_plus = np.sum(ranks[diff > 0])
    w_minus = np.sum(ranks[diff < 0])
    w = min(w_plus, w_minus)

    # Normal approximation for large n
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    if std_w > 0:
        z = (w - mean_w) / std_w
        from math import erf, sqrt
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    else:
        z = 0
        p_value = 1.0

    # Correlation
    if x_std > 0 and y_std > 0 and len(x) > 1:
        correlation = np.corrcoef(x[nonzero], y[nonzero])[0, 1] if np.sum(nonzero) > 1 else 0
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
        t_statistic=z,
        p_value=p_value,
        correlation=correlation,
        significance=significance,
        n=n
    )

def _chi2_cdf(x: float, df: int) -> float:
    """Approximate CDF of chi-squared distribution."""
    if x < 0 or df <= 0:
        return 0.0

    from math import gamma
    return _incomplete_gamma(df / 2, x / 2) / gamma(df / 2)

def _incomplete_gamma(a: float, x: float, max_iter: int = 100) -> float:
    """Incomplete gamma function approximation."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0

    from math import exp, log, gamma

    # Series expansion for small x
    if x < a + 1:
        ap = a
        s = 1.0 / a
        ds = s
        for _ in range(max_iter):
            ap += 1
            ds *= x / ap
            s += ds
            if abs(ds) < abs(s) * 1e-10:
                break
        return s * exp(-x + a * log(x))

    # Continued fraction for large x
    b = x + 1 - a
    c = 1e30
    d = 1 / b
    h = d
    for i in range(1, max_iter):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-10:
            break

    return gamma(a) - h * exp(-x + a * log(x))

def friedman_test(
    results: Dict[str, np.ndarray],
    significance_level: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Friedman test for comparing multiple models.

    Non-parametric test for comparing more than two related samples.

    Parameters
    ----------
    results : dict
        Dictionary of {model_name: scores_array}.
        All arrays must have the same length.
    significance_level : float
        Significance level.

    Returns
    -------
    chi2 : float
        Chi-squared statistic.
    p_value : float
        P-value.
    significant : bool
        Whether the difference is significant.

    Examples
    --------
    >>> results = {
    ...     'ModelA': np.array([0.85, 0.87, 0.83]),
    ...     'ModelB': np.array([0.82, 0.84, 0.81]),
    ...     'ModelC': np.array([0.80, 0.82, 0.79])
    ... }
    >>> chi2, p_value, sig = friedman_test(results)
    """
    model_names = list(results.keys())
    k = len(model_names)  # Number of models
    n = len(results[model_names[0]])  # Number of datasets/folds

    # Create matrix of results
    matrix = np.array([results[name] for name in model_names])

    # Rank each column (dataset)
    ranks = np.zeros_like(matrix)
    for j in range(n):
        ranks[:, j] = k + 1 - np.argsort(np.argsort(-matrix[:, j])) - 1

    # Average ranks
    avg_ranks = np.mean(ranks, axis=1)

    # Friedman statistic
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(avg_ranks ** 2) - k * (k + 1) ** 2 / 4)

    # P-value
    df = k - 1
    p_value = 1 - _chi2_cdf(chi2, df)

    return chi2, p_value, p_value < significance_level

def nemenyi_post_hoc(
    results: Dict[str, np.ndarray],
    significance_level: float = 0.05
) -> Dict[Tuple[str, str], bool]:
    """
    Nemenyi post-hoc test after Friedman test.

    Parameters
    ----------
    results : dict
        Dictionary of {model_name: scores_array}.
    significance_level : float
        Significance level.

    Returns
    -------
    pairwise : dict
        Dictionary of {(model1, model2): is_significant}.
    """
    model_names = list(results.keys())
    k = len(model_names)
    n = len(results[model_names[0]])

    # Calculate ranks
    matrix = np.array([results[name] for name in model_names])
    ranks = np.zeros_like(matrix)
    for j in range(n):
        ranks[:, j] = k + 1 - np.argsort(np.argsort(-matrix[:, j])) - 1

    avg_ranks = np.mean(ranks, axis=1)

    # Critical difference
    q_alpha = {
        0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
        0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
               7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}
    }

    alpha = 0.05 if significance_level <= 0.05 else 0.10
    q = q_alpha[alpha].get(k, 3.0)

    cd = q * np.sqrt(k * (k + 1) / (6 * n))

    # Pairwise comparisons
    pairwise = {}
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i < j:
                diff = abs(avg_ranks[i] - avg_ranks[j])
                pairwise[(name1, name2)] = diff > cd

    return pairwise

def friedman_aligned_ranks_test(
    results: Dict[str, np.ndarray],
    significance_level: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Friedman Aligned Ranks test.

    More powerful alternative to standard Friedman test.

    Parameters
    ----------
    results : dict
        Dictionary of {model_name: scores_array}.
    significance_level : float
        Significance level.

    Returns
    -------
    statistic : float
        Test statistic.
    p_value : float
        P-value.
    significant : bool
        Whether the difference is significant.
    """
    model_names = list(results.keys())
    k = len(model_names)
    n = len(results[model_names[0]])

    # Create matrix
    matrix = np.array([results[name] for name in model_names])

    # Calculate aligned observations
    row_means = np.mean(matrix, axis=0)
    aligned = matrix - row_means

    # Rank all aligned observations
    all_aligned = aligned.flatten()
    ranks = np.argsort(np.argsort(-all_aligned)) + 1
    ranks = ranks.reshape(k, n)

    # Sum of ranks per algorithm
    R = np.sum(ranks, axis=1)

    # Test statistic
    numerator = (k - 1) * (np.sum(R**2) - k * n**2 * (k * n + 1)**2 / 4)
    denominator = (k * n * (k * n + 1) * (2 * k * n + 1) / 6 -
                   np.sum(np.sum(ranks, axis=0)**2) / k)

    if denominator == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = numerator / denominator
        p_value = 1 - _chi2_cdf(statistic, k - 1)

    return statistic, p_value, p_value < significance_level

def quade_test(
    results: Dict[str, np.ndarray],
    significance_level: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Quade test for comparing multiple algorithms.

    Similar to Friedman but accounts for dataset difficulty.

    Parameters
    ----------
    results : dict
        Dictionary of {model_name: scores_array}.
    significance_level : float
        Significance level.

    Returns
    -------
    f_statistic : float
        F statistic.
    p_value : float
        P-value.
    significant : bool
        Whether the difference is significant.
    """
    model_names = list(results.keys())
    k = len(model_names)
    n = len(results[model_names[0]])

    # Create matrix
    matrix = np.array([results[name] for name in model_names])

    # Rank within each dataset
    ranks = np.zeros_like(matrix)
    for j in range(n):
        ranks[:, j] = k + 1 - np.argsort(np.argsort(-matrix[:, j])) - 1

    # Compute range for each dataset (difficulty)
    ranges = np.max(matrix, axis=0) - np.min(matrix, axis=0)

    # Rank the ranges
    range_ranks = np.argsort(np.argsort(ranges)) + 1

    # Compute weighted ranks
    S = np.zeros(k)
    for i in range(k):
        S[i] = np.sum(range_ranks * (ranks[i] - (k + 1) / 2))

    # Test statistic
    A = np.sum(range_ranks**2) * k * (k + 1)**2 / 4
    B = np.sum(S**2) / n

    if n * A - B == 0:
        f_statistic = 0.0
        p_value = 1.0
    else:
        f_statistic = (n - 1) * B / (n * A - B)
        # Approximate F-distribution p-value
        df1 = k - 1
        df2 = (k - 1) * (n - 1)
        p_value = 1 - _f_cdf(f_statistic, df1, df2)

    return f_statistic, p_value, p_value < significance_level

def _f_cdf(f: float, d1: int, d2: int) -> float:
    """Approximate CDF of F-distribution."""
    if f <= 0:
        return 0.0

    x = d1 * f / (d1 * f + d2)

    from math import lgamma, exp

    def incomplete_beta(x, a, b, max_iter=100):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0

        if x > (a + 1) / (a + b + 2):
            return 1 - incomplete_beta(1 - x, b, a, max_iter)

        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        front = exp(a * np.log(x) + b * np.log(1 - x) - lbeta) / a

        result = 1.0
        term = 1.0
        for n in range(1, max_iter):
            term *= (a + b + n - 1) * x / (a + n)
            result += term
            if abs(term) < 1e-10:
                break

        return front * result

    return incomplete_beta(x, d1 / 2, d2 / 2)
