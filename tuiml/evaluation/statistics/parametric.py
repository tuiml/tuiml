"""
Parametric statistical tests.
- PairedTTester.java
- PairedStats.java
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

class SignificanceLevel(Enum):
    """Significance level indicators."""
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
    """
    Statistics for paired comparison between two sets of results.

    Attributes
    ----------
    x_mean : float
        Mean of first sample.
    y_mean : float
        Mean of second sample.
    x_std : float
        Std of first sample.
    y_std : float
        Std of second sample.
    diff_mean : float
        Mean of differences.
    diff_std : float
        Std of differences.
    t_statistic : float
        T-test statistic.
    p_value : float
        P-value from t-test.
    correlation : float
        Correlation between samples.
    significance : SignificanceLevel
        Significance indicator.
    n : int
        Number of paired observations.
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
        """Check if difference is significant."""
        return self.significance != SignificanceLevel.TIE

    def x_better(self) -> bool:
        """Check if x is significantly better than y."""
        return self.significance == SignificanceLevel.WIN

    def y_better(self) -> bool:
        """Check if y is significantly better than x."""
        return self.significance == SignificanceLevel.LOSS

def _t_distribution_cdf(t: float, df: int) -> float:
    """Approximate CDF of t-distribution."""
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
    """Incomplete beta function approximation."""
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
    """
    Perform paired t-test between two sets of results.

    Parameters
    ----------
    x : ndarray
        First set of results (e.g., model A accuracies).
    y : ndarray
        Second set of results (e.g., model B accuracies).
    significance_level : float, default=0.05
        Significance level (alpha).
    higher_better : bool, default=True
        If True, higher values are better.

    Returns
    -------
    stats : PairedStats
        Paired comparison statistics.

    Examples
    --------
    >>> from tuiml.evaluation.statistics import paired_t_test
    >>> import numpy as np
    >>> model_a_acc = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
    >>> model_b_acc = np.array([0.82, 0.84, 0.81, 0.83, 0.82])
    >>> stats = paired_t_test(model_a_acc, model_b_acc)
    >>> print(f"Significant: {stats.is_significant()}")
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
    """
    Corrected resampled paired t-test.

    Accounts for the dependency introduced by overlapping training sets
    in cross-validation using Nadeau & Bengio (2003) correction.

    Parameters
    ----------
    x, y : ndarray
        Results from two models.
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    significance_level : float
        Significance level.
    higher_better : bool
        If True, higher is better.

    Returns
    -------
    stats : PairedStats

    References
    ----------
    Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error.
    Machine Learning, 52(3), 239-281.
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
    """
    One-way ANOVA test for comparing multiple groups.

    Parameters
    ----------
    *groups : array-like
        Groups to compare.
    significance_level : float
        Significance level.

    Returns
    -------
    f_statistic : float
        F-statistic.
    p_value : float
        P-value.
    significant : bool
        Whether the difference is significant.
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
    """Approximate CDF of F-distribution."""
    if f <= 0 or d1 <= 0 or d2 <= 0:
        return 0.0

    x = d1 * f / (d1 * f + d2)
    return 1 - _incomplete_beta(1 - x, d2 / 2, d1 / 2)
