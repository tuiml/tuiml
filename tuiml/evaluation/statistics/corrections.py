"""
Multiple comparison correction methods.

Used to control family-wise error rate (FWER) or false discovery rate (FDR)
when performing multiple statistical tests.
"""

import numpy as np
from typing import List, Tuple

def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bonferroni correction for multiple comparisons.

    Most conservative method. Controls FWER.

    Parameters
    ----------
    p_values : ndarray
        Array of p-values from multiple tests.
    alpha : float
        Significance level.

    Returns
    -------
    adjusted_p : ndarray
        Adjusted p-values.
    reject : ndarray
        Boolean array indicating which hypotheses to reject.

    Examples
    --------
    >>> from tuiml.evaluation.statistics import bonferroni_correction
    >>> import numpy as np
    >>> p_values = np.array([0.01, 0.04, 0.03, 0.005])
    >>> adjusted, reject = bonferroni_correction(p_values, alpha=0.05)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    adjusted_p = np.minimum(p_values * n, 1.0)
    reject = adjusted_p < alpha

    return adjusted_p, reject

def holm_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Holm-Bonferroni step-down correction.

    Less conservative than Bonferroni while still controlling FWER.

    Parameters
    ----------
    p_values : ndarray
        Array of p-values.
    alpha : float
        Significance level.

    Returns
    -------
    adjusted_p : ndarray
        Adjusted p-values.
    reject : ndarray
        Boolean array indicating which hypotheses to reject.

    Examples
    --------
    >>> from tuiml.evaluation.statistics import holm_correction
    >>> import numpy as np
    >>> p_values = np.array([0.01, 0.04, 0.03, 0.005])
    >>> adjusted, reject = holm_correction(p_values, alpha=0.05)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Calculate adjusted p-values
    adjusted_sorted = np.zeros(n)
    for i in range(n):
        adjusted_sorted[i] = min(sorted_p[i] * (n - i), 1.0)

    # Enforce monotonicity (adjusted p-values should be non-decreasing)
    for i in range(1, n):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i-1])

    # Unsort to original order
    adjusted_p = np.zeros(n)
    adjusted_p[sorted_idx] = adjusted_sorted

    reject = adjusted_p < alpha

    return adjusted_p, reject

def hochberg_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hochberg step-up correction.

    More powerful than Holm but requires independence assumption.

    Parameters
    ----------
    p_values : ndarray
        Array of p-values.
    alpha : float
        Significance level.

    Returns
    -------
    adjusted_p : ndarray
        Adjusted p-values.
    reject : ndarray
        Boolean array indicating which hypotheses to reject.
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sort p-values in descending order
    sorted_idx = np.argsort(p_values)[::-1]
    sorted_p = p_values[sorted_idx]

    # Calculate adjusted p-values
    adjusted_sorted = np.zeros(n)
    for i in range(n):
        adjusted_sorted[i] = min(sorted_p[i] * (i + 1), 1.0)

    # Enforce monotonicity (should be non-increasing when sorted descending)
    for i in range(1, n):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i-1])

    # Unsort to original order
    adjusted_p = np.zeros(n)
    adjusted_p[sorted_idx] = adjusted_sorted

    reject = adjusted_p < alpha

    return adjusted_p, reject

def hommel_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hommel correction.

    Most powerful step-wise method that controls FWER.

    Parameters
    ----------
    p_values : ndarray
        Array of p-values.
    alpha : float
        Significance level.

    Returns
    -------
    adjusted_p : ndarray
        Adjusted p-values.
    reject : ndarray
        Boolean array indicating which hypotheses to reject.
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    if n == 1:
        return p_values, p_values < alpha

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Initialize adjusted p-values
    adjusted_sorted = np.ones(n)

    # Hommel's iterative procedure
    for i in range(n, 0, -1):
        # Calculate gamma_i
        gamma = sorted_p[:i] * i / np.arange(1, i + 1)
        min_gamma = np.min(gamma)

        # Update adjusted p-values
        for j in range(i):
            adjusted_sorted[j] = min(adjusted_sorted[j], min_gamma * (n - j) / i)

    # Ensure monotonicity
    for i in range(1, n):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i-1])

    # Clip to [0, 1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    # Unsort to original order
    adjusted_p = np.zeros(n)
    adjusted_p[sorted_idx] = adjusted_sorted

    reject = adjusted_p < alpha

    return adjusted_p, reject

def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg procedure for controlling False Discovery Rate (FDR).

    Less conservative than FWER-controlling methods. Controls the expected
    proportion of false discoveries among rejected hypotheses.

    Parameters
    ----------
    p_values : ndarray
        Array of p-values.
    alpha : float
        Significance level (FDR level).

    Returns
    -------
    adjusted_p : ndarray
        Adjusted p-values (q-values).
    reject : ndarray
        Boolean array indicating which hypotheses to reject.

    Examples
    --------
    >>> from tuiml.evaluation.statistics import benjamini_hochberg
    >>> import numpy as np
    >>> p_values = np.array([0.01, 0.04, 0.03, 0.005])
    >>> adjusted, reject = benjamini_hochberg(p_values, alpha=0.05)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Calculate adjusted p-values (q-values)
    adjusted_sorted = np.zeros(n)
    for i in range(n):
        adjusted_sorted[i] = sorted_p[i] * n / (i + 1)

    # Enforce monotonicity (from the end)
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    # Clip to [0, 1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    # Unsort to original order
    adjusted_p = np.zeros(n)
    adjusted_p[sorted_idx] = adjusted_sorted

    reject = adjusted_p < alpha

    return adjusted_p, reject
