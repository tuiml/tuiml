"""
Feature scoring metrics for univariate feature selection.

This module provides metrics for scoring individual features in feature selection tasks.
Compatible with WEKA's attribute evaluation methods.
- chi2: ChiSquaredAttributeEval.java
- f_classif: ANOVA F-test
- f_regression: F-statistic for regression
- oner_score: OneRAttributeEval.java
- relief_f: ReliefFAttributeEval.java
"""

import numpy as np
from typing import Tuple, Optional, Any
from scipy import stats

def chi2(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute chi-squared statistics between each feature and the class.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (must be non-negative).
    y : ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    chi2_scores : ndarray of shape (n_features,)
        Chi-squared statistic for each feature.
    pvalues : ndarray of shape (n_features,)
        p-values corresponding to the chi-squared statistics.
    """
    if np.any(X < 0):
        raise ValueError("chi2 requires non-negative feature values")

    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    chi2_scores = np.zeros(n_features)
    pvalues = np.ones(n_features)

    for i in range(n_features):
        x_i = X[:, i]

        # Discretize continuous features using bins
        if np.issubdtype(x_i.dtype, np.floating) and len(np.unique(x_i)) > 10:
            n_bins = min(10, len(np.unique(x_i)))
            x_discrete = np.digitize(x_i, np.percentile(x_i, np.linspace(0, 100, n_bins + 1)[1:-1]))
        else:
            x_discrete = x_i.astype(int) if not np.issubdtype(x_i.dtype, np.integer) else x_i

        x_values = np.unique(x_discrete)
        n_vals = len(x_values)

        if n_vals <= 1:
            continue

        # Build contingency table
        observed = np.zeros((n_vals, n_classes))
        for j, val in enumerate(x_values):
            for k, cls in enumerate(classes):
                observed[j, k] = np.sum((x_discrete == val) & (y == cls))

        # Expected frequencies
        row_sums = np.sum(observed, axis=1)
        col_sums = np.sum(observed, axis=0)
        total = np.sum(observed)

        if total == 0:
            continue

        expected = np.outer(row_sums, col_sums) / total

        # Chi-squared statistic
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2_stat = np.sum(np.where(expected > 0,
                                        (observed - expected) ** 2 / expected,
                                        0))

        # Degrees of freedom
        df = (n_vals - 1) * (n_classes - 1)

        if df > 0:
            chi2_scores[i] = chi2_stat
            pvalues[i] = 1 - stats.chi2.cdf(chi2_stat, df)

    return chi2_scores, pvalues

def f_classif(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ANOVA F-value between each feature and the class.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values (class labels).

    Returns
    -------
    f_scores : ndarray of shape (n_features,)
        F-statistic for each feature.
    pvalues : ndarray of shape (n_features,)
        p-values associated with the F-statistic.
    """
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    f_scores = np.zeros(n_features)
    pvalues = np.ones(n_features)

    df_between = n_classes - 1
    df_within = n_samples - n_classes

    if df_within <= 0:
        return f_scores, pvalues

    for i in range(n_features):
        x_i = X[:, i]

        # Handle missing values
        valid_mask = ~np.isnan(x_i)
        if np.sum(valid_mask) < n_classes + 1:
            continue

        x_valid = x_i[valid_mask]
        y_valid = y[valid_mask]

        # Group by class
        groups = [x_valid[y_valid == c] for c in classes]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            continue

        # Grand mean
        grand_mean = np.mean(x_valid)
        n_total = len(x_valid)

        # Between-group sum of squares
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)

        # Within-group sum of squares
        ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

        # F-statistic
        if ss_within == 0:
            f_scores[i] = np.inf if ss_between > 0 else 0
            pvalues[i] = 0 if ss_between > 0 else 1
        else:
            df_b = len(groups) - 1
            df_w = n_total - len(groups)

            if df_w > 0 and df_b > 0:
                ms_between = ss_between / df_b
                ms_within = ss_within / df_w
                f_scores[i] = ms_between / ms_within
                pvalues[i] = 1 - stats.f.cdf(f_scores[i], df_b, df_w)

    return f_scores, pvalues

def f_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute F-statistic and p-value for regression on each feature.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    f_scores : ndarray of shape (n_features,)
        F-statistic for each feature.
    pvalues : ndarray of shape (n_features,)
        p-values associated with the F-statistic.
    """
    n_samples, n_features = X.shape
    f_scores = np.zeros(n_features)
    pvalues = np.ones(n_features)

    df_regression = 1
    df_residual = n_samples - 2

    if df_residual <= 0:
        return f_scores, pvalues

    y_mean = np.nanmean(y)
    ss_total = np.nansum((y - y_mean) ** 2)

    if ss_total == 0:
        return f_scores, pvalues

    for i in range(n_features):
        x_i = X[:, i]

        # Handle missing values
        valid_mask = ~(np.isnan(x_i) | np.isnan(y))
        n_valid = np.sum(valid_mask)

        if n_valid < 3:
            continue

        x_valid = x_i[valid_mask]
        y_valid = y[valid_mask]

        # Simple linear regression: y = a + b*x
        x_mean = np.mean(x_valid)
        y_mean_valid = np.mean(y_valid)

        # Compute slope
        ss_xy = np.sum((x_valid - x_mean) * (y_valid - y_mean_valid))
        ss_xx = np.sum((x_valid - x_mean) ** 2)

        if ss_xx == 0:
            continue

        b = ss_xy / ss_xx
        a = y_mean_valid - b * x_mean

        # Predicted values
        y_pred = a + b * x_valid

        # Sum of squares
        ss_reg = np.sum((y_pred - y_mean_valid) ** 2)
        ss_res = np.sum((y_valid - y_pred) ** 2)

        # F-statistic
        if ss_res == 0:
            f_scores[i] = np.inf
            pvalues[i] = 0
        else:
            ms_reg = ss_reg / df_regression
            ms_res = ss_res / (n_valid - 2)
            f_scores[i] = ms_reg / ms_res
            pvalues[i] = 1 - stats.f.cdf(f_scores[i], df_regression, n_valid - 2)

    return f_scores, pvalues

def correlation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation coefficient between each feature and the target.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    scores : ndarray of shape (n_features,)
        Absolute Pearson correlation coefficient for each feature.
    """
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)

    y_centered = y - np.nanmean(y)
    y_std = np.nanstd(y)

    if y_std == 0:
        return scores

    for i in range(n_features):
        x_i = X[:, i]

        # Handle missing values
        valid_mask = ~(np.isnan(x_i) | np.isnan(y))
        if np.sum(valid_mask) < 3:
            scores[i] = 0
            continue

        x_valid = x_i[valid_mask]
        y_valid = y[valid_mask]

        # Compute correlation
        x_centered = x_valid - np.mean(x_valid)
        y_centered_valid = y_valid - np.mean(y_valid)

        x_std = np.std(x_valid)
        y_std_valid = np.std(y_valid)

        if x_std == 0 or y_std_valid == 0:
            scores[i] = 0
        else:
            corr = np.sum(x_centered * y_centered_valid) / (len(x_valid) * x_std * y_std_valid)
            scores[i] = np.abs(corr)

    return scores

def oner_score(X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Evaluate features using OneR classifier accuracy.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.
    n_bins : int, default=10
        Number of bins for discretizing continuous features.

    Returns
    -------
    scores : ndarray of shape (n_features,)
        OneR accuracy for each feature (between 0 and 1).
    """
    n_samples, n_features = X.shape
    classes, y_encoded = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    scores = np.zeros(n_features)

    for i in range(n_features):
        x_i = X[:, i]

        # Discretize continuous features
        if np.issubdtype(x_i.dtype, np.floating):
            nan_mask = np.isnan(x_i)
            x_discrete = x_i.copy()
            if np.any(~nan_mask):
                x_valid = x_i[~nan_mask]
                n_unique = len(np.unique(x_valid))
                if n_unique > n_bins:
                    percentiles = np.linspace(0, 100, n_bins + 1)
                    bin_edges = np.percentile(x_valid, percentiles)
                    x_discrete = np.digitize(x_i, bin_edges[:-1])
                else:
                    x_discrete = x_i
                x_discrete = np.where(nan_mask, -999, x_discrete)
        else:
            x_discrete = np.where(np.isnan(x_i.astype(float)), -999, x_i)

        # Build OneR classifier
        x_values = np.unique(x_discrete)
        correct = 0

        for val in x_values:
            mask = x_discrete == val
            if np.sum(mask) == 0:
                continue

            # Find most frequent class for this value
            class_counts = np.bincount(y_encoded[mask], minlength=n_classes)
            predicted_class = np.argmax(class_counts)

            # Count correct predictions
            correct += class_counts[predicted_class]

        # Accuracy
        scores[i] = correct / n_samples

    return scores

def relief_f(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 10,
    n_samples: int = -1,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Compute ReliefF scores for each feature.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.
    n_neighbors : int, default=10
        Number of nearest neighbors to consider.
    n_samples : int, default=-1
        Number of instances to sample (-1 for all).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    scores : ndarray of shape (n_features,)
        ReliefF score for each feature.
    """
    n_total, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    rng = np.random.RandomState(random_state)

    # Determine number of samples to use
    if n_samples <= 0 or n_samples > n_total:
        sample_indices = np.arange(n_total)
    else:
        sample_indices = rng.choice(n_total, size=n_samples, replace=False)

    # Compute min/max for normalization
    min_vals = np.nanmin(X, axis=0)
    max_vals = np.nanmax(X, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1

    # Class probabilities
    class_probs = {}
    for c in classes:
        class_probs[c] = np.sum(y == c) / n_total

    # Initialize weights
    weights = np.zeros(n_features)

    # Process each sampled instance
    for idx in sample_indices:
        inst = X[idx]
        inst_class = y[idx]

        # Find nearest hits and misses
        for c in classes:
            class_mask = y == c
            class_indices = np.where(class_mask)[0]

            # Remove current instance if same class
            if c == inst_class:
                class_indices = class_indices[class_indices != idx]

            if len(class_indices) == 0:
                continue

            # Compute distances to all instances in this class
            class_X = X[class_indices]
            diffs = np.abs(class_X - inst)
            diffs = diffs / ranges
            diffs = np.nan_to_num(diffs, nan=1.0)
            distances = np.sum(diffs, axis=1)

            # Find k nearest neighbors
            k = min(n_neighbors, len(class_indices))
            nearest_idx = np.argsort(distances)[:k]
            nearest_instances = class_X[nearest_idx]

            # Compute attribute differences
            for j in range(n_features):
                diff = np.abs(inst[j] - nearest_instances[:, j])
                diff = diff / ranges[j]
                diff = np.nan_to_num(diff, nan=1.0)
                avg_diff = np.mean(diff)

                if c == inst_class:
                    # Hit: decrease weight if values are different
                    weights[j] -= avg_diff / len(sample_indices)
                else:
                    # Miss: increase weight if values are different
                    if n_classes == 2:
                        weights[j] += avg_diff / len(sample_indices)
                    else:
                        # Weight by class probability
                        p_miss = class_probs[c] / (1 - class_probs[inst_class])
                        weights[j] += p_miss * avg_diff / len(sample_indices)

    return weights
