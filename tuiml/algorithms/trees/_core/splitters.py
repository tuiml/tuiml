"""Vectorized split search for tree building."""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def compute_max_features(max_features, n_features: int) -> int:
    """Compute actual number of features to consider at each split.

    Parameters
    ----------
    max_features : str, int, float, or None
        Feature selection strategy: ``'sqrt'``, ``'log2'``, int count,
        float fraction, or ``None`` for all features.
    n_features : int
        Total number of features.

    Returns
    -------
    k : int
        Number of features to sample at each split.
    """
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else:
            return n_features
    elif isinstance(max_features, float):
        return max(1, int(max_features * n_features))
    elif isinstance(max_features, int):
        return min(max_features, n_features)
    else:
        return n_features


def best_split_classifier(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    n_classes: int,
    min_samples_leaf: int,
    rng: np.random.RandomState,
    max_features: Optional[int] = None,
) -> Tuple[int, float, float]:
    """Find the best binary split for classification using vectorised cumulative sums.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Integer-encoded labels.
    criterion : str
        ``"gini"``, ``"entropy"``, ``"log_loss"``, or ``"gain_ratio"``.
    n_classes : int
        Number of classes.
    min_samples_leaf : int
        Minimum samples at a leaf.
    rng : np.random.RandomState
        Random number generator for tie-breaking.
    max_features : int or None
        Number of features to consider (None = all).

    Returns
    -------
    best_feature : int
        Index of the best feature (``-1`` if no valid split).
    best_threshold : float
        Threshold value for the best split.
    best_gain : float
        Impurity reduction of the best split.
    """
    from .criteria import classifier_node_impurity

    n_samples, n_features = X.shape
    parent_impurity = classifier_node_impurity(y, criterion, n_classes)
    best_gain = -np.inf
    best_feature = -1
    best_threshold = 0.0

    feature_order = rng.permutation(n_features)
    if max_features is not None and max_features < n_features:
        feature_order = feature_order[:max_features]

    for feat_idx in feature_order:
        col = X[:, feat_idx]
        sorted_indices = np.argsort(col)
        sorted_col = col[sorted_indices]
        sorted_y = y[sorted_indices]

        # One-hot encode sorted labels -> (n_samples, n_classes)
        one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
        one_hot[np.arange(n_samples), sorted_y] = 1.0

        # Cumulative class counts
        cum_counts = np.cumsum(one_hot, axis=0)
        total_counts = cum_counts[-1]

        left_counts = cum_counts[:-1]
        right_counts = total_counts - left_counts

        n_left = np.arange(1, n_samples, dtype=np.float64)
        n_right = n_samples - n_left

        left_probs = left_counts / n_left[:, None]
        right_probs = right_counts / n_right[:, None]

        if criterion == "gini":
            imp_left = 1.0 - np.sum(left_probs ** 2, axis=1)
            imp_right = 1.0 - np.sum(right_probs ** 2, axis=1)
        else:  # entropy / log_loss / gain_ratio
            safe_left = np.where(left_probs > 0, left_probs, 1.0)
            safe_right = np.where(right_probs > 0, right_probs, 1.0)
            imp_left = -np.sum(
                np.where(left_probs > 0, left_probs * np.log2(safe_left), 0.0),
                axis=1,
            )
            imp_right = -np.sum(
                np.where(right_probs > 0, right_probs * np.log2(safe_right), 0.0),
                axis=1,
            )

        gains = (
            parent_impurity
            - (n_left / n_samples) * imp_left
            - (n_right / n_samples) * imp_right
        )

        if criterion == "gain_ratio":
            p_left = n_left / n_samples
            p_right = n_right / n_samples
            split_info = -(p_left * np.log2(p_left) + p_right * np.log2(p_right))
            gains = np.where(split_info > 0, gains / split_info, -np.inf)

        # Mask invalid splits
        valid = sorted_col[1:] != sorted_col[:-1]
        valid &= n_left >= min_samples_leaf
        valid &= n_right >= min_samples_leaf
        gains = np.where(valid, gains, -np.inf)

        if np.all(gains == -np.inf):
            continue

        idx = int(np.argmax(gains))
        if gains[idx] > best_gain:
            best_gain = gains[idx]
            best_feature = feat_idx
            best_threshold = (sorted_col[idx] + sorted_col[idx + 1]) / 2.0

    return best_feature, best_threshold, best_gain


def best_split_regressor(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    min_samples_leaf: int,
    rng: np.random.RandomState,
    max_features: Optional[int] = None,
) -> Tuple[int, float, float]:
    """Find the best binary split for regression using vectorised cumulative sums.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target values.
    criterion : str
        ``"squared_error"``, ``"friedman_mse"``, or ``"absolute_error"``.
    min_samples_leaf : int
        Minimum samples at a leaf.
    rng : np.random.RandomState
        Random number generator for tie-breaking.
    max_features : int or None
        Number of features to consider (None = all).

    Returns
    -------
    best_feature : int
        Index of the best feature (``-1`` if no valid split).
    best_threshold : float
        Threshold value for the best split.
    best_gain : float
        Impurity reduction of the best split.
    """
    n_samples, n_features = X.shape
    parent_impurity = float(np.mean((y - np.mean(y)) ** 2))
    best_gain = -np.inf
    best_feature = -1
    best_threshold = 0.0

    feature_order = rng.permutation(n_features)
    if max_features is not None and max_features < n_features:
        feature_order = feature_order[:max_features]

    for feat_idx in feature_order:
        col = X[:, feat_idx]
        sorted_indices = np.argsort(col)
        sorted_col = col[sorted_indices]
        sorted_y = y[sorted_indices]

        cum_sum = np.cumsum(sorted_y)
        cum_sq_sum = np.cumsum(sorted_y ** 2)
        total_sum = cum_sum[-1]
        total_sq_sum = cum_sq_sum[-1]

        left_sum = cum_sum[:-1]
        left_sq_sum = cum_sq_sum[:-1]

        n_left = np.arange(1, n_samples, dtype=np.float64)
        n_right = n_samples - n_left

        right_sum = total_sum - left_sum
        right_sq_sum = total_sq_sum - left_sq_sum

        left_mean = left_sum / n_left
        right_mean = right_sum / n_right

        if criterion == "friedman_mse":
            gains = (n_left * n_right / n_samples ** 2) * (left_mean - right_mean) ** 2
        else:
            left_mse = left_sq_sum / n_left - left_mean ** 2
            right_mse = right_sq_sum / n_right - right_mean ** 2
            gains = (
                parent_impurity
                - (n_left / n_samples) * left_mse
                - (n_right / n_samples) * right_mse
            )

        # Mask invalid splits
        valid = sorted_col[1:] != sorted_col[:-1]
        valid &= n_left >= min_samples_leaf
        valid &= n_right >= min_samples_leaf
        gains = np.where(valid, gains, -np.inf)

        if np.all(gains == -np.inf):
            continue

        idx = int(np.argmax(gains))
        if gains[idx] > best_gain:
            best_gain = gains[idx]
            best_feature = feat_idx
            best_threshold = (sorted_col[idx] + sorted_col[idx + 1]) / 2.0

    return best_feature, best_threshold, best_gain


def best_split_stump(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[int, float, float, bool]:
    """Find the best single-level split for a decision stump.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Class labels.
    sample_weight : np.ndarray or None
        Sample weights.

    Returns
    -------
    best_feature : int
        Index of the best feature.
    best_threshold : float
        Best split threshold.
    best_error : float
        Weighted classification error.
    is_numeric : bool
        Always True for this implementation.
    """
    from collections import defaultdict

    n_samples, n_features = X.shape
    if sample_weight is None:
        sample_weight = np.ones(n_samples)

    best_error = float("inf")
    best_feature = 0
    best_threshold = 0.0

    for feat_idx in range(n_features):
        col = X[:, feat_idx].astype(float)
        sorted_indices = np.argsort(col)
        sorted_col = col[sorted_indices]
        sorted_y = y[sorted_indices]
        sorted_weights = sample_weight[sorted_indices]

        right_counts = defaultdict(float)
        left_counts = defaultdict(float)
        for i, cls in enumerate(sorted_y):
            right_counts[cls] += sorted_weights[i]

        total_weight = sum(right_counts.values())

        for i in range(len(sorted_col) - 1):
            cls = sorted_y[i]
            left_counts[cls] += sorted_weights[i]
            right_counts[cls] -= sorted_weights[i]

            if sorted_col[i] == sorted_col[i + 1]:
                continue

            left_total = sum(left_counts.values())
            right_total = sum(right_counts.values())

            left_error = (left_total - max(left_counts.values())) if left_total > 0 else 0
            right_error = (right_total - max(right_counts.values())) if right_total > 0 else 0

            total_error = (left_error + right_error) / total_weight
            if total_error < best_error:
                best_error = total_error
                best_feature = feat_idx
                best_threshold = (sorted_col[i] + sorted_col[i + 1]) / 2

    return best_feature, best_threshold, best_error, True
