"""Impurity criteria for tree splitting and node evaluation."""

from __future__ import annotations

import numpy as np


# =========================================================================
# Classification criteria
# =========================================================================

def gini_impurity(y: np.ndarray, n_classes: int = 0) -> float:
    """Compute Gini impurity of a label distribution.

    Parameters
    ----------
    y : np.ndarray
        Integer-encoded class labels.
    n_classes : int
        Total number of classes (used for minlength in bincount).

    Returns
    -------
    gini : float
        Gini impurity value in [0, 1].
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes)
    probs = counts / len(y)
    return float(1.0 - np.sum(probs ** 2))


def entropy(y: np.ndarray, n_classes: int = 0) -> float:
    """Compute Shannon entropy of a label distribution.

    Parameters
    ----------
    y : np.ndarray
        Integer-encoded class labels.
    n_classes : int
        Total number of classes.

    Returns
    -------
    entropy : float
        Shannon entropy in bits.
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes)
    probs = counts / len(y)
    nonzero = probs > 0
    return float(-np.sum(probs[nonzero] * np.log2(probs[nonzero])))


def entropy_from_counts(class_counts: dict, total: int) -> float:
    """Compute Shannon entropy from a class-count dictionary.

    Parameters
    ----------
    class_counts : dict
        Mapping from class label to count.
    total : int
        Total number of samples.

    Returns
    -------
    entropy : float
        Shannon entropy in bits.
    """
    if total == 0:
        return 0.0
    ent = 0.0
    for count in class_counts.values():
        if count > 0:
            p = count / total
            ent -= p * np.log2(p + 1e-10)
    return ent


def gain_ratio_score(
    parent_entropy: float,
    child_entropies: np.ndarray,
    child_sizes: np.ndarray,
    n_total: int,
) -> float:
    """Compute the C4.5 gain ratio from pre-computed entropies.

    Parameters
    ----------
    parent_entropy : float
        Entropy of the parent node.
    child_entropies : np.ndarray
        Entropy of each child partition.
    child_sizes : np.ndarray
        Number of samples in each child.
    n_total : int
        Total samples at the parent.

    Returns
    -------
    gain_ratio : float
        Gain ratio value.
    """
    if n_total == 0:
        return 0.0
    weights = child_sizes / n_total
    info_gain = parent_entropy - float(np.sum(weights * child_entropies))
    # Split information
    split_info = 0.0
    for w in weights:
        if w > 0:
            split_info -= w * np.log2(w + 1e-10)
    if split_info < 1e-10:
        return 0.0
    return info_gain / split_info


# =========================================================================
# Classification node impurity dispatcher
# =========================================================================

def classifier_node_impurity(
    y: np.ndarray, criterion: str, n_classes: int
) -> float:
    """Compute node impurity for a classifier.

    Parameters
    ----------
    y : np.ndarray
        Integer-encoded class labels.
    criterion : str
        ``"gini"``, ``"entropy"``, ``"log_loss"``, or ``"gain_ratio"``.
    n_classes : int
        Number of classes.

    Returns
    -------
    impurity : float
        Node impurity value.
    """
    if criterion == "gini":
        return gini_impurity(y, n_classes)
    else:
        return entropy(y, n_classes)


# =========================================================================
# Regression criteria
# =========================================================================

def squared_error(y: np.ndarray) -> float:
    """Compute mean squared error (variance) of target values.

    Parameters
    ----------
    y : np.ndarray
        Target values.

    Returns
    -------
    mse : float
        Mean squared error around the mean.
    """
    if len(y) == 0:
        return 0.0
    return float(np.mean((y - np.mean(y)) ** 2))


def friedman_mse(y_left: np.ndarray, y_right: np.ndarray, n_total: int) -> float:
    """Compute Friedman's improvement score for a split.

    Parameters
    ----------
    y_left : np.ndarray
        Left child target values.
    y_right : np.ndarray
        Right child target values.
    n_total : int
        Total number of samples.

    Returns
    -------
    score : float
        Friedman MSE improvement.
    """
    n_left = len(y_left)
    n_right = len(y_right)
    if n_left == 0 or n_right == 0 or n_total == 0:
        return 0.0
    return (n_left * n_right / n_total ** 2) * (np.mean(y_left) - np.mean(y_right)) ** 2


def absolute_error(y: np.ndarray) -> float:
    """Compute mean absolute error around the median.

    Parameters
    ----------
    y : np.ndarray
        Target values.

    Returns
    -------
    mae : float
        Mean absolute error.
    """
    if len(y) == 0:
        return 0.0
    return float(np.mean(np.abs(y - np.median(y))))


def sdr(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute Standard Deviation Reduction (M5P criterion).

    Parameters
    ----------
    y : np.ndarray
        Parent node target values.
    y_left : np.ndarray
        Left child target values.
    y_right : np.ndarray
        Right child target values.

    Returns
    -------
    sdr_val : float
        Standard deviation reduction.
    """
    n = len(y)
    if n == 0 or len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    std_total = np.std(y, ddof=1) if len(y) > 1 else 0.0
    std_left = np.std(y_left, ddof=1) if len(y_left) > 1 else 0.0
    std_right = np.std(y_right, ddof=1) if len(y_right) > 1 else 0.0
    return std_total - (len(y_left) / n) * std_left - (len(y_right) / n) * std_right


# =========================================================================
# Regression node impurity dispatcher
# =========================================================================

def regressor_node_impurity(y: np.ndarray, criterion: str) -> float:
    """Compute node impurity for a regressor.

    Parameters
    ----------
    y : np.ndarray
        Target values.
    criterion : str
        ``"squared_error"``, ``"friedman_mse"``, or ``"absolute_error"``.

    Returns
    -------
    impurity : float
        Node impurity value.
    """
    if criterion in ("squared_error", "friedman_mse"):
        return squared_error(y)
    else:
        return absolute_error(y)
