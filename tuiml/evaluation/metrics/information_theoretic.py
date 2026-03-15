"""
Information-theoretic evaluation metrics.

These metrics are used by Weka for measuring information gain, entropy, etc.
Equivalent to Weka's information-theoretic evaluation methods.
"""

from typing import Optional
import numpy as np
from tuiml.base.metrics import safe_divide

def entropy(labels: np.ndarray, base: Optional[float] = None) -> float:
    """
    Calculate entropy of a label distribution.
    
    Equivalent to Weka's information-theoretic calculations.
    
    Args:
        labels: Array of class labels
        base: Logarithm base (None for natural log, 2 for bits)
        
    Returns:
        Entropy value
        
    Example:
        >>> entropy([0, 0, 1, 1])
        0.693...  # ln(2)
        >>> entropy([0, 0, 1, 1], base=2)
        1.0  # 1 bit
    """
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    
    probs = counts / len(labels)
    
    # Remove zero probabilities
    probs = probs[probs > 0]
    
    if base is None:
        return float(-np.sum(probs * np.log(probs)))
    elif base == 2:
        return float(-np.sum(probs * np.log2(probs)))
    else:
        return float(-np.sum(probs * np.log(probs) / np.log(base)))

def conditional_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base: Optional[float] = None
) -> float:
    """
    Calculate conditional entropy H(Y|X).
    
    Args:
        y_true: True labels (Y)
        y_pred: Predicted labels (X)
        base: Logarithm base
        
    Returns:
        Conditional entropy H(Y|X)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    cond_ent = 0.0
    
    for pred_label in np.unique(y_pred):
        mask = y_pred == pred_label
        p_x = np.sum(mask) / len(y_pred)
        
        if p_x > 0:
            h_y_given_x = entropy(y_true[mask], base=base)
            cond_ent += p_x * h_y_given_x
    
    return float(cond_ent)

def mutual_information(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base: Optional[float] = None
) -> float:
    """
    Calculate mutual information I(Y;X) = H(Y) - H(Y|X).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        base: Logarithm base
        
    Returns:
        Mutual information value
    """
    h_y = entropy(y_true, base=base)
    h_y_given_x = conditional_entropy(y_true, y_pred, base=base)
    
    return float(h_y - h_y_given_x)

def information_gain(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base: Optional[float] = 2
) -> float:
    """
    Calculate information gain (same as mutual information).
    
    Equivalent to Weka's information gain calculation.
    
    Args:
        y_true: True labels
        y_pred: Split/predicted labels
        base: Logarithm base (default 2 for bits)
        
    Returns:
        Information gain in bits (if base=2)
    """
    return mutual_information(y_true, y_pred, base=base)

def gain_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base: Optional[float] = 2
) -> float:
    """
    Calculate gain ratio (information gain normalized by split information).
    
    Equivalent to Weka's gain ratio used in C4.5/J48.
    
    Args:
        y_true: True labels
        y_pred: Split labels
        base: Logarithm base
        
    Returns:
        Gain ratio value
    """
    ig = information_gain(y_true, y_pred, base=base)
    split_info = entropy(y_pred, base=base)
    
    if split_info == 0:
        return 0.0
    
    return float(ig / split_info)

def kullback_leibler_divergence(
    y_true_proba: np.ndarray,
    y_pred_proba: np.ndarray,
    base: Optional[float] = None
) -> float:
    """
    Calculate Kullback-Leibler divergence KL(P||Q).
    
    Equivalent to Weka's KB Information metric.
    
    Args:
        y_true_proba: True probability distribution (P)
        y_pred_proba: Predicted probability distribution (Q)
        base: Logarithm base
        
    Returns:
        KL divergence value
        
    Example:
        >>> kullback_leibler_divergence([0.5, 0.5], [0.4, 0.6])
        0.020...
    """
    p = np.asarray(y_true_proba)
    q = np.asarray(y_pred_proba)
    
    # Clip to avoid log(0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    
    if base is None:
        return float(np.sum(p * np.log(p / q)))
    elif base == 2:
        return float(np.sum(p * np.log2(p / q)))
    else:
        return float(np.sum(p * np.log(p / q) / np.log(base)))

def jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: Optional[float] = None
) -> float:
    """
    Calculate Jensen-Shannon divergence (symmetric version of KL divergence).
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        base: Logarithm base
        
    Returns:
        JS divergence value
    """
    p = np.asarray(p)
    q = np.asarray(q)
    
    m = (p + q) / 2.0
    
    kl_pm = kullback_leibler_divergence(p, m, base=base)
    kl_qm = kullback_leibler_divergence(q, m, base=base)
    
    return float((kl_pm + kl_qm) / 2.0)

def cross_entropy(
    y_true_proba: np.ndarray,
    y_pred_proba: np.ndarray,
    base: Optional[float] = None
) -> float:
    """
    Calculate cross-entropy H(P,Q) = H(P) + KL(P||Q).
    
    Args:
        y_true_proba: True probability distribution
        y_pred_proba: Predicted probability distribution
        base: Logarithm base
        
    Returns:
        Cross-entropy value
    """
    p = np.asarray(y_true_proba)
    q = np.asarray(y_pred_proba)
    
    # Clip to avoid log(0)
    q = np.clip(q, 1e-10, 1.0)
    
    if base is None:
        return float(-np.sum(p * np.log(q)))
    elif base == 2:
        return float(-np.sum(p * np.log2(q)))
    else:
        return float(-np.sum(p * np.log(q) / np.log(base)))

def symmetrical_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base: Optional[float] = 2
) -> float:
    """
    Calculate symmetrical uncertainty (normalized mutual information).
    
    Equivalent to Weka's SymmetricalUncertAttributeEval.
    
    SU(X,Y) = 2 * IG(Y|X) / (H(X) + H(Y))
    
    Args:
        y_true: True labels
        y_pred: Predicted/feature labels
        base: Logarithm base
        
    Returns:
        Symmetrical uncertainty in range [0, 1]
    """
    ig = information_gain(y_true, y_pred, base=base)
    h_true = entropy(y_true, base=base)
    h_pred = entropy(y_pred, base=base)
    
    if h_true + h_pred == 0:
        return 0.0
    
    return float(2.0 * ig / (h_true + h_pred))

def prior_entropy(
    y_true: np.ndarray,
    base: Optional[float] = 2
) -> float:
    """
    Calculate prior entropy (entropy of class distribution).
    
    Equivalent to Weka's Evaluation.SFPriorEntropy().
    
    Args:
        y_true: Class labels
        base: Logarithm base
        
    Returns:
        Prior entropy
    """
    return entropy(y_true, base=base)

def scheme_entropy(
    y_pred_proba: np.ndarray,
    base: Optional[float] = 2
) -> float:
    """
    Calculate scheme entropy (average entropy of predicted distributions).
    
    Equivalent to Weka's Evaluation.SFSchemeEntropy().
    
    Args:
        y_pred_proba: Predicted probability distributions, shape (n_samples, n_classes)
        base: Logarithm base
        
    Returns:
        Mean entropy of predictions
    """
    y_pred_proba = np.asarray(y_pred_proba)
    
    if y_pred_proba.ndim == 1:
        # Binary case
        y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T
    
    entropies = []
    for proba in y_pred_proba:
        proba = proba[proba > 0]  # Remove zeros
        if base is None:
            ent = -np.sum(proba * np.log(proba))
        elif base == 2:
            ent = -np.sum(proba * np.log2(proba))
        else:
            ent = -np.sum(proba * np.log(proba) / np.log(base))
        entropies.append(ent)
    
    return float(np.mean(entropies))

def entropy_gain(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    base: Optional[float] = 2
) -> float:
    """
    Calculate entropy gain (prior entropy - scheme entropy).
    
    Equivalent to Weka's Evaluation.SFEntropyGain().
    
    Args:
        y_true: True class labels
        y_pred_proba: Predicted probability distributions
        base: Logarithm base
        
    Returns:
        Entropy gain
    """
    prior_ent = prior_entropy(y_true, base=base)
    scheme_ent = scheme_entropy(y_pred_proba, base=base)
    
    return float(prior_ent - scheme_ent)

def kb_information(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    base: Optional[float] = 2
) -> float:
    """
    Calculate KB information (Kullback-Leibler information).
    
    Equivalent to Weka's Evaluation.KBInformation().
    
    This measures the information gain from using the learned model
    instead of the prior distribution.
    
    Args:
        y_true: True class labels
        y_pred_proba: Predicted probability distributions
        base: Logarithm base
        
    Returns:
        KB information value
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    if y_pred_proba.ndim == 1:
        y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T
    
    # Compute prior distribution
    _, counts = np.unique(y_true, return_counts=True)
    prior_proba = counts / len(y_true)
    
    kb_sum = 0.0
    
    for i, true_label in enumerate(y_true):
        pred_prob = y_pred_proba[i, int(true_label)]
        prior_prob = prior_proba[int(true_label)]
        
        # Clip to avoid log(0)
        pred_prob = max(pred_prob, 1e-10)
        prior_prob = max(prior_prob, 1e-10)
        
        if base is None:
            kb_sum += np.log(pred_prob / prior_prob)
        elif base == 2:
            kb_sum += np.log2(pred_prob / prior_prob)
        else:
            kb_sum += np.log(pred_prob / prior_prob) / np.log(base)
    
    return float(kb_sum / len(y_true))

def mean_kb_information(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    base: Optional[float] = 2
) -> float:
    """
    Calculate mean KB information per instance.
    
    Equivalent to Weka's Evaluation.SFMeanKBInformation().
    """
    return kb_information(y_true, y_pred_proba, base=base)
