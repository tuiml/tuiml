"""
Information-theoretic evaluation metrics.

Entropy, information gain, and the information-score family used to measure how
much a model's predictions reduce uncertainty about the target. Logarithms
default to base 2, so results are expressed in bits.
"""

from typing import Optional
import numpy as np
from tuiml.base.metrics import safe_divide

def entropy(labels: np.ndarray, base: Optional[float] = None) -> float:
    """
    Calculate entropy of a label distribution, the average information
    content (uncertainty) of the labels.

    .. math::
        H(X) = -\\sum_i p_i \\log p_i

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Array of class labels.
    base : int or None, default=None
        Logarithm base. None gives natural log (nats), 2 gives bits.

    Returns
    -------
    value : float
        Entropy in nats when ``base`` is None, or in bits when ``base=2``.
        Non-negative; 0 when all labels are identical, maximal when labels
        are uniformly distributed.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import entropy
    >>> round(entropy([0, 0, 1, 1]), 4)          # ln(2) nats
    0.6931
    >>> entropy([0, 0, 1, 1], base=2)            # 1 bit
    1.0
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
    Calculate conditional entropy H(Y|X), the remaining uncertainty in Y once X is known.

    .. math::
        H(Y|X) = \\sum_x p(x) H(Y|X=x)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels (Y).
    y_pred : array-like of shape (n_samples,)
        Predicted or feature labels (X), used to partition the samples.
    base : int or None, default=None
        Logarithm base. None gives natural log (nats), 2 gives bits.

    Returns
    -------
    value : float
        Conditional entropy H(Y|X), in the units of ``base``. Ranges from 0
        (X fully determines Y) up to H(Y) (X and Y are independent).

    Examples
    --------
    >>> from tuiml.evaluation.metrics import conditional_entropy
    >>> round(conditional_entropy([0, 0, 1, 1], [0, 0, 1, 1], base=2), 4)  # X determines Y
    0.0
    >>> round(conditional_entropy([0, 0, 1, 1], [0, 1, 0, 1], base=2), 4)  # X independent of Y
    1.0
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
    Calculate mutual information I(Y;X) = H(Y) - H(Y|X), the reduction in
    uncertainty about Y from observing X.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels (Y).
    y_pred : array-like of shape (n_samples,)
        Predicted or feature labels (X).
    base : int or None, default=None
        Logarithm base. None gives natural log (nats), 2 gives bits.

    Returns
    -------
    value : float
        Mutual information, in the units of ``base``. Non-negative; 0 when
        X and Y are independent, and equal to H(Y) when X fully determines Y.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import mutual_information
    >>> round(mutual_information([0, 0, 1, 1], [0, 0, 1, 1], base=2), 4)  # X determines Y
    1.0
    >>> round(mutual_information([0, 0, 1, 1], [0, 1, 0, 1], base=2), 4)  # X independent of Y
    0.0
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
    Calculate information gain, numerically identical to mutual information.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Split or predicted labels.
    base : int or None, default=2
        Logarithm base. 2 gives bits (the default), None gives natural log (nats).

    Returns
    -------
    value : float
        Information gain, in bits when ``base=2``. Non-negative; 0 means the
        split carries no information about ``y_true``.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import information_gain
    >>> round(information_gain([0, 0, 1, 1], [0, 0, 1, 1]), 4)  # split matches labels exactly
    1.0
    """
    return mutual_information(y_true, y_pred, base=base)

def gain_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base: Optional[float] = 2
) -> float:
    """
    Calculate gain ratio, information gain normalized by split information.

    Introduced by Quinlan for C4.5 to counteract information gain's bias
    toward high-cardinality splits, and defined as

    .. math::
        \\text{GainRatio} = \\frac{IG(Y, X)}{H(X)}

    which corrects information gain's bias toward splits with many values.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Split labels.
    base : int or None, default=2
        Logarithm base used for both the information gain and the split
        entropy. 2 gives bits, None gives natural log (nats).

    Returns
    -------
    value : float
        Gain ratio, typically in [0, 1]. Returns 0.0 when the split entropy
        is 0 (a single-valued split) to avoid division by zero.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import gain_ratio
    >>> round(gain_ratio([0, 0, 1, 1], [0, 0, 1, 1]), 4)  # split matches labels exactly
    1.0
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
    Calculate Kullback-Leibler divergence KL(P||Q), the expected extra
    information needed to encode samples from P using a code optimized
    for Q instead of P.

    .. math::
        KL(P \\| Q) = \\sum_i p_i \\log \\frac{p_i}{q_i}

    Parameters
    ----------
    y_true_proba : array-like of shape (n_classes,)
        True probability distribution (P).
    y_pred_proba : array-like of shape (n_classes,)
        Predicted probability distribution (Q). Both distributions are
        clipped to [1e-10, 1] to avoid ``log(0)``.
    base : int or None, default=None
        Logarithm base. None gives natural log (nats), 2 gives bits.

    Returns
    -------
    divergence : float
        Kullback-Leibler divergence, always >= 0 and 0 only when the two
        distributions are identical. Not symmetric in its arguments.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import kullback_leibler_divergence
    >>> round(kullback_leibler_divergence([0.5, 0.5], [0.4, 0.6]), 4)
    0.0204
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
    Calculate Jensen-Shannon divergence, a smoothed and symmetric version of
    KL divergence.

    .. math::
        JSD(P \\| Q) = \\frac{1}{2} KL(P \\| M) + \\frac{1}{2} KL(Q \\| M),
        \\quad M = \\frac{1}{2}(P + Q)

    Unlike KL divergence, JSD is symmetric in P and Q, always finite, and its
    square root is a proper metric.

    Parameters
    ----------
    p : array-like of shape (n_classes,)
        First probability distribution.
    q : array-like of shape (n_classes,)
        Second probability distribution.
    base : int or None, default=None
        Logarithm base. None gives natural log (nats), 2 gives bits.

    Returns
    -------
    value : float
        Jensen-Shannon divergence, bounded in [0, log(2)] (nats) or [0, 1]
        (bits). 0 when the two distributions are identical.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import jensen_shannon_divergence
    >>> round(jensen_shannon_divergence([0.5, 0.5], [0.5, 0.5]), 4)  # identical distributions
    0.0
    >>> round(jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0], base=2), 4)  # disjoint, base 2
    1.0
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
    Calculate cross-entropy H(P,Q) = H(P) + KL(P||Q), the average number of
    units needed to identify an event drawn from P when using a code
    optimized for Q.

    Parameters
    ----------
    y_true_proba : array-like of shape (n_classes,)
        True probability distribution (P).
    y_pred_proba : array-like of shape (n_classes,)
        Predicted probability distribution (Q). Clipped to [1e-10, 1] to
        avoid ``log(0)``.
    base : int or None, default=None
        Logarithm base. None gives natural log (nats), 2 gives bits.

    Returns
    -------
    value : float
        Cross-entropy, in the units of ``base``. Always >= H(P), with
        equality only when Q equals P.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import cross_entropy
    >>> round(cross_entropy([0.5, 0.5], [0.5, 0.5]), 4)  # Q equals P: reduces to H(P)
    0.6931
    >>> round(cross_entropy([1.0, 0.0], [0.9, 0.1], base=2), 4)
    0.152
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
    Calculate symmetrical uncertainty, a normalized, symmetric variant of
    mutual information.

    .. math::
        SU(X,Y) = \\frac{2 \\cdot IG(Y,X)}{H(X) + H(Y)}

    Normalizing by H(X) + H(Y) compensates for mutual information's bias
    toward variables with more distinct values.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted or feature labels.
    base : int or None, default=2
        Logarithm base used for the underlying entropy and information gain
        calculations. Cancels out in the ratio, but must be consistent
        across both terms.

    Returns
    -------
    value : float
        Symmetrical uncertainty, in range [0, 1]. 0 means independence, 1
        means each variable fully determines the other. Returns 0.0 when
        H(X) + H(Y) is 0.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import symmetrical_uncertainty
    >>> round(symmetrical_uncertainty([0, 0, 1, 1], [0, 0, 1, 1]), 4)  # X determines Y
    1.0
    >>> round(symmetrical_uncertainty([0, 0, 1, 1], [0, 1, 0, 1]), 4)  # X independent of Y
    0.0
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
    Calculate prior entropy, the entropy of the class distribution before
    any model prediction is taken into account.

    A thin wrapper around :func:`entropy` that defaults to base 2.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Class labels.
    base : int or None, default=2
        Logarithm base. 2 gives bits (the default), None gives natural log (nats).

    Returns
    -------
    value : float
        Prior entropy, in bits when ``base=2``. Non-negative; 0 when all
        labels are identical.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import prior_entropy
    >>> round(prior_entropy([0, 0, 1, 1]), 4)  # balanced two-class labels: 1 bit
    1.0
    """
    return entropy(y_true, base=base)

def prediction_entropy(
    y_pred_proba: np.ndarray,
    base: Optional[float] = 2
) -> float:
    """
    Calculate prediction entropy, the average entropy of the model's predicted
    class distributions.

    Low prediction entropy means the model makes confident (low-entropy), peaked
    predictions; high prediction entropy means predictions are close to uniform.

    Parameters
    ----------
    y_pred_proba : array-like of shape (n_samples, n_classes)
        Predicted probability distributions, one row per sample. A 1D input
        of shape (n_samples,) is treated as binary positive-class
        probabilities and expanded to two columns.
    base : int or None, default=2
        Logarithm base. 2 gives bits (the default), None gives natural log (nats).

    Returns
    -------
    value : float
        Mean per-sample entropy, in bits when ``base=2``. Non-negative;
        0 when every prediction is fully confident (a one-hot distribution).

    Examples
    --------
    >>> from tuiml.evaluation.metrics import prediction_entropy
    >>> round(prediction_entropy([[0.9, 0.1], [0.5, 0.5]]), 4)  # confident + maximally uncertain
    0.7345
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
    Calculate entropy gain, the reduction in entropy from using the model's
    predictions instead of the prior class distribution.

    Computed as :func:`prior_entropy` minus :func:`prediction_entropy`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred_proba : array-like of shape (n_samples, n_classes)
        Predicted probability distributions, one row per sample. A 1D input
        is treated as binary positive-class probabilities.
    base : int or None, default=2
        Logarithm base used for both the prior and prediction entropy terms.
        2 gives bits (the default), None gives natural log (nats).

    Returns
    -------
    value : float
        Entropy gain, in bits when ``base=2``. Positive when the model's
        predictions are more confident (lower entropy) than the prior;
        can be negative if the model is less confident than the prior.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import entropy_gain
    >>> y_true = [0, 0, 1, 1]
    >>> y_pred_proba = [[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]]
    >>> round(entropy_gain(y_true, y_pred_proba), 4)  # confident, correct predictions
    0.531
    """
    prior_ent = prior_entropy(y_true, base=base)
    pred_ent = prediction_entropy(y_pred_proba, base=base)
    
    return float(prior_ent - pred_ent)

def kb_information(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    base: Optional[float] = 2
) -> float:
    """
    Calculate KB (Kononenko-Bratko) information, the average per-instance
    information gained by using the model's predicted probability for the
    true class instead of the prior probability of that class.

    For each instance, the log-ratio of the predicted probability of the true
    class to the prior probability of that class is accumulated and averaged.

    References
    ----------
    .. [Kononenko1991] Kononenko, I. and Bratko, I. (1991).
           **Information-Based Evaluation Criterion for Classifier's
           Performance.** *Machine Learning*, 6(1), 67-80.
           DOI: `10.1007/BF00153760 <https://doi.org/10.1007/BF00153760>`_

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels, used as integer indices into ``y_pred_proba``'s
        columns and to estimate the prior class distribution.
    y_pred_proba : array-like of shape (n_samples, n_classes)
        Predicted probability distributions, one row per sample. A 1D input
        is treated as binary positive-class probabilities. Probabilities
        are clipped to a minimum of 1e-10 to avoid ``log(0)``.
    base : int or None, default=2
        Logarithm base. 2 gives bits (the default), None gives natural log (nats).

    Returns
    -------
    value : float
        Mean KB information per instance, in bits when ``base=2``. Positive
        when predictions assign more probability to the true class than the
        prior does; negative when predictions are worse than the prior.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import kb_information
    >>> y_true = [0, 0, 1, 1]
    >>> y_pred_proba = [[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]]
    >>> round(kb_information(y_true, y_pred_proba), 4)  # confident, correct predictions
    0.848
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
