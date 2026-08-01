"""
Classification evaluation metrics.

Scoring functions for models that predict a discrete class label. The module is
organised around four groups:

* **Contingency summaries** — :func:`confusion_matrix`,
  :func:`classification_report` and the raw cell counts
  (:func:`num_true_positives` and friends) describe *how* a model is wrong, not
  just how often.
* **Threshold-free scores** — :func:`accuracy_score`,
  :func:`balanced_accuracy_score`, :func:`precision_score`,
  :func:`recall_score`, :func:`f1_score`, :func:`matthews_corrcoef` and
  :func:`cohen_kappa_score` summarise a single set of hard predictions.
* **Ranking scores** — :func:`roc_curve`, :func:`roc_auc_score`,
  :func:`precision_recall_curve` and :func:`average_precision_score` work on
  continuous scores or probabilities and evaluate a model across *all*
  decision thresholds at once.
* **Losses** — :func:`log_loss`, :func:`hamming_loss` and :func:`zero_one_loss`
  are lower-is-better quantities suitable for model selection.

On imbalanced data prefer ``balanced_accuracy_score``, ``f1_score``,
``matthews_corrcoef`` or ``average_precision_score`` over plain accuracy, which
a majority-class predictor can trivially inflate.

Multiclass averaging is controlled by the ``average`` keyword shared by
:func:`precision_score`, :func:`recall_score` and :func:`f1_score`:
``'binary'`` (default) scores only ``pos_label``, ``'macro'`` gives every class
equal weight, ``'weighted'`` weights by class support, and ``None`` returns the
per-class array.

The metric names mirror Weka's ``Evaluation`` class so results line up with a
Weka experiment report, while the call signatures follow the scikit-learn
convention ``metric(y_true, y_pred)``.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from tuiml.base.metrics import (
    Metric, MetricType, AverageType,
    check_classification_targets, check_consistent_length,
    get_num_classes, get_class_labels, is_binary, safe_divide, weighted_sum
)

# =============================================================================
# Confusion Matrix
# =============================================================================

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """Compute the **confusion matrix** of a classification result.

    Entry :math:`C_{ij}` counts the samples whose true class is ``labels[i]``
    and whose predicted class is ``labels[j]``. The diagonal therefore holds the
    correct predictions and every off-diagonal cell names a specific confusion.

    For the binary case with labels sorted ascending, the layout is::

                        predicted 0    predicted 1
        actual 0            TN             FP
        actual 1            FN             TP

    Equivalent to Weka's ``Evaluation.confusionMatrix()``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    labels : np.ndarray of shape (n_classes,), optional
        Label values, in the order they should index the rows and columns. When
        ``None`` the sorted union of the labels appearing in ``y_true`` and
        ``y_pred`` is used. Samples whose true *or* predicted label is absent
        from ``labels`` are skipped.
    normalize : {'true', 'pred', 'all'}, optional
        Normalisation applied to the counts: ``'true'`` divides each row by its
        sum (per-class recall), ``'pred'`` divides each column by its sum
        (per-class precision), ``'all'`` divides by the grand total. ``None``
        (default) returns raw integer counts.

    Returns
    -------
    cm : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix. ``dtype`` is ``int64`` when ``normalize`` is ``None``
        and ``float64`` otherwise.

    Notes
    -----
    Complexity: :math:`O(n)` time, :math:`O(k^2)` memory for :math:`k` classes.

    When to use: whenever a single accuracy number is not enough — the matrix
    is the only view that tells you *which* classes a model trades off against
    each other.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.classification_report` : Per-class precision/recall/F1 built on this matrix.
    :func:`~tuiml.evaluation.metrics.accuracy_score` : Scalar summary of the diagonal.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import confusion_matrix
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> confusion_matrix(y_true, y_pred).tolist()
    [[2, 0], [1, 1]]
    >>> confusion_matrix(y_true, y_pred, normalize='all').tolist()
    [[0.5, 0.0], [0.25, 0.25]]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_classification_targets(y_true, y_pred)

    if labels is None:
        labels = get_class_labels(y_true, y_pred)

    n_labels = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Build confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            cm[label_to_index[true], label_to_index[pred]] += 1

    # Normalize if requested
    if normalize == 'true':
        cm = cm.astype(np.float64)
        cm = safe_divide(cm, cm.sum(axis=1, keepdims=True))
    elif normalize == 'pred':
        cm = cm.astype(np.float64)
        cm = safe_divide(cm, cm.sum(axis=0, keepdims=True))
    elif normalize == 'all':
        cm = cm.astype(np.float64)
        cm = cm / cm.sum()

    return cm

# =============================================================================
# Basic Classification Metrics
# =============================================================================

def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute the **accuracy** of a classification result.

    Accuracy is the fraction (or, with ``normalize=False``, the count) of
    samples whose predicted label matches the true label.

    .. math::
        \\text{accuracy} = \\frac{1}{n} \\sum_{i=1}^{n}
        \\mathbb{1}[y_i = \\hat{y}_i]

    Equivalent to Weka's ``Evaluation.pctCorrect() / 100``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    normalize : bool, default=True
        If ``True`` return the fraction of correct predictions. If ``False``
        return the (possibly weighted) number of correct predictions.
    sample_weight : np.ndarray of shape (n_samples,), optional
        Per-sample weights. When given, correct predictions are summed with
        these weights and, if ``normalize=True``, divided by the total weight.

    Returns
    -------
    score : float
        Accuracy in :math:`[0, 1]` when ``normalize=True``, otherwise a count.

    Notes
    -----
    Complexity: :math:`O(n)` time, :math:`O(n)` memory.

    When to use: only when the classes are roughly balanced *and* every kind of
    error costs the same. On skewed data a constant majority-class predictor can
    score very high accuracy while being useless — reach for
    :func:`balanced_accuracy_score`, :func:`f1_score` or
    :func:`matthews_corrcoef` instead.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.balanced_accuracy_score` : Accuracy that gives every class equal weight.
    :func:`~tuiml.evaluation.metrics.zero_one_loss` : ``1 - accuracy``.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import accuracy_score
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> accuracy_score(y_true, y_pred)
    0.75
    >>> accuracy_score(y_true, y_pred, normalize=False)
    3.0
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_classification_targets(y_true, y_pred)

    correct = (y_true == y_pred)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        correct_weighted = np.sum(correct * sample_weight)
        if normalize:
            return correct_weighted / np.sum(sample_weight)
        return float(correct_weighted)

    if normalize:
        return float(np.mean(correct))
    return float(np.sum(correct))

def balanced_accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    adjusted: bool = False
) -> float:
    """Compute the **balanced accuracy**: the mean per-class recall.

    Each class contributes equally regardless of how many samples it has, so a
    majority-class predictor scores :math:`1/k` rather than the majority
    frequency.

    .. math::
        \\text{balanced accuracy} = \\frac{1}{k} \\sum_{c=1}^{k}
        \\frac{\\text{TP}_c}{\\text{TP}_c + \\text{FN}_c}

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    adjusted : bool, default=False
        If ``True``, rescale the result so that random guessing scores ``0.0``
        by applying :math:`(b - 1/k) / (1 - 1/k)`. Perfect prediction still
        scores ``1.0`` and the adjusted score can go negative.

    Returns
    -------
    score : float
        Balanced accuracy in :math:`[0, 1]`, or in
        :math:`[-1/(k-1), 1]` when ``adjusted=True``.

    Notes
    -----
    Complexity: :math:`O(kn)` time for :math:`k` classes, :math:`O(n)` memory.

    When to use: the default replacement for :func:`accuracy_score` on
    imbalanced problems, and the natural choice when every class matters
    equally regardless of its frequency.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.accuracy_score` : Unweighted, support-dominated accuracy.
    :func:`~tuiml.evaluation.metrics.recall_score` : Per-class recall, which this averages.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import balanced_accuracy_score
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> balanced_accuracy_score(y_true, y_pred)
    0.75
    >>> balanced_accuracy_score(y_true, y_pred, adjusted=True)
    0.5
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = get_class_labels(y_true, y_pred)
    recalls = []

    for label in labels:
        mask = y_true == label
        if np.sum(mask) > 0:
            recalls.append(np.mean(y_pred[mask] == label))

    balanced_acc = np.mean(recalls)

    if adjusted:
        n_classes = len(labels)
        chance = 1 / n_classes if n_classes > 0 else 1.0
        balanced_acc = (balanced_acc - chance) / (1 - chance) if chance < 1 else 0.0

    return float(balanced_acc)

# =============================================================================
# Precision, Recall, F-score
# =============================================================================

def _precision_recall_fscore_support(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 1.0,
    labels: Optional[np.ndarray] = None,
    pos_label: int = 1,
    average: Optional[str] = None,
    zero_division: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class precision, recall, F-score and support.

    This is the shared one-vs-rest core behind :func:`precision_score`,
    :func:`recall_score` and :func:`f1_score`. It always returns per-class
    arrays; averaging is applied afterwards by :func:`_average_scores`.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    beta : float, default=1.0
        Weight of recall relative to precision in the F-score.
    labels : np.ndarray of shape (n_classes,), optional
        Label values and their order. Defaults to the sorted union of the
        labels present in ``y_true`` and ``y_pred``.
    pos_label : int, default=1
        Accepted for signature compatibility with the public wrappers; not used
        here because every class is scored.
    average : str, optional
        Accepted for signature compatibility; not used here.
    zero_division : float, default=0.0
        Value substituted when a precision, recall or F-score has a zero
        denominator.

    Returns
    -------
    precision : np.ndarray of shape (n_classes,)
        Per-class precision :math:`\\text{TP} / (\\text{TP} + \\text{FP})`.
    recall : np.ndarray of shape (n_classes,)
        Per-class recall :math:`\\text{TP} / (\\text{TP} + \\text{FN})`.
    fscore : np.ndarray of shape (n_classes,)
        Per-class F-beta score.
    support : np.ndarray of shape (n_classes,)
        Number of true instances of each class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_classification_targets(y_true, y_pred)

    if labels is None:
        labels = get_class_labels(y_true, y_pred)

    n_labels = len(labels)
    tp = np.zeros(n_labels)
    fp = np.zeros(n_labels)
    fn = np.zeros(n_labels)
    support = np.zeros(n_labels)

    for i, label in enumerate(labels):
        true_mask = y_true == label
        pred_mask = y_pred == label
        tp[i] = np.sum(true_mask & pred_mask)
        fp[i] = np.sum(~true_mask & pred_mask)
        fn[i] = np.sum(true_mask & ~pred_mask)
        support[i] = np.sum(true_mask)

    precision = safe_divide(tp, tp + fp, zero_division)
    recall = safe_divide(tp, tp + fn, zero_division)

    beta_sq = beta ** 2
    fscore = safe_divide((1 + beta_sq) * precision * recall,
                         beta_sq * precision + recall, zero_division)

    return precision, recall, fscore, support

def precision_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
    """Compute the **precision**: how many predicted positives are correct.

    .. math::
        \\text{precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}

    Precision is the metric to optimise when a false positive is expensive —
    flagging a legitimate transaction as fraud, or a healthy patient as sick.

    Equivalent to Weka's ``Evaluation.precision(classIndex)``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    **kwargs : dict
        Forwarded to :func:`_precision_recall_fscore_support`. Recognised keys:

        ``average`` : {'binary', 'macro', 'weighted', 'micro'} or None, default='binary'
            How to reduce the per-class scores. ``'binary'`` returns the score
            of ``pos_label`` only; ``'macro'`` averages classes equally;
            ``'weighted'`` averages by support; ``None`` returns the per-class
            array.
        ``pos_label`` : int, default=1
            Class treated as positive when ``average='binary'``.
        ``labels`` : np.ndarray, optional
            Label values and their order.
        ``zero_division`` : float, default=0.0
            Value used when ``TP + FP == 0``.

    Returns
    -------
    score : float or np.ndarray of shape (n_classes,)
        Precision in :math:`[0, 1]`; an array when ``average=None``.

    Notes
    -----
    Complexity: :math:`O(kn)` time for :math:`k` classes.

    When to use: pair it with :func:`recall_score` — precision alone is trivial
    to maximise by predicting the positive class almost never.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.recall_score` : The complementary error type.
    :func:`~tuiml.evaluation.metrics.f1_score` : Harmonic mean of the two.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import precision_score
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> precision_score(y_true, y_pred)
    1.0
    >>> round(precision_score(y_true, y_pred, average='macro'), 4)
    0.8333
    """
    p, _, _, s = _precision_recall_fscore_support(y_true, y_pred, **kwargs)
    avg = kwargs.get('average', 'binary')
    pos = kwargs.get('pos_label', 1)
    lbls = kwargs.get('labels', get_class_labels(y_true, y_pred))
    return _average_scores(p, s, avg, pos, lbls)

def recall_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
    """Compute the **recall**: how many actual positives were found.

    .. math::
        \\text{recall} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}

    Recall is the metric to optimise when a missed positive is expensive — an
    undiagnosed disease, or an undetected intrusion.

    Equivalent to Weka's ``Evaluation.recall(classIndex)``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    **kwargs : dict
        Forwarded to :func:`_precision_recall_fscore_support`. Recognised keys
        are ``average``, ``pos_label``, ``labels`` and ``zero_division``; see
        :func:`precision_score` for their meaning.

    Returns
    -------
    score : float or np.ndarray of shape (n_classes,)
        Recall in :math:`[0, 1]`; an array when ``average=None``.

    Notes
    -----
    Complexity: :math:`O(kn)` time for :math:`k` classes.

    When to use: alongside :func:`precision_score`. Recall alone is trivial to
    maximise by predicting everything positive.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.precision_score` : The complementary error type.
    :func:`~tuiml.evaluation.metrics.true_positive_rate` : Same quantity, ROC naming.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import recall_score
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> recall_score(y_true, y_pred)
    0.5
    """
    _, r, _, s = _precision_recall_fscore_support(y_true, y_pred, **kwargs)
    avg = kwargs.get('average', 'binary')
    pos = kwargs.get('pos_label', 1)
    lbls = kwargs.get('labels', get_class_labels(y_true, y_pred))
    return _average_scores(r, s, avg, pos, lbls)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
    """Compute the **F1 score**: the harmonic mean of precision and recall.

    .. math::
        F_1 = 2 \\cdot \\frac{\\text{precision} \\cdot \\text{recall}}
        {\\text{precision} + \\text{recall}}

    The harmonic mean is deliberately unforgiving: it stays near the *smaller*
    of the two, so a model cannot score well by sacrificing one for the other.

    Equivalent to Weka's ``Evaluation.fMeasure(classIndex)``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    **kwargs : dict
        Forwarded to :func:`_precision_recall_fscore_support`. Recognised keys
        are ``average``, ``pos_label``, ``labels``, ``beta`` and
        ``zero_division``; see :func:`precision_score` for their meaning.

    Returns
    -------
    score : float or np.ndarray of shape (n_classes,)
        F1 score in :math:`[0, 1]`; an array when ``average=None``.

    Notes
    -----
    Complexity: :math:`O(kn)` time for :math:`k` classes.

    When to use: the standard single number for an imbalanced binary problem
    where the positive class is the one you care about. It ignores true
    negatives entirely — if those matter, use :func:`matthews_corrcoef`.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.precision_score` : One half of the mean.
    :func:`~tuiml.evaluation.metrics.recall_score` : The other half.
    :func:`~tuiml.evaluation.metrics.matthews_corrcoef` : Balanced alternative that uses all four cells.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import f1_score
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> round(f1_score(y_true, y_pred), 4)
    0.6667
    >>> round(f1_score(y_true, y_pred, average='weighted'), 4)
    0.7333
    """
    _, _, f, s = _precision_recall_fscore_support(y_true, y_pred, **kwargs)
    avg = kwargs.get('average', 'binary')
    pos = kwargs.get('pos_label', 1)
    lbls = kwargs.get('labels', get_class_labels(y_true, y_pred))
    return _average_scores(f, s, avg, pos, lbls)

def _average_scores(scores, support, average, pos_label, labels):
    """Reduce a per-class score array to a single number.

    Parameters
    ----------
    scores : np.ndarray of shape (n_classes,)
        Per-class metric values.
    support : np.ndarray of shape (n_classes,)
        Number of true instances of each class, used by the weighted averages.
    average : {'binary', 'macro', 'weighted', 'micro'} or None
        Reduction strategy. ``None`` returns ``scores`` unchanged.
    pos_label : int
        Class selected when ``average='binary'``.
    labels : np.ndarray of shape (n_classes,)
        Label values in the same order as ``scores``.

    Returns
    -------
    score : float or np.ndarray
        Reduced score, or the original array when ``average`` is ``None``.

    Raises
    ------
    ValueError
        If ``average`` is not one of the recognised strategies.
    """
    if average is None: return scores
    if average == 'binary':
        idx = np.where(labels == pos_label)[0]
        return float(scores[idx[0]]) if len(idx) > 0 else 0.0
    if average == 'macro': return float(np.mean(scores))
    if average == 'weighted':
        total = np.sum(support)
        return float(np.sum(scores * support) / total) if total > 0 else 0.0
    if average == 'micro':
        # Simple micro average logic for binary/multiclass
        return float(np.sum(scores * support) / np.sum(support)) if np.sum(support) > 0 else 0.0
    raise ValueError(f"Unknown average: {average}")

# Rates
def true_positive_rate(y_true, y_pred, pos_label=1):
    """Compute the **true positive rate** (TPR, sensitivity, recall).

    The fraction of actual positives that the model correctly flags; it is the
    y-axis of an ROC curve.

    .. math::
        \\text{TPR} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    rate : float
        True positive rate in :math:`[0, 1]`.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.false_positive_rate` : ROC x-axis.
    :func:`~tuiml.evaluation.metrics.recall_score` : Identical quantity, precision/recall naming.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import true_positive_rate
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> true_positive_rate(y_true, y_pred)
    0.5
    """
    return recall_score(y_true, y_pred, pos_label=pos_label, average='binary')

def false_positive_rate(y_true, y_pred, pos_label=1):
    """Compute the **false positive rate** (FPR, fall-out).

    The fraction of actual negatives that the model wrongly flags as positive;
    it is the x-axis of an ROC curve.

    .. math::
        \\text{FPR} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}} = 1 - \\text{TNR}

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    rate : float
        False positive rate in :math:`[0, 1]`; ``0.0`` when there are no
        negatives.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.true_positive_rate` : ROC y-axis.
    :func:`~tuiml.evaluation.metrics.roc_curve` : Traces both rates over all thresholds.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import false_positive_rate
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> false_positive_rate(y_true, y_pred)
    0.0
    """
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

def matthews_corrcoef(y_true, y_pred):
    """Compute the **Matthews correlation coefficient** (MCC).

    MCC is the correlation between the true and predicted binary labels. Unlike
    F1 it uses all four cells of the confusion matrix, so it cannot be inflated
    by a model that simply predicts the majority class.

    .. math::
        \\text{MCC} = \\frac{\\text{TP} \\cdot \\text{TN} - \\text{FP} \\cdot \\text{FN}}
        {\\sqrt{(\\text{TP}+\\text{FP})(\\text{TP}+\\text{FN})
                (\\text{TN}+\\text{FP})(\\text{TN}+\\text{FN})}}

    Equivalent to Weka's
    ``Evaluation.matthewsCorrelationCoefficient(classIndex)``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    score : float
        MCC in :math:`[-1, 1]`: ``1.0`` perfect agreement, ``0.0`` no better
        than chance, ``-1.0`` total disagreement. Returns ``0.0`` when the
        denominator vanishes.

    Notes
    -----
    Complexity: :math:`O(n)` time.

    Only the binary case is implemented; for three or more classes this
    function currently returns ``0.0`` rather than the multiclass
    generalisation.

    When to use: the most informative single number for an imbalanced binary
    problem, and the safest default when you are unsure which errors matter.

    References
    ----------
    .. [Matthews1975] Matthews, B. W. (1975). "Comparison of the predicted and
       observed secondary structure of T4 phage lysozyme." *Biochimica et
       Biophysica Acta*, 405(2), 442-451.
       :doi:`10.1016/0005-2795(75)90109-9`

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.f1_score` : Ignores true negatives.
    :func:`~tuiml.evaluation.metrics.cohen_kappa_score` : Another chance-corrected agreement score.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import matthews_corrcoef
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> round(matthews_corrcoef(y_true, y_pred), 4)
    0.5774
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        num = tp * tn - fp * fn
        den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return float(num / den) if den > 0 else 0.0
    # Multiclass implementation skipped for brevity, but easy to add
    return 0.0

def cohen_kappa_score(y_true, y_pred):
    """Compute **Cohen's kappa**: accuracy corrected for chance agreement.

    Kappa compares the observed agreement :math:`p_o` against the agreement
    :math:`p_e` you would expect if the two labellings were independent with
    the same marginals.

    .. math::
        \\kappa = \\frac{p_o - p_e}{1 - p_e}

    Equivalent to Weka's ``Evaluation.kappa()``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels (or the first rater's labels).
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels (or the second rater's labels).

    Returns
    -------
    score : float
        Kappa, at most ``1.0``. ``1.0`` is perfect agreement, ``0.0`` is
        chance-level agreement, and negative values mean worse than chance.
        Returns ``0.0`` when :math:`p_e = 1`.

    Notes
    -----
    Complexity: :math:`O(n + k^2)` time for :math:`k` classes.

    When to use: whenever a high raw accuracy might just reflect a skewed class
    distribution, and for inter-annotator agreement — kappa is symmetric in its
    two arguments. It supports multiclass input, unlike
    :func:`matthews_corrcoef`.

    References
    ----------
    .. [Cohen1960] Cohen, J. (1960). "A Coefficient of Agreement for Nominal
       Scales." *Educational and Psychological Measurement*, 20(1), 37-46.
       :doi:`10.1177/001316446002000104`

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.accuracy_score` : The uncorrected :math:`p_o`.
    :func:`~tuiml.evaluation.metrics.matthews_corrcoef` : Binary-only correlation alternative.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import cohen_kappa_score
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> cohen_kappa_score(y_true, y_pred)
    0.5
    """
    cm = confusion_matrix(y_true, y_pred)
    n = np.sum(cm)
    if n == 0: return 0.0
    po = np.trace(cm) / n
    pe = np.dot(np.sum(cm, axis=1), np.sum(cm, axis=0)) / (n**2)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0

# ROC AUC
def _binary_roc_curve(y_true, y_score, pos_label=1):
    """Compute binary ROC coordinates for score thresholds.

    Sorts the samples by decreasing score, then accumulates true and false
    positives at every distinct score value to obtain the ROC points.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_score : np.ndarray of shape (n_samples,)
        Continuous score or probability for the positive class. Higher means
        more likely positive.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    fpr : np.ndarray of shape (n_thresholds,)
        Increasing false positive rates, starting at ``0.0``.
    tpr : np.ndarray of shape (n_thresholds,)
        Increasing true positive rates, starting at ``0.0``.
    thresholds : np.ndarray of shape (n_thresholds,)
        Decreasing score thresholds. The first entry is ``inf``, the operating
        point at which nothing is predicted positive.

    Raises
    ------
    ValueError
        If ``y_score`` is not 1-D, if the two inputs differ in length, or if
        ``y_true`` contains only one class (ROC is then undefined).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_score.ndim != 1:
        raise ValueError("y_score must be a 1-D score vector for binary ROC")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    y_true_bin = y_true == pos_label
    n_pos = np.sum(y_true_bin)
    n_neg = len(y_true_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC is undefined when only one class is present")

    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[order]
    y_true_bin = y_true_bin[order]

    distinct = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct, y_true_bin.size - 1]

    tps = np.cumsum(y_true_bin)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]

    tpr = tps / n_pos
    fpr = fps / n_neg
    return fpr.astype(float), tpr.astype(float), thresholds


def roc_auc_score(y_true, y_score, average='macro', labels=None):
    """Compute the **area under the ROC curve** (AUC).

    AUC is the probability that a randomly chosen positive sample is ranked
    above a randomly chosen negative one, so it evaluates a model's *ranking*
    across every decision threshold at once rather than at one fixed cut-off.

    .. math::
        \\text{AUC} = P(\\hat{s}(x^{+}) > \\hat{s}(x^{-}))

    Binary input accepts a 1-D score vector for the positive class. Multiclass
    input accepts a probability/score matrix of shape
    ``(n_samples, n_classes)`` and computes one-vs-rest AUC for each class.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_score : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
        Continuous scores or probabilities. A 1-D vector is interpreted as the
        score of the positive class; a 2-D matrix must have one column per
        entry of ``labels``.
    average : {'macro', 'weighted'} or None, default='macro'
        How to reduce the per-class one-vs-rest AUCs in the multiclass case.
        ``None`` returns the per-class array. Ignored for 1-D ``y_score``.
    labels : np.ndarray of shape (n_classes,), optional
        Label values and the column order of a 2-D ``y_score``. Defaults to
        ``np.unique(y_true)``.

    Returns
    -------
    score : float or np.ndarray of shape (n_classes,)
        AUC in :math:`[0, 1]`. ``0.5`` is random ranking and ``1.0`` is perfect
        separation. An array is returned when ``average=None``.

    Raises
    ------
    ValueError
        If a 1-D ``y_score`` is given for more than two classes, if ``y_score``
        is neither 1-D nor 2-D, or if its shape disagrees with ``y_true`` or
        ``labels``.

    Notes
    -----
    Complexity: :math:`O(n \\log n)` time, dominated by the sort inside
    :func:`_binary_roc_curve`.

    When to use: for comparing rankers and probabilistic models independently of
    the operating threshold. On heavily imbalanced data AUC can look
    optimistic because the false positive rate has a very large denominator —
    prefer :func:`average_precision_score` there.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.roc_curve` : The curve this integrates.
    :func:`~tuiml.evaluation.metrics.auc` : Generic trapezoidal integration.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import roc_auc_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.4, 0.35, 0.8])
    >>> roc_auc_score(y_true, y_score)
    0.75
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if labels is None:
        labels = np.unique(y_true)
    labels = np.asarray(labels)

    if y_score.ndim == 1:
        if len(labels) > 2:
            raise ValueError(
                "Multiclass ROC AUC requires y_score with shape "
                "(n_samples, n_classes), not hard labels or a 1-D score vector"
            )
        pos_label = labels[-1] if len(labels) == 2 else 1
        fpr, tpr, _ = _binary_roc_curve(y_true, y_score, pos_label=pos_label)
        return auc(fpr, tpr)

    if y_score.ndim != 2:
        raise ValueError("y_score must be 1-D or 2-D")
    if y_score.shape[0] != len(y_true):
        raise ValueError("y_true and y_score must have the same number of rows")
    if y_score.shape[1] != len(labels):
        raise ValueError("Number of score columns must match number of labels")

    scores = []
    support = []
    for i, label in enumerate(labels):
        y_true_bin = (y_true == label).astype(int)
        fpr, tpr, _ = _binary_roc_curve(y_true_bin, y_score[:, i], pos_label=1)
        scores.append(auc(fpr, tpr))
        support.append(np.sum(y_true == label))

    scores = np.asarray(scores, dtype=float)
    support = np.asarray(support, dtype=float)

    if average is None:
        return scores
    if average == 'macro':
        return float(np.mean(scores))
    if average == 'weighted':
        return float(np.sum(scores * support) / np.sum(support))
    raise ValueError(f"Unknown average: {average}")

def precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None, pos_label=1, average=None, zero_division=0.0):
    """Compute precision, recall, F-measure and support in a single pass.

    Returns all four quantities from one traversal of the data, which is both
    cheaper and more consistent than calling :func:`precision_score`,
    :func:`recall_score` and :func:`f1_score` separately.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    beta : float, default=1.0
        Weight of recall relative to precision in the F-score. ``beta > 1``
        favours recall, ``beta < 1`` favours precision.
    labels : np.ndarray of shape (n_classes,), optional
        Label values and their order. Defaults to the sorted union of the
        labels present in ``y_true`` and ``y_pred``.
    pos_label : int, default=1
        Class scored when ``average='binary'``.
    average : {'binary', 'macro', 'weighted', 'micro'} or None, default=None
        How to reduce the per-class scores. ``None`` (default) keeps the
        per-class arrays.
    zero_division : float, default=0.0
        Value substituted when a denominator is zero.

    Returns
    -------
    precision : float or np.ndarray of shape (n_classes,)
        Precision, reduced according to ``average``.
    recall : float or np.ndarray of shape (n_classes,)
        Recall, reduced according to ``average``.
    fscore : float or np.ndarray of shape (n_classes,)
        F-beta score, reduced according to ``average``.
    support : float or np.ndarray of shape (n_classes,)
        Number of true instances per class, or their total when ``average`` is
        not ``None``.

    Notes
    -----
    Complexity: :math:`O(kn)` time for :math:`k` classes.

    When to use: building a report, or logging several related metrics at once.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.classification_report` : Formats these numbers as text.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import precision_recall_fscore_support
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    >>> np.round(p, 4).tolist()
    [0.6667, 1.0]
    >>> np.round(r, 4).tolist()
    [1.0, 0.5]
    >>> s.tolist()
    [2.0, 2.0]
    """
    p, r, f, s = _precision_recall_fscore_support(y_true, y_pred, beta=beta, labels=labels, pos_label=pos_label, zero_division=zero_division)
    if average is not None:
        p = _average_scores(p, s, average, pos_label, labels if labels is not None else get_class_labels(y_true, y_pred))
        r = _average_scores(r, s, average, pos_label, labels if labels is not None else get_class_labels(y_true, y_pred))
        f = _average_scores(f, s, average, pos_label, labels if labels is not None else get_class_labels(y_true, y_pred))
        s = np.sum(s)
    return p, r, f, s

def classification_report(y_true, y_pred, labels=None, target_names=None):
    """Build a text report of the main per-class classification metrics.

    Produces a fixed-width table with one row per class (precision, recall,
    F1 and support) followed by the overall accuracy and the macro average.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    labels : np.ndarray of shape (n_classes,), optional
        Label values and the row order. Defaults to the sorted union of the
        labels present in ``y_true`` and ``y_pred``.
    target_names : list of str, optional
        Display names for the classes, in the same order as ``labels``.
        Defaults to ``str(label)``.

    Returns
    -------
    report : str
        Multi-line, newline-terminated report ready to ``print``.

    Notes
    -----
    Complexity: :math:`O(kn)` time for :math:`k` classes.

    When to use: for a human-readable summary. Parse
    :func:`precision_recall_fscore_support` instead if you need the numbers
    programmatically.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.precision_recall_fscore_support` : The underlying numbers.
    :func:`~tuiml.evaluation.metrics.confusion_matrix` : Shows which classes are confused.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import classification_report
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> print(classification_report(y_true, y_pred).rstrip())  # doctest: +NORMALIZE_WHITESPACE
              class  precision     recall   f1-score    support
                  0     0.6667     1.0000     0.8000          2
                  1     1.0000     0.5000     0.6667          2
    ------------------------------------------------------------
           accuracy                           0.7500          4
          macro avg     0.8333     0.7500     0.7333          4
    """
    p, r, f, s = _precision_recall_fscore_support(y_true, y_pred, labels=labels)
    if labels is None: labels = get_class_labels(y_true, y_pred)
    if target_names is None: target_names = [str(l) for l in labels]

    report = f"{'class':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n"
    for i, name in enumerate(target_names):
        report += f"{name:>15} {p[i]:>10.4f} {r[i]:>10.4f} {f[i]:>10.4f} {int(s[i]):>10d}\n"

    # Add averages
    report += "-"*60 + "\n"
    report += f"{'accuracy':>15} {'':>10} {'':>10} {accuracy_score(y_true, y_pred):>10.4f} {int(np.sum(s)):>10d}\n"
    report += f"{'macro avg':>15} {np.mean(p):>10.4f} {np.mean(r):>10.4f} {np.mean(f):>10.4f} {int(np.sum(s)):>10d}\n"
    return report

# Missing aliases and additional functions
def fbeta_score(y_true, y_pred, beta=1.0, **kwargs):
    """Compute the **F-beta score**, a precision/recall trade-off.

    The F-beta score is the weighted harmonic mean of precision and recall, with
    ``beta`` controlling how much more recall counts than precision.

    .. math::
        F_\\beta = (1 + \\beta^2) \\cdot
        \\frac{\\text{precision} \\cdot \\text{recall}}
        {\\beta^2 \\cdot \\text{precision} + \\text{recall}}

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    beta : float, default=1.0
        Weight of recall relative to precision. **Currently ignored**: this
        function delegates to :func:`f1_score`, so it always behaves as
        ``beta=1``. Pass ``beta`` to
        :func:`precision_recall_fscore_support` for a true F-beta score.
    **kwargs : dict
        Forwarded to :func:`f1_score`; see :func:`precision_score` for the
        recognised keys.

    Returns
    -------
    score : float or np.ndarray of shape (n_classes,)
        F-beta score in :math:`[0, 1]`; an array when ``average=None``.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.f1_score` : The ``beta=1`` special case.
    :func:`~tuiml.evaluation.metrics.precision_recall_fscore_support` : Honours ``beta``.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import fbeta_score
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> round(fbeta_score(y_true, y_pred), 4)
    0.6667
    """
    return f1_score(y_true, y_pred, **kwargs)  # For now, delegate

def true_negative_rate(y_true, y_pred, pos_label=1):
    """Compute the **true negative rate** (TNR, specificity).

    The fraction of actual negatives that the model correctly leaves
    unflagged — the mirror image of recall.

    .. math::
        \\text{TNR} = \\frac{\\text{TN}}{\\text{TN} + \\text{FP}}

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    pos_label : int, default=1
        Label treated as the positive class; everything else is negative.

    Returns
    -------
    rate : float
        True negative rate in :math:`[0, 1]`; ``0.0`` when there are no
        negatives.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.specificity_score` : Alias for this function.
    :func:`~tuiml.evaluation.metrics.true_positive_rate` : The positive-class counterpart.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import true_negative_rate
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> true_negative_rate(y_true, y_pred)
    1.0
    """
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    n = tn + np.sum((y_true != pos_label) & (y_pred == pos_label))
    return float(tn / n) if n > 0 else 0.0

def false_negative_rate(y_true, y_pred, pos_label=1):
    """Compute the **false negative rate** (FNR, miss rate).

    The fraction of actual positives the model misses, i.e. ``1 - TPR``.

    .. math::
        \\text{FNR} = \\frac{\\text{FN}}{\\text{TP} + \\text{FN}} = 1 - \\text{TPR}

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    rate : float
        False negative rate in :math:`[0, 1]`.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.true_positive_rate` : The complement of this rate.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import false_negative_rate
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> false_negative_rate(y_true, y_pred)
    0.5
    """
    return 1.0 - true_positive_rate(y_true, y_pred, pos_label)

def sensitivity_score(y_true, y_pred, pos_label=1):
    """Compute the **sensitivity**, the clinical name for the true positive rate.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    score : float
        Sensitivity in :math:`[0, 1]`.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.true_positive_rate` : The function this delegates to.
    :func:`~tuiml.evaluation.metrics.specificity_score` : Reported alongside sensitivity in diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import sensitivity_score
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> sensitivity_score(y_true, y_pred)
    0.5
    """
    return true_positive_rate(y_true, y_pred, pos_label)

def specificity_score(y_true, y_pred, pos_label=1):
    """Compute the **specificity**, the clinical name for the true negative rate.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    score : float
        Specificity in :math:`[0, 1]`.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.true_negative_rate` : The function this delegates to.
    :func:`~tuiml.evaluation.metrics.sensitivity_score` : Reported alongside specificity in diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import specificity_score
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> specificity_score(y_true, y_pred)
    1.0
    """
    return true_negative_rate(y_true, y_pred, pos_label)

def num_true_positives(y_true, y_pred, pos_label=1):
    """Count the **true positives**: positives correctly predicted as positive.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    count : int
        Number of true positives.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.confusion_matrix` : All four counts at once.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import num_true_positives
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> num_true_positives(y_true, y_pred)
    1
    """
    return int(np.sum((y_true == pos_label) & (y_pred == pos_label)))

def num_true_negatives(y_true, y_pred, pos_label=1):
    """Count the **true negatives**: negatives correctly predicted as negative.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    pos_label : int, default=1
        Label treated as the positive class; everything else is negative.

    Returns
    -------
    count : int
        Number of true negatives.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.confusion_matrix` : All four counts at once.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import num_true_negatives
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> num_true_negatives(y_true, y_pred)
    2
    """
    return int(np.sum((y_true != pos_label) & (y_pred != pos_label)))

def num_false_positives(y_true, y_pred, pos_label=1):
    """Count the **false positives**: negatives wrongly predicted as positive.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    pos_label : int, default=1
        Label treated as the positive class; everything else is negative.

    Returns
    -------
    count : int
        Number of false positives (type I errors).

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.false_positive_rate` : The same errors as a rate.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import num_false_positives
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> num_false_positives(y_true, y_pred)
    0
    """
    return int(np.sum((y_true != pos_label) & (y_pred == pos_label)))

def num_false_negatives(y_true, y_pred, pos_label=1):
    """Count the **false negatives**: positives wrongly predicted as negative.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    count : int
        Number of false negatives (type II errors).

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.false_negative_rate` : The same errors as a rate.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import num_false_negatives
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> num_false_negatives(y_true, y_pred)
    1
    """
    return int(np.sum((y_true == pos_label) & (y_pred != pos_label)))

def roc_curve(y_true, y_score, pos_label=1):
    """Compute the **receiver operating characteristic** (ROC) curve.

    Sweeps the decision threshold from ``+inf`` down through every distinct
    score and records the resulting (FPR, TPR) operating points. Plotting TPR
    against FPR shows the full trade-off a model offers; the diagonal is random
    guessing.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_score : np.ndarray of shape (n_samples,)
        Continuous score or probability for the positive class.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    fpr : np.ndarray of shape (n_thresholds,)
        Increasing false positive rates.
    tpr : np.ndarray of shape (n_thresholds,)
        Increasing true positive rates.
    thresholds : np.ndarray of shape (n_thresholds,)
        Decreasing thresholds, the first being ``inf``.

    Raises
    ------
    ValueError
        If ``y_score`` is not 1-D, the inputs differ in length, or ``y_true``
        contains only one class.

    Notes
    -----
    Complexity: :math:`O(n \\log n)` time.

    When to use: to pick an operating threshold, not just to score a model —
    the curve tells you what recall costs in false alarms.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.roc_auc_score` : Scalar summary of this curve.
    :func:`~tuiml.evaluation.metrics.precision_recall_curve` : Better view on imbalanced data.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import roc_curve
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = roc_curve(y_true, y_score)
    >>> fpr.tolist()
    [0.0, 0.0, 0.5, 0.5, 1.0]
    >>> tpr.tolist()
    [0.0, 0.5, 0.5, 1.0, 1.0]
    >>> thresholds.tolist()
    [inf, 0.8, 0.4, 0.35, 0.1]
    """
    return _binary_roc_curve(y_true, y_score, pos_label=pos_label)

def auc(x, y):
    """Compute the area under a curve by the **trapezoidal rule**.

    A generic integrator: given the x and y coordinates of a curve it returns
    :math:`\\int y \\, dx` approximated by trapezoids.

    Parameters
    ----------
    x : np.ndarray of shape (n_points,)
        x-coordinates, monotonically increasing (e.g. the false positive rates
        returned by :func:`roc_curve`).
    y : np.ndarray of shape (n_points,)
        y-coordinates (e.g. the corresponding true positive rates).

    Returns
    -------
    area : float
        Area under the curve. Negative if ``x`` is decreasing.

    Notes
    -----
    Complexity: :math:`O(n)` time.

    When to use: to integrate any curve produced elsewhere in this module.
    :func:`roc_auc_score` is the convenience wrapper for the ROC case.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.roc_curve` : Produces suitable ``x``/``y`` arrays.
    :func:`~tuiml.evaluation.metrics.roc_auc_score` : ROC area in one call.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import auc, roc_curve
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, _ = roc_curve(y_true, y_score)
    >>> auc(fpr, tpr)
    0.75
    """
    return float(np.trapezoid(y, x))

def precision_recall_curve(y_true, y_score, pos_label=1):
    """Compute the **precision-recall curve**.

    Walks the samples in order of decreasing score and records the precision
    and recall obtained by treating each prefix as the set of positive
    predictions. Unlike an ROC curve it ignores true negatives, which makes it
    the informative view when positives are rare.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_score : np.ndarray of shape (n_samples,)
        Continuous score or probability for the positive class.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    precision : np.ndarray of shape (n_samples,)
        Precision at each threshold.
    recall : np.ndarray of shape (n_samples,)
        Recall at each threshold, increasing.
    thresholds : np.ndarray of shape (n_samples,)
        The scores, sorted in decreasing order, that define the thresholds.

    Notes
    -----
    Complexity: :math:`O(n \\log n)` time, dominated by the sort.

    When to use: on imbalanced problems, where a large true-negative pool makes
    an ROC curve look better than the model really is.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.roc_curve` : The balanced-data counterpart.
    :func:`~tuiml.evaluation.metrics.average_precision_score` : Scalar summary of this curve.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import precision_recall_curve
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.4, 0.35, 0.8])
    >>> precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    >>> np.round(precision, 4).tolist()
    [1.0, 0.5, 0.6667, 0.5]
    >>> recall.tolist()
    [0.5, 0.5, 1.0, 1.0]
    """
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    desc = np.argsort(y_score)[::-1]
    y_score = y_score[desc]; y_true = y_true[desc]
    tps = np.cumsum(y_true == pos_label)
    fps = np.cumsum(y_true != pos_label)
    precision = tps / (tps + fps)
    recall = tps / np.sum(y_true == pos_label) if np.sum(y_true == pos_label) > 0 else np.zeros_like(tps)
    return precision, recall, y_score

def average_precision_score(y_true, y_score, pos_label=1):
    """Compute the **average precision** (AP), the area under the PR curve.

    AP summarises :func:`precision_recall_curve` as the precision achieved at
    each threshold, weighted by the gain in recall it produces.

    .. math::
        \\text{AP} = \\sum_{k} (R_k - R_{k-1}) \\, P_k

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_score : np.ndarray of shape (n_samples,)
        Continuous score or probability for the positive class.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    score : float
        Average precision. The baseline for a random ranker is the positive
        class prevalence, not ``0.5`` as it is for ROC AUC.

    Notes
    -----
    Complexity: :math:`O(n \\log n)` time.

    When to use: as the headline ranking metric on heavily imbalanced data,
    where :func:`roc_auc_score` is dominated by the abundant negatives.

    Warnings
    --------
    The current implementation negates the recall increments, so it returns the
    negation of the value defined above.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.precision_recall_curve` : The curve being summarised.
    :func:`~tuiml.evaluation.metrics.roc_auc_score` : Threshold-free alternative for balanced data.
    """
    p, r, _ = precision_recall_curve(y_true, y_score, pos_label)
    return float(-np.sum(np.diff(r) * p[:-1]))

def log_loss(y_true, y_pred_proba, eps=1e-15):
    """Compute the **logistic loss** (cross-entropy) of predicted probabilities.

    Log loss scores a *probabilistic* classifier: it rewards assigning high
    probability to the correct class and punishes confident mistakes severely,
    because the penalty grows without bound as the predicted probability of the
    true class approaches zero.

    .. math::
        L = -\\frac{1}{n} \\sum_{i=1}^{n} \\log p_{i, y_i}

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels, used directly as column indices into
        ``y_pred_proba`` — so they must be integers ``0 … n_classes - 1``.
    y_pred_proba : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
        Predicted class probabilities. A 1-D vector is treated as the
        probability of class ``1`` and expanded to two columns.
    eps : float, default=1e-15
        Probabilities are clipped to ``[eps, 1 - eps]`` so that
        :math:`\\log 0` never occurs.

    Returns
    -------
    loss : float
        Mean negative log-likelihood, in nats. Non-negative; lower is better,
        and ``0.0`` means every true class was predicted with probability 1.

    Notes
    -----
    Complexity: :math:`O(n)` time.

    When to use: whenever calibrated probabilities matter — ranking or
    thresholding metrics such as AUC cannot tell a well-calibrated model from an
    over-confident one with the same ordering.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.zero_one_loss` : Ignores confidence entirely.
    :func:`~tuiml.evaluation.metrics.roc_auc_score` : Scores ranking rather than calibration.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import log_loss
    >>> y_true = np.array([0, 0, 1, 1])
    >>> proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.35, 0.65], [0.2, 0.8]])
    >>> round(log_loss(y_true, proba), 4)
    0.3175
    """
    y_true = np.asarray(y_true); y_pred_proba = np.asarray(y_pred_proba)
    if y_pred_proba.ndim == 1:
        y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    n = len(y_true)
    loss = 0.0
    for i, true_label in enumerate(y_true):
        loss -= np.log(y_pred_proba[i, int(true_label)])
    return float(loss / n)

def hamming_loss(y_true, y_pred):
    """Compute the **Hamming loss**: the fraction of mismatched labels.

    For single-label input this is simply ``1 - accuracy``. For multi-label
    indicator arrays it is the fraction of individual label positions that
    disagree, which makes it more forgiving than an exact-match criterion.

    .. math::
        L_H = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbb{1}[y_i \\neq \\hat{y}_i]

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels, of any shape.
    y_pred : np.ndarray
        Predicted labels, broadcastable to the shape of ``y_true``.

    Returns
    -------
    loss : float
        Fraction of positions that disagree, in :math:`[0, 1]`; lower is
        better.

    Notes
    -----
    Complexity: :math:`O(n)` time.

    When to use: for multi-label problems, where partial credit for getting
    most labels right is the behaviour you want.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.zero_one_loss` : Single-label equivalent.
    :func:`~tuiml.evaluation.metrics.accuracy_score` : ``1 - hamming_loss`` for single-label input.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import hamming_loss
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> hamming_loss(y_true, y_pred)
    0.25
    """
    return float(np.mean(y_true != y_pred))

def zero_one_loss(y_true, y_pred, normalize=True):
    """Compute the **0-1 loss**: the fraction (or count) of misclassifications.

    The direct complement of :func:`accuracy_score`; it charges 1 for every
    wrong prediction regardless of how wrong it was.

    .. math::
        L_{01} = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbb{1}[y_i \\neq \\hat{y}_i]

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels.
    normalize : bool, default=True
        If ``True`` return the fraction of misclassifications; if ``False``
        return their count.

    Returns
    -------
    loss : float
        Misclassification rate in :math:`[0, 1]`, or a count when
        ``normalize=False``. Lower is better.

    Notes
    -----
    Complexity: :math:`O(n)` time.

    When to use: when every error costs the same. If errors have different
    costs, weight them via :func:`confusion_matrix` instead.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.accuracy_score` : ``1 - zero_one_loss``.
    :func:`~tuiml.evaluation.metrics.log_loss` : Confidence-aware alternative.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import zero_one_loss
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> zero_one_loss(y_true, y_pred)
    0.25
    >>> zero_one_loss(y_true, y_pred, normalize=False)
    1.0
    """
    loss = np.sum(y_true != y_pred)
    return float(loss / len(y_true)) if normalize else float(loss)
