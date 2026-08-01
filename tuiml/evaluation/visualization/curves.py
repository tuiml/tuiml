"""
Diagnostic curves for a single fitted model: ROC, precision-recall, and
learning curves.

Where a metric collapses a model to one number, a curve shows the whole
trade-off. This module covers the three curves worth plotting for most projects:

- :func:`plot_roc_curve` — true positive rate against false positive rate as the
  decision threshold sweeps from strict to permissive. The area under it (AUC)
  is the probability that a random positive is scored above a random negative.
  Handles binary problems and, given a full probability matrix, multiclass
  one-vs-rest with a macro-average overlay.
- :func:`plot_pr_curve` — precision against recall over the same threshold
  sweep, summarised by average precision. Prefer this to ROC on **imbalanced**
  data: ROC looks flatteringly good when negatives vastly outnumber positives,
  because a large absolute number of false positives is still a small false
  positive *rate*.
- :func:`plot_learning_curve` — train and validation score as the training set
  grows. A validation curve still climbing at the right edge means more data
  would help; a wide, persistent gap between the two curves means overfitting,
  and two low curves converging means underfitting.

All three take arrays you already have — no model is refitted here. Each calls
``matplotlib.pyplot.show()`` and optionally writes a PNG via ``save_path``.
matplotlib is imported lazily, so every function raises :exc:`ImportError` when
it is missing.

See Also
--------
:func:`~tuiml.evaluation.metrics.roc_auc_score` : The scalar AUC without a plot.
:func:`~tuiml.evaluation.metrics.average_precision_score` : Scalar summary of
    the precision-recall curve.
:func:`~tuiml.evaluation.visualization.plot_confusion_matrix` : What happens at
    one specific threshold.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# NumPy 2.x compatibility: trapz was renamed to trapezoid
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
if _trapz is None:
    def _trapz(y, x):
        """Integrate ``y`` over ``x`` by the trapezoidal rule.

        Last-resort fallback used only when neither ``numpy.trapezoid``
        (NumPy >= 2.0) nor the legacy ``numpy.trapz`` is available.

        Parameters
        ----------
        y : ndarray of shape (n_points,)
            Values to integrate, sampled at ``x``.
        x : ndarray of shape (n_points,)
            Sample positions, assumed sorted ascending.

        Returns
        -------
        area : float
            Approximate area under the sampled curve.
        """
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ._style import get_colors, setup_figure, style_axis, SEMANTIC_COLORS
from tuiml.evaluation.metrics.classification import _binary_roc_curve, auc

def _roc_curve_binary(y_true_bin: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the ROC curve and its AUC for one binary label vector.

    Parameters
    ----------
    y_true_bin : ndarray of shape (n_samples,)
        Binary ground truth encoded as 0/1, where 1 is the positive class.
    y_score : ndarray of shape (n_samples,)
        Continuous score or probability for the positive class; larger means
        more positive.

    Returns
    -------
    fpr : ndarray of shape (n_thresholds,)
        False positive rate at each threshold, increasing from 0 to 1.
    tpr : ndarray of shape (n_thresholds,)
        True positive rate at the same thresholds.
    auc_score : float
        Area under the ``(fpr, tpr)`` curve.
    """
    fpr, tpr, _ = _binary_roc_curve(y_true_bin, y_score, pos_label=1)
    return fpr, tpr, auc(fpr, tpr)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = 'ROC Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: str = None,
    show_auc: bool = True,
    label: str = None,
    show_grid: bool = False,
    classes: Optional[List] = None,
):
    """Plot a Receiver Operating Characteristic curve, binary or one-vs-rest.

    The curve traces true positive rate against false positive rate as the
    decision threshold sweeps from "predict nothing positive" (bottom left) to
    "predict everything positive" (top right). A perfect ranker hugs the top-left
    corner; the dashed diagonal is what random guessing achieves. The shaded
    area under the curve is the AUC, and equals the probability that a randomly
    chosen positive sample is scored higher than a randomly chosen negative one,
    so 0.5 is chance and 1.0 is perfect.

    The plot is threshold-free — it says how well the model *ranks*, not how
    well any particular cut-off classifies. Because the false positive rate is
    normalised by the number of negatives, ROC stays optimistic on heavily
    imbalanced data; pair it with
    :func:`~tuiml.evaluation.visualization.plot_pr_curve` there.

    Two modes are selected automatically from the shape of ``y_score``. With a
    1-D score vector (or a 2-column probability matrix, whose second column is
    used) a single binary curve is drawn. With a probability matrix of three or
    more columns, one one-vs-rest curve is drawn per class plus a dotted
    macro-average curve interpolated on a common 200-point FPR grid.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True class labels, binary or multiclass. In the binary path, labels
        that are not already 0/1 are binarised against the largest label as the
        positive class.
    y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Positive-class probabilities (1-D) for binary problems, or per-class
        probabilities (2-D) as returned by ``predict_proba``. Predicted *labels*
        will not produce a meaningful curve.
    title : str, default='ROC Curve'
        Axis title. Title-cased when rendered; ``' (one-vs-rest)'`` is appended
        in the multiclass path.
    figsize : tuple of (float, float), default=(8, 6)
        Figure size in inches.
    save_path : str, optional
        If given, the figure is written to this path as a 300-dpi PNG with a
        tight bounding box before being shown.
    show_auc : bool, default=True
        Append ``(AUC = ...)`` to the legend entry. Binary path only.
    label : str, optional
        Legend text for the curve. Defaults to ``'ROC'``. Binary path only.
    show_grid : bool, default=False
        Currently has no effect — grid lines are always drawn.
    classes : list, optional
        Class labels in the column order of ``y_score``, used to label the
        per-class curves. Defaults to ``np.unique(y_true)``. Multiclass path
        only.

    Returns
    -------
    fpr : ndarray of shape (n_thresholds,)
        False positive rates. In the multiclass path this is the shared
        200-point grid the macro-average was computed on.
    tpr : ndarray of shape (n_thresholds,)
        Matching true positive rates, macro-averaged in the multiclass path.
    auc_score : float
        Area under the returned curve; the macro-average AUC in the multiclass
        path.

    Raises
    ------
    ImportError
        If matplotlib is not installed (it is imported lazily).
    ValueError
        If ``y_true`` has more than two classes but ``y_score`` is 1-D, or if
        ``len(classes)`` does not match the number of ``y_score`` columns.

    Notes
    -----
    Side effects: mutates the global matplotlib style, calls
    ``matplotlib.pyplot.show()`` before returning, and writes a file when
    ``save_path`` is given. The figure object is not returned. Use a
    non-interactive backend such as ``Agg`` for headless rendering.

    See Also
    --------
    :func:`~tuiml.evaluation.visualization.plot_pr_curve` : Better view of the
        same model on imbalanced data.
    :func:`~tuiml.evaluation.metrics.roc_auc_score` : The scalar AUC alone.
    :func:`~tuiml.evaluation.visualization.plot_confusion_matrix` : Behaviour at
        one chosen threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.visualization import plot_roc_curve
    >>> y_true = np.array([0, 0, 1, 1, 0, 1])
    >>> y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    >>> fpr, tpr, auc_score = plot_roc_curve(y_true, y_score)   # doctest: +SKIP

    Multiclass, straight from a fitted TuiML classifier:

    >>> from tuiml.datasets import load_iris
    >>> from tuiml.evaluation.splitting import train_test_split
    >>> from tuiml.algorithms import NaiveBayesClassifier
    >>> X, y = load_iris()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0)
    >>> clf = NaiveBayesClassifier().fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    >>> fpr, tpr, macro_auc = plot_roc_curve(
    ...     y_test, proba, classes=[0, 1, 2],
    ...     save_path='roc.png')                               # doctest: +SKIP
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    unique_labels = np.unique(y_true)

    if y_score.ndim == 1 and len(unique_labels) > 2:
        raise ValueError(
            "Multiclass ROC curves require class probabilities/scores with "
            "shape (n_samples, n_classes). A 1-D vector, such as predicted "
            "class labels, cannot produce meaningful ROC curves."
        )

    # ── Multiclass path (one-vs-rest) ────────────────────────────────
    if y_score.ndim == 2 and y_score.shape[1] > 2:
        if classes is None:
            classes = list(unique_labels)
        n_classes = y_score.shape[1]
        if len(classes) != n_classes:
            raise ValueError(
                "Length of classes must match the number of y_score columns"
            )
        colors = get_colors(n_classes)

        fig, ax = setup_figure(figsize=figsize)
        # Common FPR grid for macro-averaging
        all_fpr = np.linspace(0.0, 1.0, 200)
        mean_tpr = np.zeros_like(all_fpr)
        per_class_auc = []

        for k in range(n_classes):
            y_true_bin = (y_true == classes[k]).astype(int)
            fpr_k, tpr_k, auc_k = _roc_curve_binary(y_true_bin, y_score[:, k])
            per_class_auc.append(auc_k)
            ax.plot(fpr_k, tpr_k, lw=2.0, color=colors[k],
                    label=f'{classes[k]} (AUC = {auc_k:.3f})')
            mean_tpr += np.interp(all_fpr, fpr_k, tpr_k)

        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, lw=3.0, linestyle=':',
                color=SEMANTIC_COLORS.get('primary', 'k'),
                label=f'macro-avg (AUC = {macro_auc:.3f})')
        ax.plot([0, 1], [0, 1], '--', lw=1.5, color=SEMANTIC_COLORS['neutral'], label='Random')

        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
        style_axis(ax, title=f'{title} (one-vs-rest)',
                   xlabel='False Positive Rate', ylabel='True Positive Rate',
                   legend=True, legend_loc='lower right', grid=True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
        plt.show()
        return all_fpr, mean_tpr, macro_auc

    # ── Binary path ──────────────────────────────────────────────────
    # If a 2-column proba matrix was passed, use the positive-class column.
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        y_score = y_score[:, 1]

    # Normalise true labels to {0,1}
    uniq = np.unique(y_true)
    if not (set(uniq.tolist()) <= {0, 1}):
        pos_label = uniq[-1]
        y_true_bin = (y_true == pos_label).astype(int)
    else:
        y_true_bin = y_true.astype(int)

    fpr, tpr, auc_score = _roc_curve_binary(y_true_bin, y_score)

    colors = get_colors(2)
    fig, ax = setup_figure(figsize=figsize)
    if label is None:
        label = f'ROC (AUC = {auc_score:.3f})' if show_auc else 'ROC'
    elif show_auc:
        label = f'{label} (AUC = {auc_score:.3f})'

    ax.plot(fpr, tpr, lw=3.0, label=label, color=colors[0])
    ax.plot([0, 1], [0, 1], '--', lw=2.0, label='Random', color=SEMANTIC_COLORS['neutral'])
    ax.fill_between(fpr, tpr, alpha=0.2, color=colors[0])
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
    style_axis(ax, title=title, xlabel='False Positive Rate', ylabel='True Positive Rate',
               legend=True, legend_loc='lower right', grid=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.show()
    return fpr, tpr, auc_score

def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = 'Precision-Recall Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: str = None,
    show_ap: bool = True,
    label: str = None,
    show_grid: bool = False,
):
    """
    Plot a precision-recall curve for a binary classifier.

    Each point corresponds to one decision threshold: recall (x) is the fraction
    of true positives found, precision (y) is the fraction of positive
    predictions that were right. Lowering the threshold moves you right — more
    positives found — usually at the cost of precision, so the curve slopes down
    to the right. A model that is useful everywhere stays high across the whole
    width; a model that only works when it is very confident starts high and
    collapses.

    The dashed horizontal **baseline** is the positive class prevalence, which
    is what a random classifier achieves. On imbalanced data that line sits low,
    which is exactly why this plot is more honest than ROC: beating it is a real
    achievement, and the gap above it is visible at a glance. The shaded area is
    summarised by the average precision (AP) reported in the legend.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Binary ground truth encoded as 0/1, where 1 is the positive class.
    y_score : ndarray of shape (n_samples,)
        Predicted probability or score for the positive class.
    title : str, default='Precision-Recall Curve'
        Axis title; title-cased when rendered.
    figsize : tuple of (float, float), default=(8, 6)
        Figure size in inches.
    save_path : str, optional
        If given, the figure is written to this path as a 300-dpi PNG with a
        tight bounding box before being shown.
    show_ap : bool, default=True
        Append ``(AP = ...)`` to the legend entry.
    label : str, optional
        Legend text for the curve. Defaults to ``'PR'``.
    show_grid : bool, default=False
        Currently has no effect — grid lines are always drawn.

    Returns
    -------
    recall : ndarray of shape (n_points,)
        Recall values, sorted ascending.
    precision : ndarray of shape (n_points,)
        Precision at the same points.
    ap : float
        Average precision, computed as the trapezoidal area under the
        recall-sorted curve.

    Raises
    ------
    ImportError
        If matplotlib is not installed (it is imported lazily).

    Notes
    -----
    The curve is evaluated at every distinct value in ``y_score``, so cost is
    :math:`O(n^2)` in the number of unique scores — subsample before plotting
    very large score vectors.

    ``y_true`` must be 0/1: unlike :func:`plot_roc_curve`, no binarisation of
    other label encodings is performed, and arbitrary labels silently yield an
    all-zero curve.

    Side effects: mutates the global matplotlib style, calls
    ``matplotlib.pyplot.show()`` before returning, and writes a file when
    ``save_path`` is given. The figure object is not returned.

    See Also
    --------
    :func:`~tuiml.evaluation.visualization.plot_roc_curve` : The complementary
        threshold sweep.
    :func:`~tuiml.evaluation.metrics.average_precision_score` : The scalar AP
        alone.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.visualization import plot_pr_curve
    >>> y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    >>> y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.05, 0.3])
    >>> recall, precision, ap = plot_pr_curve(y_true, y_score)   # doctest: +SKIP
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    colors = get_colors(2)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Calculate PR curve
    thresholds = np.unique(y_score)
    thresholds = np.sort(thresholds)[::-1]

    precision_list = [1.0]
    recall_list = [0.0]

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    precision = np.array(precision_list)
    recall = np.array(recall_list)

    # Calculate Average Precision
    sorted_idx = np.argsort(recall)
    recall_sorted = recall[sorted_idx]
    precision_sorted = precision[sorted_idx]
    ap = _trapz(precision_sorted, recall_sorted)

    # Plot
    fig, ax = setup_figure(figsize=figsize)

    if label is None:
        label = f'PR (AP = {ap:.3f})' if show_ap else 'PR'
    elif show_ap:
        label = f'{label} (AP = {ap:.3f})'

    ax.plot(recall_sorted, precision_sorted, lw=3.0, label=label, color=colors[0])

    # Fill area under curve
    ax.fill_between(recall_sorted, precision_sorted, alpha=0.2, color=colors[0])

    # Baseline
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color=SEMANTIC_COLORS['neutral'], linestyle='--', lw=2.0,
               label=f'Baseline ({baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    style_axis(
        ax,
        title=title,
        xlabel='Recall',
        ylabel='Precision',
        legend=True,
        legend_loc='lower left',
        grid=True,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()

    return recall_sorted, precision_sorted, ap

def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    title: str = 'Learning Curve',
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None,
    metric_name: str = 'Score',
    show_std: bool = True,
    show_grid: bool = False,
):
    """
    Plot training and validation score as a function of training set size.

    Two curves are drawn against the number of training samples: training score
    (circles) and cross-validation score (squares), each optionally with a
    :math:`\\pm 1` standard deviation band across CV folds. This is the plot
    that tells you whether to collect more data or change the model:

    - Validation curve still rising at the right edge — **more data will help**.
    - Validation curve flat and a wide gap below the training curve —
      **overfitting**; regularise or simplify.
    - Both curves flat, close together and low — **underfitting**; the model is
      too simple or the features are too weak, and more data will not help.

    Nothing is fitted here: you supply the sizes and the scores, typically
    collected by refitting an estimator on growing subsets and scoring each fit
    with a splitter from :mod:`tuiml.evaluation.splitting`.

    Parameters
    ----------
    train_sizes : ndarray of shape (n_sizes,)
        Number of training samples used at each point, in increasing order;
        used directly as the x coordinates.
    train_scores : ndarray of shape (n_sizes,) or (n_sizes, n_splits)
        Training scores. If 2-D, the mean over axis 1 is plotted and the
        standard deviation becomes the shaded band.
    test_scores : ndarray of shape (n_sizes,) or (n_sizes, n_splits)
        Validation/test scores, same shape convention as ``train_scores``.
    title : str, default='Learning Curve'
        Axis title; title-cased when rendered.
    figsize : tuple of (float, float), default=(10, 6)
        Figure size in inches.
    save_path : str, optional
        If given, the figure is written to this path as a 300-dpi PNG with a
        tight bounding box before being shown.
    metric_name : str, default='Score'
        Name of the metric, used as the y-axis label (e.g. ``'Accuracy'``).
    show_std : bool, default=True
        Draw the standard deviation bands. Ignored for 1-D score arrays, which
        carry no spread.
    show_grid : bool, default=False
        Currently has no effect — grid lines are always drawn.

    Returns
    -------
    None
        The figure is shown (and optionally saved) rather than returned.

    Raises
    ------
    ImportError
        If matplotlib is not installed (it is imported lazily).

    Notes
    -----
    Side effects: mutates the global matplotlib style, calls
    ``matplotlib.pyplot.show()``, and writes a file when ``save_path`` is given.

    See Also
    --------
    :func:`~tuiml.evaluation.visualization.plot_roc_curve` : Threshold behaviour
        of a single fitted model.
    :func:`~tuiml.evaluation.visualization.plot_boxplot_comparison` : Score
        spread across algorithms rather than across training sizes.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.visualization import plot_learning_curve
    >>> train_sizes = np.array([20, 40, 60, 80, 100])
    >>> train_scores = np.array([[0.99, 0.98, 1.00],
    ...                          [0.97, 0.97, 0.98],
    ...                          [0.96, 0.96, 0.97],
    ...                          [0.96, 0.95, 0.96],
    ...                          [0.95, 0.95, 0.96]])
    >>> test_scores = np.array([[0.72, 0.70, 0.75],
    ...                         [0.80, 0.79, 0.82],
    ...                         [0.85, 0.84, 0.86],
    ...                         [0.88, 0.87, 0.88],
    ...                         [0.89, 0.89, 0.90]])
    >>> plot_learning_curve(train_sizes, train_scores, test_scores,
    ...                     metric_name='Accuracy')   # doctest: +SKIP
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    colors = get_colors(2)

    train_sizes = np.asarray(train_sizes)
    train_scores = np.asarray(train_scores)
    test_scores = np.asarray(test_scores)

    # Handle both 1D and 2D arrays
    if train_scores.ndim == 1:
        train_mean = train_scores
        train_std = np.zeros_like(train_mean)
        test_mean = test_scores
        test_std = np.zeros_like(test_mean)
    else:
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

    fig, ax = setup_figure(figsize=figsize)

    # Training curve
    ax.plot(train_sizes, train_mean, 'o-', color=colors[0], lw=3.0,
            markersize=10, label='Training score')
    if show_std and train_scores.ndim > 1:
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color=colors[0])

    # Validation curve
    ax.plot(train_sizes, test_mean, 's-', color=colors[1], lw=3.0,
            markersize=10, label='Cross-validation score')
    if show_std and test_scores.ndim > 1:
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                        alpha=0.2, color=colors[1])

    style_axis(
        ax,
        title=title,
        xlabel='Training Set Size',
        ylabel=metric_name,
        legend=True,
        legend_loc='best',
        grid=True,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()
