"""
Regression evaluation metrics.

Scoring functions for models that predict a continuous target. The module
covers two families:

* **Absolute-scale errors** — :func:`mean_absolute_error`,
  :func:`mean_squared_error` and :func:`root_mean_squared_error` report error in
  the units of the target, so they are directly interpretable but not comparable
  across datasets.
* **Relative / normalized scores** — :func:`relative_absolute_error`,
  :func:`root_relative_squared_error`, :func:`r2_score` and
  :func:`correlation_coefficient` divide the model error by the error of a
  trivial baseline (predicting the mean), which makes them comparable across
  datasets with different target scales.

The relative errors are reported as percentages and the naming mirrors Weka's
``Evaluation`` class, so the numbers line up with a Weka experiment report.
Every function takes ``(y_true, y_pred)`` as 1-D arrays of equal length and
returns a plain Python ``float``.

Examples
--------
>>> import numpy as np
>>> from tuiml.evaluation.metrics import mean_absolute_error, r2_score
>>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
>>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
>>> mean_absolute_error(y_true, y_pred)
0.5
>>> round(r2_score(y_true, y_pred), 4)
0.9486
"""

from typing import Optional, Union
import numpy as np
from tuiml.base.metrics import check_consistent_length, safe_divide

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the **Mean Absolute Error** (MAE).

    MAE is the average magnitude of the residuals. Because the errors are not
    squared, it weights every mistake linearly and is therefore far less
    sensitive to outliers than :func:`mean_squared_error`.

    .. math::
        \\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|

    Equivalent to Weka's ``Evaluation.meanAbsoluteError()``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth target values.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    error : float
        Mean absolute error, in the units of the target. Non-negative; ``0.0``
        means a perfect fit.

    Notes
    -----
    Complexity: :math:`O(n)` time, :math:`O(n)` memory.

    When to use: report MAE when the cost of an error grows linearly with its
    size, or when the target contains outliers you do not want to dominate the
    score.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.mean_squared_error` : Squared-error analogue.
    :func:`~tuiml.evaluation.metrics.relative_absolute_error` : MAE relative to a mean predictor.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import mean_absolute_error
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> mean_absolute_error(y_true, y_pred)
    0.5
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, squared: bool = True) -> float:
    """Compute the **Mean Squared Error** (MSE), or its square root (RMSE).

    Squaring the residuals penalises large mistakes quadratically, which makes
    MSE the natural loss for least-squares models but also makes it sensitive
    to outliers.

    .. math::
        \\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth target values.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values.
    squared : bool, default=True
        If ``True`` return MSE. If ``False`` return the root mean squared error
        :math:`\\sqrt{\\text{MSE}}`, which is back on the scale of the target.

    Returns
    -------
    error : float
        Mean squared error (or RMSE when ``squared=False``). Non-negative;
        ``0.0`` means a perfect fit.

    Notes
    -----
    Complexity: :math:`O(n)` time, :math:`O(n)` memory.

    When to use: MSE is the right choice when large errors are
    disproportionately costly, and it is the quantity linear regression
    actually minimises. Use ``squared=False`` (or
    :func:`~tuiml.evaluation.metrics.root_mean_squared_error`) when you want a
    number readers can compare against the target's own units.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.root_mean_squared_error` : Thin wrapper for ``squared=False``.
    :func:`~tuiml.evaluation.metrics.mean_absolute_error` : Outlier-robust alternative.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import mean_squared_error
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> round(mean_squared_error(y_true, y_pred, squared=False), 4)
    0.6124
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse if squared else np.sqrt(mse))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the **Root Mean Squared Error** (RMSE).

    RMSE is the square root of :func:`mean_squared_error`, which puts the score
    back into the units of the target while keeping MSE's quadratic penalty on
    large residuals.

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}

    Equivalent to Weka's ``Evaluation.rootMeanSquaredError()``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth target values.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    error : float
        Root mean squared error, in the units of the target. Non-negative;
        ``0.0`` means a perfect fit.

    Notes
    -----
    Complexity: :math:`O(n)` time, :math:`O(n)` memory.

    When to use: the default headline number for a regression report — same
    ranking as MSE, but readable on the target's scale.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.mean_squared_error` : The underlying squared error.
    :func:`~tuiml.evaluation.metrics.root_relative_squared_error` : RMSE expressed relative to a mean predictor.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import root_mean_squared_error
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> round(root_mean_squared_error(y_true, y_pred), 4)
    0.6124
    """
    return mean_squared_error(y_true, y_pred, squared=False)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute :math:`R^2`, the **coefficient of determination**.

    :math:`R^2` is the fraction of the target's variance that the model
    explains, measured against the trivial baseline that always predicts
    :math:`\\bar{y}`.

    .. math::
        R^2 = 1 - \\frac{\\sum_i (y_i - \\hat{y}_i)^2}{\\sum_i (y_i - \\bar{y})^2}

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth target values.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    score : float
        Coefficient of determination. ``1.0`` is a perfect fit, ``0.0`` matches
        a constant mean predictor, and negative values mean the model is worse
        than predicting the mean. Unlike a correlation this is **not** bounded
        below by ``-1``.

    Notes
    -----
    Complexity: :math:`O(n)` time, :math:`O(n)` memory.

    When to use: the standard scale-free summary of regression quality. Note
    that :math:`R^2` equals the squared Pearson correlation only for an
    unbiased linear fit; for a general model the two can differ, because
    :func:`correlation_coefficient` ignores systematic bias while :math:`R^2`
    does not.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.correlation_coefficient` : Pearson correlation between truth and prediction.
    :func:`~tuiml.evaluation.metrics.root_relative_squared_error` : Related normalized error, reported as a percentage.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import r2_score
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> round(r2_score(y_true, y_pred), 4)
    0.9486
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - safe_divide(ss_res, ss_tot))

def relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the **Relative Absolute Error** (RAE), as a percentage.

    RAE divides the model's total absolute error by the total absolute error of
    the trivial predictor that always outputs :math:`\\bar{y}`. Dividing out the
    target's scale makes the score comparable across datasets.

    .. math::
        \\text{RAE} = 100 \\cdot
        \\frac{\\sum_i |y_i - \\hat{y}_i|}{\\sum_i |y_i - \\bar{y}|}

    Equivalent to Weka's ``Evaluation.relativeAbsoluteError()``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth target values.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    error : float
        Relative absolute error **as a percentage**. Below ``100.0`` means the
        model beats the mean predictor; ``0.0`` is a perfect fit.

    Notes
    -----
    Complexity: :math:`O(n)` time, :math:`O(n)` memory.

    When to use: comparing a model across datasets whose targets have different
    units or magnitudes, where a raw MAE would be meaningless.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.mean_absolute_error` : The un-normalized absolute error.
    :func:`~tuiml.evaluation.metrics.root_relative_squared_error` : Squared-error counterpart.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import relative_absolute_error
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> round(relative_absolute_error(y_true, y_pred), 4)
    23.5294
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mae = np.sum(np.abs(y_true - y_pred))
    mae_mean = np.sum(np.abs(y_true - np.mean(y_true)))
    return float(safe_divide(mae, mae_mean) * 100)

def root_relative_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the **Root Relative Squared Error** (RRSE), as a percentage.

    RRSE is the model's squared error divided by the squared error of the mean
    predictor, square-rooted and expressed as a percentage. It is the
    squared-error counterpart of :func:`relative_absolute_error` and is tied
    directly to :math:`R^2` by
    :math:`\\text{RRSE} = 100\\sqrt{1 - R^2}`.

    .. math::
        \\text{RRSE} = 100 \\cdot \\sqrt{
        \\frac{\\sum_i (y_i - \\hat{y}_i)^2}{\\sum_i (y_i - \\bar{y})^2}}

    Equivalent to Weka's ``Evaluation.rootRelativeSquaredError()``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth target values.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    error : float
        Root relative squared error **as a percentage**. Below ``100.0`` means
        the model beats the mean predictor; ``0.0`` is a perfect fit.

    Notes
    -----
    Complexity: :math:`O(n)` time, :math:`O(n)` memory.

    When to use: the scale-free companion to RMSE, and the number Weka prints
    alongside it in an experiment summary.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.root_mean_squared_error` : The un-normalized squared error.
    :func:`~tuiml.evaluation.metrics.r2_score` : Equivalent information on a ``1.0``-is-best scale.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import root_relative_squared_error
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> round(root_relative_squared_error(y_true, y_pred), 4)
    22.6698
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mse = np.sum((y_true - y_pred) ** 2)
    mse_mean = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(np.sqrt(safe_divide(mse, mse_mean)) * 100)

def correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the **Pearson correlation coefficient** between truth and prediction.

    Measures how well the predictions track the targets *up to an arbitrary
    linear rescaling*: a model that predicts :math:`2y + 5` still scores
    ``1.0``. That makes it a measure of ranking/shape agreement rather than of
    calibrated accuracy.

    .. math::
        r = \\frac{\\sum_i (y_i - \\bar{y})(\\hat{y}_i - \\bar{\\hat{y}})}
        {\\sqrt{\\sum_i (y_i - \\bar{y})^2}\\;
         \\sqrt{\\sum_i (\\hat{y}_i - \\bar{\\hat{y}})^2}}

    Equivalent to Weka's ``Evaluation.correlationCoefficient()``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth target values.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    score : float
        Correlation in :math:`[-1, 1]`. ``1.0`` is perfect positive linear
        agreement, ``0.0`` no linear relationship, ``-1.0`` perfect inversion.
        Returns ``0.0`` when the correlation is undefined, which happens when
        either input is constant.

    Notes
    -----
    Complexity: :math:`O(n)` time, :math:`O(n)` memory.

    When to use: when only the ordering or shape of the predictions matters and
    a constant offset or scale factor is acceptable. Pair it with
    :func:`r2_score` if calibration matters — a high correlation with a low
    :math:`R^2` is the signature of a systematically biased model.

    See Also
    --------
    :func:`~tuiml.evaluation.metrics.r2_score` : Penalizes bias and scale errors that this metric ignores.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import correlation_coefficient
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> round(correlation_coefficient(y_true, y_pred), 4)
    0.9849
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0
