"""Base class and input handling for time-series classification."""

from __future__ import annotations

from typing import Optional

import numpy as np

from tuiml.base.algorithms import Classifier


def as_panel(X: np.ndarray) -> np.ndarray:
    """Coerce time-series input to the canonical 3-D panel layout.

    TuiML represents a collection of time series as ``(n_samples, n_channels,
    n_timepoints)``. A 2-D array is read as univariate, one series per row,
    which is how nearly every public benchmark ships its data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
        Input series.

    Returns
    -------
    panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
        Contiguous float64 panel.

    Raises
    ------
    ValueError
        If the input is not 2-D or 3-D.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import as_panel
    >>> as_panel(np.zeros((5, 100))).shape
    (5, 1, 100)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[:, None, :]
    elif X.ndim != 3:
        raise ValueError(
            "time-series input must be (n_samples, n_timepoints) or "
            f"(n_samples, n_channels, n_timepoints), got {X.ndim}-D"
        )
    return np.ascontiguousarray(X)


class TimeSeriesClassifier(Classifier):
    """Base class for classifiers over whole time series.

    Unlike the rest of :mod:`tuiml.algorithms`, these take a **panel** of
    shape ``(n_samples, n_channels, n_timepoints)`` rather than a feature
    matrix, and the ordering of the time axis carries the signal. Flattening a
    series into columns and handing it to an ordinary classifier throws that
    ordering away, which is why this family exists.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.
    n_channels_ : int
        Number of channels seen during :meth:`fit`.
    n_timepoints_ : int
        Series length seen during :meth:`fit`.

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.classification.DTWNeighborsClassifier` : Elastic-distance baseline.
    :func:`~tuiml.algorithms.timeseries.classification.as_panel` : The input coercion used throughout.
    """

    def __init__(self) -> None:
        """Initialise the classifier in an unfitted state."""
        super().__init__()
        self.classes_: Optional[np.ndarray] = None
        self.n_channels_: Optional[int] = None
        self.n_timepoints_: Optional[int] = None

    def _validate_fit(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Coerce the training panel and record its shape and classes.

        Parameters
        ----------
        X : np.ndarray
            Training series, 2-D or 3-D.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
            Canonical training panel.
        y : np.ndarray of shape (n_samples,)
            Labels as an array.
        """
        panel = as_panel(X)
        y = np.asarray(y)
        if len(panel) != len(y):
            raise ValueError(
                f"X has {len(panel)} series but y has {len(y)} labels"
            )
        self.classes_ = np.unique(y)
        self.n_channels_ = panel.shape[1]
        self.n_timepoints_ = panel.shape[2]
        return panel, y

    def _validate_predict(self, X: np.ndarray) -> np.ndarray:
        """Coerce a prediction panel and check it matches the training shape.

        Parameters
        ----------
        X : np.ndarray
            Series to classify, 2-D or 3-D.

        Returns
        -------
        panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
            Canonical panel.
        """
        self._check_is_fitted()
        panel = as_panel(X)
        if panel.shape[1] != self.n_channels_:
            raise ValueError(
                f"X has {panel.shape[1]} channels but the model was fitted on "
                f"{self.n_channels_}"
            )
        return panel
