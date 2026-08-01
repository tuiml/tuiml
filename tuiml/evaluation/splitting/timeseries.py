"""
Time series cross-validation splitters.
"""

import numpy as np
from typing import Iterator, Optional, Tuple
from tuiml.base.splitting import BaseSplitter

class TimeSeriesSplit(BaseSplitter):
    """
    Time Series cross-validation splitter.

    Provides train/test indices for time series data where test set
    is always in the future relative to training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits.
    test_size : int, optional
        Size of test set. If None, uses n_samples // (n_splits + 1).
    gap : int, default=0
        Number of samples to skip between train and test.
    max_train_size : int, optional
        Maximum size for a single training set.

    Raises
    ------
    ValueError
        From :meth:`split` when ``n_splits * test_size + gap`` leaves no room
        for a training set, since that configuration cannot yield the promised
        number of folds.

    Notes
    -----
    Unlike regular K-Fold, training set grows with each split:
    - Split 1: train=[0:n], test=[n:n+test_size]
    - Split 2: train=[0:n+test_size], test=[n+test_size:n+2*test_size]
    - etc.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import TimeSeriesSplit
    >>> import numpy as np
    >>> X = np.arange(10).reshape(-1, 1)
    >>> tss = TimeSeriesSplit(n_splits=3)
    >>> for train_idx, test_idx in tss.split(X):
    ...     print(f"Train: {train_idx}, Test: {test_idx}")
    Train: [0 1 2 3], Test: [4 5]
    Train: [0 1 2 3 4 5], Test: [6 7]
    Train: [0 1 2 3 4 5 6 7], Test: [8 9]
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None
    ):
        """Store the split count, test size, gap and max train size. See the class docstring."""
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        if gap < 0:
            raise ValueError("gap must be non-negative")

        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "n_splits": {
                "type": "integer",
                "default": 5,
                "description": "Number of splits"
            },
            "test_size": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Size of test set (if None, uses n_samples // (n_splits + 1))"
            },
            "gap": {
                "type": "integer",
                "default": 0,
                "description": "Number of samples to skip between train and test"
            },
            "max_train_size": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Maximum size for a single training set"
            }
        }

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series split indices."""
        X, y = self._validate_input(X, y)
        n_samples = len(X)
        indices = np.arange(n_samples)

        test_size = self.test_size or n_samples // (self.n_splits + 1)
        test_size = max(1, test_size)

        # The first fold needs room for a non-empty training set before it:
        # test_start = n_samples - n_splits * test_size, and train_end is that
        # minus the gap. Without this check the leading folds are silently
        # dropped and split() yields fewer pairs than get_n_splits() promises.
        if n_samples <= self.n_splits * test_size + self.gap:
            raise ValueError(
                f"Cannot produce {self.n_splits} splits from {n_samples} samples "
                f"with test_size={test_size} and gap={self.gap}: "
                f"{self.n_splits} * {test_size} + {self.gap} leaves no room for a "
                f"training set. Reduce n_splits, test_size, or gap."
            )

        # Calculate test start positions
        test_starts = range(
            n_samples - self.n_splits * test_size,
            n_samples,
            test_size
        )

        for test_start in test_starts:
            train_end = test_start - self.gap

            if train_end <= 0:
                continue

            # Apply max_train_size if specified
            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)

            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_start + test_size]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Get the number of splits this splitter yields.

        Parameters
        ----------
        X : np.ndarray, optional
            Ignored, present for API consistency.
        y : np.ndarray, optional
            Ignored, present for API consistency.
        groups : np.ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        n_splits : int
            Number of train/test pairs :meth:`split` yields. Configurations
            that could not deliver this count raise in :meth:`split` rather
            than silently yielding fewer folds.
        """
        return self.n_splits

    def __repr__(self) -> str:
        """Return a reproducible string form of the splitter.

        Returns
        -------
        repr_str : str
            Constructor-style representation, e.g.
            ``TimeSeriesSplit(n_splits=5, test_size=None, gap=0)``.
        """
        return (
            f"TimeSeriesSplit(n_splits={self.n_splits}, "
            f"test_size={self.test_size}, gap={self.gap})"
        )
