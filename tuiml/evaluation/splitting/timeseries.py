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
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None
    ):
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
        """Get number of splits."""
        return self.n_splits

    def __repr__(self) -> str:
        return (
            f"TimeSeriesSplit(n_splits={self.n_splits}, "
            f"test_size={self.test_size}, gap={self.gap})"
        )
