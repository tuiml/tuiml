"""
Leave-One-Out and Leave-P-Out cross-validation splitters.
"""

import numpy as np
from typing import Iterator, Optional, Tuple
from itertools import combinations
from tuiml.base.splitting import BaseSplitter

class LeaveOneOut(BaseSplitter):
    """
    Leave-One-Out cross-validation.

    Each sample is used once as test while remaining samples form training set.

    Notes
    -----
    This can be very slow for large datasets as it creates n splits.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import LeaveOneOut
    >>> import numpy as np
    >>> X = np.arange(5).reshape(-1, 1)
    >>> loo = LeaveOneOut()
    >>> for train_idx, test_idx in loo.split(X):
    ...     print(f"Train: {train_idx}, Test: {test_idx}")
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {}

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate LOO indices."""
        X, y = self._validate_input(X, y)
        n_samples = len(X)
        indices = np.arange(n_samples)

        for i in range(n_samples):
            test_idx = np.array([i])
            train_idx = np.concatenate([indices[:i], indices[i+1:]])
            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits (equals n_samples)."""
        if X is None:
            raise ValueError("X is required to determine n_splits for LeaveOneOut")
        return len(X)

    def __repr__(self) -> str:
        return "LeaveOneOut()"

class LeavePOut(BaseSplitter):
    """
    Leave-P-Out cross-validation.

    P samples are used as test while remaining samples form training set.
    Generates all possible combinations.

    Parameters
    ----------
    p : int
        Size of test set.

    Notes
    -----
    Number of splits is C(n, p) = n! / (p! * (n-p)!)
    This grows very fast and can be impractical for large n or p.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import LeavePOut
    >>> import numpy as np
    >>> X = np.arange(5).reshape(-1, 1)
    >>> lpo = LeavePOut(p=2)
    >>> for train_idx, test_idx in lpo.split(X):
    ...     print(f"Train: {train_idx}, Test: {test_idx}")
    """

    def __init__(self, p: int = 2):
        if p < 1:
            raise ValueError("p must be at least 1")
        self.p = p

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "p": {
                "type": "integer",
                "default": 2,
                "description": "Size of test set (number of samples to leave out)"
            }
        }

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate LPO indices."""
        X, y = self._validate_input(X, y)
        n_samples = len(X)

        if self.p >= n_samples:
            raise ValueError(
                f"p={self.p} must be less than n_samples={n_samples}"
            )

        indices = np.arange(n_samples)

        for test_idx in combinations(range(n_samples), self.p):
            test_idx = np.array(test_idx)
            train_idx = np.array([i for i in indices if i not in test_idx])
            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits C(n, p)."""
        if X is None:
            raise ValueError("X is required to determine n_splits for LeavePOut")

        n = len(X)
        p = self.p

        # Calculate C(n, p) = n! / (p! * (n-p)!)
        from math import factorial
        return factorial(n) // (factorial(p) * factorial(n - p))

    def __repr__(self) -> str:
        return f"LeavePOut(p={self.p})"
