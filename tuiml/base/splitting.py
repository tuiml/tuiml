"""
Base classes for data splitting.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Iterator

class BaseSplitter(ABC):
    """
    Base class for all data splitters.

    All splitters implement the split() method that yields
    (train_indices, test_indices) tuples.
    """

    @abstractmethod
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,), optional
            Target values.
        groups : ndarray of shape (n_samples,), optional
            Group labels for samples.

        Yields
        ------
        train_indices : ndarray
            Indices for training set.
        test_indices : ndarray
            Indices for test set.
        """
        pass

    @abstractmethod
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """
        Get the number of splits.

        Parameters
        ----------
        X : ndarray, optional
            Training data (needed for some splitters).
        y : ndarray, optional
            Target values.
        groups : ndarray, optional
            Group labels.

        Returns
        -------
        n_splits : int
            Number of splits.
        """
        pass

    def _validate_input(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate and convert input to numpy arrays."""
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
            if len(X) != len(y):
                raise ValueError(
                    f"X and y must have same length. Got {len(X)} and {len(y)}"
                )
        return X, y

    def _is_classification(self, y: np.ndarray) -> bool:
        """Check if task is classification based on y values."""
        unique = np.unique(y)
        # Classification if few unique values or integer type
        return (
            len(unique) < len(y) * 0.1 or
            y.dtype in [np.int32, np.int64, np.int_, object, bool]
        )
