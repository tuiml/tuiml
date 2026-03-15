"""
Shuffle (random permutation) splitters.
"""

import numpy as np
from typing import Iterator, Optional, Tuple, Union
from tuiml.base.splitting import BaseSplitter

class ShuffleSplit(BaseSplitter):
    """
    Random permutation cross-validation.

    Generates random train/test splits multiple times.

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.
    test_size : float or int, default=0.1
        If float, proportion for test set.
        If int, absolute number of test samples.
    train_size : float or int, optional
        If float, proportion for train set.
        If int, absolute number of train samples.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import ShuffleSplit
    >>> import numpy as np
    >>> X = np.arange(10).reshape(-1, 1)
    >>> ss = ShuffleSplit(n_splits=5, test_size=0.3)
    >>> for train_idx, test_idx in ss.split(X):
    ...     print(f"Train: {train_idx}, Test: {test_idx}")
    """

    def __init__(
        self,
        n_splits: int = 10,
        test_size: Optional[Union[float, int]] = 0.1,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None
    ):
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")

        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "n_splits": {
                "type": "integer",
                "default": 10,
                "description": "Number of re-shuffling and splitting iterations"
            },
            "test_size": {
                "type": ["number", "integer", "null"],
                "default": 0.1,
                "description": "Proportion (float) or absolute number (int) for test set"
            },
            "train_size": {
                "type": ["number", "integer", "null"],
                "default": None,
                "description": "Proportion (float) or absolute number (int) for train set"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def _resolve_sizes(self, n_samples: int) -> Tuple[int, int]:
        """Calculate train and test sizes."""
        if self.test_size is not None:
            if isinstance(self.test_size, float):
                n_test = int(n_samples * self.test_size)
            else:
                n_test = int(self.test_size)
        else:
            n_test = int(n_samples * 0.1)

        if self.train_size is not None:
            if isinstance(self.train_size, float):
                n_train = int(n_samples * self.train_size)
            else:
                n_train = int(self.train_size)
        else:
            n_train = n_samples - n_test

        # Ensure valid sizes
        n_test = max(1, min(n_test, n_samples - 1))
        n_train = max(1, min(n_train, n_samples - n_test))

        return n_train, n_test

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate shuffle split indices."""
        X, y = self._validate_input(X, y)
        n_samples = len(X)
        n_train, n_test = self._resolve_sizes(n_samples)

        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n_samples)

        for _ in range(self.n_splits):
            # Shuffle and split
            permutation = rng.permutation(n_samples)
            test_idx = indices[permutation[:n_test]]
            train_idx = indices[permutation[n_test:n_test + n_train]]
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
            f"ShuffleSplit(n_splits={self.n_splits}, "
            f"test_size={self.test_size}, train_size={self.train_size})"
        )

class StratifiedShuffleSplit(BaseSplitter):
    """
    Stratified random permutation cross-validation.

    Generates random train/test splits while preserving class proportions.

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.
    test_size : float or int, default=0.1
        If float, proportion for test set.
        If int, absolute number of test samples.
    train_size : float or int, optional
        If float, proportion for train set.
        If int, absolute number of train samples.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import StratifiedShuffleSplit
    >>> import numpy as np
    >>> X = np.arange(10).reshape(-1, 1)
    >>> y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3)
    >>> for train_idx, test_idx in sss.split(X, y):
    ...     print(f"Test class distribution: {np.bincount(y[test_idx])}")
    """

    def __init__(
        self,
        n_splits: int = 10,
        test_size: Optional[Union[float, int]] = 0.1,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None
    ):
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")

        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "n_splits": {
                "type": "integer",
                "default": 10,
                "description": "Number of re-shuffling and splitting iterations"
            },
            "test_size": {
                "type": ["number", "integer", "null"],
                "default": 0.1,
                "description": "Proportion (float) or absolute number (int) for test set"
            },
            "train_size": {
                "type": ["number", "integer", "null"],
                "default": None,
                "description": "Proportion (float) or absolute number (int) for train set"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified shuffle split indices."""
        X, y = self._validate_input(X, y)

        if y is None:
            raise ValueError("y is required for StratifiedShuffleSplit")

        n_samples = len(X)
        y = np.asarray(y)

        # Calculate sizes
        if self.test_size is not None:
            if isinstance(self.test_size, float):
                test_ratio = self.test_size
            else:
                test_ratio = self.test_size / n_samples
        else:
            test_ratio = 0.1

        if self.train_size is not None:
            if isinstance(self.train_size, float):
                train_ratio = self.train_size
            else:
                train_ratio = self.train_size / n_samples
        else:
            train_ratio = 1.0 - test_ratio

        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n_samples)

        # Get unique classes
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)

        # Get class indices
        class_indices = [indices[y_indices == c] for c in range(n_classes)]

        for _ in range(self.n_splits):
            train_idx_list = []
            test_idx_list = []

            for ci in class_indices:
                n_class = len(ci)
                n_class_test = max(1, int(n_class * test_ratio))
                n_class_train = max(1, int(n_class * train_ratio))

                # Shuffle class indices
                shuffled = rng.permutation(ci)

                test_idx_list.extend(shuffled[:n_class_test])
                train_idx_list.extend(
                    shuffled[n_class_test:n_class_test + n_class_train]
                )

            train_idx = np.array(train_idx_list)
            test_idx = np.array(test_idx_list)

            # Shuffle the final indices
            rng.shuffle(train_idx)
            rng.shuffle(test_idx)

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
            f"StratifiedShuffleSplit(n_splits={self.n_splits}, "
            f"test_size={self.test_size})"
        )
