"""
Holdout (train/test split) splitters.
"""

import numpy as np
from typing import Iterator, Optional, Tuple, Union
from tuiml.base.splitting import BaseSplitter

def train_test_split(
    *arrays,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    shuffle: bool = True,
    stratify: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
):
    """
    Split arrays into train and test subsets.

    Parameters
    ----------
    *arrays : sequence of arrays
        Arrays to split (X, y, or more).
    test_size : float or int, optional
        If float, proportion for test (0-1).
        If int, absolute number of test samples.
        Default is 0.25 if train_size not specified.
    train_size : float or int, optional
        If float, proportion for train (0-1).
        If int, absolute number of train samples.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    stratify : ndarray, optional
        Array to use for stratification (usually y).
    random_state : int, optional
        Random seed.

    Returns
    -------
    splits : list of arrays
        Train and test splits for each input array.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import train_test_split
    >>> import numpy as np
    >>> X = np.arange(10).reshape(-1, 1)
    >>> y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>>
    >>> # Simple split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    >>>
    >>> # Stratified split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, stratify=y
    ... )
    """
    if len(arrays) == 0:
        raise ValueError("At least one array is required")

    # Validate arrays have same length
    n_samples = len(arrays[0])
    for arr in arrays[1:]:
        if len(arr) != n_samples:
            raise ValueError("All arrays must have the same length")

    # Determine test/train sizes
    if test_size is None and train_size is None:
        test_size = 0.25

    if test_size is not None:
        if isinstance(test_size, float):
            if not 0 < test_size < 1:
                raise ValueError("test_size as float must be between 0 and 1")
            n_test = int(n_samples * test_size)
        else:
            n_test = int(test_size)
    elif train_size is not None:
        if isinstance(train_size, float):
            if not 0 < train_size < 1:
                raise ValueError("train_size as float must be between 0 and 1")
            n_test = n_samples - int(n_samples * train_size)
        else:
            n_test = n_samples - int(train_size)
    else:
        n_test = int(n_samples * 0.25)

    n_test = max(1, min(n_test, n_samples - 1))
    n_train = n_samples - n_test

    # Generate indices
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)

    if stratify is not None:
        # Stratified split
        stratify = np.asarray(stratify)
        classes, y_indices = np.unique(stratify, return_inverse=True)

        train_indices = []
        test_indices = []

        for c in range(len(classes)):
            class_idx = indices[y_indices == c]
            if shuffle:
                rng.shuffle(class_idx)

            n_class_test = max(1, int(len(class_idx) * n_test / n_samples))
            test_indices.extend(class_idx[:n_class_test])
            train_indices.extend(class_idx[n_class_test:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)
    else:
        # Simple split
        if shuffle:
            rng.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    # Split all arrays
    result = []
    for arr in arrays:
        arr = np.asarray(arr)
        result.append(arr[train_indices])
        result.append(arr[test_indices])

    return result

class HoldoutSplit(BaseSplitter):
    """
    Holdout (train/test) split.

    Generates a single train/test split.

    Parameters
    ----------
    test_size : float, default=0.3
        Proportion of data for test set.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import HoldoutSplit
    >>> hs = HoldoutSplit(test_size=0.3, shuffle=True)
    >>> for train_idx, test_idx in hs.split(X):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """

    def __init__(
        self,
        test_size: float = 0.3,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "test_size": {
                "type": "number",
                "default": 0.3,
                "description": "Proportion of data for test set (between 0 and 1)"
            },
            "shuffle": {
                "type": "boolean",
                "default": True,
                "description": "Whether to shuffle before splitting"
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
        """Generate holdout split indices."""
        X, y = self._validate_input(X, y)
        n_samples = len(X)
        n_test = int(n_samples * self.test_size)

        indices = np.arange(n_samples)
        rng = np.random.RandomState(self.random_state)

        if self.shuffle:
            rng.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Returns 1 (single split)."""
        return 1

    def __repr__(self) -> str:
        return f"HoldoutSplit(test_size={self.test_size}, shuffle={self.shuffle})"

class StratifiedHoldoutSplit(BaseSplitter):
    """
    Stratified holdout split.

    Preserves class proportions in train/test split.

    Parameters
    ----------
    test_size : float, default=0.3
        Proportion of data for test set.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        test_size: float = 0.3,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "test_size": {
                "type": "number",
                "default": 0.3,
                "description": "Proportion of data for test set (between 0 and 1)"
            },
            "shuffle": {
                "type": "boolean",
                "default": True,
                "description": "Whether to shuffle before splitting"
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
        """Generate stratified holdout split indices."""
        X, y = self._validate_input(X, y)

        if y is None:
            raise ValueError("y is required for stratified split")

        n_samples = len(X)
        indices = np.arange(n_samples)
        rng = np.random.RandomState(self.random_state)

        # Get unique classes
        classes, y_indices = np.unique(y, return_inverse=True)

        train_indices = []
        test_indices = []

        for c in range(len(classes)):
            class_idx = indices[y_indices == c]
            if self.shuffle:
                rng.shuffle(class_idx)

            n_class_test = max(1, int(len(class_idx) * self.test_size))
            test_indices.extend(class_idx[:n_class_test])
            train_indices.extend(class_idx[n_class_test:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if self.shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)

        yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Returns 1 (single split)."""
        return 1

    def __repr__(self) -> str:
        return f"StratifiedHoldoutSplit(test_size={self.test_size})"
