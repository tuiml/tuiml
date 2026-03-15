"""
Group-based cross-validation splitters.

Ensures that samples from the same group are not split across train and test.
"""

import numpy as np
from typing import Iterator, Optional, Tuple
from tuiml.base.splitting import BaseSplitter

class GroupKFold(BaseSplitter):
    """
    K-Fold cross-validation with non-overlapping groups.

    Ensures that samples from the same group are not split across
    train and test sets.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.

    Notes
    -----
    The same group will not appear in two different folds.
    The number of distinct groups must be at least equal to n_splits.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import GroupKFold
    >>> import numpy as np
    >>> X = np.arange(6).reshape(-1, 1)
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> groups = np.array([1, 1, 2, 2, 3, 3])  # 3 groups
    >>> gkf = GroupKFold(n_splits=3)
    >>> for train_idx, test_idx in gkf.split(X, y, groups):
    ...     print(f"Train groups: {groups[train_idx]}, Test groups: {groups[test_idx]}")
    """

    def __init__(self, n_splits: int = 5):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "n_splits": {
                "type": "integer",
                "default": 5,
                "description": "Number of folds (must be at least 2)"
            }
        }

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate group K-fold indices."""
        X, y = self._validate_input(X, y)

        if groups is None:
            raise ValueError("groups is required for GroupKFold")

        groups = np.asarray(groups)
        if len(groups) != len(X):
            raise ValueError("groups must have same length as X")

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if n_groups < self.n_splits:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than "
                f"the number of groups={n_groups}"
            )

        # Assign groups to folds
        group_to_fold = {}
        for i, group in enumerate(unique_groups):
            group_to_fold[group] = i % self.n_splits

        # Create fold indices
        indices = np.arange(len(X))

        for fold in range(self.n_splits):
            test_mask = np.array([group_to_fold[g] == fold for g in groups])
            test_idx = indices[test_mask]
            train_idx = indices[~test_mask]
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
        return f"GroupKFold(n_splits={self.n_splits})"

class StratifiedGroupKFold(BaseSplitter):
    """
    Stratified K-Fold cross-validation with non-overlapping groups.

    Tries to preserve class proportions while ensuring groups don't
    appear in multiple folds.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle groups before splitting.
    random_state : int, optional
        Random seed.

    Notes
    -----
    This is a best-effort stratification. Perfect stratification may
    not be possible when group sizes vary significantly.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import StratifiedGroupKFold
    >>> import numpy as np
    >>> X = np.arange(12).reshape(-1, 1)
    >>> y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    >>> sgkf = StratifiedGroupKFold(n_splits=3)
    >>> for train_idx, test_idx in sgkf.split(X, y, groups):
    ...     print(f"Test class distribution: {np.bincount(y[test_idx])}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "n_splits": {
                "type": "integer",
                "default": 5,
                "description": "Number of folds (must be at least 2)"
            },
            "shuffle": {
                "type": "boolean",
                "default": False,
                "description": "Whether to shuffle groups before splitting"
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
        """Generate stratified group K-fold indices."""
        X, y = self._validate_input(X, y)

        if y is None:
            raise ValueError("y is required for StratifiedGroupKFold")
        if groups is None:
            raise ValueError("groups is required for StratifiedGroupKFold")

        groups = np.asarray(groups)
        y = np.asarray(y)

        if len(groups) != len(X):
            raise ValueError("groups must have same length as X")

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if n_groups < self.n_splits:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than "
                f"the number of groups={n_groups}"
            )

        rng = np.random.RandomState(self.random_state)

        # Calculate class distribution per group
        group_class_counts = {}
        for group in unique_groups:
            mask = groups == group
            group_y = y[mask]
            classes, counts = np.unique(group_y, return_counts=True)
            group_class_counts[group] = dict(zip(classes, counts))

        # Sort groups by their majority class for better stratification
        def get_majority_class(g):
            counts = group_class_counts[g]
            return max(counts.keys(), key=lambda c: counts.get(c, 0))

        sorted_groups = sorted(unique_groups, key=get_majority_class)

        if self.shuffle:
            # Shuffle within each class group
            rng.shuffle(sorted_groups)

        # Assign groups to folds trying to balance classes
        fold_class_counts = [{} for _ in range(self.n_splits)]
        group_to_fold = {}

        for group in sorted_groups:
            # Find fold with lowest count of this group's majority class
            majority_class = get_majority_class(group)
            fold_counts = [
                fold_class_counts[f].get(majority_class, 0)
                for f in range(self.n_splits)
            ]
            best_fold = np.argmin(fold_counts)

            group_to_fold[group] = best_fold

            # Update fold class counts
            for cls, cnt in group_class_counts[group].items():
                fold_class_counts[best_fold][cls] = \
                    fold_class_counts[best_fold].get(cls, 0) + cnt

        # Generate splits
        indices = np.arange(len(X))

        for fold in range(self.n_splits):
            test_mask = np.array([group_to_fold[g] == fold for g in groups])
            test_idx = indices[test_mask]
            train_idx = indices[~test_mask]
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
        return f"StratifiedGroupKFold(n_splits={self.n_splits}, shuffle={self.shuffle})"
