"""Random-permutation cross-validators (Monte-Carlo cross-validation).

A shuffle splitter draws a **fresh random train/test partition on every
iteration** instead of rotating over a fixed set of folds. That is the key
difference from :class:`~tuiml.evaluation.splitting.KFold`: the number of
iterations and the test proportion are decoupled, so you can ask for 50 splits
with 10% test each, and the test sets are *independent draws* that may overlap
rather than a disjoint cover of the data. Some samples may be tested many
times, others never.

Use these when you want more repetitions than ``1 / test_size`` folds would
allow, or a test fraction that K-fold cannot express. Prefer
:class:`~tuiml.evaluation.splitting.KFold` when you need each sample tested
exactly once.

* :class:`~tuiml.evaluation.splitting.ShuffleSplit` -- ignores ``y``.
* :class:`~tuiml.evaluation.splitting.StratifiedShuffleSplit` -- requires ``y``
  and samples within each class so every draw keeps the class proportions.

Both also allow ``train_size`` to be smaller than the complement of
``test_size``, which yields a sub-sampled training set -- handy for learning
curves.
"""

import numpy as np
from typing import Iterator, Optional, Tuple, Union
from tuiml.base.splitting import BaseSplitter

class ShuffleSplit(BaseSplitter):
    """**Unstratified** random-permutation splitter: ``n_splits`` independent draws.

    Each iteration reshuffles all sample indices and cuts the permutation into
    a test block of ``n_test`` rows followed by a train block of ``n_train``
    rows. Unlike :class:`~tuiml.evaluation.splitting.KFold`, the test sets of
    different iterations are **not** disjoint and need not cover the data: a
    sample can be tested repeatedly or never. Test size and iteration count are
    independent knobs, so ``n_splits`` may be far larger than ``1 / test_size``.

    Labels are ignored -- a draw can be badly class-skewed on imbalanced data.
    Use :class:`~tuiml.evaluation.splitting.StratifiedShuffleSplit` for
    classification.

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffle-and-split iterations. Must be at least 1.
    test_size : float or int, default=0.1
        If float, the proportion of rows per test set. If int, the absolute
        number. Clipped to ``[1, n_samples - 1]``.
    train_size : float or int, optional
        If given, the size of the training block taken after the test block.
        If None, all remaining rows are used. Setting it smaller sub-samples
        the training set, so some rows appear in neither part -- useful for
        learning curves.
    random_state : int, optional
        Random seed. One generator drives all ``n_splits`` permutations, so a
        fixed seed reproduces the whole sequence.

    Notes
    -----
    Three draws with ``test_size=0.3`` over 10 samples. Test sets overlap
    across iterations, which never happens with K-fold::

        sample:   0 1 2 3 4 5 6 7 8 9
        split 0:  . . T . T . . . T .
        split 1:  . T . T . T . . . .
        split 2:  . . T T . . . . T .

        T = test, . = train

    See Also
    --------
    :class:`~tuiml.evaluation.splitting.StratifiedShuffleSplit` : Same, with class proportions preserved.
    :class:`~tuiml.evaluation.splitting.KFold` : Disjoint folds covering every sample once.
    :class:`~tuiml.evaluation.splitting.HoldoutSplit` : A single random split.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.splitting import ShuffleSplit
    >>> X = np.arange(20).reshape(10, 2)
    >>> cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    >>> print(cv.get_n_splits())
    3
    >>> for train_idx, test_idx in cv.split(X):
    ...     print(len(train_idx), len(test_idx), sorted(test_idx.tolist()))
    7 3 [2, 4, 8]
    7 3 [1, 3, 5]
    7 3 [2, 3, 8]

    Note that samples 2, 3 and 8 are tested more than once while others are
    never tested -- the defining behaviour of a shuffle split.
    """

    def __init__(
        self,
        n_splits: int = 10,
        test_size: Optional[Union[float, int]] = 0.1,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None
    ):
        """Store the iteration count, part sizes and seed. See the class docstring."""
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")

        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for constructor parameters."""
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
        """Return a reproducible string form of the splitter.

        Returns
        -------
        repr_str : str
            Constructor-style representation, e.g.
            ``ShuffleSplit(n_splits=10, test_size=0.1, train_size=None)``.
        """
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
    Test class distribution: [1 1]
    Test class distribution: [1 1]
    Test class distribution: [1 1]
    Test class distribution: [1 1]
    Test class distribution: [1 1]
    """

    def __init__(
        self,
        n_splits: int = 10,
        test_size: Optional[Union[float, int]] = 0.1,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None
    ):
        """Store the iteration count, part sizes and seed. See the class docstring."""
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")

        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for constructor parameters."""
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
        """Return a reproducible string form of the splitter.

        Returns
        -------
        repr_str : str
            Constructor-style representation, e.g.
            ``StratifiedShuffleSplit(n_splits=10, test_size=0.1)``.
        """
        return (
            f"StratifiedShuffleSplit(n_splits={self.n_splits}, "
            f"test_size={self.test_size})"
        )
