"""Group-aware cross-validation: never split a group across train and test.

Reach for these splitters whenever rows are **not** independent because several
of them come from the same source -- multiple readings per patient, several
sessions per user, many sentences from one document, repeated measurements of
one subject. With an ordinary
:class:`~tuiml.evaluation.splitting.KFold`, sibling rows land on both sides of
the split and the model can score well simply by recognising the source, which
silently inflates the estimate. Grouped splitters keep every group whole inside
a single fold, so the test fold always measures generalisation to *unseen*
groups.

Two variants are provided:

* :class:`~tuiml.evaluation.splitting.GroupKFold` -- assigns groups to folds
  round-robin; ignores ``y``.
* :class:`~tuiml.evaluation.splitting.StratifiedGroupKFold` -- also tries to
  balance the class distribution across folds; requires ``y``.

Both require the ``groups`` argument to ``split()`` and need at least
``n_splits`` distinct groups.
"""

import numpy as np
from typing import Iterator, Optional, Tuple
from tuiml.base.splitting import BaseSplitter

class GroupKFold(BaseSplitter):
    """K-fold over **whole groups**: a group never spans train and test.

    Folding happens at the level of the distinct values in ``groups``, not at
    the level of rows. Each unique group is assigned to exactly one fold, so
    when that fold is the test set every row of the group is held out together,
    and when it is not, every row is in training. As a result no group ever
    appears on both sides of a split -- the property that makes this splitter
    the right choice for clustered data.

    Groups are assigned round-robin in sorted order (``i % n_splits`` over
    ``np.unique(groups)``), so folds hold a similar *number of groups* but not
    necessarily a similar number of *rows*: unequal group sizes give unequal
    fold sizes. Labels are ignored; use
    :class:`~tuiml.evaluation.splitting.StratifiedGroupKFold` if class balance
    matters too. ``groups`` is required and there must be at least ``n_splits``
    distinct groups.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2 and no larger than the number of
        distinct groups.

    Notes
    -----
    6 samples in 3 groups, ``n_splits=3``. Rows of group 1 always move
    together::

        sample:  0 1 2 3 4 5
        group:   1 1 2 2 3 3
        fold 0:  T T . . . .      test group {1}
        fold 1:  . . T T . .      test group {2}
        fold 2:  . . . . T T      test group {3}

        T = test, . = train

    See Also
    --------
    :class:`~tuiml.evaluation.splitting.StratifiedGroupKFold` : Grouped folds that also balance classes.
    :class:`~tuiml.evaluation.splitting.KFold` : Plain folds, no group awareness.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.splitting import GroupKFold
    >>> X = np.arange(12).reshape(6, 2)
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> groups = np.array([1, 1, 2, 2, 3, 3])
    >>> cv = GroupKFold(n_splits=3)
    >>> print(cv.get_n_splits())
    3
    >>> for train_idx, test_idx in cv.split(X, y, groups):
    ...     print(test_idx.tolist(),
    ...           sorted(set(groups[test_idx].tolist())),
    ...           sorted(set(groups[train_idx].tolist())))
    [0, 1] [1] [2, 3]
    [2, 3] [2] [1, 3]
    [4, 5] [3] [1, 2]

    The test and train group sets are always disjoint.
    """

    def __init__(self, n_splits: int = 5):
        """Store the fold count. See the class docstring."""
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for constructor parameters."""
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
        """Yield the ``n_splits`` train/test index pairs of a grouped K-fold.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Only its length is used.
        y : np.ndarray of shape (n_samples,), optional
            Ignored; accepted for a uniform splitter API.
        groups : np.ndarray of shape (n_samples,)
            Group label of each sample. **Required.** Samples sharing a label
            always end up in the same fold.

        Yields
        ------
        train_index : np.ndarray of shape (n_train,)
            Positional indices of the training rows for this fold.
        test_index : np.ndarray of shape (n_test,)
            Positional indices of the held-out rows: every row of the groups
            assigned to this fold, and no row of any other group.

        Raises
        ------
        ValueError
            If ``groups`` is None, has a different length than ``X``, or
            contains fewer distinct values than ``n_splits``.
        """
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
        """Return the number of folds, i.e. how many pairs ``split`` will yield.

        Parameters
        ----------
        X : np.ndarray, optional
            Ignored; the count is fixed at construction time.
        y : np.ndarray, optional
            Ignored.
        groups : np.ndarray, optional
            Ignored.

        Returns
        -------
        n_splits : int
            The configured ``n_splits``.
        """
        return self.n_splits

    def __repr__(self) -> str:
        """Return a short string showing the fold count."""
        return f"GroupKFold(n_splits={self.n_splits})"

class StratifiedGroupKFold(BaseSplitter):
    """Grouped K-fold that **also** tries to balance classes across folds.

    Combines the two guarantees of
    :class:`~tuiml.evaluation.splitting.GroupKFold` and
    :class:`~tuiml.evaluation.splitting.StratifiedKFold`, but only the first is
    exact. A group is still never split across train and test -- that is a hard
    constraint. Class balance is **best effort**: groups are sorted by their
    majority class and then assigned greedily, each group going to whichever
    fold currently holds the fewest samples of that group's majority class.

    Because whole groups are indivisible, exact stratification is generally
    impossible, and the greedy pass optimises class counts rather than fold
    sizes. Expect folds that differ noticeably in the number of groups and rows
    when group sizes or class mixes vary. Both ``y`` and ``groups`` are
    required, and there must be at least ``n_splits`` distinct groups.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2 and no larger than the number of
        distinct groups.
    shuffle : bool, default=False
        Whether to shuffle the group order before the greedy assignment,
        which varies the resulting partition between runs.
    random_state : int, optional
        Random seed for reproducibility. Only used when ``shuffle=True``.

    Notes
    -----
    Fold assignment is driven by each group's majority class, so a group whose
    labels are mixed contributes its minority rows wherever its majority class
    sends it. Verify the realised balance rather than assuming it::

        for train_idx, test_idx in cv.split(X, y, groups):
            print(np.bincount(y[test_idx]))

    See Also
    --------
    :class:`~tuiml.evaluation.splitting.GroupKFold` : Grouped folds without stratification.
    :class:`~tuiml.evaluation.splitting.StratifiedKFold` : Exact stratification, no group awareness.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.splitting import StratifiedGroupKFold
    >>> X = np.arange(24).reshape(12, 2)
    >>> y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    >>> cv = StratifiedGroupKFold(n_splits=3)
    >>> print(cv.get_n_splits())
    3
    >>> for train_idx, test_idx in cv.split(X, y, groups):
    ...     print(sorted(set(groups[test_idx].tolist())),
    ...           np.bincount(y[test_idx], minlength=2).tolist())
    [1, 3, 6] [2, 4]
    [2, 4] [2, 2]
    [5] [0, 2]

    Groups stay intact, but the folds are visibly uneven -- the best-effort
    nature of the stratification in action.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ):
        """Store the fold count and shuffling policy. See the class docstring."""
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for constructor parameters."""
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
        """Yield the ``n_splits`` grouped, class-balanced train/test index pairs.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Only its length is used.
        y : np.ndarray of shape (n_samples,)
            Class labels used for the best-effort balancing. **Required.**
        groups : np.ndarray of shape (n_samples,)
            Group label of each sample. **Required.** Samples sharing a label
            always end up in the same fold.

        Yields
        ------
        train_index : np.ndarray of shape (n_train,)
            Positional indices of the training rows for this fold.
        test_index : np.ndarray of shape (n_test,)
            Positional indices of the held-out rows: all rows of the groups
            assigned to this fold. Group-disjoint from the training rows;
            class proportions are approximate, not guaranteed.

        Raises
        ------
        ValueError
            If ``y`` or ``groups`` is None, if ``groups`` has a different
            length than ``X``, or if there are fewer distinct groups than
            ``n_splits``.
        """
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
            """Return the class label with the most rows inside group ``g``.

            Looks up the per-class counts of group ``g`` (already tallied in
            the enclosing ``group_class_counts``) and returns whichever class
            has the highest count. Ties resolve to whichever class ``max``
            visits first, i.e. the first class label in ``counts``.

            Parameters
            ----------
            g : object
                A group label, i.e. one entry of ``unique_groups``.

            Returns
            -------
            majority_class : object
                The class label that occurs most often among the samples of
                group ``g``.
            """
            counts = group_class_counts[g]
            return max(counts.keys(), key=lambda c: counts.get(c, 0))

        sorted_groups = sorted(unique_groups, key=get_majority_class)

        if self.shuffle:
            # Shuffle within each class group
            rng.shuffle(sorted_groups)

        # Assign groups to folds trying to balance classes
        fold_class_counts = [{} for _ in range(self.n_splits)]
        fold_sizes = [0] * self.n_splits
        group_to_fold = {}

        for group in sorted_groups:
            # Fewest rows of this group's majority class first, then the
            # emptiest fold.
            #
            # The size tiebreak is what keeps every fold populated. Choosing on
            # the majority-class counter alone lets each class fill folds
            # independently from the left: with three class-0 groups and three
            # class-1 groups over five folds, both classes pick folds 0, 1 and
            # 2, and folds 3 and 4 are never assigned anything. That yielded
            # test folds of [20, 20, 20, 0, 0] where GroupKFold managed
            # [20, 10, 10, 10, 10] on the same data, and a fold with no test
            # rows scores nothing while still counting as a fold.
            majority_class = get_majority_class(group)
            best_fold = min(
                range(self.n_splits),
                key=lambda f: (
                    fold_class_counts[f].get(majority_class, 0),
                    fold_sizes[f],
                    f,
                ),
            )

            group_to_fold[group] = best_fold
            fold_sizes[best_fold] += sum(group_class_counts[group].values())

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
        """Return the number of folds, i.e. how many pairs ``split`` will yield.

        Parameters
        ----------
        X : np.ndarray, optional
            Ignored; the count is fixed at construction time.
        y : np.ndarray, optional
            Ignored.
        groups : np.ndarray, optional
            Ignored.

        Returns
        -------
        n_splits : int
            The configured ``n_splits``.
        """
        return self.n_splits

    def __repr__(self) -> str:
        """Return a short string showing the fold count and shuffle flag."""
        return f"StratifiedGroupKFold(n_splits={self.n_splits}, shuffle={self.shuffle})"
