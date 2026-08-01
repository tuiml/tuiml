"""Single train/test holdout splitting.

Holdout evaluation carves the data into **one** train part and **one** test
part instead of rotating over folds. It costs a single fit, so it is the right
default for a quick sanity check, for a large dataset where one split is
already a reliable estimate, or for carving off a final untouched test set
before any cross-validation happens.

This module offers two shapes of the same idea:

* :func:`~tuiml.evaluation.splitting.train_test_split` -- an eager helper that
  slices the arrays you hand it and returns the pieces directly.
* :class:`~tuiml.evaluation.splitting.HoldoutSplit` and
  :class:`~tuiml.evaluation.splitting.StratifiedHoldoutSplit` -- lazy splitters
  implementing the ``BaseSplitter`` protocol, so they can be dropped into
  :func:`~tuiml.evaluation.splitting.cross_val_score` or any other code that
  expects ``split()``/``get_n_splits()``. Both yield exactly one pair.

Use the stratified variants when ``y`` is a class label; the unstratified ones
can leave a rare class entirely out of the training half. When one split is too
noisy to trust, move to :class:`~tuiml.evaluation.splitting.KFold` or
:class:`~tuiml.evaluation.splitting.ShuffleSplit`.
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
    """Split one or more equal-length arrays into a single train and test part.

    All input arrays are indexed with the *same* pair of index vectors, so rows
    stay aligned across ``X``, ``y`` and any extra arrays (sample weights, ids).
    The output is flattened: for inputs ``(X, y)`` the return order is
    ``X_train, X_test, y_train, y_test``.

    Parameters
    ----------
    *arrays : sequence of array-like
        One or more arrays to split. All must have the same first dimension.
    test_size : float or int, optional
        If float in ``(0, 1)``, the proportion of rows held out for test. If
        int, the absolute number of test rows. Defaults to ``0.25`` when
        neither ``test_size`` nor ``train_size`` is given. The resulting count
        is clipped to ``[1, n_samples - 1]`` so neither part is ever empty.
    train_size : float or int, optional
        Complementary specification of the training part, used only when
        ``test_size`` is None.
    shuffle : bool, default=True
        Whether to permute the rows before splitting. With ``shuffle=False``
        and no ``stratify``, the test part is the **leading** block of rows,
        which is only meaningful if the row order already carries meaning.
    stratify : array-like of shape (n_samples,), optional
        Labels to stratify on (normally ``y``). When given, the split is done
        per class so both parts keep roughly the class proportions of
        ``stratify``, and at least one row of each class lands in the test part.
    random_state : int, optional
        Random seed for reproducible shuffling.

    Returns
    -------
    splits : list of np.ndarray
        ``2 * len(arrays)`` arrays, alternating train part then test part for
        each input array in the order given.

    See Also
    --------
    :class:`~tuiml.evaluation.splitting.HoldoutSplit` : Same split exposed as a lazy splitter object.
    :class:`~tuiml.evaluation.splitting.KFold` : Rotate over folds instead of one split.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.splitting import train_test_split
    >>> X = np.arange(20).reshape(10, 2)
    >>> y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0
    ... )
    >>> print(X_train.shape, X_test.shape)
    (7, 2) (3, 2)
    >>> print(sorted(y_test.tolist()))
    [0, 0, 1]

    Passing ``stratify=y`` forces both parts to mirror the class balance:

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.4, stratify=y, random_state=0
    ... )
    >>> print(np.bincount(y_test).tolist(), np.bincount(y_train).tolist())
    [2, 2] [3, 3]
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
    """Single, **unstratified** train/test holdout exposed as a splitter.

    ``split()`` yields exactly **one** ``(train_index, test_index)`` pair and
    ``get_n_splits()`` always returns ``1``, so this class plugs a plain
    holdout into any code written against the cross-validator protocol.

    The first ``int(n_samples * test_size)`` positions of the (optionally
    shuffled) index vector become the test set and the rest become the training
    set. Labels are ignored, so on class-sorted data an unshuffled holdout can
    put an entire class on one side; use
    :class:`~tuiml.evaluation.splitting.StratifiedHoldoutSplit` for classification.

    Parameters
    ----------
    test_size : float, default=0.3
        Proportion of rows held out for testing. Must be strictly between 0
        and 1; unlike
        :func:`~tuiml.evaluation.splitting.train_test_split`, an absolute count
        is not accepted here.
    shuffle : bool, default=True
        Whether to permute the sample order before cutting. With
        ``shuffle=False`` the test set is the leading block of rows.
    random_state : int, optional
        Random seed for reproducibility. Only used when ``shuffle=True``.

    Notes
    -----
    Layout for ``test_size=0.3`` over 10 samples, one column per position of
    the (possibly shuffled) index vector::

        position:  0 1 2 3 4 5 6 7 8 9
        split 0:   T T T . . . . . . .

        T = test, . = train

    See Also
    --------
    :class:`~tuiml.evaluation.splitting.StratifiedHoldoutSplit` : Same, with class proportions preserved.
    :func:`~tuiml.evaluation.splitting.train_test_split` : Eager function returning the sliced arrays.
    :class:`~tuiml.evaluation.splitting.ShuffleSplit` : Repeat this random holdout ``n_splits`` times.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.splitting import HoldoutSplit
    >>> X = np.arange(20).reshape(10, 2)
    >>> cv = HoldoutSplit(test_size=0.3, random_state=0)
    >>> print(cv.get_n_splits())
    1
    >>> for train_idx, test_idx in cv.split(X):
    ...     print(len(train_idx), len(test_idx), sorted(test_idx.tolist()))
    7 3 [2, 4, 8]
    """

    def __init__(
        self,
        test_size: float = 0.3,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        """Store the holdout proportion and shuffling policy. See the class docstring."""
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for constructor parameters."""
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
        """Yield the single train/test index pair of an unstratified holdout.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to split. Only its length is used.
        y : np.ndarray of shape (n_samples,), optional
            Ignored; accepted for a uniform splitter API.
        groups : np.ndarray of shape (n_samples,), optional
            Ignored; accepted for a uniform splitter API.

        Yields
        ------
        train_index : np.ndarray of shape (n_train,)
            Positional indices of the training rows.
        test_index : np.ndarray of shape (n_test,)
            Positional indices of the held-out rows, where
            ``n_test == int(n_samples * test_size)``. Exactly one pair is
            produced.
        """
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
        """Return 1: a holdout produces a single train/test pair.

        Parameters
        ----------
        X : np.ndarray, optional
            Ignored.
        y : np.ndarray, optional
            Ignored.
        groups : np.ndarray, optional
            Ignored.

        Returns
        -------
        n_splits : int
            Always ``1``.
        """
        return 1

    def __repr__(self) -> str:
        """Return a short string showing the test proportion and shuffle flag."""
        return f"HoldoutSplit(test_size={self.test_size}, shuffle={self.shuffle})"

class StratifiedHoldoutSplit(BaseSplitter):
    """Single, **stratified** train/test holdout exposed as a splitter.

    Like :class:`~tuiml.evaluation.splitting.HoldoutSplit`, ``split()`` yields
    exactly one ``(train_index, test_index)`` pair and ``get_n_splits()``
    returns ``1``. The difference is that ``y`` is **required** and the holdout
    is taken *within each class*: from every class,
    ``max(1, int(n_class * test_size))`` samples go to the test part. Both parts
    therefore keep roughly the class distribution of ``y``, and every class is
    guaranteed at least one test sample even when it is tiny.

    Parameters
    ----------
    test_size : float, default=0.3
        Proportion of each class held out for testing. Must be strictly
        between 0 and 1.
    shuffle : bool, default=True
        Whether to permute each class's indices before slicing, and to shuffle
        the two resulting index vectors. With ``shuffle=False`` the test part
        of each class is its leading block.
    random_state : int, optional
        Random seed for reproducibility. Only used when ``shuffle=True``.

    Notes
    -----
    10 samples, classes ``0 0 0 0 0 1 1 1 1 1``, ``test_size=0.4``. Two samples
    are drawn from each class rather than four from one end::

        class:    0 0 0 0 0 1 1 1 1 1
        split 0:  T T . . . T T . . .

        T = test, . = train

    Because the per-class count is rounded up with ``max(1, ...)``, the total
    test size can slightly exceed ``test_size * n_samples`` when there are many
    small classes.

    See Also
    --------
    :class:`~tuiml.evaluation.splitting.HoldoutSplit` : Same, ignoring class labels.
    :func:`~tuiml.evaluation.splitting.train_test_split` : Eager equivalent via ``stratify=y``.
    :class:`~tuiml.evaluation.splitting.StratifiedKFold` : Rotate over stratified folds instead.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.splitting import StratifiedHoldoutSplit
    >>> X = np.arange(20).reshape(10, 2)
    >>> y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> cv = StratifiedHoldoutSplit(test_size=0.4, random_state=0)
    >>> print(cv.get_n_splits())
    1
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     print(len(train_idx), len(test_idx), np.bincount(y[test_idx]).tolist())
    6 4 [2, 2]
    """

    def __init__(
        self,
        test_size: float = 0.3,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        """Store the holdout proportion and shuffling policy. See the class docstring."""
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for constructor parameters."""
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
        """Yield the single train/test index pair of a stratified holdout.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to split. Only its length is used.
        y : np.ndarray of shape (n_samples,)
            Class labels driving the stratification. **Required.**
        groups : np.ndarray of shape (n_samples,), optional
            Ignored; accepted for a uniform splitter API.

        Yields
        ------
        train_index : np.ndarray of shape (n_train,)
            Positional indices of the training rows.
        test_index : np.ndarray of shape (n_test,)
            Positional indices of the held-out rows, holding approximately the
            class proportions of ``y`` and at least one sample per class.
            Exactly one pair is produced.

        Raises
        ------
        ValueError
            If ``y`` is None.
        """
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
        """Return 1: a holdout produces a single train/test pair.

        Parameters
        ----------
        X : np.ndarray, optional
            Ignored.
        y : np.ndarray, optional
            Ignored.
        groups : np.ndarray, optional
            Ignored.

        Returns
        -------
        n_splits : int
            Always ``1``.
        """
        return 1

    def __repr__(self) -> str:
        """Return a short string showing the test proportion."""
        return f"StratifiedHoldoutSplit(test_size={self.test_size})"
