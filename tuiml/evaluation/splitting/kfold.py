"""Fold-based cross-validation splitters and the ``cross_val_score`` driver.

This module contains the cross-validators that partition a dataset into
``n_splits`` mutually exclusive test folds, plus a convenience function that
fits and scores an estimator across any splitter:

* :class:`~tuiml.evaluation.splitting.KFold` -- consecutive (optionally
  shuffled) folds; ignores ``y`` entirely.
* :class:`~tuiml.evaluation.splitting.StratifiedKFold` -- folds built per class
  so every fold keeps roughly the class distribution of ``y``.
* :class:`~tuiml.evaluation.splitting.RepeatedKFold` and
  :class:`~tuiml.evaluation.splitting.RepeatedStratifiedKFold` -- run the above
  ``n_repeats`` times with a fresh shuffle each time, yielding
  ``n_splits * n_repeats`` train/test pairs.
* :func:`~tuiml.evaluation.splitting.cross_val_score` -- clone, fit and score an
  estimator once per split and return the array of fold scores.

Reach for ``KFold`` on regression or well-balanced data, ``StratifiedKFold``
whenever class labels are imbalanced (a plain ``KFold`` fold can otherwise miss
a rare class completely), and the repeated variants when a single K-fold
estimate is too noisy to separate two candidate models.

Every splitter here assumes samples are **exchangeable**. For temporally
ordered data use :class:`~tuiml.evaluation.splitting.TimeSeriesSplit`; when
samples are clustered (same patient, same user, same document) use
:class:`~tuiml.evaluation.splitting.GroupKFold` instead, otherwise leakage
across folds inflates the score.
"""

import numpy as np
from typing import Any, Callable, Iterator, Optional, Tuple, Union
from tuiml.base.splitting import BaseSplitter


def cross_val_score(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: Union[int, "BaseSplitter"] = 5,
    scoring: Optional[Union[str, Callable]] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Fit and score an estimator once per cross-validation split.

    For every ``(train_index, test_index)`` pair produced by ``cv`` the
    estimator is cloned (via ``estimator.__class__(**estimator.get_params())``),
    fitted on the training rows and scored on the held-out rows. The original
    ``estimator`` is left unfitted whenever cloning succeeds.

    Parameters
    ----------
    estimator : estimator object
        The object to fit. Must implement ``fit(X, y)`` and ``predict(X)``; a
        ``get_params()`` method is used to clone it between folds.
    X : np.ndarray of shape (n_samples, n_features)
        The data to fit.
    y : np.ndarray of shape (n_samples,)
        The target variable.
    cv : int or BaseSplitter, default=5
        If int, the number of folds for an internal
        :class:`~tuiml.evaluation.splitting.KFold` with ``shuffle=True``.
        If a splitter, any object exposing ``split(X, y)``.
    scoring : str or callable, optional
        If None, accuracy is used. If str, one of ``'accuracy'``, ``'f1'``,
        ``'precision'``, ``'recall'``, ``'r2'``, ``'mse'`` (alias
        ``'neg_mean_squared_error'``) or ``'mae'`` (alias
        ``'neg_mean_absolute_error'``); the two error metrics are negated so
        that larger is always better. If callable, a function with signature
        ``scoring(y_true, y_pred) -> float``.
    random_state : int, optional
        Random seed for the internal shuffle. Only used when ``cv`` is an int;
        a splitter object carries its own seed.

    Returns
    -------
    scores : np.ndarray of shape (n_splits,)
        One score per split, in the order the splitter yielded them.

    See Also
    --------
    :class:`~tuiml.evaluation.splitting.KFold` : Default splitter when ``cv`` is an int.
    :class:`~tuiml.evaluation.splitting.StratifiedKFold` : Class-balanced folds.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.splitting import cross_val_score, StratifiedKFold
    >>> from tuiml.algorithms.bayesian import NaiveBayesClassifier
    >>> rng = np.random.RandomState(0)
    >>> X = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(4, 1, (30, 2))])
    >>> y = np.array([0] * 30 + [1] * 30)
    >>> scores = cross_val_score(NaiveBayesClassifier(), X, y, cv=5, random_state=0)
    >>> print(scores.shape)
    (5,)
    >>> print(round(float(scores.mean()), 3))
    1.0

    Pass a splitter instance for full control over the folds and metric:

    >>> cv = StratifiedKFold(n_splits=4)
    >>> scores = cross_val_score(NaiveBayesClassifier(), X, y, cv=cv, scoring='f1')
    >>> print(len(scores), round(float(scores.mean()), 3))
    4 1.0
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Handle cv parameter
    if isinstance(cv, int):
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        splitter = cv

    # Handle scoring parameter
    if scoring is None:
        # Default to accuracy
        def scorer(y_true, y_pred):
            """Score predictions with accuracy (the default metric).

            Parameters
            ----------
            y_true : np.ndarray of shape (n_samples,)
                True targets for the held-out fold.
            y_pred : np.ndarray of shape (n_samples,)
                Predicted targets for the held-out fold.

            Returns
            -------
            score : float
                Fraction of predictions that exactly match ``y_true``.
            """
            return np.mean(y_true == y_pred)
    elif isinstance(scoring, str):
        scoring_lower = scoring.lower()
        if scoring_lower == 'accuracy':
            def scorer(y_true, y_pred):
                """Score predictions with accuracy (``scoring='accuracy'``).

                Parameters
                ----------
                y_true : np.ndarray of shape (n_samples,)
                    True targets for the held-out fold.
                y_pred : np.ndarray of shape (n_samples,)
                    Predicted targets for the held-out fold.

                Returns
                -------
                score : float
                    Fraction of predictions that exactly match ``y_true``.
                """
                return np.mean(y_true == y_pred)
        elif scoring_lower == 'f1':
            def scorer(y_true, y_pred):
                """Score predictions with macro-averaged F1 (``scoring='f1'``).

                Parameters
                ----------
                y_true : np.ndarray of shape (n_samples,)
                    True targets for the held-out fold.
                y_pred : np.ndarray of shape (n_samples,)
                    Predicted targets for the held-out fold.

                Returns
                -------
                score : float
                    Macro-averaged F1 score.
                """
                from tuiml.evaluation.metrics import f1_score
                return f1_score(y_true, y_pred, average='macro')
        elif scoring_lower == 'precision':
            def scorer(y_true, y_pred):
                """Score predictions with macro-averaged precision (``scoring='precision'``).

                Parameters
                ----------
                y_true : np.ndarray of shape (n_samples,)
                    True targets for the held-out fold.
                y_pred : np.ndarray of shape (n_samples,)
                    Predicted targets for the held-out fold.

                Returns
                -------
                score : float
                    Macro-averaged precision score.
                """
                from tuiml.evaluation.metrics import precision_score
                return precision_score(y_true, y_pred, average='macro')
        elif scoring_lower == 'recall':
            def scorer(y_true, y_pred):
                """Score predictions with macro-averaged recall (``scoring='recall'``).

                Parameters
                ----------
                y_true : np.ndarray of shape (n_samples,)
                    True targets for the held-out fold.
                y_pred : np.ndarray of shape (n_samples,)
                    Predicted targets for the held-out fold.

                Returns
                -------
                score : float
                    Macro-averaged recall score.
                """
                from tuiml.evaluation.metrics import recall_score
                return recall_score(y_true, y_pred, average='macro')
        elif scoring_lower == 'r2':
            def scorer(y_true, y_pred):
                """Score predictions with the coefficient of determination (``scoring='r2'``).

                Parameters
                ----------
                y_true : np.ndarray of shape (n_samples,)
                    True targets for the held-out fold.
                y_pred : np.ndarray of shape (n_samples,)
                    Predicted targets for the held-out fold.

                Returns
                -------
                score : float
                    :math:`R^2` score.
                """
                from tuiml.evaluation.metrics import r2_score
                return r2_score(y_true, y_pred)
        elif scoring_lower in ('mse', 'neg_mean_squared_error'):
            def scorer(y_true, y_pred):
                """Score predictions with negated mean squared error (``scoring='mse'``).

                Negated so that, like every other scorer here, larger is better.

                Parameters
                ----------
                y_true : np.ndarray of shape (n_samples,)
                    True targets for the held-out fold.
                y_pred : np.ndarray of shape (n_samples,)
                    Predicted targets for the held-out fold.

                Returns
                -------
                score : float
                    Negative mean squared error.
                """
                from tuiml.evaluation.metrics import mean_squared_error
                return -mean_squared_error(y_true, y_pred)
        elif scoring_lower in ('mae', 'neg_mean_absolute_error'):
            def scorer(y_true, y_pred):
                """Score predictions with negated mean absolute error (``scoring='mae'``).

                Negated so that, like every other scorer here, larger is better.

                Parameters
                ----------
                y_true : np.ndarray of shape (n_samples,)
                    True targets for the held-out fold.
                y_pred : np.ndarray of shape (n_samples,)
                    Predicted targets for the held-out fold.

                Returns
                -------
                score : float
                    Negative mean absolute error.
                """
                from tuiml.evaluation.metrics import mean_absolute_error
                return -mean_absolute_error(y_true, y_pred)
        else:
            raise ValueError(
                f"Unknown scoring '{scoring}'. Valid options: "
                "'accuracy', 'f1', 'precision', 'recall', 'r2', 'mse', 'mae'"
            )
    else:
        scorer = scoring

    scores = []
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone the estimator
        try:
            est = estimator.__class__(**estimator.get_params())
        except (AttributeError, TypeError):
            # Fallback: use the estimator directly (not ideal but works)
            est = estimator

        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        scores.append(scorer(y_test, y_pred))

    return np.array(scores)

class KFold(BaseSplitter):
    """**Unstratified** K-fold cross-validator over consecutive index blocks.

    The ``n_samples`` rows are cut into ``n_splits`` folds. Each fold serves
    once as the test set while the remaining ``n_splits - 1`` folds form the
    training set, so every sample is tested exactly once and appears in
    ``n_splits - 1`` training sets. Labels are ignored: ``y`` is accepted only
    for API compatibility and never influences the partition. If class balance
    matters, use :class:`~tuiml.evaluation.splitting.StratifiedKFold`.

    With ``shuffle=False`` (the default) the folds are consecutive slices of
    ``0..n_samples-1``, which makes the split reproducible without a seed but
    dangerous on data that arrives sorted by label. With ``shuffle=True`` the
    index vector is permuted once, before folding, so the folds are random but
    still disjoint.

    When ``n_samples`` is not divisible by ``n_splits`` the first
    ``n_samples % n_splits`` folds get one extra sample, so fold sizes differ by
    at most one.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2 and no larger than ``n_samples``.
    shuffle : bool, default=False
        Whether to permute the sample order once before cutting the folds.
    random_state : int, optional
        Random seed for reproducibility. Only has an effect when
        ``shuffle=True``.

    Notes
    -----
    Layout for ``n_splits=5`` over 10 samples, one column per sample::

        fold 0:  T T . . . . . . . .
        fold 1:  . . T T . . . . . .
        fold 2:  . . . . T T . . . .
        fold 3:  . . . . . . T T . .
        fold 4:  . . . . . . . . T T

        T = test, . = train

    Each split is roughly :math:`(k-1)/k` train and :math:`1/k` test, so larger
    ``n_splits`` means more training data per fit, lower bias, higher variance
    and :math:`k` times the compute.

    See Also
    --------
    :class:`~tuiml.evaluation.splitting.StratifiedKFold` : Same folds, but class proportions preserved.
    :class:`~tuiml.evaluation.splitting.RepeatedKFold` : Repeat this splitter with a new shuffle each time.
    :class:`~tuiml.evaluation.splitting.GroupKFold` : Keep samples of a group inside one fold.
    :class:`~tuiml.evaluation.splitting.TimeSeriesSplit` : Ordered splits that never train on the future.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.splitting import KFold
    >>> X = np.arange(20).reshape(10, 2)
    >>> cv = KFold(n_splits=5)
    >>> print(cv.get_n_splits())
    5
    >>> for train_idx, test_idx in cv.split(X):
    ...     print(test_idx.tolist(), len(train_idx))
    [0, 1] 8
    [2, 3] 8
    [4, 5] 8
    [6, 7] 8
    [8, 9] 8

    Shuffling scatters the folds but keeps them disjoint:

    >>> cv = KFold(n_splits=2, shuffle=True, random_state=0)
    >>> for train_idx, test_idx in cv.split(X):
    ...     print(sorted(test_idx.tolist()))
    [1, 2, 4, 8, 9]
    [0, 3, 5, 6, 7]
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
                "minimum": 2,
                "description": "Number of folds"
            },
            "shuffle": {
                "type": "boolean",
                "default": False,
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
        """Yield the ``n_splits`` train/test index pairs of a plain K-fold.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Only its length is used.
        y : np.ndarray of shape (n_samples,), optional
            Ignored; accepted for a uniform splitter API.
        groups : np.ndarray of shape (n_samples,), optional
            Ignored; accepted for a uniform splitter API.

        Yields
        ------
        train_index : np.ndarray of shape (n_train,)
            Positional indices into ``X`` of the training rows for this fold.
        test_index : np.ndarray of shape (n_test,)
            Positional indices of the held-out rows. Test folds across the
            whole iteration are disjoint and cover every sample exactly once.

        Raises
        ------
        ValueError
            If ``n_splits`` exceeds ``n_samples``.
        """
        X, y = self._validate_input(X, y)
        n_samples = len(X)

        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than "
                f"n_samples={n_samples}"
            )

        indices = np.arange(n_samples)
        rng = np.random.RandomState(self.random_state)

        if self.shuffle:
            rng.shuffle(indices)

        # Calculate fold sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_indices = indices[current:current + fold_size]
            train_indices = np.concatenate([
                indices[:current],
                indices[current + fold_size:]
            ])
            yield train_indices, test_indices
            current += fold_size

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
        return f"KFold(n_splits={self.n_splits}, shuffle={self.shuffle})"

class StratifiedKFold(BaseSplitter):
    """**Stratified** K-fold cross-validator: every fold mirrors the class mix.

    Unlike :class:`~tuiml.evaluation.splitting.KFold`, this splitter requires
    ``y`` and folds **each class separately**, then concatenates the per-class
    slices into ``n_splits`` test folds. A class holding a fraction :math:`p` of
    the data therefore holds roughly :math:`p` of every fold, up to rounding.
    That keeps a rare class from vanishing out of a training fold (which makes
    the fitted model unable to predict it at all) or piling up in a single test
    fold (which makes the fold score meaningless).

    Test folds remain disjoint and jointly cover all samples, exactly as in
    plain K-fold; only the assignment of samples to folds differs. With
    ``shuffle=True`` the indices of each class are permuted before slicing, so
    the composition of a fold changes but its class proportions do not.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2. For clean stratification it should
        not exceed the size of the smallest class.
    shuffle : bool, default=False
        Whether to permute the indices within each class before folding.
    random_state : int, optional
        Random seed for reproducibility. Only used when ``shuffle=True``.

    Notes
    -----
    10 samples, classes ``0 0 0 0 0 1 1 1 1 1``, ``n_splits=5``. Each fold takes
    one sample from each class rather than a contiguous block::

        class:   0 0 0 0 0 1 1 1 1 1
        fold 0:  T . . . . T . . . .
        fold 1:  . T . . . . T . . .
        fold 2:  . . T . . . . T . .
        fold 3:  . . . T . . . . T .
        fold 4:  . . . . T . . . . T

        T = test, . = train

    See Also
    --------
    :class:`~tuiml.evaluation.splitting.KFold` : Same folding, without stratification.
    :class:`~tuiml.evaluation.splitting.RepeatedStratifiedKFold` : Repeat this splitter with a new shuffle each time.
    :class:`~tuiml.evaluation.splitting.StratifiedShuffleSplit` : Stratified random splits that may overlap.
    :class:`~tuiml.evaluation.splitting.StratifiedGroupKFold` : Stratified folds that also respect groups.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.splitting import StratifiedKFold
    >>> X = np.arange(20).reshape(10, 2)
    >>> y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> cv = StratifiedKFold(n_splits=5)
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     print(sorted(test_idx.tolist()), np.bincount(y[test_idx]).tolist())
    [0, 5] [1, 1]
    [1, 6] [1, 1]
    [2, 7] [1, 1]
    [3, 8] [1, 1]
    [4, 9] [1, 1]

    Every test fold holds one sample of each class, which a plain ``KFold``
    would not guarantee on this label ordering.
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
                "minimum": 2,
                "description": "Number of folds"
            },
            "shuffle": {
                "type": "boolean",
                "default": False,
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
        """Yield the ``n_splits`` train/test index pairs of a stratified K-fold.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Only its length is used.
        y : np.ndarray of shape (n_samples,)
            Class labels driving the stratification. **Required.**
        groups : np.ndarray of shape (n_samples,), optional
            Ignored; accepted for a uniform splitter API.

        Yields
        ------
        train_index : np.ndarray of shape (n_train,)
            Positional indices of the training rows for this fold.
        test_index : np.ndarray of shape (n_test,)
            Positional indices of the held-out rows, holding approximately the
            same class proportions as ``y`` as a whole.

        Raises
        ------
        ValueError
            If ``y`` is None.
        """
        X, y = self._validate_input(X, y)

        if y is None:
            raise ValueError("y is required for StratifiedKFold")

        n_samples = len(X)
        indices = np.arange(n_samples)
        rng = np.random.RandomState(self.random_state)

        # Get unique classes and their indices
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)

        # Group indices by class
        class_indices = [indices[y_indices == c] for c in range(n_classes)]

        if self.shuffle:
            for ci in class_indices:
                rng.shuffle(ci)

        # Create stratified folds
        test_folds = [[] for _ in range(self.n_splits)]

        for ci in class_indices:
            fold_sizes = np.full(self.n_splits, len(ci) // self.n_splits)
            fold_sizes[:len(ci) % self.n_splits] += 1

            current = 0
            for fold_idx, fold_size in enumerate(fold_sizes):
                test_folds[fold_idx].extend(ci[current:current + fold_size])
                current += fold_size

        # Generate splits
        all_indices = set(indices)
        for test_fold in test_folds:
            test_indices = np.array(test_fold)
            train_indices = np.array(list(all_indices - set(test_fold)))
            yield train_indices, test_indices

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
        return f"StratifiedKFold(n_splits={self.n_splits}, shuffle={self.shuffle})"

class RepeatedKFold(BaseSplitter):
    """Run an **unstratified** K-fold ``n_repeats`` times with a fresh shuffle.

    Each repeat builds a
    :class:`~tuiml.evaluation.splitting.KFold` with ``shuffle=True`` and a seed
    drawn from this splitter's own generator, then yields all of its folds
    before the next repeat starts. The result is
    ``n_splits * n_repeats`` train/test pairs, emitted repeat by repeat.

    Within one repeat the test folds are disjoint and cover the data exactly
    once; **across** repeats they overlap, since every repeat re-partitions the
    same samples. Averaging over all pairs therefore shrinks the variance of a
    cross-validation estimate without giving more independent data, which is
    what makes it useful when a single K-fold score is too noisy to rank two
    models. Labels are ignored -- use
    :class:`~tuiml.evaluation.splitting.RepeatedStratifiedKFold` for classification.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds within each repeat.
    n_repeats : int, default=10
        Number of times the whole K-fold is repeated.
    random_state : int, optional
        Seed for the generator that produces one sub-seed per repeat, making
        the entire sequence of ``n_splits * n_repeats`` folds reproducible.

    Notes
    -----
    Total fits when driven by
    :func:`~tuiml.evaluation.splitting.cross_val_score` is
    ``n_splits * n_repeats``, so cost grows linearly in both.

    See Also
    --------
    :class:`~tuiml.evaluation.splitting.KFold` : The splitter repeated here.
    :class:`~tuiml.evaluation.splitting.RepeatedStratifiedKFold` : Stratified counterpart.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.splitting import RepeatedKFold
    >>> X = np.arange(20).reshape(10, 2)
    >>> cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=0)
    >>> print(cv.get_n_splits())
    6
    >>> print(sum(1 for _ in cv.split(X)))
    6
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[int] = None
    ):
        """Store the fold count, repeat count and seed. See the class docstring."""
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``n_splits * n_repeats`` train/test index pairs, repeat by repeat.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Only its length is used.
        y : np.ndarray of shape (n_samples,), optional
            Ignored by the underlying
            :class:`~tuiml.evaluation.splitting.KFold`; passed through only for
            API symmetry.
        groups : np.ndarray of shape (n_samples,), optional
            Ignored; passed through for API symmetry.

        Yields
        ------
        train_index : np.ndarray of shape (n_train,)
            Positional indices of the training rows for this fold.
        test_index : np.ndarray of shape (n_test,)
            Positional indices of the held-out rows. Disjoint within a repeat,
            overlapping across repeats.
        """
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_repeats):
            kf = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=rng.randint(0, 2**31)
            )
            yield from kf.split(X, y, groups)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the total number of folds across all repeats.

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
            ``n_splits * n_repeats``.
        """
        return self.n_splits * self.n_repeats

    def __repr__(self) -> str:
        """Return a short string showing the fold and repeat counts."""
        return f"RepeatedKFold(n_splits={self.n_splits}, n_repeats={self.n_repeats})"

class RepeatedStratifiedKFold(BaseSplitter):
    """Run a **stratified** K-fold ``n_repeats`` times with a fresh shuffle.

    Each repeat builds a
    :class:`~tuiml.evaluation.splitting.StratifiedKFold` with ``shuffle=True``
    and a seed drawn from this splitter's own generator, then yields all of its
    folds before the next repeat begins. Every one of the
    ``n_splits * n_repeats`` test folds therefore keeps approximately the class
    proportions of ``y``, and ``y`` is **required**.

    Use this instead of :class:`~tuiml.evaluation.splitting.RepeatedKFold`
    whenever the target is a class label, especially with imbalanced classes or
    small datasets where a single stratified K-fold estimate is still noisy.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds within each repeat.
    n_repeats : int, default=10
        Number of times the whole stratified K-fold is repeated.
    random_state : int, optional
        Seed for the generator that produces one sub-seed per repeat, making
        the entire sequence of folds reproducible.

    See Also
    --------
    :class:`~tuiml.evaluation.splitting.StratifiedKFold` : The splitter repeated here.
    :class:`~tuiml.evaluation.splitting.RepeatedKFold` : Unstratified counterpart.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.splitting import RepeatedStratifiedKFold
    >>> X = np.arange(20).reshape(10, 2)
    >>> y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=0)
    >>> print(cv.get_n_splits())
    6
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     print(np.bincount(y[test_idx]).tolist())
    [3, 3]
    [2, 2]
    [3, 3]
    [2, 2]
    [3, 3]
    [2, 2]

    The two classes stay balanced in every fold of every repeat.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[int] = None
    ):
        """Store the fold count, repeat count and seed. See the class docstring."""
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``n_splits * n_repeats`` stratified train/test pairs, repeat by repeat.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Only its length is used.
        y : np.ndarray of shape (n_samples,)
            Class labels driving the stratification. **Required.**
        groups : np.ndarray of shape (n_samples,), optional
            Ignored; passed through for API symmetry.

        Yields
        ------
        train_index : np.ndarray of shape (n_train,)
            Positional indices of the training rows for this fold.
        test_index : np.ndarray of shape (n_test,)
            Positional indices of the held-out rows, class-balanced like ``y``.
            Disjoint within a repeat, overlapping across repeats.
        """
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_repeats):
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=rng.randint(0, 2**31)
            )
            yield from skf.split(X, y, groups)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the total number of folds across all repeats.

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
            ``n_splits * n_repeats``.
        """
        return self.n_splits * self.n_repeats

    def __repr__(self) -> str:
        """Return a short string showing the fold and repeat counts."""
        return f"RepeatedStratifiedKFold(n_splits={self.n_splits}, n_repeats={self.n_repeats})"
