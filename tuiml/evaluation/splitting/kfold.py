"""
K-Fold cross-validation splitters.
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
    """
    Evaluate an estimator using cross-validation.

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data. Must implement fit and predict methods.
    X : ndarray of shape (n_samples, n_features)
        The data to fit.
    y : ndarray of shape (n_samples,)
        The target variable.
    cv : int or cross-validator, default=5
        If int, number of folds for KFold cross-validation.
        If cross-validator, a splitter object with split method.
    scoring : str or callable, optional
        If None, uses accuracy for classification.
        If str, one of 'accuracy', 'f1', 'precision', 'recall', 'r2', 'mse', 'mae'.
        If callable, a function with signature scoring(y_true, y_pred) -> float.
    random_state : int, optional
        Random seed for reproducibility when cv is int.

    Returns
    -------
    scores : ndarray of shape (n_splits,)
        Array of scores for each fold.

    Examples
    --------
    >>> from tuiml.evaluation import cross_val_score, KFold
    >>> from tuiml.algorithms.trees import RandomForestClassifier
    >>> from tuiml.datasets import load_iris
    >>>
    >>> X, y = load_iris()
    >>> model = RandomForestClassifier(n_estimators=100)
    >>>
    >>> # Using int for cv
    >>> scores = cross_val_score(model, X, y, cv=5)
    >>> print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    >>>
    >>> # Using a splitter object
    >>> kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    >>> scores = cross_val_score(model, X, y, cv=kfold)
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
            return np.mean(y_true == y_pred)
    elif isinstance(scoring, str):
        scoring_lower = scoring.lower()
        if scoring_lower == 'accuracy':
            def scorer(y_true, y_pred):
                return np.mean(y_true == y_pred)
        elif scoring_lower == 'f1':
            def scorer(y_true, y_pred):
                from tuiml.evaluation.metrics import f1_score
                return f1_score(y_true, y_pred, average='macro')
        elif scoring_lower == 'precision':
            def scorer(y_true, y_pred):
                from tuiml.evaluation.metrics import precision_score
                return precision_score(y_true, y_pred, average='macro')
        elif scoring_lower == 'recall':
            def scorer(y_true, y_pred):
                from tuiml.evaluation.metrics import recall_score
                return recall_score(y_true, y_pred, average='macro')
        elif scoring_lower == 'r2':
            def scorer(y_true, y_pred):
                from tuiml.evaluation.metrics import r2_score
                return r2_score(y_true, y_pred)
        elif scoring_lower in ('mse', 'neg_mean_squared_error'):
            def scorer(y_true, y_pred):
                from tuiml.evaluation.metrics import mean_squared_error
                return -mean_squared_error(y_true, y_pred)
        elif scoring_lower in ('mae', 'neg_mean_absolute_error'):
            def scorer(y_true, y_pred):
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
    """
    K-Fold cross-validation splitter.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle before splitting.
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import KFold
    >>> import numpy as np
    >>> X = np.arange(10).reshape(-1, 1)
    >>> kf = KFold(n_splits=5)
    >>> for train_idx, test_idx in kf.split(X):
    ...     print(f"Train: {train_idx}, Test: {test_idx}")
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
        """Generate K-fold indices."""
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
        """Get number of splits."""
        return self.n_splits

    def __repr__(self) -> str:
        return f"KFold(n_splits={self.n_splits}, shuffle={self.shuffle})"

class StratifiedKFold(BaseSplitter):
    """
    Stratified K-Fold cross-validation.

    Preserves class proportions in each fold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle before splitting.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import StratifiedKFold
    >>> import numpy as np
    >>> X = np.arange(10).reshape(-1, 1)
    >>> y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> skf = StratifiedKFold(n_splits=5)
    >>> for train_idx, test_idx in skf.split(X, y):
    ...     print(f"Train classes: {y[train_idx]}, Test classes: {y[test_idx]}")
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
        """Generate stratified K-fold indices."""
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
        """Get number of splits."""
        return self.n_splits

    def __repr__(self) -> str:
        return f"StratifiedKFold(n_splits={self.n_splits}, shuffle={self.shuffle})"

class RepeatedKFold(BaseSplitter):
    """
    Repeated K-Fold cross-validation.

    Repeats K-Fold n_repeats times with different randomization.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    n_repeats : int, default=10
        Number of times to repeat.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> from tuiml.evaluation.splitting import RepeatedKFold
    >>> rkf = RepeatedKFold(n_splits=5, n_repeats=3)
    >>> # This gives 15 total splits (5 folds × 3 repeats)
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate repeated K-fold indices."""
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
        """Get number of splits."""
        return self.n_splits * self.n_repeats

    def __repr__(self) -> str:
        return f"RepeatedKFold(n_splits={self.n_splits}, n_repeats={self.n_repeats})"

class RepeatedStratifiedKFold(BaseSplitter):
    """
    Repeated Stratified K-Fold cross-validation.

    Repeats Stratified K-Fold n_repeats times.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    n_repeats : int, default=10
        Number of times to repeat.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate repeated stratified K-fold indices."""
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
        """Get number of splits."""
        return self.n_splits * self.n_repeats

    def __repr__(self) -> str:
        return f"RepeatedStratifiedKFold(n_splits={self.n_splits}, n_repeats={self.n_repeats})"
