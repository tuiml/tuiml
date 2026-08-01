"""
Base classes for hyperparameter tuning.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from copy import deepcopy
import time
import warnings

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

@dataclass
class TuningResult:
    """Encapsulates the outcome of a hyperparameter optimization run.

    Attributes
    ----------
    best_params : dict
        The configuration of parameters yielding the highest score.
    best_score : float
        The validation score achieved by the ``best_params``.
    best_estimator : Any
        The model instance retrained on the full dataset using ``best_params``.
    cv_results : dict
        A comprehensive log of all evaluation scores, split times, and
        parameter combinations.
    n_iterations : int
        The total number of parameter variations explored.
    total_time : float
        The wall-clock time consumed by the entire search in seconds.

    See Also
    --------
    :class:`~tuiml.base.tuning.BaseTuner` : Produces these results.
    """
    best_params: Dict[str, Any]
    best_score: float
    best_estimator: Any
    cv_results: Dict[str, Any] = field(default_factory=dict)
    n_iterations: int = 0
    total_time: float = 0.0

class ParameterGrid:
    """Cartesian product of parameter values for exhaustive search.

    Generates every possible combination of parameter values defined in the
    search space.

    Parameters
    ----------
    param_grid : dict or list of dict
        A dictionary where keys are parameter names and values are lists
        of settings to try. Alternatively, a list of such dictionaries
        to define disjoint search spaces.

    See Also
    --------
    :class:`~tuiml.base.tuning.ParameterDistribution` : Sampled search spaces
        for random search.

    Examples
    --------
    Exhaustively search a support vector machine space:

    >>> from tuiml.base import ParameterGrid
    >>> grid = ParameterGrid({'C': [1, 10], 'kernel': ['linear', 'rbf']})
    >>> len(grid)
    4
    >>> list(grid)[0]
    {'C': 1, 'kernel': 'linear'}
    """

    def __init__(self, param_grid: Union[Dict, List[Dict]]):
        """Initialize the grid from one or more parameter mappings."""
        if isinstance(param_grid, dict):
            self.param_grids = [param_grid]
        else:
            self.param_grids = list(param_grid)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all parameter combinations.

        Returns
        -------
        combinations : Iterator of dict
            One dictionary per parameter combination.
        """
        for grid in self.param_grids:
            yield from self._product_dict(grid)

    def _product_dict(self, d: Dict) -> Iterator[Dict[str, Any]]:
        """Yield the cartesian product of a dictionary's value lists."""
        keys = list(d.keys())
        if not keys:
            yield {}
            return

        values = [d[k] if isinstance(d[k], (list, tuple)) else [d[k]] for k in keys]

        # Generate all combinations
        indices = [0] * len(keys)
        while True:
            yield {k: values[i][indices[i]] for i, k in enumerate(keys)}

            # Increment indices
            for i in range(len(keys) - 1, -1, -1):
                indices[i] += 1
                if indices[i] < len(values[i]):
                    break
                indices[i] = 0
            else:
                break

    def __len__(self) -> int:
        """Return the total number of parameter combinations.

        Returns
        -------
        n_combinations : int
            Number of distinct parameter combinations in the grid.
        """
        total = 0
        for grid in self.param_grids:
            n = 1
            for v in grid.values():
                if isinstance(v, (list, tuple)):
                    n *= len(v)
            total += n
        return total

class ParameterDistribution:
    """Distribution of parameters for random search.

    Parameters
    ----------
    param_distributions : dict
        Dictionary with parameters as keys and distributions as values.
        Values can be:

        - List: uniform choice from list
        - Tuple ``(low, high)``: uniform continuous
        - Tuple ``(low, high, 'log')``: log-uniform
        - Tuple ``(low, high, 'int')``: uniform integer
        - Callable: distribution that returns a sample

    See Also
    --------
    :class:`~tuiml.base.tuning.ParameterGrid` : Exhaustive search spaces
        for grid search.

    Examples
    --------
    >>> from tuiml.base import ParameterDistribution
    >>> dist = ParameterDistribution({
    ...     'C': (0.1, 10, 'log'),      # Log-uniform between 0.1 and 10
    ...     'kernel': ['linear', 'rbf'], # Uniform choice
    ...     'gamma': (0.001, 1)          # Uniform continuous
    ... })
    >>> sample = dist.sample()
    """

    def __init__(self, param_distributions: Dict):
        """Initialize the distribution from a parameter mapping."""
        self.param_distributions = param_distributions

    def sample(self, random_state: Optional[int] = None) -> Dict[str, Any]:
        """Sample one parameter combination from the distributions.

        Parameters
        ----------
        random_state : int, optional
            Seed for the random number generator used for sampling.

        Returns
        -------
        params : dict
            One sampled value per parameter.
        """
        rng = np.random.RandomState(random_state)
        params = {}

        for key, dist in self.param_distributions.items():
            if callable(dist):
                params[key] = dist()
            elif isinstance(dist, (list, tuple)) and not self._is_range(dist):
                params[key] = rng.choice(dist)
            elif isinstance(dist, tuple):
                if len(dist) == 3 and dist[2] == 'log':
                    # Log-uniform
                    low, high = np.log(dist[0]), np.log(dist[1])
                    params[key] = np.exp(rng.uniform(low, high))
                elif len(dist) == 3 and dist[2] == 'int':
                    # Uniform integer
                    params[key] = rng.randint(dist[0], dist[1] + 1)
                else:
                    # Uniform continuous
                    params[key] = rng.uniform(dist[0], dist[1])

        return params

    def _is_range(self, t: tuple) -> bool:
        """Check if a tuple represents a range ``(low, high)`` or ``(low, high, type)``."""
        if len(t) == 2:
            return isinstance(t[0], (int, float)) and isinstance(t[1], (int, float))
        if len(t) == 3:
            return (isinstance(t[0], (int, float)) and
                    isinstance(t[1], (int, float)) and
                    isinstance(t[2], str))
        return False

class BaseTuner(ABC):
    """Abstract base class for hyperparameter optimization strategies.

    Tuners manage the selection of parameters, the execution of
    cross-validation, and the selection of the 'best' model configuration.

    Parameters
    ----------
    estimator : Algorithm
        The model template to optimize.
    scoring : str or callable, default="accuracy"
        The evaluation metric used to rank parameter combinations. Built-in
        options are ``"accuracy"``, ``"neg_mse"``, and ``"r2"``; a callable
        must accept ``(y_true, y_pred)`` and return a float.
    cv : int, default=5
        The number of folds for cross-validation.
    refit : bool, default=True
        Whether to retrain the model on the entire training set using the
        best found parameters after the search completes.
    verbose : int, default=0
        The level of progress logging (higher values produce more detail).
    n_jobs : int, default=1
        The number of parallel processes to use during cross-validation.
        Requires ``joblib``; falls back to sequential execution otherwise.
    random_seed : int, optional
        Seed for reproducible random sampling. If ``None``, the global TuiML
        seed is used, falling back to ``42``.
    progress_callback : callable, optional
        Function invoked with a progress dictionary after each evaluated
        parameter combination.

    Attributes
    ----------
    best_params_ : dict
        Parameter combination with the highest mean cross-validation score.
    best_score_ : float
        Mean cross-validation score of ``best_params_``.
    best_estimator_ : Any
        Estimator refitted on the full training data with ``best_params_``
        (available when ``refit=True``).
    cv_results_ : dict
        Log of scores, timings, and parameters for every combination tried.

    See Also
    --------
    :class:`~tuiml.base.tuning.ParameterGrid` : Search space for grid search.
    :class:`~tuiml.base.tuning.ParameterDistribution` : Search space for
        random search.
    :class:`~tuiml.base.tuning.TuningResult` : Structured result container.
    """

    def __init__(
        self,
        estimator,
        scoring: Union[str, Callable] = 'accuracy',
        cv: int = 5,
        refit: bool = True,
        verbose: int = 0,
        n_jobs: int = 1,
        random_seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ):
        """Initialize the tuner with an estimator and search settings."""
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.verbose = verbose
        self.n_jobs = n_jobs

        if random_seed is None:
            from tuiml.utils.seed import get_global_seed
            seed = get_global_seed()
        else:
            seed = random_seed
        if seed is None:
            seed = 42
        self.random_state = seed
        self.progress_callback = progress_callback

        self.best_params_: Optional[Dict] = None
        self.best_score_: Optional[float] = None
        self.best_estimator_: Optional[Any] = None
        self.cv_results_: Optional[Dict] = None

    def _notify_progress(self, iteration: int, total: int, params: Dict,
                         mean_score: float, std_score: float, best_score: float):
        """Invoke the progress callback with the current search state, if set."""
        if self.progress_callback is not None:
            self.progress_callback({
                'type': 'tune_progress',
                'iteration': iteration,
                'total': total,
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'best_score': best_score,
            })

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseTuner":
        """Run the search to find the best parameters.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BaseTuner
            The fitted tuner, with ``best_params_``, ``best_score_``,
            ``best_estimator_``, and ``cv_results_`` populated.
        """
        pass

    def _get_scorer(self) -> Callable:
        """Resolve ``self.scoring`` to a scoring callable.

        Returns
        -------
        scorer : callable
            Function accepting ``(y_true, y_pred)`` and returning a float.
        """
        if callable(self.scoring):
            return self.scoring

        # Built-in scorers
        scorers = {
            'accuracy': lambda y_true, y_pred: np.mean(y_true == y_pred),
            'neg_mse': lambda y_true, y_pred: -np.mean((y_true - y_pred) ** 2),
            'r2': lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
        }

        return scorers.get(self.scoring, scorers['accuracy'])

    def _cross_validate(
        self,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """Perform cross-validation for a parameter combination.

        Parameters
        ----------
        estimator : Algorithm
            Estimator template to evaluate (copied for each fold).
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.
        params : dict
            Parameter combination to set on the estimator before fitting.

        Returns
        -------
        mean_score : float
            Mean score across folds.
        std_score : float
            Standard deviation of the fold scores.
        fit_time : float
            Total wall-clock time for the cross-validation, in seconds.
        """
        from tuiml.evaluation.splitting import StratifiedKFold, KFold

        scorer = self._get_scorer()

        # Choose CV strategy
        if self._is_classification(y):
            cv = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                 random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv, shuffle=True,
                       random_state=self.random_state)

        start_time = time.time()

        if self.n_jobs != 1 and JOBLIB_AVAILABLE:
            # Parallel execution
            parallel = Parallel(n_jobs=self.n_jobs)
            scores = parallel(
                delayed(_fit_and_score)(
                    deepcopy(estimator), X, y, train_idx, test_idx, scorer, params
                )
                for train_idx, test_idx in cv.split(X, y)
            )
        else:
            # Sequential execution
            if self.n_jobs != 1 and not JOBLIB_AVAILABLE:
                warnings.warn("joblib not installed, falling back to sequential execution.")

            scores = []
            for train_idx, test_idx in cv.split(X, y):
                scores.append(_fit_and_score(
                    deepcopy(estimator), X, y, train_idx, test_idx, scorer, params
                ))

        total_time = time.time() - start_time
        return np.mean(scores), np.std(scores), total_time

    def _is_classification(self, y: np.ndarray) -> bool:
        """Heuristically check whether the target represents a classification task."""
        unique = np.unique(y)
        if np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.bool_) or y.dtype == object:
            return True
        return len(unique) < max(20, len(y) * 0.05)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the best estimator found by the search.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predictions from ``best_estimator_``.

        Raises
        ------
        RuntimeError
            If the tuner has not been fitted yet.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("Tuner has not been fitted yet")
        return self.best_estimator_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score the best estimator on the given data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            Score of ``best_estimator_`` under the tuner's scoring metric.
        """
        y_pred = self.predict(X)
        scorer = self._get_scorer()
        return scorer(y, y_pred)

__all__ = [
    "TuningResult",
    "ParameterGrid",
    "ParameterDistribution",
    "BaseTuner",
    "_fit_and_score",
    "JOBLIB_AVAILABLE",
]

def _fit_and_score(estimator, X, y, train_idx, test_idx, scorer, params):
    """Fit an estimator on one fold and return its test score (parallel helper)."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Set parameters
    for k, v in params.items():
        setattr(estimator, k, v)

    # Fit and score
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    return scorer(y_test, y_pred)
