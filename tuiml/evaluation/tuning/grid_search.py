"""Exhaustive grid search over a discrete hyperparameter grid.

This module provides :class:`~tuiml.evaluation.tuning.GridSearchCV`, the
"try everything" tuner. It expands a mapping of parameter name to a list of
candidate values into the full cartesian product, scores every combination
with k-fold cross-validation, ranks them, and optionally refits the winner on
the whole training set.

Reach for grid search when the space is small, discrete, and you want a
complete, deterministic sweep that is trivial to explain and reproduce (for
example ``criterion`` x ``max_depth`` with three values each). The cost is the
*product* of the list lengths times the number of folds, so it degrades badly
in higher dimensions: swap to :class:`~tuiml.evaluation.tuning.RandomSearchCV`
when parameters are continuous or unequally important, and to
:class:`~tuiml.evaluation.tuning.BayesianSearchCV` when each fit is expensive
enough that the choice of the next candidate is worth modelling.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy
import time

from tuiml.base.tuning import BaseTuner, ParameterGrid, TuningResult

class GridSearchCV(BaseTuner):
    """Exhaustive **cross-validated search** over every point of a parameter grid.

    Overview
    --------
    1. Expand ``param_grid`` into the cartesian product of its value lists.
    2. For each combination, copy the estimator, set the parameters on the
       copy, and run ``cv``-fold cross-validation (stratified for
       classification targets, plain k-fold otherwise).
    3. Record the mean and standard deviation of the fold scores, plus the
       mean per-fold fit time, in ``cv_results_``.
    4. Keep the combination with the highest mean score in ``best_params_``
       and rank every combination in ``cv_results_['rank_test_score']``
       (rank 1 is best).
    5. If ``refit=True``, refit a fresh copy of the estimator with the winning
       parameters on the *full* training set and store it as
       ``best_estimator_``, which then backs ``predict`` and ``score``.

    Search space
    ------------
    ``param_grid`` maps a parameter name to the list of values to try. Scalars
    are treated as one-element lists. A list of dictionaries defines disjoint
    sub-grids, which lets you avoid invalid combinations::

        # single grid: 3 x 2 = 6 candidates
        {'max_depth': [3, 5, 10], 'criterion': ['gini', 'entropy']}

        # disjoint grids: 2 + 3 = 5 candidates
        [{'kernel': ['linear'], 'C': [1, 10]},
         {'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1.0]}]

    Cost
    ----
    The search performs :math:`n_{\\text{candidates}} \\times \\text{cv}` model
    fits, plus one more when ``refit=True``. ``n_candidates`` is the *product*
    of the list lengths (summed over sub-grids), so it grows exponentially in
    the number of tuned parameters::

        parameters   values each   candidates   fits at cv=5
        ----------   -----------   ----------   ------------
                 2             3            9             46
                 3             3           27            136
                 4             4          256           1281

    Parameters
    ----------
    estimator : object
        Estimator to tune. Must expose ``fit(X, y)`` and ``predict(X)`` and
        accept the grid's parameter names as writable attributes. The instance
        is never modified; every evaluation works on a deep copy.
    param_grid : dict or list of dict
        Search space. Keys are parameter names, values are lists of candidate
        settings. A list of dicts defines disjoint sub-grids. Wrapped in
        :class:`~tuiml.base.tuning.ParameterGrid`.
    scoring : str or callable, default='accuracy'
        Metric maximized by the search. Built-ins are ``'accuracy'``,
        ``'neg_mse'`` and ``'r2'``; a callable must have the signature
        ``scorer(y_true, y_pred) -> float`` and follow the
        higher-is-better convention. An unrecognized string falls back to
        ``'accuracy'``.
    cv : int, default=5
        Number of cross-validation folds per candidate.
    refit : bool, default=True
        Whether to refit the estimator on the full training data with
        ``best_params_`` after the search. Required for ``predict``/``score``.
    verbose : int, default=0
        Verbosity. ``0`` is silent; any value above ``0`` prints one line per
        candidate plus a final summary.
    n_jobs : int, default=1
        Number of folds evaluated in parallel via ``joblib``. ``1`` runs
        sequentially; other values fall back to sequential execution with a
        warning when ``joblib`` is not installed.
    random_seed : int, optional
        Seed controlling the cross-validation shuffling. If ``None``, the
        global TuiML seed is used, falling back to ``42``. The legacy keyword
        ``random_state`` is still accepted as an alias and is stored on the
        instance as ``self.random_state``.
    progress_callback : callable, optional
        Called after each candidate with a dictionary containing
        ``'type'``, ``'iteration'``, ``'total'``, ``'params'``,
        ``'mean_score'``, ``'std_score'`` and ``'best_score'``.

    Attributes
    ----------
    param_grid : ParameterGrid
        The expanded search space. ``len(self.param_grid)`` is the number of
        candidates the search will evaluate.
    best_params_ : dict
        Parameter combination with the highest mean cross-validation score.
        ``None`` before ``fit`` is called.
    best_score_ : float
        Mean cross-validation score of ``best_params_``.
    best_estimator_ : object
        Copy of ``estimator`` refitted on the full training data with
        ``best_params_``. Only set when ``refit=True``.
    cv_results_ : dict
        Per-candidate log with parallel lists, one entry per candidate:

        - ``'params'`` : list of dict, the parameter combination.
        - ``'mean_test_score'`` : list of float, mean score across folds.
        - ``'std_test_score'`` : list of float, standard deviation of the
          fold scores.
        - ``'mean_fit_time'`` : list of float, mean seconds per fold.
        - ``'rank_test_score'`` : list of int, 1 for the best candidate.
    total_time_ : float
        Wall-clock seconds consumed by the whole search, including the refit.

    Notes
    -----
    **Complexity.** :math:`O(n_{\\text{candidates}} \\cdot \\text{cv} \\cdot C)`
    where :math:`C` is the cost of a single estimator fit; memory is
    :math:`O(n_{\\text{candidates}})` for the result log.

    **When to use.** Small discrete spaces where a complete sweep is
    affordable and reproducibility matters. For continuous parameters, more
    than three or four tuned parameters, or expensive fits, prefer
    :class:`~tuiml.evaluation.tuning.RandomSearchCV` or
    :class:`~tuiml.evaluation.tuning.BayesianSearchCV`.

    See Also
    --------
    :class:`~tuiml.evaluation.tuning.RandomSearchCV` : Samples a fixed budget
        of configurations from distributions instead of enumerating a grid.
    :class:`~tuiml.evaluation.tuning.BayesianSearchCV` : Models the score
        surface with a Gaussian Process to pick each next configuration.
    :class:`~tuiml.base.tuning.ParameterGrid` : The grid expansion used here.
    :class:`~tuiml.base.tuning.TuningResult` : Structured result returned by
        :meth:`get_results`.

    Examples
    --------
    Sweep a two-value grid for Naive Bayes on iris:

    >>> from tuiml.evaluation.tuning import GridSearchCV
    >>> from tuiml.algorithms.bayesian import NaiveBayesClassifier
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> X, y = data.X, data.y
    >>> search = GridSearchCV(
    ...     estimator=NaiveBayesClassifier(),
    ...     param_grid={'use_kernel_estimator': [True, False]},
    ...     cv=3,
    ...     scoring='accuracy',
    ...     random_seed=0,
    ... )
    >>> search = search.fit(X, y)
    >>> sorted(search.best_params_)
    ['use_kernel_estimator']
    >>> round(float(search.best_score_), 2)
    0.96
    >>> len(search.cv_results_['params'])
    2
    >>> search.cv_results_['rank_test_score']
    [1, 2]

    The refitted best estimator backs ``predict``:

    >>> y_pred = search.predict(X)
    >>> y_pred.shape
    (150,)
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for constructor parameters."""
        return {
            "estimator": {
                "type": "object",
                "description": "Estimator to tune. Must have fit() and predict() methods."
            },
            "param_grid": {
                "type": ["object", "array"],
                "description": "Dictionary with parameters as keys and lists of values, or list of such dictionaries."
            },
            "scoring": {
                "type": ["string", "callable"],
                "default": "accuracy",
                "description": "Scoring metric. Options: 'accuracy', 'neg_mse', 'r2', or a callable."
            },
            "cv": {
                "type": "integer",
                "default": 5,
                "minimum": 2,
                "description": "Number of cross-validation folds."
            },
            "refit": {
                "type": "boolean",
                "default": True,
                "description": "Refit estimator with best parameters on full data."
            },
            "verbose": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": "Verbosity level."
            },
            "n_jobs": {
                "type": "integer",
                "default": 1,
                "description": "Number of parallel jobs (not implemented yet)."
            },
            "random_seed": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility."
            }
        }

    def __init__(
        self,
        estimator,
        param_grid: Union[Dict, List[Dict]],
        scoring: Union[str, Callable] = 'accuracy',
        cv: int = 5,
        refit: bool = True,
        verbose: int = 0,
        n_jobs: int = 1,
        random_seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ):
        """Initialize the grid search; see the class docstring for parameters."""
        legacy_random_state = kwargs.pop('random_state', None)
        if random_seed is None:
            random_seed = legacy_random_state
            
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            cv=cv,
            refit=refit,
            verbose=verbose,
            n_jobs=n_jobs,
            random_seed=random_seed,
            progress_callback=progress_callback
        )
        self.param_grid = ParameterGrid(param_grid)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GridSearchCV":
        """Evaluate every grid point and keep the best-scoring one.

        Runs ``len(self.param_grid) * cv`` estimator fits, then one extra fit
        on the full data when ``refit=True``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values. Integer, boolean, or object dtype (or few unique
            values) selects stratified folds; otherwise plain k-fold is used.

        Returns
        -------
        self : GridSearchCV
            The fitted searcher, with ``best_params_``, ``best_score_``,
            ``cv_results_`` and ``total_time_`` populated (and
            ``best_estimator_`` when ``refit=True``).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        start_time = time.time()

        # Initialize results
        results = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'mean_fit_time': [],
            'rank_test_score': []
        }

        best_score = -np.inf
        best_params = None
        n_combinations = len(self.param_grid)

        if self.verbose > 0:
            print(f"Fitting {self.cv} folds for {n_combinations} candidates...")

        for i, params in enumerate(self.param_grid):
            if self.verbose > 0:
                print(f"  [{i+1}/{n_combinations}] Testing: {params}")

            mean_score, std_score, fit_time = self._cross_validate(
                self.estimator, X, y, params
            )

            results['params'].append(params)
            results['mean_test_score'].append(mean_score)
            results['std_test_score'].append(std_score)
            results['mean_fit_time'].append(fit_time / self.cv)

            if self.verbose > 0:
                print(f"    Score: {mean_score:.4f} ± {std_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

            self._notify_progress(i + 1, n_combinations, params,
                                  mean_score, std_score, best_score)

        # Compute ranks
        scores = np.array(results['mean_test_score'])
        ranks = len(scores) - np.argsort(np.argsort(scores))
        results['rank_test_score'] = ranks.tolist()

        self.cv_results_ = results
        self.best_params_ = best_params
        self.best_score_ = best_score

        # Refit with best parameters
        if self.refit and best_params is not None:
            self.best_estimator_ = deepcopy(self.estimator)
            for k, v in best_params.items():
                setattr(self.best_estimator_, k, v)
            self.best_estimator_.fit(X, y)

        self.total_time_ = time.time() - start_time

        if self.verbose > 0:
            print(f"\nBest parameters: {best_params}")
            print(f"Best score: {best_score:.4f}")
            print(f"Total time: {self.total_time_:.2f}s")

        return self

    def get_results(self) -> TuningResult:
        """Bundle the fitted search state into a :class:`~tuiml.base.tuning.TuningResult`.

        Returns
        -------
        result : TuningResult
            Container holding ``best_params``, ``best_score``,
            ``best_estimator``, ``cv_results``, the number of candidates
            evaluated, and the total search time in seconds.

        Raises
        ------
        TypeError
            If called before :meth:`fit`, because ``cv_results_`` is still
            ``None``.
        """
        return TuningResult(
            best_params=self.best_params_,
            best_score=self.best_score_,
            best_estimator=self.best_estimator_,
            cv_results=self.cv_results_,
            n_iterations=len(self.cv_results_['params']),
            total_time=self.total_time_
        )

    def __repr__(self) -> str:
        """Return a short summary of the searcher's configuration.

        Returns
        -------
        text : str
            String of the form
            ``GridSearchCV(estimator=..., cv=..., scoring='...')``.
        """
        return (
            f"GridSearchCV(estimator={self.estimator.__class__.__name__}, "
            f"cv={self.cv}, scoring='{self.scoring}')"
        )
