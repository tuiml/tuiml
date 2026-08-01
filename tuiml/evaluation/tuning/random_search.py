"""Randomized cross-validated search over hyperparameter distributions.

This module provides :class:`~tuiml.evaluation.tuning.RandomSearchCV`, which
draws a fixed budget of ``n_iter`` configurations from the distributions you
declare instead of enumerating a grid. Because the budget is decoupled from
the dimensionality of the space, it is the practical default when some
parameters are continuous, when you do not know in advance which parameters
matter, or when you simply want to cap the number of model fits.

The classic argument (Bergstra & Bengio, 2012) is that when only a few of the
tuned parameters actually influence the score, a random draw of :math:`n`
points explores :math:`n` distinct values of *each* important parameter, while
a grid re-tests the same handful of values over and over.

Search spaces are expressed with :class:`~tuiml.base.tuning.ParameterDistribution`,
which accepts explicit value lists as well as ``(low, high)``,
``(low, high, 'log')`` and ``(low, high, 'int')`` range tuples.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy
import time

from tuiml.base.tuning import BaseTuner, ParameterDistribution, TuningResult

class RandomSearchCV(BaseTuner):
    """**Randomized** cross-validated search over hyperparameter distributions.

    Draws ``n_iter`` independent configurations from the declared
    distributions, cross-validates each, and keeps the best. Unlike
    :class:`~tuiml.evaluation.tuning.GridSearchCV`, the cost is fixed by
    ``n_iter`` rather than by the size of the space, so continuous parameters
    and high-dimensional spaces are affordable.

    Overview
    --------
    1. Seed a random generator from ``random_seed`` (or the global TuiML seed).
    2. Repeat ``n_iter`` times: sample one value per parameter from its
       distribution, then cross-validate a copy of the estimator configured
       with that sample.
    3. Log the mean score, its standard deviation, and the mean per-fold fit
       time into ``cv_results_``, and track the running best.
    4. Rank all samples in ``cv_results_['rank_test_score']`` (rank 1 is best).
    5. If ``refit=True``, refit a fresh copy of the estimator on the full
       training set with ``best_params_`` and store it as ``best_estimator_``.

    Search space
    ------------
    ``param_distributions`` maps a parameter name to one of the forms accepted
    by :class:`~tuiml.base.tuning.ParameterDistribution`::

        {'criterion':  ['gini', 'entropy'],   # list  -> uniform choice
         'max_depth':  (2, 20, 'int'),        # 3-tuple 'int'   -> uniform integer, high INCLUSIVE
         'C':          (0.001, 100, 'log'),   # 3-tuple 'log'   -> log-uniform (low > 0 required)
         'max_features': (0.1, 1.0),          # 2-tuple numeric -> uniform continuous, high exclusive
         'alpha':      lambda: 10 ** -3}      # callable        -> called with no arguments

    Two subtleties follow from how a tuple is classified. A 2-tuple of
    *numbers* is always read as a continuous range, so use a **list** when you
    mean "choose one of these two numbers". A tuple of non-numbers such as
    ``('linear', 'rbf')`` is not a range and is treated as a choice. Sampled
    choices come back as NumPy scalars (``np.str_('gini')``, ``np.int64(5)``)
    because the draw goes through ``numpy.random.RandomState.choice``.

    Cost
    ----
    The search performs exactly :math:`\\text{n\\_iter} \\times \\text{cv}` model
    fits, plus one more when ``refit=True`` — independent of how many
    parameters are tuned. That is the whole point: with ``n_iter=20, cv=5``
    you pay 101 fits whether the space has two dimensions or twenty, whereas
    :class:`~tuiml.evaluation.tuning.GridSearchCV` would need the full
    cartesian product.

    Parameters
    ----------
    estimator : object
        Estimator to tune. Must expose ``fit(X, y)`` and ``predict(X)`` and
        accept the sampled parameter names as writable attributes. The
        instance is never modified; every evaluation works on a deep copy.
    param_distributions : dict
        Search space. Keys are parameter names; values are lists, range
        tuples, or zero-argument callables as described above. Wrapped in
        :class:`~tuiml.base.tuning.ParameterDistribution`.
    n_iter : int, default=10
        Number of configurations to sample, i.e. the search budget. Samples
        are independent, so repeated values are possible in small discrete
        spaces.
    scoring : str or callable, default='accuracy'
        Metric maximized by the search. Built-ins are ``'accuracy'``,
        ``'neg_mse'`` and ``'r2'``; a callable must have the signature
        ``scorer(y_true, y_pred) -> float`` and follow the higher-is-better
        convention. An unrecognized string falls back to ``'accuracy'``.
    cv : int, default=5
        Number of cross-validation folds per sampled configuration.
    refit : bool, default=True
        Whether to refit the estimator on the full training data with
        ``best_params_`` after the search. Required for ``predict``/``score``.
    verbose : int, default=0
        Verbosity. ``0`` is silent; any value above ``0`` prints one line per
        sample plus a final summary.
    n_jobs : int, default=1
        Number of folds evaluated in parallel via ``joblib``. ``1`` runs
        sequentially; other values fall back to sequential execution with a
        warning when ``joblib`` is not installed.
    random_seed : int, optional
        Seed for both the parameter sampling and the cross-validation
        shuffling, making a whole search reproducible. If ``None``, the global
        TuiML seed is used, falling back to ``42``. The legacy keyword
        ``random_state`` is still accepted as an alias and is stored on the
        instance as ``self.random_state``.
    progress_callback : callable, optional
        Called after each sample with a dictionary containing ``'type'``,
        ``'iteration'``, ``'total'``, ``'params'``, ``'mean_score'``,
        ``'std_score'`` and ``'best_score'``.

    Attributes
    ----------
    param_distributions : ParameterDistribution
        The wrapped search space that configurations are drawn from.
    n_iter : int
        The configured search budget.
    best_params_ : dict
        Sampled configuration with the highest mean cross-validation score.
        ``None`` before ``fit`` is called.
    best_score_ : float
        Mean cross-validation score of ``best_params_``.
    best_estimator_ : object
        Copy of ``estimator`` refitted on the full training data with
        ``best_params_``. Only set when ``refit=True``.
    cv_results_ : dict
        Per-sample log with parallel lists of length ``n_iter``:

        - ``'params'`` : list of dict, the sampled configuration.
        - ``'mean_test_score'`` : list of float, mean score across folds.
        - ``'std_test_score'`` : list of float, standard deviation of the
          fold scores.
        - ``'mean_fit_time'`` : list of float, mean seconds per fold.
        - ``'rank_test_score'`` : list of int, 1 for the best sample.
    total_time_ : float
        Wall-clock seconds consumed by the whole search, including the refit.

    Notes
    -----
    **Complexity.** :math:`O(\\text{n\\_iter} \\cdot \\text{cv} \\cdot C)` where
    :math:`C` is the cost of a single estimator fit; memory is
    :math:`O(\\text{n\\_iter})` for the result log.

    **When to use.** The pragmatic default: any continuous parameter, more
    than a couple of tuned parameters, or a hard budget on training time.
    Prefer :class:`~tuiml.evaluation.tuning.GridSearchCV` when the space is
    tiny and you want guaranteed coverage, and
    :class:`~tuiml.evaluation.tuning.BayesianSearchCV` when each fit is
    expensive enough to justify modelling the score surface.

    References
    ----------
    .. [Bergstra2012] Bergstra, J., & Bengio, Y. (2012). Random Search for
       Hyper-Parameter Optimization. *Journal of Machine Learning Research*,
       13, 281-305.

    See Also
    --------
    :class:`~tuiml.evaluation.tuning.GridSearchCV` : Exhaustive sweep of a
        discrete grid.
    :class:`~tuiml.evaluation.tuning.BayesianSearchCV` : Gaussian-Process
        guided search that chooses each next configuration.
    :class:`~tuiml.base.tuning.ParameterDistribution` : The sampler backing
        ``param_distributions``.
    :class:`~tuiml.base.tuning.TuningResult` : Structured result returned by
        :meth:`get_results`.

    Examples
    --------
    Sample four decision-tree configurations from a mixed space:

    >>> from tuiml.evaluation.tuning import RandomSearchCV
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> X, y = data.X, data.y
    >>> search = RandomSearchCV(
    ...     estimator=DecisionTreeClassifier(),
    ...     param_distributions={
    ...         'max_depth': (2, 8, 'int'),
    ...         'criterion': ['gini', 'entropy'],
    ...     },
    ...     n_iter=4,
    ...     cv=3,
    ...     scoring='accuracy',
    ...     random_seed=0,
    ... )
    >>> search = search.fit(X, y)
    >>> sorted(search.best_params_)
    ['criterion', 'max_depth']
    >>> str(search.best_params_['criterion'])
    'entropy'
    >>> round(float(search.best_score_), 2)
    0.94

    The budget, not the size of the space, fixes the amount of work:

    >>> len(search.cv_results_['params'])
    4
    >>> min(search.cv_results_['rank_test_score'])
    1
    >>> search.predict(X).shape
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
            "param_distributions": {
                "type": "object",
                "description": "Dictionary with parameters as keys and distributions as values. Values can be: List (uniform choice), Tuple (low, high) for continuous, Tuple (low, high, 'log') for log-uniform, or Tuple (low, high, 'int') for integer."
            },
            "n_iter": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "description": "Number of parameter combinations to sample."
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
        param_distributions: Dict,
        n_iter: int = 10,
        scoring: Union[str, Callable] = 'accuracy',
        cv: int = 5,
        refit: bool = True,
        verbose: int = 0,
        n_jobs: int = 1,
        random_seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ):
        """Initialize the random search; see the class docstring for parameters."""
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
        self.param_distributions = ParameterDistribution(param_distributions)
        self.n_iter = n_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomSearchCV":
        """Sample ``n_iter`` configurations and keep the best-scoring one.

        Runs exactly ``n_iter * cv`` estimator fits, then one extra fit on the
        full data when ``refit=True``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values. Integer, boolean, or object dtype (or few unique
            values) selects stratified folds; otherwise plain k-fold is used.

        Returns
        -------
        self : RandomSearchCV
            The fitted searcher, with ``best_params_``, ``best_score_``,
            ``cv_results_`` and ``total_time_`` populated (and
            ``best_estimator_`` when ``refit=True``).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        start_time = time.time()
        rng = np.random.RandomState(self.random_state)

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

        if self.verbose > 0:
            print(f"Fitting {self.cv} folds for {self.n_iter} candidates...")

        for i in range(self.n_iter):
            # Sample parameters
            params = self.param_distributions.sample(
                random_state=rng.randint(0, 2**31)
            )

            if self.verbose > 0:
                print(f"  [{i+1}/{self.n_iter}] Testing: {params}")

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

            self._notify_progress(i + 1, self.n_iter, params,
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
            ``best_estimator``, ``cv_results``, ``n_iterations`` (equal to
            ``n_iter``), and the total search time in seconds.

        Raises
        ------
        AttributeError
            If called before :meth:`fit`, because ``total_time_`` does not
            exist yet.
        """
        return TuningResult(
            best_params=self.best_params_,
            best_score=self.best_score_,
            best_estimator=self.best_estimator_,
            cv_results=self.cv_results_,
            n_iterations=self.n_iter,
            total_time=self.total_time_
        )

    def __repr__(self) -> str:
        """Return a short summary of the searcher's configuration.

        Returns
        -------
        text : str
            String of the form ``RandomSearchCV(estimator=..., n_iter=...,
            cv=..., scoring='...')``.
        """
        return (
            f"RandomSearchCV(estimator={self.estimator.__class__.__name__}, "
            f"n_iter={self.n_iter}, cv={self.cv}, scoring='{self.scoring}')"
        )
