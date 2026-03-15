"""
Random search cross-validation.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy
import time

from tuiml.base.tuning import BaseTuner, ParameterDistribution, TuningResult

class RandomSearchCV(BaseTuner):
    """
    Random search over hyperparameter distributions.

    More efficient than grid search when not all parameters
    are equally important.

    Parameters
    ----------
    estimator : object
        Estimator to tune. Must have fit() and predict() methods.
    param_distributions : dict
        Dictionary with parameters as keys and distributions as values.
        Values can be:
        - List: uniform choice from list
        - Tuple (low, high): uniform continuous
        - Tuple (low, high, 'log'): log-uniform
        - Tuple (low, high, 'int'): uniform integer
    n_iter : int, default=10
        Number of parameter combinations to sample.
    scoring : str or callable, default='accuracy'
        Scoring metric.
    cv : int, default=5
        Number of cross-validation folds.
    refit : bool, default=True
        Refit estimator with best parameters.
    verbose : int, default=0
        Verbosity level.
    random_state : int, optional
        Random seed.

    Attributes
    ----------
    best_params_ : dict
        Best parameters found.
    best_score_ : float
        Best cross-validation score.
    best_estimator_ : object
        Estimator fitted with best parameters.
    cv_results_ : dict
        Cross-validation results.

    Examples
    --------
    >>> from tuiml.evaluation.tuning import RandomSearchCV
    >>> from tuiml.algorithms.trees import RandomForestClassifier
    >>>
    >>> # Define parameter distributions
    >>> param_distributions = {
    ...     'n_estimators': (10, 200, 'int'),     # Uniform integer 10-200
    ...     'max_depth': [None, 5, 10, 20],       # Choice from list
    ...     'min_samples_split': (2, 20, 'int'),  # Uniform integer 2-20
    ...     'max_features': (0.1, 1.0)            # Uniform continuous 0.1-1.0
    ... }
    >>>
    >>> # Create and fit random search
    >>> search = RandomSearchCV(
    ...     estimator=RandomForestClassifier(),
    ...     param_distributions=param_distributions,
    ...     n_iter=20,
    ...     cv=5,
    ...     scoring='accuracy'
    ... )
    >>> search.fit(X_train, y_train)
    >>>
    >>> print(f"Best params: {search.best_params_}")
    >>> print(f"Best score: {search.best_score_:.4f}")
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """
        Return JSON Schema for RandomSearchCV parameters.

        Returns
        -------
        dict
            JSON Schema describing all __init__ parameters.
        """
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
            "random_state": {
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
        random_state: Optional[int] = 42,
        progress_callback: Optional[Callable] = None
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            cv=cv,
            refit=refit,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_callback=progress_callback
        )
        self.param_distributions = ParameterDistribution(param_distributions)
        self.n_iter = n_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomSearchCV":
        """
        Fit random search to find best parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self
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
        """
        Get tuning results as TuningResult object.

        Returns
        -------
        result : TuningResult
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
        return (
            f"RandomSearchCV(estimator={self.estimator.__class__.__name__}, "
            f"n_iter={self.n_iter}, cv={self.cv}, scoring='{self.scoring}')"
        )
