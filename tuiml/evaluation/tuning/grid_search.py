"""
Grid search cross-validation.
- CVParameterSelection.java
- GridSearch.java
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy
import time

from tuiml.base.tuning import BaseTuner, ParameterGrid, TuningResult

class GridSearchCV(BaseTuner):
    """
    Exhaustive search over specified parameter grid.

    Parameters
    ----------
    estimator : object
        Estimator to tune. Must have fit() and predict() methods.
    param_grid : dict or list of dicts
        Dictionary with parameters as keys and lists of values.
    scoring : str or callable, default='accuracy'
        Scoring metric. Options: 'accuracy', 'neg_mse', 'r2'
    cv : int, default=5
        Number of cross-validation folds.
    refit : bool, default=True
        Refit estimator with best parameters on full data.
    verbose : int, default=0
        Verbosity level.
    n_jobs : int, default=1
        Number of parallel jobs (not implemented yet).
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
        Cross-validation results for all parameter combinations.

    Examples
    --------
    >>> from tuiml.evaluation.tuning import GridSearchCV
    >>> from tuiml.algorithms.bayesian import NaiveBayesClassifier
    >>>
    >>> # Define parameter grid
    >>> param_grid = {
    ...     'use_kernel_density': [True, False],
    ...     'use_supervised_discretization': [True, False]
    ... }
    >>>
    >>> # Create and fit grid search
    >>> grid = GridSearchCV(
    ...     estimator=NaiveBayesClassifier(),
    ...     param_grid=param_grid,
    ...     cv=5,
    ...     scoring='accuracy'
    ... )
    >>> grid.fit(X_train, y_train)
    >>>
    >>> print(f"Best params: {grid.best_params_}")
    >>> print(f"Best score: {grid.best_score_:.4f}")
    >>>
    >>> # Predict with best model
    >>> y_pred = grid.predict(X_test)
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """
        Return JSON Schema for GridSearchCV parameters.

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
            "random_state": {
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
        self.param_grid = ParameterGrid(param_grid)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GridSearchCV":
        """
        Fit grid search to find best parameters.

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
            n_iterations=len(self.cv_results_['params']),
            total_time=self.total_time_
        )

    def __repr__(self) -> str:
        return (
            f"GridSearchCV(estimator={self.estimator.__class__.__name__}, "
            f"cv={self.cv}, scoring='{self.scoring}')"
        )
