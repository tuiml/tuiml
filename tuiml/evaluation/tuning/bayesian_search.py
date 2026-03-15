"""
Bayesian optimization for hyperparameter tuning.

Implements Gaussian Process-based Bayesian Optimization from scratch.
No external optimization libraries required.

References
----------
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian 
  optimization of machine learning algorithms. NeurIPS.
- Brochu, E., Cora, V. M., & De Freitas, N. (2010). A tutorial on 
  Bayesian optimization of expensive cost functions.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy
import time
from scipy.stats import norm
from scipy.optimize import minimize
import warnings

from tuiml.base.tuning import BaseTuner, ParameterDistribution, TuningResult

class GaussianProcess:
    """
    Gaussian Process for Bayesian Optimization.
    
    Simple GP implementation with RBF (squared exponential) kernel.
    
    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type. Currently only 'rbf' supported.
    length_scale : float, default=1.0
        Length scale for RBF kernel.
    noise : float, default=1e-10
        Noise level for numerical stability.
    """
    
    @classmethod
    def get_parameter_schema(cls) -> dict:
        """
        Return JSON Schema for GaussianProcess parameters.

        Returns
        -------
        dict
            JSON Schema describing all __init__ parameters.
        """
        return {
            "kernel": {
                "type": "string",
                "default": "rbf",
                "enum": ["rbf"],
                "description": "Kernel type. Currently only 'rbf' (squared exponential) is supported."
            },
            "length_scale": {
                "type": "number",
                "default": 1.0,
                "exclusiveMinimum": 0,
                "description": "Length scale for RBF kernel."
            },
            "noise": {
                "type": "number",
                "default": 1e-10,
                "minimum": 0,
                "description": "Noise level for numerical stability."
            }
        }

    def __init__(
        self,
        kernel: str = 'rbf',
        length_scale: float = 1.0,
        noise: float = 1e-10
    ):
        self.kernel = kernel
        self.length_scale = length_scale
        self.noise = noise
        
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute RBF (Radial Basis Function) kernel.
        
        k(x1, x2) = exp(-||x1 - x2||^2 / (2 * length_scale^2))
        """
        # Compute squared Euclidean distances
        dists = np.sum(X1**2, axis=1)[:, np.newaxis] + \
                np.sum(X2**2, axis=1)[np.newaxis, :] - \
                2 * np.dot(X1, X2.T)
        
        # Apply RBF kernel
        return np.exp(-dists / (2 * self.length_scale**2))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Gaussian Process to training data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Training targets.
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        
        # Compute kernel matrix
        K = self._rbf_kernel(self.X_train, self.X_train)
        
        # Add noise to diagonal for numerical stability
        K += self.noise * np.eye(len(self.X_train))
        
        # Compute inverse of kernel matrix
        self.K_inv = np.linalg.inv(K)
        
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using Gaussian Process.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test features.
        return_std : bool, default=False
            If True, return standard deviation along with mean.
            
        Returns
        -------
        y_mean : ndarray
            Predicted mean.
        y_std : ndarray (if return_std=True)
            Predicted standard deviation.
        """
        if self.X_train is None:
            raise ValueError("GP not fitted yet!")
        
        X = np.asarray(X)
        
        # Compute kernel between test and training points
        K_star = self._rbf_kernel(X, self.X_train)
        
        # Predict mean
        y_mean = K_star @ self.K_inv @ self.y_train
        
        if not return_std:
            return y_mean
        
        # Compute variance
        K_star_star = self._rbf_kernel(X, X)
        y_var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
        y_std = np.sqrt(np.maximum(y_var, 0))  # Ensure non-negative
        
        return y_mean, y_std

class AcquisitionFunction:
    """
    Acquisition functions for Bayesian Optimization.
    
    Parameters
    ----------
    kind : str, default='ei'
        Type of acquisition function:
        - 'ei': Expected Improvement
        - 'ucb': Upper Confidence Bound
        - 'poi': Probability of Improvement
    xi : float, default=0.01
        Exploration-exploitation trade-off parameter.
    kappa : float, default=2.576
        Exploration parameter for UCB (higher = more exploration).
    """
    
    @classmethod
    def get_parameter_schema(cls) -> dict:
        """
        Return JSON Schema for AcquisitionFunction parameters.

        Returns
        -------
        dict
            JSON Schema describing all __init__ parameters.
        """
        return {
            "kind": {
                "type": "string",
                "default": "ei",
                "enum": ["ei", "ucb", "poi"],
                "description": "Type of acquisition function: 'ei' (Expected Improvement), 'ucb' (Upper Confidence Bound), or 'poi' (Probability of Improvement)."
            },
            "xi": {
                "type": "number",
                "default": 0.01,
                "minimum": 0,
                "description": "Exploration-exploitation trade-off parameter for EI and POI."
            },
            "kappa": {
                "type": "number",
                "default": 2.576,
                "minimum": 0,
                "description": "Exploration parameter for UCB (higher = more exploration)."
            }
        }

    def __init__(
        self,
        kind: str = 'ei',
        xi: float = 0.01,
        kappa: float = 2.576
    ):
        self.kind = kind
        self.xi = xi
        self.kappa = kappa
        
    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcess,
        y_best: float
    ) -> np.ndarray:
        """
        Compute acquisition function value.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Points to evaluate.
        gp : GaussianProcess
            Fitted Gaussian Process.
        y_best : float
            Best observed value so far.
            
        Returns
        -------
        values : ndarray
            Acquisition function values (higher is better).
        """
        if self.kind == 'ei':
            return self._expected_improvement(X, gp, y_best)
        elif self.kind == 'ucb':
            return self._upper_confidence_bound(X, gp)
        elif self.kind == 'poi':
            return self._probability_of_improvement(X, gp, y_best)
        else:
            raise ValueError(f"Unknown acquisition function: {self.kind}")
    
    def _expected_improvement(
        self,
        X: np.ndarray,
        gp: GaussianProcess,
        y_best: float
    ) -> np.ndarray:
        """
        Expected Improvement acquisition function.
        
        EI(x) = E[max(f(x) - f(x_best) - xi, 0)]
        """
        mu, sigma = gp.predict(X, return_std=True)
        
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)
        
        # Compute improvement
        improvement = mu - y_best - self.xi
        Z = improvement / sigma
        
        # Expected improvement
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei
    
    def _upper_confidence_bound(
        self,
        X: np.ndarray,
        gp: GaussianProcess
    ) -> np.ndarray:
        """
        Upper Confidence Bound acquisition function.
        
        UCB(x) = mu(x) + kappa * sigma(x)
        """
        mu, sigma = gp.predict(X, return_std=True)
        return mu + self.kappa * sigma
    
    def _probability_of_improvement(
        self,
        X: np.ndarray,
        gp: GaussianProcess,
        y_best: float
    ) -> np.ndarray:
        """
        Probability of Improvement acquisition function.
        
        PI(x) = P(f(x) >= f(x_best) + xi)
        """
        mu, sigma = gp.predict(X, return_std=True)
        
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)
        
        # Compute improvement
        improvement = mu - y_best - self.xi
        Z = improvement / sigma
        
        # Probability of improvement
        pi = norm.cdf(Z)
        
        return pi

class BayesianSearchCV(BaseTuner):
    """
    Bayesian optimization search for hyperparameter tuning.
    
    Uses Gaussian Process regression to model the objective function
    and acquisition functions to select promising hyperparameters.
    
    Parameters
    ----------
    estimator : object
        Estimator to tune. Must have fit() and predict() methods.
    param_space : dict
        Dictionary with parameters as keys and search spaces as values.
        Values can be:
        - List of values: discrete search
        - Tuple (min, max): continuous search
        - Tuple (min, max, 'int'): integer search
    n_iterations : int, default=50
        Number of iterations to run.
    acquisition : str, default='ei'
        Acquisition function: 'ei' (Expected Improvement), 
        'ucb' (Upper Confidence Bound), or 'poi' (Probability of Improvement).
    n_random_starts : int, default=10
        Number of random evaluations before using GP.
    xi : float, default=0.01
        Exploration parameter for EI and POI.
    kappa : float, default=2.576
        Exploration parameter for UCB.
    scoring : str or callable, default='accuracy'
        Scoring metric.
    cv : int, default=5
        Number of cross-validation folds.
    refit : bool, default=True
        Refit estimator with best parameters on full data.
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
        Cross-validation results for all evaluated parameters.
    gp_ : GaussianProcess
        Fitted Gaussian Process model.
        
    Examples
    --------
    >>> from tuiml.algorithms.ensemble import XGBoostClassifier
    >>> from tuiml.evaluation.tuning import BayesianSearchCV
    >>> 
    >>> # Define search space
    >>> param_space = {
    ...     'n_estimators': (50, 500, 'int'),
    ...     'max_depth': (3, 10, 'int'),
    ...     'learning_rate': (0.01, 0.3),
    ... }
    >>> 
    >>> # Create Bayesian search
    >>> bayes = BayesianSearchCV(
    ...     estimator=XGBoostClassifier(),
    ...     param_space=param_space,
    ...     n_iterations=30,
    ...     cv=5
    ... )
    >>> bayes.fit(X_train, y_train)
    >>> 
    >>> print(f"Best params: {bayes.best_params_}")
    >>> print(f"Best score: {bayes.best_score_:.4f}")
    """
    
    @classmethod
    def get_parameter_schema(cls) -> dict:
        """
        Return JSON Schema for BayesianSearchCV parameters.

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
            "param_space": {
                "type": "object",
                "description": "Dictionary with parameters as keys and search spaces as values. Values can be: List (discrete search), Tuple (min, max) for continuous, or Tuple (min, max, 'int') for integer."
            },
            "n_iterations": {
                "type": "integer",
                "default": 50,
                "minimum": 1,
                "description": "Number of iterations to run."
            },
            "acquisition": {
                "type": "string",
                "default": "ei",
                "enum": ["ei", "ucb", "poi"],
                "description": "Acquisition function: 'ei' (Expected Improvement), 'ucb' (Upper Confidence Bound), or 'poi' (Probability of Improvement)."
            },
            "n_random_starts": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "description": "Number of random evaluations before using Gaussian Process."
            },
            "xi": {
                "type": "number",
                "default": 0.01,
                "minimum": 0,
                "description": "Exploration parameter for EI and POI acquisition functions."
            },
            "kappa": {
                "type": "number",
                "default": 2.576,
                "minimum": 0,
                "description": "Exploration parameter for UCB (higher = more exploration)."
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
        param_space: Dict,
        n_iterations: int = 50,
        acquisition: str = 'ei',
        n_random_starts: int = 10,
        xi: float = 0.01,
        kappa: float = 2.576,
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
        
        self.param_space = param_space
        self.n_iterations = n_iterations
        self.acquisition = acquisition
        self.n_random_starts = n_random_starts
        self.xi = xi
        self.kappa = kappa
        
        self.gp_ = None
        self._param_names = list(param_space.keys())
        self._bounds = self._get_bounds()
        
    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        for param_name in self._param_names:
            space = self.param_space[param_name]
            
            if isinstance(space, (list, tuple)):
                if len(space) == 2 or (len(space) == 3 and space[2] in ['int', 'float']):
                    # Continuous or integer range
                    bounds.append((float(space[0]), float(space[1])))
                else:
                    # Discrete values
                    bounds.append((0.0, float(len(space) - 1)))
            else:
                raise ValueError(f"Invalid parameter space for {param_name}")
                
        return bounds
    
    def _vector_to_params(self, X: np.ndarray) -> Dict[str, Any]:
        """Convert parameter vector to parameter dict."""
        params = {}
        
        for i, param_name in enumerate(self._param_names):
            space = self.param_space[param_name]
            value = X[i]
            
            if isinstance(space, list):
                # Discrete values
                idx = int(round(value))
                idx = np.clip(idx, 0, len(space) - 1)
                params[param_name] = space[idx]
            elif isinstance(space, tuple):
                if len(space) == 3 and space[2] == 'int':
                    # Integer range
                    params[param_name] = int(round(value))
                else:
                    # Continuous range
                    params[param_name] = float(value)
            
        return params
    
    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dict to parameter vector."""
        X = np.zeros(len(self._param_names))
        
        for i, param_name in enumerate(self._param_names):
            space = self.param_space[param_name]
            value = params[param_name]
            
            if isinstance(space, list):
                # Discrete values
                X[i] = float(space.index(value))
            else:
                # Continuous or integer
                X[i] = float(value)
                
        return X
    
    def _random_sample(self, rng: np.random.RandomState) -> np.ndarray:
        """Sample random point from parameter space."""
        X = np.zeros(len(self._param_names))
        
        for i, (low, high) in enumerate(self._bounds):
            X[i] = rng.uniform(low, high)
            
        return X
    
    def _optimize_acquisition(
        self,
        acquisition_func: AcquisitionFunction,
        gp: GaussianProcess,
        y_best: float,
        n_restarts: int = 25
    ) -> np.ndarray:
        """
        Optimize acquisition function to find next point to evaluate.
        
        Uses random restarts with L-BFGS-B optimization.
        """
        rng = np.random.RandomState(self.random_state)
        
        best_x = None
        best_acq = -np.inf
        
        # Try multiple random restarts
        for _ in range(n_restarts):
            # Random starting point
            x0 = self._random_sample(rng)
            
            # Optimize (minimize negative acquisition function)
            def neg_acquisition(x):
                x_2d = x.reshape(1, -1)
                return -acquisition_func(x_2d, gp, y_best)[0]
            
            try:
                result = minimize(
                    neg_acquisition,
                    x0,
                    method='L-BFGS-B',
                    bounds=self._bounds
                )
                
                if result.success:
                    acq_value = -result.fun
                    if acq_value > best_acq:
                        best_acq = acq_value
                        best_x = result.x
            except Exception:
                continue
        
        # Fallback to random if optimization failed
        if best_x is None:
            best_x = self._random_sample(rng)
            
        return best_x
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianSearchCV":
        """
        Run Bayesian optimization to find best parameters.
        
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
            'param_vectors': [],
            'mean_test_score': [],
            'std_test_score': [],
            'mean_fit_time': [],
        }
        
        best_score = -np.inf
        best_params = None
        
        if self.verbose > 0:
            print(f"Running Bayesian Optimization for {self.n_iterations} iterations...")
            print(f"Random initialization: {self.n_random_starts} iterations")
            print(f"Acquisition function: {self.acquisition}")
            print()
        
        # Phase 1: Random exploration
        for i in range(min(self.n_random_starts, self.n_iterations)):
            # Sample random parameters
            X_sample = self._random_sample(rng)
            params = self._vector_to_params(X_sample)
            
            if self.verbose > 0:
                print(f"[{i+1}/{self.n_iterations}] Random: {params}")
            
            # Evaluate
            mean_score, std_score, fit_time = self._cross_validate(
                self.estimator, X, y, params
            )
            
            # Store results
            results['params'].append(params)
            results['param_vectors'].append(X_sample)
            results['mean_test_score'].append(mean_score)
            results['std_test_score'].append(std_score)
            results['mean_fit_time'].append(fit_time / self.cv)
            
            if self.verbose > 0:
                print(f"    Score: {mean_score:.4f} ± {std_score:.4f}\n")

            # Update best
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

            self._notify_progress(i + 1, self.n_iterations, params,
                                  mean_score, std_score, best_score)

        # Phase 2: Bayesian optimization
        if self.n_iterations > self.n_random_starts:
            # Initialize GP
            self.gp_ = GaussianProcess(length_scale=1.0, noise=1e-6)
            acquisition_func = AcquisitionFunction(
                kind=self.acquisition,
                xi=self.xi,
                kappa=self.kappa
            )
            
            for i in range(self.n_random_starts, self.n_iterations):
                # Fit GP to observed data
                X_observed = np.array(results['param_vectors'])
                y_observed = np.array(results['mean_test_score'])
                
                self.gp_.fit(X_observed, y_observed)
                
                # Optimize acquisition function
                X_next = self._optimize_acquisition(
                    acquisition_func,
                    self.gp_,
                    best_score
                )
                
                params = self._vector_to_params(X_next)
                
                if self.verbose > 0:
                    print(f"[{i+1}/{self.n_iterations}] Bayesian: {params}")
                
                # Evaluate
                mean_score, std_score, fit_time = self._cross_validate(
                    self.estimator, X, y, params
                )
                
                # Store results
                results['params'].append(params)
                results['param_vectors'].append(X_next)
                results['mean_test_score'].append(mean_score)
                results['std_test_score'].append(std_score)
                results['mean_fit_time'].append(fit_time / self.cv)
                
                if self.verbose > 0:
                    print(f"    Score: {mean_score:.4f} ± {std_score:.4f}\n")

                # Update best
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params

                self._notify_progress(i + 1, self.n_iterations, params,
                                      mean_score, std_score, best_score)

        # Store results
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
            n_iterations=self.n_iterations,
            total_time=self.total_time_
        )
    
    def __repr__(self) -> str:
        return (
            f"BayesianSearchCV(estimator={self.estimator.__class__.__name__}, "
            f"n_iterations={self.n_iterations}, acquisition='{self.acquisition}')"
        )
