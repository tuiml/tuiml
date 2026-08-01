"""Sequential model-based hyperparameter search using Gaussian Processes.

Where :class:`~tuiml.evaluation.tuning.GridSearchCV` enumerates and
:class:`~tuiml.evaluation.tuning.RandomSearchCV` samples blindly, Bayesian
optimization *learns as it goes*: it fits a cheap probabilistic surrogate to
the (configuration, score) pairs observed so far and spends its next expensive
model fit wherever that surrogate says the payoff is greatest.

The module supplies the three pieces of that loop:

- :class:`GaussianProcess` — the surrogate. A from-scratch GP regressor with
  an RBF kernel that returns a posterior mean and standard deviation for any
  untried configuration.
- :class:`AcquisitionFunction` — the decision rule. Turns the surrogate's mean
  and uncertainty into a single "how promising is this point" score, trading
  exploitation of good regions against exploration of uncertain ones.
- :class:`~tuiml.evaluation.tuning.BayesianSearchCV` — the tuner. Warms up
  with random configurations, then repeatedly refits the GP, maximizes the
  acquisition function with L-BFGS-B restarts, and cross-validates the winner.

Reach for this module when a single estimator fit is expensive enough that
thinking about the next configuration is cheaper than trying another one at
random — large models, big datasets, or budgets in the tens of evaluations
rather than thousands. For cheap fits the GP overhead dominates and random
search is the better tool.

Everything is implemented on NumPy and SciPy; no external Bayesian
optimization library is required.

References
----------
.. [Snoek2012] Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical
   Bayesian Optimization of Machine Learning Algorithms. *Advances in Neural
   Information Processing Systems (NeurIPS)*, 25, 2951-2959.
.. [Brochu2010] Brochu, E., Cora, V. M., & de Freitas, N. (2010). A Tutorial
   on Bayesian Optimization of Expensive Cost Functions, with Application to
   Active User Modeling and Hierarchical Reinforcement Learning.
   *arXiv:1012.2599*. https://doi.org/10.48550/arXiv.1012.2599
.. [Rasmussen2006] Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian
   Processes for Machine Learning*. MIT Press.
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
    """**Gaussian Process** surrogate model with an RBF kernel.

    A GP places a distribution over functions and, conditioned on the points
    observed so far, returns for any new point both a predicted mean and an
    honest estimate of how uncertain that prediction is. That uncertainty is
    exactly what an :class:`AcquisitionFunction` needs in order to decide
    where to sample next.

    Overview
    --------
    1. ``fit`` builds the kernel matrix :math:`K` over the training inputs and
       adds ``noise`` to its diagonal for numerical stability.
    2. It inverts that matrix once and caches the result in ``K_inv``.
    3. ``predict`` forms the cross-kernel :math:`K_*` between the query points
       and the training inputs and applies the standard GP posterior formulas.

    Theory
    ------
    The squared-exponential (RBF) kernel with length scale :math:`\\ell` is

    .. math::
        k(x, x') = \\exp\\!\\left(
            -\\frac{\\lVert x - x' \\rVert^{2}}{2\\,\\ell^{2}}
        \\right)

    Conditioning a zero-mean GP prior on observations
    :math:`(X, y)` gives the posterior at test points :math:`X_*`

    .. math::
        \\mu_* = K_*\\,(K + \\sigma_n^{2} I)^{-1}\\, y

    .. math::
        \\Sigma_* = K_{**} - K_*\\,(K + \\sigma_n^{2} I)^{-1}\\,K_*^{\\top}

    where :math:`K = k(X, X)`, :math:`K_* = k(X_*, X)`,
    :math:`K_{**} = k(X_*, X_*)` and :math:`\\sigma_n^{2}` is ``noise``.
    ``predict`` returns :math:`\\mu_*` and, on request, the square root of the
    diagonal of :math:`\\Sigma_*`, clipped at zero so round-off cannot produce
    a negative variance.

    A small :math:`\\ell` makes the surrogate wiggly and trusts observations
    only very locally; a large :math:`\\ell` smooths the surface and lets a
    single observation influence distant regions.

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type. Only ``'rbf'`` (squared exponential) is implemented; the
        value is stored but not otherwise consulted.
    length_scale : float, default=1.0
        Length scale :math:`\\ell` of the RBF kernel. Inputs are used on their
        raw scale, so a single length scale is shared by all dimensions.
    noise : float, default=1e-10
        Variance :math:`\\sigma_n^{2}` added to the diagonal of :math:`K`. Acts
        as a jitter that keeps the inversion well conditioned; raise it if
        ``numpy.linalg.inv`` complains about a singular matrix.

    Attributes
    ----------
    X_train : np.ndarray of shape (n_samples, n_features) or None
        Training inputs retained from the last :meth:`fit`. ``None`` before
        fitting.
    y_train : np.ndarray of shape (n_samples,) or None
        Training targets retained from the last :meth:`fit`.
    K_inv : np.ndarray of shape (n_samples, n_samples) or None
        Cached inverse of the noise-augmented kernel matrix.

    Notes
    -----
    **Complexity.** :meth:`fit` is :math:`O(n^{3})` for the explicit matrix
    inversion and :math:`O(n^{2})` in memory; :meth:`predict` is
    :math:`O(m\\,n)` for the mean and :math:`O(m^{2} + m\\,n^{2})` when
    standard deviations are requested. This is fine for the tens to low
    hundreds of observations a hyperparameter search produces, and unsuitable
    for large-scale regression.

    **When to use.** As the surrogate inside
    :class:`~tuiml.evaluation.tuning.BayesianSearchCV`. It is a deliberately
    minimal implementation: no kernel hyperparameter marginal-likelihood
    optimization, no Cholesky solve, no per-dimension length scales.

    References
    ----------
    .. [Rasmussen2006] Rasmussen, C. E., & Williams, C. K. I. (2006).
       *Gaussian Processes for Machine Learning*, chapter 2. MIT Press.

    See Also
    --------
    :class:`~tuiml.evaluation.tuning.bayesian_search.AcquisitionFunction` :
        Consumes this model's mean and standard deviation.
    :class:`~tuiml.evaluation.tuning.BayesianSearchCV` : Uses the GP as the
        surrogate for the score surface.

    Examples
    --------
    Fit the surrogate to four observations and query an untried point:

    >>> import numpy as np
    >>> from tuiml.evaluation.tuning.bayesian_search import GaussianProcess
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1])
    >>> gp = GaussianProcess(length_scale=1.0, noise=1e-6)
    >>> gp.fit(X, y)
    >>> mu, sigma = gp.predict(np.array([[1.5]]), return_std=True)
    >>> round(float(mu[0]), 3)
    1.019
    >>> round(float(sigma[0]), 3)
    0.1

    The posterior interpolates the observations it has already seen:

    >>> round(float(gp.predict(np.array([[1.0]]))[0]), 3)
    0.8
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for constructor parameters."""
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
        """Initialize the GP; see the class docstring for parameters."""
        self.kernel = kernel
        self.length_scale = length_scale
        self.noise = noise
        
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the RBF (squared exponential) kernel matrix between two sets of points.

        Implements
        :math:`k(x_1, x_2) = \\exp(-\\lVert x_1 - x_2 \\rVert^{2} / (2\\,\\ell^{2}))`
        using the expanded form of the squared Euclidean distance so the whole
        matrix is produced with one matrix product.

        Parameters
        ----------
        X1 : np.ndarray of shape (n1, n_features)
            First set of points.
        X2 : np.ndarray of shape (n2, n_features)
            Second set of points.

        Returns
        -------
        K : np.ndarray of shape (n1, n2)
            Kernel (covariance) matrix with entries in :math:`(0, 1]`.
        """
        # Compute squared Euclidean distances
        dists = np.sum(X1**2, axis=1)[:, np.newaxis] + \
                np.sum(X2**2, axis=1)[np.newaxis, :] - \
                2 * np.dot(X1, X2.T)
        
        # Apply RBF kernel
        return np.exp(-dists / (2 * self.length_scale**2))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Condition the Gaussian Process on observed points.

        Stores the training data, builds the noise-augmented kernel matrix
        :math:`K + \\sigma_n^{2} I`, and caches its inverse in ``K_inv``. Each
        call replaces any previous fit.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Observed inputs.
        y : np.ndarray of shape (n_samples,)
            Observed targets.

        Returns
        -------
        None
            The model is updated in place; ``X_train``, ``y_train`` and
            ``K_inv`` are set.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the kernel matrix is singular. Increase ``noise`` or drop
            duplicate rows from ``X``.
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
        """Evaluate the GP posterior at new points.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points. Must have the same number of columns as the data
            passed to :meth:`fit`.
        return_std : bool, default=False
            If ``True``, also return the posterior standard deviation, which
            is what acquisition functions use to quantify exploration value.

        Returns
        -------
        y_mean : np.ndarray of shape (n_samples,)
            Posterior mean :math:`\\mu_*`. Returned on its own when
            ``return_std=False``.
        y_std : np.ndarray of shape (n_samples,)
            Posterior standard deviation, clipped at zero. Only returned when
            ``return_std=True``, as the second element of a tuple.

        Raises
        ------
        ValueError
            If :meth:`fit` has not been called yet.
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
    """**Acquisition functions** that turn a GP posterior into a sampling decision.

    A callable policy for Bayesian optimization. Given a fitted
    :class:`GaussianProcess` and the best score observed so far, it scores
    candidate points so that *higher is better*; the optimizer then maximizes
    it to choose the next configuration to actually train. All three variants
    balance the same tension: the posterior mean :math:`\\mu(x)` pulls toward
    regions already known to be good (exploitation) while the posterior
    standard deviation :math:`\\sigma(x)` pulls toward regions the surrogate
    has not pinned down (exploration).

    Theory
    ------
    Write :math:`y^{+}` for the incumbent best observed score and

    .. math::
        Z = \\frac{\\mu(x) - y^{+} - \\xi}{\\sigma(x)}

    **Expected Improvement** (``kind='ei'``) is the expected amount by which
    :math:`x` beats the incumbent, which has the closed form

    .. math::
        \\text{EI}(x) = \\left(\\mu(x) - y^{+} - \\xi\\right)\\,\\Phi(Z)
                      + \\sigma(x)\\,\\phi(Z)

    where :math:`\\Phi` and :math:`\\phi` are the standard normal CDF and PDF.
    The first term rewards a high mean, the second rewards uncertainty, and
    :math:`\\xi` sets how much improvement must be expected before a point
    counts as promising. EI is the usual default because it is scale-aware:
    it answers "by how much", not merely "how likely".

    **Probability of Improvement** (``kind='poi'``) keeps only the probability
    that the incumbent is beaten,

    .. math::
        \\text{PI}(x) = P\\!\\left(f(x) \\geq y^{+} + \\xi\\right) = \\Phi(Z)

    which ignores the size of the gain and therefore exploits greedily unless
    :math:`\\xi` is raised.

    **Upper Confidence Bound** (``kind='ucb'``) is an explicit optimistic
    bound and the only variant that does not reference :math:`y^{+}`,

    .. math::
        \\text{UCB}(x) = \\mu(x) + \\kappa\\,\\sigma(x)

    with :math:`\\kappa` directly dialing exploration. The default
    :math:`\\kappa = 2.576` is the two-sided 99% normal quantile.

    Parameters
    ----------
    kind : {'ei', 'ucb', 'poi'}, default='ei'
        Which policy to evaluate: ``'ei'`` for Expected Improvement,
        ``'ucb'`` for Upper Confidence Bound, ``'poi'`` for Probability of
        Improvement. Any other value raises at call time, not at construction.
    xi : float, default=0.01
        Minimum improvement margin :math:`\\xi` for ``'ei'`` and ``'poi'``.
        Larger values explore more; ``0.0`` is purely greedy. Ignored by
        ``'ucb'``.
    kappa : float, default=2.576
        Exploration weight :math:`\\kappa` for ``'ucb'``; larger values favor
        uncertain regions. Ignored by ``'ei'`` and ``'poi'``.

    Notes
    -----
    **Complexity.** One GP prediction with standard deviations plus
    :math:`O(m)` arithmetic for :math:`m` candidate points, so the cost is
    dominated by :meth:`GaussianProcess.predict`.

    **When to use.** Keep ``'ei'`` unless you have a reason not to. Choose
    ``'ucb'`` when you want a single, interpretable exploration knob, and
    ``'poi'`` when any improvement at all is what matters. Standard deviations
    are floored at ``1e-9`` before division, so already-observed points where
    :math:`\\sigma \\to 0` yield a near-zero score rather than a divide-by-zero.

    References
    ----------
    .. [Jones1998] Jones, D. R., Schonlau, M., & Welch, W. J. (1998).
       Efficient Global Optimization of Expensive Black-Box Functions.
       *Journal of Global Optimization*, 13(4), 455-492.
       https://doi.org/10.1023/A:1008306431147
    .. [Srinivas2010] Srinivas, N., Krause, A., Kakade, S., & Seeger, M.
       (2010). Gaussian Process Optimization in the Bandit Setting: No Regret
       and Experimental Design. *ICML*, 1015-1022.

    See Also
    --------
    :class:`~tuiml.evaluation.tuning.bayesian_search.GaussianProcess` :
        Supplies the mean and standard deviation this class consumes.
    :class:`~tuiml.evaluation.tuning.BayesianSearchCV` : Maximizes this
        function to pick each next configuration.

    Examples
    --------
    Score two candidate points against a surrogate whose incumbent is 0.9:

    >>> import numpy as np
    >>> from tuiml.evaluation.tuning.bayesian_search import (
    ...     AcquisitionFunction, GaussianProcess)
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1])
    >>> gp = GaussianProcess(length_scale=1.0, noise=1e-6)
    >>> gp.fit(X, y)
    >>> candidates = np.array([[1.5], [2.5]])
    >>> ei = AcquisitionFunction(kind='ei', xi=0.01)
    >>> [round(float(v), 4) for v in ei(candidates, gp, y_best=0.9)]
    [0.1157, 0.0]

    UCB and POI rank the same candidates with different appetites for risk:

    >>> ucb = AcquisitionFunction(kind='ucb', kappa=2.576)
    >>> [round(float(v), 4) for v in ucb(candidates, gp, y_best=0.9)]
    [1.2752, 0.8316]
    >>> poi = AcquisitionFunction(kind='poi')
    >>> [round(float(v), 4) for v in poi(candidates, gp, y_best=0.9)]
    [0.8626, 0.0006]
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for constructor parameters."""
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
        """Initialize the acquisition policy; see the class docstring for parameters."""
        self.kind = kind
        self.xi = xi
        self.kappa = kappa
        
    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcess,
        y_best: float
    ) -> np.ndarray:
        """Score candidate points with the configured acquisition policy.

        Dispatches on ``self.kind`` to Expected Improvement, Upper Confidence
        Bound, or Probability of Improvement.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Candidate points to score.
        gp : GaussianProcess
            Surrogate already fitted on the observations so far.
        y_best : float
            Incumbent best observed score :math:`y^{+}`. Used by ``'ei'`` and
            ``'poi'``; ignored by ``'ucb'``.

        Returns
        -------
        values : np.ndarray of shape (n_samples,)
            Acquisition values, higher meaning more promising. Not comparable
            across different ``kind`` settings.

        Raises
        ------
        ValueError
            If ``self.kind`` is not one of ``'ei'``, ``'ucb'`` or ``'poi'``.
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
    >>> from tuiml.algorithms.ensemble import BaggingClassifier
    >>> from tuiml.datasets import load_iris
    >>> from tuiml.evaluation.splitting import train_test_split
    >>> from tuiml.evaluation.tuning import BayesianSearchCV
    >>> X, y = load_iris()
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    A search space maps each parameter to a ``(low, high)`` range for floats
    or a ``(low, high, 'int')`` triple for integers:

    >>> param_space = {
    ...     'n_estimators': (5, 20, 'int'),
    ...     'bag_size_percent': (50.0, 100.0),
    ... }
    >>> search = BayesianSearchCV(
    ...     estimator=BaggingClassifier(),
    ...     param_space=param_space,
    ...     n_iterations=5,
    ...     cv=2,
    ... )
    >>> search = search.fit(X_train, y_train)              # doctest: +SKIP
    >>> sorted(search.best_params_)                        # doctest: +SKIP
    ['bag_size_percent', 'n_estimators']
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
            "random_seed": {
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
        random_seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ):
        """Initialize the tuner; see the class docstring for parameters."""
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
                """Negated acquisition value at one point, for use with a minimizer.

                :func:`scipy.optimize.minimize` only minimizes, but the search
                wants the point where ``acquisition_func`` is *largest*. Negating
                the score turns "maximize acquisition" into the minimization
                problem L-BFGS-B actually solves, without changing where the
                optimum sits.

                Parameters
                ----------
                x : np.ndarray of shape (n_params,)
                    Flat parameter vector proposed by the optimizer, in the same
                    order as ``self._param_names``.

                Returns
                -------
                neg_value : float
                    ``-acquisition_func(x, gp, y_best)``, i.e. the negated
                    acquisition score at ``x``. Smaller is more promising.
                """
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
        """Return a short constructor-like summary of the search.

        Returns
        -------
        repr : str
            ``"BayesianSearchCV(estimator=..., n_iterations=..., acquisition='...')"``,
            showing the wrapped estimator's class name and the two settings
            that most affect how the search behaves.
        """
        return (
            f"BayesianSearchCV(estimator={self.estimator.__class__.__name__}, "
            f"n_iterations={self.n_iterations}, acquisition='{self.acquisition}')"
        )
