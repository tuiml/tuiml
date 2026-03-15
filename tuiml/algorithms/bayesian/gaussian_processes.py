"""Gaussian Process regression implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from tuiml.base.algorithms import Regressor, regressor

@regressor(tags=["bayesian", "regression", "kernel", "uncertainty"], version="1.0.0")
class GaussianProcessesRegressor(Regressor):
    """Gaussian Process Regression (**GPR**) with **uncertainty estimation**.

    Gaussian Processes are a **non-parametric Bayesian** approach to regression.
    They provide a probabilistic model that not only makes predictions but
    also estimates the **predictive variance** (uncertainty) of those predictions.

    Overview
    --------
    The algorithm performs regression through the following steps:

    1. Optionally **normalise** features and targets to zero mean, unit variance
    2. Compute the **kernel matrix** :math:`K` between all training points
    3. Add noise to the diagonal for regularisation: :math:`K + \\sigma_n^2 I`
    4. Solve for dual coefficients via **Cholesky decomposition**
    5. At prediction time, compute the predictive **mean** and **variance**
       using the kernel between test and training points

    Theory
    ------
    A Gaussian Process defines a distribution over functions:

    .. math::

        f(\\mathbf{x}) \\sim \\mathcal{GP}\\bigl(m(\\mathbf{x}),\\, k(\\mathbf{x}, \\mathbf{x}')\\bigr)

    Given training data :math:`(X, \\mathbf{y})` with noise
    :math:`\\sigma_n^2`, the **predictive distribution** at test points
    :math:`X_*` is Gaussian:

    .. math::

        \\bar{f}_* &= K_*^\\top (K + \\sigma_n^2 I)^{-1} \\mathbf{y} \\\\
        \\text{cov}(f_*) &= K_{**} - K_*^\\top (K + \\sigma_n^2 I)^{-1} K_*

    The **log marginal likelihood** used for model comparison is:

    .. math::

        \\log p(\\mathbf{y} \\mid X) = -\\tfrac{1}{2} \\mathbf{y}^\\top K_y^{-1} \\mathbf{y}
        - \\tfrac{1}{2} \\log |K_y| - \\tfrac{n}{2} \\log 2\\pi

    where :math:`K_y = K + \\sigma_n^2 I`.

    Parameters
    ----------
    kernel : str, default='rbf'
        The kernel function to use:

        - ``'rbf'`` -- Radial Basis Function (Squared Exponential)
        - ``'poly'`` -- Polynomial kernel
        - ``'linear'`` -- Linear (dot product) kernel
    gamma : float or "scale", default="scale"
        Kernel coefficient for 'rbf' and 'poly'.
        If "scale", uses ``1.0 / (n_features * X.var())``.
    degree : int, default=2
        The degree of the polynomial kernel. Ignored for other kernels.
    noise : float, default=1.0
        Noise level (Tikhonov regularisation). Added to the diagonal of the
        kernel matrix to account for noise in the observations and for
        numerical stability. Larger values increase smoothing.
    normalize : bool, default=True
        Whether to normalise the features and target variables to zero mean
        and unit variance before fitting. Highly recommended for GPR.

    Attributes
    ----------
    alpha_ : np.ndarray
        Dual coefficients (weights) assigned to each training sample.
    L_ : np.ndarray
        Lower triangular Cholesky factor of the kernel matrix.
    X_train_ : np.ndarray
        Features used during training (normalised if ``normalize=True``).
    y_train_ : np.ndarray
        Targets used during training (normalised if ``normalize=True``).
    y_mean_ : float
        The mean value of the training targets.
    y_std_ : float
        The standard deviation of the training targets.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n^3)` due to Cholesky decomposition of the :math:`n \\times n` kernel matrix
    - Prediction: :math:`O(n^2)` per test point for the mean; :math:`O(n^2)` for the variance
    - Memory: :math:`O(n^2)` for storing the kernel matrix

    **When to use GaussianProcessesRegressor:**

    - When you need **calibrated uncertainty estimates** alongside predictions
    - Small-to-medium datasets (up to a few thousand samples)
    - When the relationship between features and target is smooth
    - Bayesian optimisation and active learning settings
    - When model selection via the log marginal likelihood is desirable

    References
    ----------
    .. [Rasmussen2006] Rasmussen, C.E. and Williams, C.K.I. (2006).
           **Gaussian Processes for Machine Learning.**
           *MIT Press*.

    .. [Scholkopf2002] Scholkopf, B. and Smola, A.J. (2002).
           **Learning with Kernels: Support Vector Machines, Regularization,
           Optimization, and Beyond.**
           *MIT Press*.

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.NaiveBayesClassifier` : Probabilistic classifier using Bayes' theorem.
    :class:`~tuiml.algorithms.bayesian.BayesianNetworkClassifier` : Graphical model-based Bayesian classifier.

    Examples
    --------
    Basic regression with RBF kernel:

    >>> from tuiml.algorithms.bayesian import GaussianProcessesRegressor
    >>> import numpy as np
    >>>
    >>> # Create sinusoidal training data
    >>> X = np.array([[1], [3], [5], [7], [9]])
    >>> y = np.sin(X).ravel()
    >>>
    >>> # Fit the Gaussian Process model
    >>> gp = GaussianProcessesRegressor(kernel='rbf', noise=0.1)
    >>> gp.fit(X, y)
    GaussianProcessesRegressor(kernel='rbf', noise=0.1, n_train=5)
    >>>
    >>> # Predict with uncertainty
    >>> mean, std = gp.predict([[4]], return_std=True)
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: Any = "scale",
        degree: int = 2,
        noise: float = 1.0,
        normalize: bool = True,
    ):
        """Initialize GaussianProcessesRegressor.

        Parameters
        ----------
        kernel : str, default='rbf'
            Kernel type ('rbf', 'poly', or 'linear').
        gamma : float or "scale", default="scale"
            Kernel coefficient.
        degree : int, default=2
            Polynomial degree (only for 'poly' kernel).
        noise : float, default=1.0
            Noise level (regularisation).
        normalize : bool, default=True
            Whether to normalise inputs.
        """
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.noise = noise
        self.normalize = normalize

        # Fitted attributes
        self.alpha_ = None
        self.L_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.y_mean_ = None
        self.y_std_ = None
        self._gamma_value = None
        self._X_mean = None
        self._X_std = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "kernel": {
                "type": "string",
                "default": "rbf",
                "enum": ["rbf", "poly", "linear"],
                "description": "Kernel type"
            },
            "gamma": {
                "type": ["number", "string"],
                "default": "scale",
                "description": "Kernel coefficient ('scale' or float)"
            },
            "degree": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Polynomial kernel degree"
            },
            "noise": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "description": "Noise level (regularization)"
            },
            "normalize": {
                "type": "boolean",
                "default": True,
                "description": "Whether to normalize inputs"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return [
            "numeric",
            "numeric_class"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n^3) training, O(n^2) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes "
            "for machine learning. MIT Press."
        ]

    def _compute_gamma(self, X: np.ndarray) -> float:
        """Compute the gamma kernel coefficient based on the chosen strategy.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data used to compute variance-based gamma.

        Returns
        -------
        gamma : float
            The computed gamma value for the kernel function.
        """
        n_features = X.shape[1]
        if self.gamma == "scale":
            var = np.var(X)
            if var > 0:
                return 1.0 / (n_features * var)
            return 1.0 / n_features
        elif self.gamma == "auto":
            return 1.0 / n_features
        else:
            return float(self.gamma)

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the kernel (Gram) matrix between two sets of vectors.

        Parameters
        ----------
        X1 : np.ndarray of shape (n_samples_1, n_features)
            First set of input vectors.
        X2 : np.ndarray of shape (n_samples_2, n_features)
            Second set of input vectors.

        Returns
        -------
        K : np.ndarray of shape (n_samples_1, n_samples_2)
            The kernel matrix where ``K[i, j] = k(X1[i], X2[j])``.
        """
        if self.kernel == "linear":
            return X1 @ X2.T

        elif self.kernel == "poly":
            return (self._gamma_value * (X1 @ X2.T) + 1) ** self.degree

        elif self.kernel == "rbf":
            # Efficient RBF kernel computation
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y
            X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            sq_dist = X1_sq + X2_sq - 2 * X1 @ X2.T
            sq_dist = np.maximum(sq_dist, 0)  # Handle numerical errors
            return np.exp(-self._gamma_value * sq_dist)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessesRegressor":
        """Fit the Gaussian Process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GaussianProcessesRegressor
            Returns the fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Normalize inputs
        if self.normalize:
            self._X_mean = np.mean(X, axis=0)
            self._X_std = np.std(X, axis=0)
            self._X_std[self._X_std == 0] = 1.0
            X = (X - self._X_mean) / self._X_std
        else:
            self._X_mean = np.zeros(n_features)
            self._X_std = np.ones(n_features)

        # Normalize targets
        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y)
        if self.y_std_ == 0:
            self.y_std_ = 1.0
        y_normalized = (y - self.y_mean_) / self.y_std_

        self.X_train_ = X
        self.y_train_ = y_normalized
        self._gamma_value = self._compute_gamma(X)

        # Compute kernel matrix
        K = self._kernel_matrix(X, X)

        # Add noise to diagonal (regularization)
        K += self.noise * np.eye(n_samples)

        # Cholesky decomposition
        try:
            self.L_ = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # If Cholesky fails, add more regularization
            K += 1e-6 * np.eye(n_samples)
            self.L_ = np.linalg.cholesky(K)

        # Solve for alpha: K @ alpha = y
        # Using Cholesky: L @ L.T @ alpha = y
        # Solve L @ v = y, then L.T @ alpha = v
        v = np.linalg.solve(self.L_, y_normalized)
        self.alpha_ = np.linalg.solve(self.L_.T, v)

        self._is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict using the Gaussian Process regression model.

        This method can return both the mean of the predictive distribution 
        and its standard deviation (uncertainty).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points where the GP is evaluated.
        return_std : bool, default=False
            Whether or not to return the standard deviation of the 
            predictive distribution at the query points.

        Returns
        -------
        y_mean : np.ndarray of shape (n_samples,)
            Mean of predictive distribution at query points.
        y_std : np.ndarray of shape (n_samples,)
            Standard deviation of predictive distribution at query points. 
            Only returned if ``return_std`` is True.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Normalize inputs
        if self.normalize:
            X = (X - self._X_mean) / self._X_std

        # Compute kernel between test and training points
        K_star = self._kernel_matrix(X, self.X_train_)

        # Predictive mean: K_star @ alpha
        mean = K_star @ self.alpha_

        # Denormalize
        mean = mean * self.y_std_ + self.y_mean_

        if return_std:
            # Predictive variance
            # v = L^{-1} @ K_star.T
            v = np.linalg.solve(self.L_, K_star.T)

            # K_** - K_* @ K^{-1} @ K_*^T = K_** - v.T @ v
            K_star_star = self._kernel_matrix(X, X)
            var = np.diag(K_star_star) - np.sum(v ** 2, axis=0)

            # Ensure non-negative variance
            var = np.maximum(var, 0)
            std = np.sqrt(var) * self.y_std_

            return mean, std

        return mean

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates.

        Alias for ``predict(X, return_std=True)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        mean : np.ndarray of shape (n_samples,)
            Predicted mean at each query point.
        std : np.ndarray of shape (n_samples,)
            Predicted standard deviation at each query point.
        """
        return self.predict(X, return_std=True)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the R-squared (coefficient of determination) score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        r2 : float
            R-squared score. Best possible score is 1.0; it can be
            negative if the model is worse than predicting the mean.
        """
        self._check_is_fitted()
        y_pred = self.predict(X)
        y = np.asarray(y)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def log_marginal_likelihood(self) -> float:
        """Compute the log marginal likelihood of the training data.

        Returns
        -------
        log_likelihood : float
            The log marginal likelihood value. Higher values indicate
            a better fit of the kernel and noise parameters to the data.
        """
        self._check_is_fitted()
        n = len(self.y_train_)

        # -0.5 * y.T @ K^{-1} @ y - 0.5 * log|K| - n/2 * log(2*pi)
        # = -0.5 * y.T @ alpha - sum(log(diag(L))) - n/2 * log(2*pi)

        log_likelihood = -0.5 * self.y_train_ @ self.alpha_
        log_likelihood -= np.sum(np.log(np.diag(self.L_)))
        log_likelihood -= 0.5 * n * np.log(2 * np.pi)

        return log_likelihood

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            n_train = len(self.X_train_)
            return (f"GaussianProcessesRegressor(kernel='{self.kernel}', "
                    f"noise={self.noise}, n_train={n_train})")
        return f"GaussianProcessesRegressor(kernel='{self.kernel}', noise={self.noise})"
