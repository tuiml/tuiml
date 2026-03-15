"""Bayesian Linear Regression implementation."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Regressor, regressor


@regressor(tags=["bayesian", "regression", "linear", "uncertainty"], version="1.0.0")
class BayesianLinearRegressor(Regressor):
    """Bayesian Linear Regression with **conjugate prior** for uncertainty-aware predictions.

    BayesianLinearRegressor places a **Gaussian prior** over the weight
    parameters and computes a closed-form **posterior distribution** using
    the conjugate normal-inverse-gamma model. This yields both point
    predictions and **predictive uncertainty** estimates for each input.

    Overview
    --------
    The algorithm proceeds as follows:

    1. Optionally augment the feature matrix with a bias column for the intercept
    2. Compute the **posterior covariance** :math:`S_N` from the prior precision
       and the data likelihood
    3. Compute the **posterior mean** :math:`\\mathbf{m}_N` as the MAP estimate
    4. At prediction time, compute the mean prediction and optionally the
       **predictive standard deviation** that accounts for both parameter
       uncertainty and observation noise

    Theory
    ------
    Given a prior on the weights :math:`\\mathbf{w} \\sim \\mathcal{N}(\\mathbf{0}, \\alpha^{-1} I)`
    and a likelihood :math:`p(\\mathbf{y} | X, \\mathbf{w}) = \\mathcal{N}(\\Phi \\mathbf{w}, \\beta^{-1} I)`,
    the posterior is Gaussian:

    .. math::
        p(\\mathbf{w} | \\mathbf{y}, X) = \\mathcal{N}(\\mathbf{m}_N, S_N)

    where:

    .. math::
        S_N = (\\alpha I + \\beta \\, \\Phi^\\top \\Phi)^{-1}

    .. math::
        \\mathbf{m}_N = \\beta \\, S_N \\, \\Phi^\\top \\mathbf{y}

    The predictive distribution for a new input :math:`\\mathbf{x}_*` is:

    .. math::
        p(y_* | \\mathbf{x}_*, \\mathbf{y}, X) = \\mathcal{N}(\\mathbf{m}_N^\\top \\phi_*, \\sigma_*^2)

    with predictive variance:

    .. math::
        \\sigma_*^2 = \\beta^{-1} + \\phi_*^\\top S_N \\, \\phi_*

    Parameters
    ----------
    alpha : float, default=1.0
        Prior precision (inverse variance) for the weight prior. Larger
        values impose stronger regularisation toward zero weights.
    beta : float, default=1.0
        Noise precision (inverse variance) of the observation noise.
        Larger values assume less noise in the observations.
    fit_intercept : bool, default=True
        Whether to add an intercept (bias) term to the model by
        augmenting the feature matrix with a column of ones.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Posterior mean of the weight vector (excluding intercept).
    intercept_ : float
        The intercept term. Set to 0.0 if ``fit_intercept=False``.
    posterior_cov_ : np.ndarray of shape (n_params, n_params)
        Posterior covariance matrix :math:`S_N` over the weight parameters.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(m^2 n + m^3)` where :math:`n` = samples, :math:`m` = features
    - Prediction (mean): :math:`O(m)` per sample
    - Prediction (std): :math:`O(m^2)` per sample
    - Memory: :math:`O(m^2)` for storing the posterior covariance

    **When to use BayesianLinearRegressor:**

    - When you need **uncertainty estimates** alongside predictions
    - Small-to-medium datasets where a linear model is appropriate
    - When you want built-in regularisation via the prior
    - Active learning or Bayesian optimisation settings
    - When interpretability of the model weights is important

    References
    ----------
    .. [Bishop2006] Bishop, C.M. (2006).
           **Pattern Recognition and Machine Learning.**
           *Springer*, Chapter 3.3.
           DOI: `10.1007/978-0-387-45528-0 <https://doi.org/10.1007/978-0-387-45528-0>`_

    .. [Murphy2012] Murphy, K.P. (2012).
           **Machine Learning: A Probabilistic Perspective.**
           *MIT Press*, Chapter 7.6.

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.GaussianProcessesRegressor` : Non-parametric Bayesian regression with kernels.
    :class:`~tuiml.algorithms.linear.LinearRegression` : Ordinary least squares linear regression.

    Examples
    --------
    Basic Bayesian linear regression with uncertainty:

    >>> from tuiml.algorithms.bayesian import BayesianLinearRegressor
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([1.1, 2.0, 3.1, 3.9, 5.0])
    >>> reg = BayesianLinearRegressor(alpha=1.0, beta=25.0)
    >>> reg.fit(X, y)
    BayesianLinearRegressor(alpha=1.0, beta=25.0)
    >>> reg.predict(np.array([[3.5]]))
    array([...])
    >>> reg.predict_std(np.array([[3.5]]))
    array([...])
    """

    def __init__(self, alpha: float = 1.0,
                 beta: float = 1.0,
                 fit_intercept: bool = True):
        """Initialize BayesianLinearRegressor.

        Parameters
        ----------
        alpha : float, default=1.0
            Prior precision for the weight prior.
        beta : float, default=1.0
            Noise precision of the observations.
        fit_intercept : bool, default=True
            Whether to fit an intercept term.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self.posterior_cov_ = None
        self._posterior_mean = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "alpha": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "description": "Prior precision (regularisation strength)"
            },
            "beta": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "description": "Noise precision (inverse observation variance)"
            },
            "fit_intercept": {
                "type": "boolean",
                "default": True,
                "description": "Whether to fit an intercept term"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return regressor capabilities."""
        return ["numeric", "numeric_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(m^2 * n + m^3) training, O(m) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Bishop, C.M. (2006). Pattern Recognition and Machine Learning. "
            "Springer, Chapter 3.3."
        ]

    def _augment_features(self, X: np.ndarray) -> np.ndarray:
        """Augment feature matrix with a bias column if fit_intercept is True.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        Phi : np.ndarray of shape (n_samples, n_features) or (n_samples, n_features + 1)
            Augmented design matrix.
        """
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearRegressor":
        """Fit the Bayesian linear regression model.

        Computes the posterior distribution over weights given the
        training data using the conjugate Gaussian prior.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BayesianLinearRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        Phi = self._augment_features(X)
        n_params = Phi.shape[1]

        # Posterior covariance: S_N = (alpha * I + beta * Phi^T Phi)^{-1}
        prior_precision = self.alpha * np.eye(n_params)
        data_precision = self.beta * (Phi.T @ Phi)
        precision_matrix = prior_precision + data_precision
        self.posterior_cov_ = np.linalg.inv(precision_matrix)

        # Posterior mean: m_N = beta * S_N * Phi^T * y
        self._posterior_mean = self.beta * (self.posterior_cov_ @ (Phi.T @ y))

        # Extract coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = self._posterior_mean[0]
            self.coef_ = self._posterior_mean[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self._posterior_mean.copy()

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values (posterior mean predictions).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted mean target values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        Phi = self._augment_features(X)
        return Phi @ self._posterior_mean

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Predict standard deviations of the predictive distribution.

        The predictive variance accounts for both the posterior uncertainty
        over the weights and the observation noise.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_std : np.ndarray of shape (n_samples,)
            Predicted standard deviations.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        Phi = self._augment_features(X)

        # Predictive variance: 1/beta + phi^T S_N phi for each sample
        # Efficiently computed as: diag(Phi @ S_N @ Phi^T) + 1/beta
        Phi_S = Phi @ self.posterior_cov_
        predictive_var = np.sum(Phi_S * Phi, axis=1) + (1.0 / self.beta)

        return np.sqrt(np.maximum(predictive_var, 0.0))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the R-squared (coefficient of determination) score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            R-squared score.
        """
        self._check_is_fitted()
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"BayesianLinearRegressor(alpha={self.alpha}, beta={self.beta})"
        return f"BayesianLinearRegressor(alpha={self.alpha}, beta={self.beta})"
