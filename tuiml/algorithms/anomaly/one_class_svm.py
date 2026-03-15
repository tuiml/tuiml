"""One-Class SVM for novelty detection and anomaly detection."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List
from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["anomaly-detection", "svm", "novelty-detection"], version="1.0.0")
class OneClassSVMDetector(Classifier):
    """One-Class Support Vector Machine for novelty and anomaly detection.

    The One-Class SVM learns a **decision boundary** that encompasses the bulk of the 
    normal data in a high-dimensional feature space. It identifies anomalies as points 
    lying outside this boundary. It is particularly effective for **novelty detection**, 
    where the model is trained primarily on normal instances.

    Overview
    --------
    The algorithm works by mapping the input data into a high-dimensional feature 
    space (using a kernel) and finding the hyperplane that best separates the 
    data from the origin with a maximum margin:

    1. Map data to high dimensions via the selected **kernel** (e.g., RBF)
    2. Solve an optimization problem to find a "frontier" encompassing the data
    3. New points are checked against this frontier
    4. Points falling outside the frontier are labeled as anomalies

    The intuition: By separating the data from the origin in a high-dimensional 
    feature space, we capture the "support" of the distribution.

    Theory
    ------
    The One-Class SVM solves the following optimization problem:

    .. math::
        \\min_{w, \\rho, \\xi} \\frac{1}{2}\\|w\\|^2 - \\rho + \\frac{1}{\\nu n}\\sum_{i=1}^n \\xi_i

    Subject to:

    .. math::
        w^T\\phi(x_i) \\geq \\rho - \\xi_i, \\quad \\xi_i \\geq 0

    where:

    - :math:`w` — Normal vector to the separating hyperplane
    - :math:`\\rho` — Offset from the origin
    - :math:`\\nu` (nu) — Controls the trade-off between keeping data inside the frontier and the "smoothness" of the boundary
    - :math:`\\phi` — Non-linear mapping to a higher-dimensional space (defined by the kernel)
    - :math:`\\xi_i` — Slack variables allowing for points to lie outside the boundary

    **Score interpretation:**

    - Positive score → Normal point (inside the decision boundary)
    - Negative score → Anomaly (outside the decision boundary)

    Parameters
    ----------
    kernel : str, default="rbf"
        Kernel function used to map data to higher dimensions:
        
        - ``"linear"`` — Linear kernel
        - ``"rbf"`` — Radial Basis Function (Gaussian)
        - ``"poly"`` — Polynomial kernel
        - ``"sigmoid"`` — Sigmoid kernel

    nu : float, default=0.1
        Upper bound on the fraction of training errors and a lower bound on 
        the fraction of support vectors. Must be in the range ``(0, 1]``. 
        Roughly corresponds to the expected contamination rate.

    gamma : float or "auto", default="auto"
        Kernel coefficient for RBF, polynomial, and sigmoid kernels:
        
        - ``"auto"`` — Uses ``1 / n_features``
        - ``float`` — User-defined positive coefficient

    degree : int, default=3
        Degree of the polynomial kernel. Ignored for other kernels.

    coef0 : float, default=0.0
        Independent term in the polynomial and sigmoid kernel functions.

    tol : float, default=1e-3
        Tolerance for the stopping criterion in optimization.

    max_iter : int, default=1000
        Maximum number of iterations for the solver.

    Attributes
    ----------
    support_vectors_ : np.ndarray
        Points from the training set that define the decision boundary.

    dual_coef_ : np.ndarray
        Coefficients (Lagrange multipliers) associated with the support vectors.

    offset_ : float
        The learned offset (:math:`\\rho`) of the decision function.

    n_support_ : int
        Total number of support vectors found.

    gamma_ : float
        Actual gamma value used during the computation.

    n_features_in_ : int
        Number of features observed during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n^2)` to :math:`O(n^3)` (depends on kernel and tolerance)
    - Prediction: :math:`O(n_{sv} \\cdot p)` where :math:`n_{sv}` is the number of support vectors

    **When to use One-Class SVM:**

    - Novelty detection (training set contains only/mostly normal samples)
    - High-dimensional data where non-linear boundaries are required
    - Complex data distributions that can be captured with kernels
    - When precise control over the "frontier" shape is needed via hyper-parameters

    **Limitations:**

    - High sensitivity to hyper-parameters (especially ``nu`` and ``gamma``)
    - Poor scalability to very large datasets (cubic training time in worst case)
    - Not natively robust to outliers in the training set (unlike robust covariance)
    - Interpretation of results in high-dimensional kernel space is difficult

    References
    ----------
    .. [Scholkopf2001] Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J., and Williamson, R.C. (2001).
           **Estimating the support of a high-dimensional distribution.**
           *Neural Computation*, 13(7), pp. 1443-1471.
           DOI: `10.1162/089976601750264965 <https://doi.org/10.1162/089976601750264965>`_

    .. [Tax2004] Tax, D.M. and Duin, R.P. (2004).
           **Support vector data description.**
           *Machine learning*, 54(1), pp. 45-66.
           DOI: `10.1023/B:MACH.0000008084.60811.49 <https://doi.org/10.1023/B:MACH.0000008084.60811.49>`_

    See Also
    --------
    :class:`~tuiml.algorithms.anomaly.EllipticEnvelope` : Gaussian-based anomaly detection.
    :class:`~tuiml.algorithms.anomaly.IsolationForest` : Tree-based ensemble anomaly detection.

    Examples
    --------
    Basic usage for novelty detection:

    >>> from tuiml.algorithms.anomaly import OneClassSVMDetector
    >>> import numpy as np
    >>> 
    >>> # Generate normal training data
    >>> X_train = np.random.randn(100, 2)
    >>> 
    >>> # Fit the model on normal data
    >>> clf = OneClassSVMDetector(nu=0.1, kernel="rbf", gamma="auto")
    >>> clf.fit(X_train)
    >>> 
    >>> # Predict on new data (one normal, one anomaly)
    >>> X_test = np.array([[0, 0], [10, 10]])
    >>> predictions = clf.predict(X_test)
    >>> print(predictions)
    [ 1 -1]
    >>> 
    >>> # Get decision function scores
    >>> scores = clf.decision_function(X_test)
    >>> print(scores.round(2))
    [ 0.08 -0.09]
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.1,
        gamma: float | str = "auto",
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
    ):
        """Initialize One-Class SVM.

        Parameters
        ----------
        kernel : str, default="rbf"
            Kernel type.
        nu : float, default=0.1
            Upper bound on training errors.
        gamma : float or "auto", default="auto"
            Kernel coefficient.
        degree : int, default=3
            Polynomial kernel degree.
        coef0 : float, default=0.0
            Independent term in kernel.
        tol : float, default=1e-3
            Tolerance for stopping.
        max_iter : int, default=1000
            Maximum iterations.
        """
        super().__init__()
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

        # Fitted attributes
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.offset_ = None
        self.n_support_ = None
        self.gamma_ = None
        self.n_features_in_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "kernel": {
                "type": "string",
                "default": "rbf",
                "enum": ["linear", "rbf", "poly", "sigmoid"],
                "description": "Kernel type"
            },
            "nu": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Upper bound on fraction of training errors"
            },
            "gamma": {
                "oneOf": [
                    {"type": "number", "minimum": 0.0},
                    {"type": "string", "enum": ["auto"]}
                ],
                "default": "auto",
                "description": "Kernel coefficient"
            },
            "degree": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Polynomial kernel degree"
            },
            "coef0": {
                "type": "number",
                "default": 0.0,
                "description": "Independent term in kernel"
            },
            "tol": {
                "type": "number",
                "default": 1e-3,
                "minimum": 0.0,
                "description": "Tolerance for stopping criterion"
            },
            "max_iter": {
                "type": "integer",
                "default": 1000,
                "minimum": 1,
                "description": "Maximum optimization iterations"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric",
            "binary_class",
            "unsupervised",
            "anomaly_detection",
            "novelty_detection",
            "kernel_methods"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(n² to n³), Prediction: O(n_sv * d), where n=samples, n_sv=support vectors, d=features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Schölkopf et al., 2001. Estimating support of high-dimensional distribution. Neural Computation.",
            "Tax & Duin, 2004. Support vector data description. Machine Learning."
        ]

    def _kernel_function(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K(X, Y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples_X, n_features)
            First array.
        Y : np.ndarray of shape (n_samples_Y, n_features)
            Second array.

        Returns
        -------
        K : np.ndarray of shape (n_samples_X, n_samples_Y)
            Kernel matrix.
        """
        if self.kernel == "linear":
            return X @ Y.T

        elif self.kernel == "rbf":
            # RBF: exp(-gamma * ||x - y||^2)
            X_norm = np.sum(X ** 2, axis=1)[:, np.newaxis]
            Y_norm = np.sum(Y ** 2, axis=1)[np.newaxis, :]
            distances_sq = X_norm + Y_norm - 2 * (X @ Y.T)
            return np.exp(-self.gamma_ * distances_sq)

        elif self.kernel == "poly":
            # Polynomial: (gamma * x^T y + coef0)^degree
            return (self.gamma_ * (X @ Y.T) + self.coef0) ** self.degree

        elif self.kernel == "sigmoid":
            # Sigmoid: tanh(gamma * x^T y + coef0)
            return np.tanh(self.gamma_ * (X @ Y.T) + self.coef0)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: np.ndarray, _y: Optional[np.ndarray] = None) -> "OneClassSVMDetector":
        """Fit the One-Class SVM model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        _y : np.ndarray or None, default=None
            Ignored. Present for API consistency.

        Returns
        -------
        self : OneClassSVMDetector
            Fitted estimator.
        """
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Set gamma
        if self.gamma == "scale":
            X_var = X.var()
            self.gamma_ = 1.0 / (n_features * X_var) if X_var != 0 else 1.0 / n_features
        elif self.gamma == "auto":
            self.gamma_ = 1.0 / n_features
        else:
            self.gamma_ = self.gamma

        # Compute kernel matrix
        K = self._kernel_function(X, X)

        # Simplified SMO-like optimization
        # Initialize dual coefficients (alpha)
        alpha = np.random.rand(n_samples) * self.nu / n_samples
        alpha = alpha / np.sum(alpha) * self.nu  # Normalize

        # Simple coordinate descent optimization
        for iteration in range(self.max_iter):
            alpha_old = alpha.copy()

            for i in range(n_samples):
                # Simplified update rule
                error = np.sum(alpha * K[:, i]) - 1

                # Update alpha[i]
                delta = -error / (K[i, i] + 1e-10)
                alpha[i] = np.clip(alpha[i] + delta, 0, 1.0 / n_samples)

            # Normalize to satisfy constraint
            alpha = alpha / np.sum(alpha) * self.nu

            # Check convergence
            if np.max(np.abs(alpha - alpha_old)) < self.tol:
                break

        # Identify support vectors (alpha > threshold)
        sv_threshold = 1e-5
        sv_indices = alpha > sv_threshold

        self.support_vectors_ = X[sv_indices]
        self.dual_coef_ = alpha[sv_indices]
        self.n_support_ = np.sum(sv_indices)

        # Compute offset (rho)
        # Use margin support vectors (0 < alpha < C)
        margin_sv = (alpha > sv_threshold) & (alpha < 1.0 / n_samples - sv_threshold)
        if np.sum(margin_sv) > 0:
            K_sv = self._kernel_function(X[margin_sv], self.support_vectors_)
            decision_values = K_sv @ self.dual_coef_
            self.offset_ = np.median(decision_values)
        else:
            # Fallback: use all support vectors
            K_sv = self._kernel_function(self.support_vectors_, self.support_vectors_)
            decision_values = K_sv @ self.dual_coef_
            self.offset_ = np.median(decision_values)

        self._is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values.

        The decision function is the signed distance to the separating hyperplane.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Decision function values. Negative values indicate anomalies.
        """
        self._check_is_fitted()
        X = np.atleast_2d(X)

        K = self._kernel_function(X, self.support_vectors_)
        scores = K @ self.dual_coef_ - self.offset_

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if samples are inliers or outliers.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            -1 for anomalies (outliers), 1 for normal (inliers).
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Alias for decision_function for compatibility.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Decision function values.
        """
        return self.decision_function(X)
