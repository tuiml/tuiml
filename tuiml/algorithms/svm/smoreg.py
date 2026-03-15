"""Support Vector Regressor (SVR) implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Callable

from tuiml.base.algorithms import Regressor, regressor
from tuiml._cpp_ext import svm as _cpp_svm

@regressor(tags=["svm", "regression", "kernel"], version="1.0.0")
class SVR(Regressor):
    """Support Vector Regressor using **Sequential Minimal Optimization** (SMOreg).

    Implements **epsilon-insensitive** Support Vector Regression (SVR) using the
    SMO algorithm. The model finds a function :math:`f(x) = w^T \\phi(x) + b`
    that has at most :math:`\\epsilon` deviation from the actual target values,
    while remaining as **flat** as possible.

    Overview
    --------
    The SMO-based SVR training proceeds as follows:

    1. Map input features into a high-dimensional space via a **kernel function**
    2. Define an :math:`\\epsilon`-insensitive tube around the training targets
    3. Select pairs of dual variables (:math:`\\alpha_i, \\alpha_i^*`) that violate KKT conditions
    4. Solve the two-variable sub-problem **analytically** to update the multipliers
    5. Update the bias term and check for convergence
    6. Repeat until all KKT conditions are satisfied or ``max_iter`` is reached

    Theory
    ------
    The SVR solves the following **primal** optimization problem:

    .. math::
        \\min_{w, b, \\xi, \\xi^*} \\frac{1}{2}\\|w\\|^2 + C \\sum_{i=1}^{n} (\\xi_i + \\xi_i^*)

    Subject to:

    .. math::
        y_i - w^T \\phi(x_i) - b \\leq \\epsilon + \\xi_i

    .. math::
        w^T \\phi(x_i) + b - y_i \\leq \\epsilon + \\xi_i^*

    .. math::
        \\xi_i, \\xi_i^* \\geq 0

    The corresponding **dual** formulation is:

    .. math::
        \\max_{\\alpha, \\alpha^*} -\\frac{1}{2} \\sum_{i,j} (\\alpha_i - \\alpha_i^*)(\\alpha_j - \\alpha_j^*) K(x_i, x_j) - \\epsilon \\sum_{i} (\\alpha_i + \\alpha_i^*) + \\sum_{i} y_i (\\alpha_i - \\alpha_i^*)

    Subject to:

    .. math::
        0 \\leq \\alpha_i, \\alpha_i^* \\leq C

    where:

    - :math:`C` --- Regularization parameter trading off flatness against training error
    - :math:`\\epsilon` --- Width of the insensitive tube (errors within the tube are ignored)
    - :math:`K(x_i, x_j)` --- Kernel function implementing the **kernel trick**
    - :math:`\\alpha_i, \\alpha_i^*` --- Dual variables (Lagrange multipliers)
    - :math:`\\xi_i, \\xi_i^*` --- Slack variables for points outside the :math:`\\epsilon`-tube

    The prediction function is:

    .. math::
        f(x) = \\sum_{i \\in SV} (\\alpha_i - \\alpha_i^*) K(x_i, x) + b

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. Larger values allow fewer points to lie
        outside the :math:`\\epsilon`-tube but may cause overfitting.

    epsilon : float, default=0.1
        Width of the epsilon-insensitive loss tube. Predictions within
        :math:`\\epsilon` of the true value incur zero loss.

    kernel : {'linear', 'poly', 'rbf'}, default='rbf'
        Kernel type used for the non-linear mapping.

    gamma : Union[str, float], default='scale'
        Kernel coefficient for ``'rbf'`` and ``'poly'`` kernels:

        - ``'scale'`` --- Uses ``1 / (n_features * X.var())``
        - ``'auto'`` --- Uses ``1 / n_features``
        - ``float`` --- User-defined positive coefficient

    degree : int, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=0.0
        Independent term in the polynomial kernel function.

    tol : float, default=1e-3
        Tolerance for the stopping criterion in the optimization loop.

    max_iter : int, default=10000
        Maximum number of optimization iterations.

    Attributes
    ----------
    support_vectors_ : np.ndarray
        Training samples that lie on or outside the :math:`\\epsilon`-tube
        boundary.

    dual_coef_ : np.ndarray
        Coefficients :math:`(\\alpha_i - \\alpha_i^*)` of the support vectors
        in the prediction function.

    intercept_ : float
        Intercept (bias) term :math:`b`.

    n_support_ : int
        Total number of support vectors.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n^2 \\cdot p)` to :math:`O(n^3)` depending on the kernel cache, where :math:`n` = samples, :math:`p` = features
    - Prediction: :math:`O(n_{sv} \\cdot p)` per sample, where :math:`n_{sv}` is the number of support vectors

    **When to use SVR:**

    - Regression tasks where a non-linear relationship is expected
    - When robustness to outliers is important (errors within :math:`\\epsilon` are ignored)
    - Moderate-sized datasets (SVR scales quadratically with the number of samples)
    - When a sparse solution (few support vectors) is preferred

    References
    ----------
    .. [Smola2004] Smola, A.J. and Schoelkopf, B. (2004).
           **A Tutorial on Support Vector Regression.**
           *Statistics and Computing*, 14(3), pp. 199-222.
           DOI: `10.1023/B:STCO.0000035301.49549.88 <https://doi.org/10.1023/B:STCO.0000035301.49549.88>`_

    .. [Platt1999] Platt, J.C. (1999).
           **Fast Training of Support Vector Machines Using Sequential Minimal Optimization.**
           *Advances in Kernel Methods --- Support Vector Learning*, MIT Press, pp. 185-208.

    .. [Vapnik1995] Vapnik, V.N. (1995).
           **The Nature of Statistical Learning Theory.**
           *Springer-Verlag, New York*.
           DOI: `10.1007/978-1-4757-2440-0 <https://doi.org/10.1007/978-1-4757-2440-0>`_

    See Also
    --------
    :class:`~tuiml.algorithms.svm.SVC` : Support Vector Classifier using SMO optimization.
    :class:`~tuiml.algorithms.svm.kernels.RBFKernel` : Radial Basis Function kernel.
    :class:`~tuiml.algorithms.svm.kernels.PolynomialKernel` : Polynomial kernel.

    Examples
    --------
    Basic regression with an RBF kernel:

    >>> from tuiml.algorithms.svm import SVR
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.array([[1], [2], [3], [4], [5]], dtype=float)
    >>> y_train = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
    >>>
    >>> # Fit the regressor
    >>> reg = SVR(C=1.0, epsilon=0.1, kernel='rbf')
    >>> reg.fit(X_train, y_train)
    SVR(...)
    >>> predictions = reg.predict(X_train)
    """

    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        kernel: str = "rbf",
        gamma: Any = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 10000,
    ):
        """Initialize SVR.

        Parameters
        ----------
        C : float, default=1.0
            Regularization parameter.
        epsilon : float, default=0.1
            Epsilon-insensitive tube width.
        kernel : str, default='rbf'
            Kernel type.
        gamma : Union[str, float], default='scale'
            Kernel coefficient.
        degree : int, default=3
            Polynomial kernel degree.
        coef0 : float, default=0.0
            Independent term in kernel function.
        tol : float, default=1e-3
            Tolerance for convergence.
        max_iter : int, default=10000
            Maximum iterations.
        """
        super().__init__()
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

        # Fitted attributes
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.n_support_ = None
        self._X_train = None
        self._gamma_value = None
        self._kernel_obj = None
        self._cpp_model = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "C": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "description": "Regularization parameter"
            },
            "epsilon": {
                "type": "number",
                "default": 0.1,
                "minimum": 0,
                "description": "Epsilon in epsilon-insensitive loss"
            },
            "kernel": {
                "type": "string",
                "default": "rbf",
                "enum": ["linear", "poly", "rbf"],
                "description": "Kernel type"
            },
            "gamma": {
                "type": ["number", "string"],
                "default": "scale",
                "description": "Kernel coefficient ('scale', 'auto', or float)"
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
                "default": 0.001,
                "minimum": 0,
                "description": "Tolerance for stopping"
            },
            "max_iter": {
                "type": "integer",
                "default": 10000,
                "minimum": 1,
                "description": "Maximum iterations"
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
        return "O(n^2 * m) to O(n^3) depending on kernel cache"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Smola, A. J., & Scholkopf, B. (2004). A tutorial on support "
            "vector regression. Statistics and Computing, 14(3), 199-222.",
            "Platt, J. (1999). Fast training of support vector machines "
            "using sequential minimal optimization."
        ]

    def _setup_kernel(self, X: np.ndarray) -> None:
        """Set up the kernel function based on the ``kernel`` parameter.

        Resolves the kernel from a string name or uses a provided Kernel
        object, and computes the effective ``gamma`` value.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data used to compute ``gamma`` when set to
            ``'scale'`` or ``'auto'``.

        Returns
        -------
        None
            Sets ``self._kernel_obj`` and ``self._gamma_value`` in-place.
        """
        from tuiml.algorithms.svm.kernels import (
            Kernel, LinearKernel, PolynomialKernel, RBFKernel, SigmoidKernel
        )

        n_features = X.shape[1]

        # Compute gamma
        if self.gamma == 'scale':
            var = np.var(X)
            self._gamma_value = 1.0 / (n_features * var) if var > 0 else 1.0 / n_features
        elif self.gamma == 'auto':
            self._gamma_value = 1.0 / n_features
        else:
            self._gamma_value = float(self.gamma)

        # Handle kernel object or string
        if isinstance(self.kernel, Kernel):
            self._kernel_obj = self.kernel
            self._kernel_obj.build(X)
            self._use_precomputed = True
        elif isinstance(self.kernel, str):
            if self.kernel == 'linear':
                self._kernel_obj = LinearKernel()
            elif self.kernel == 'poly':
                self._kernel_obj = PolynomialKernel(
                    degree=self.degree,
                    gamma=self._gamma_value,
                    coef0=self.coef0
                )
            elif self.kernel == 'rbf':
                self._kernel_obj = RBFKernel(gamma=self._gamma_value)
            elif self.kernel == 'sigmoid':
                self._kernel_obj = SigmoidKernel(
                    gamma=self._gamma_value,
                    coef0=self.coef0
                )
            else:
                raise ValueError(f"Unknown kernel: {self.kernel}")
            self._kernel_obj.build(X)

            kt = getattr(self._kernel_obj, '_libsvm_kernel_type', None)
            self._use_precomputed = (kt is None)
        else:
            raise ValueError(f"kernel must be string or Kernel object, got {type(self.kernel)}")

    def _get_kernel_type_int(self) -> int:
        """Return integer kernel type for the C++ backend.

        Returns
        -------
        kt : int
            Kernel type integer (0=linear, 1=poly, 2=rbf, 3=sigmoid).
        """
        return getattr(self._kernel_obj, '_libsvm_kernel_type', -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVR":
        """Fit the SVR model using SMO algorithm.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SVR
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self._X_train = X
        self._setup_kernel(X)

        y_c = np.ascontiguousarray(y, dtype=np.float64)

        if self._use_precomputed:
            K = self._kernel_obj.compute_matrix_cross(X, X)
            K = np.ascontiguousarray(K, dtype=np.float64)
            self._cpp_model = _cpp_svm.svr_train_precomputed(
                K, y_c, self.C, self.epsilon, self.tol, self.max_iter
            )
        else:
            X_c = np.ascontiguousarray(X, dtype=np.float64)
            kt = self._get_kernel_type_int()
            self._cpp_model = _cpp_svm.svr_train(
                X_c, y_c, kt, self.C, self.epsilon,
                self._gamma_value, self.degree, self.coef0,
                self.tol, self.max_iter
            )

        # Extract fitted attributes
        model = self._cpp_model
        sv_indices = np.array(model.sv_indices, dtype=np.int32)
        self.support_vectors_ = X[sv_indices]
        self.dual_coef_ = np.array(model.dual_coef)
        self.n_support_ = len(sv_indices)
        self.intercept_ = -model.rho

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self._use_precomputed:
            K_test = self._kernel_obj.compute_matrix_cross(X, self._X_train)
            K_test = np.ascontiguousarray(K_test, dtype=np.float64)
            return np.asarray(_cpp_svm.svr_predict_precomputed(
                self._cpp_model, K_test
            ))
        else:
            X_c = np.ascontiguousarray(X, dtype=np.float64)
            X_train_c = np.ascontiguousarray(self._X_train, dtype=np.float64)
            kt = self._get_kernel_type_int()
            return np.asarray(_cpp_svm.svr_predict(
                self._cpp_model, X_train_c, X_c,
                kt, self._gamma_value, self.degree, self.coef0
            ))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the :math:`R^2` (coefficient of determination) score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        r2 : float
            :math:`R^2` score, where 1.0 is a perfect fit.
        """
        self._check_is_fitted()
        y_pred = self.predict(X)
        y = np.asarray(y)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"SVR(C={self.C}, epsilon={self.epsilon}, "
                    f"kernel='{self.kernel}', n_support={self.n_support_})")
        return f"SVR(C={self.C}, epsilon={self.epsilon}, kernel='{self.kernel}')"
