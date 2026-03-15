"""Support Vector Classifier (SVC) implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from tuiml.base.algorithms import Classifier, classifier
from tuiml._cpp_ext import svm as _cpp_svm

@classifier(tags=["functions", "svm", "kernel"], version="1.0.0")
class SVC(Classifier):
    """Support Vector Classifier using **Sequential Minimal Optimization** (SMO).

    SVC is an efficient algorithm for training **Support Vector Machines** that
    breaks the large quadratic programming (QP) problem into a series of the
    smallest possible QP sub-problems, each solved **analytically** in closed form.

    Overview
    --------
    The SMO-based SVC training proceeds as follows:

    1. Map input features into a (possibly high-dimensional) space via a **kernel function**
    2. Select two Lagrange multipliers that violate the **KKT conditions**
    3. Solve the two-variable QP sub-problem analytically to update the multipliers
    4. Update the bias (threshold) term and error cache
    5. Repeat steps 2-4 until all KKT conditions are satisfied or ``max_iter`` is reached
    6. For **multiclass** problems, use a one-vs-one decomposition with majority voting

    Theory
    ------
    The SVC solves the following **primal** optimization problem:

    .. math::
        \\min_{w, b, \\xi} \\frac{1}{2}\\|w\\|^2 + C \\sum_{i=1}^{n} \\xi_i

    Subject to:

    .. math::
        y_i (w^T \\phi(x_i) + b) \\geq 1 - \\xi_i, \\quad \\xi_i \\geq 0

    The corresponding **dual** formulation is:

    .. math::
        \\max_{\\alpha} \\sum_{i=1}^{n} \\alpha_i - \\frac{1}{2} \\sum_{i,j} \\alpha_i \\alpha_j y_i y_j K(x_i, x_j)

    Subject to:

    .. math::
        0 \\leq \\alpha_i \\leq C, \\quad \\sum_{i=1}^{n} \\alpha_i y_i = 0

    where:

    - :math:`C` --- Regularization parameter controlling the trade-off between margin width and training error
    - :math:`K(x_i, x_j) = \\phi(x_i)^T \\phi(x_j)` --- Kernel function (the **kernel trick**)
    - :math:`\\alpha_i` --- Lagrange multipliers (dual coefficients)
    - :math:`\\xi_i` --- Slack variables allowing for soft-margin classification

    The decision function for a new sample :math:`x` is:

    .. math::
        f(x) = \\sum_{i \\in SV} \\alpha_i y_i K(x_i, x) + b

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. Larger values penalize misclassification
        more heavily, yielding a narrower margin with fewer training errors.

    kernel : str or Kernel, default='rbf'
        Kernel function. Can be ``'linear'``, ``'poly'``, ``'rbf'``,
        ``'sigmoid'``, or a Kernel object from
        ``tuiml.algorithms.svm.kernels``.

    gamma : Union[str, float], default='scale'
        Kernel coefficient for ``'rbf'``, ``'poly'``, and ``'sigmoid'`` kernels:

        - ``'scale'`` --- Uses ``1 / (n_features * X.var())``
        - ``'auto'`` --- Uses ``1 / n_features``
        - ``float`` --- User-defined positive coefficient

    degree : int, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=0.0
        Independent term in the ``'poly'`` and ``'sigmoid'`` kernel functions.

    tol : float, default=1e-3
        Tolerance for the stopping criterion in the SMO optimization loop.

    max_iter : int, default=1000
        Maximum number of passes over the training set during optimization.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels discovered during ``fit()``.

    support_ : np.ndarray
        Indices of support vectors in the training data.
    support_vectors_ : np.ndarray
        Training samples that lie on or within the margin boundary.

    dual_coef_ : np.ndarray
        Coefficients (:math:`\\alpha_i y_i`) of the support vectors in the
        decision function.

    intercept_ : np.ndarray
        Intercept (bias) term :math:`b` of the decision function.

    n_support_ : np.ndarray
        Number of support vectors for each class.

    kernel_ : Kernel
        The actual kernel object used after ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n^2 \\cdot p)` to :math:`O(n^3)` depending on kernel cache efficiency, where :math:`n` = samples, :math:`p` = features
    - Prediction: :math:`O(n_{sv} \\cdot p)` per sample, where :math:`n_{sv}` is the number of support vectors

    **When to use SVC:**

    - Binary or multiclass classification with moderate-sized datasets
    - When a non-linear decision boundary is needed (via kernel trick)
    - High-dimensional data where the number of features exceeds the number of samples
    - When a sparse solution (few support vectors) is desirable

    References
    ----------
    .. [Platt1998] Platt, J.C. (1998).
           **Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines.**
           *Microsoft Research Technical Report MSR-TR-98-14*.

    .. [Vapnik1995] Vapnik, V.N. (1995).
           **The Nature of Statistical Learning Theory.**
           *Springer-Verlag, New York*.
           DOI: `10.1007/978-1-4757-2440-0 <https://doi.org/10.1007/978-1-4757-2440-0>`_

    .. [Keerthi2001] Keerthi, S.S., Shevade, S.K., Bhattacharyya, C. and Murthy, K.R.K. (2001).
           **Improvements to Platt's SMO Algorithm for SVM Classifier Design.**
           *Neural Computation*, 13(3), pp. 637-649.
           DOI: `10.1162/089976601300014493 <https://doi.org/10.1162/089976601300014493>`_

    See Also
    --------
    :class:`~tuiml.algorithms.svm.SVR` : Support Vector Regression using SMO optimization.
    :class:`~tuiml.algorithms.svm.kernels.RBFKernel` : Radial Basis Function kernel.
    :class:`~tuiml.algorithms.svm.kernels.LinearKernel` : Linear kernel for linearly separable data.

    Examples
    --------
    Basic binary classification with an RBF kernel:

    >>> from tuiml.algorithms.svm import SVC
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y_train = np.array([0, 0, 1, 1])
    >>>
    >>> # Fit the classifier
    >>> clf = SVC(C=1.0, kernel='rbf', gamma=0.5)
    >>> clf.fit(X_train, y_train)
    SVC(...)
    >>> predictions = clf.predict(X_train)

    Using a kernel object directly:

    >>> from tuiml.algorithms.svm import SVC
    >>> from tuiml.algorithms.svm.kernels import RBFKernel
    >>> kernel = RBFKernel(gamma=0.5)
    >>> clf = SVC(C=1.0, kernel=kernel)
    >>> clf.fit(X_train, y_train)
    SVC(...)
    """

    def __init__(self, C: float = 1.0,
                 kernel: Union[str, "Kernel"] = 'rbf',
                 gamma: Any = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 max_iter: int = 1000):
        """Initialize SVC classifier.

        Parameters
        ----------
        C : float, default=1.0
            Regularization parameter.
        kernel : str or Kernel, default='rbf'
            Kernel function (string name or Kernel object).
        gamma : Union[str, float], default='scale'
            Kernel coefficient for rbf/poly/sigmoid kernels.
        degree : int, default=3
            Polynomial kernel degree.
        coef0 : float, default=0.0
            Independent term in poly/sigmoid kernel.
        tol : float, default=1e-3
            Tolerance for stopping criterion.
        max_iter : int, default=1000
            Maximum iterations.
        """
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.classes_ = None
        self.support_ = None
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.n_support_ = None
        self._cpp_model = None
        self._gamma_actual = None
        self._kernel_obj = None  # Actual kernel object

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
            "kernel": {
                "type": ["string", "object"],
                "default": "rbf",
                "enum": ["linear", "poly", "rbf", "sigmoid"],
                "description": "Kernel function (string or Kernel object)"
            },
            "gamma": {
                "type": ["string", "number"],
                "default": "scale",
                "description": "Kernel coefficient"
            },
            "degree": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Degree for polynomial kernel"
            },
            "coef0": {
                "type": "number",
                "default": 0.0,
                "description": "Independent term in kernel function"
            },
            "tol": {
                "type": "number",
                "default": 1e-3,
                "minimum": 0,
                "description": "Tolerance for stopping criterion"
            },
            "max_iter": {
                "type": "integer",
                "default": 1000,
                "minimum": 1,
                "description": "Maximum number of iterations"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return [
            "numeric",
            "binary_class",
            "multiclass"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n^2 * m) to O(n^3) training, O(n_sv * m) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Platt, J.C. (1998). Sequential Minimal Optimization: A Fast "
            "Algorithm for Training Support Vector Machines. Microsoft Research "
            "Technical Report MSR-TR-98-14."
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
            Sets ``self._kernel_obj`` and ``self._gamma_actual`` in-place.
        """
        from tuiml.algorithms.svm.kernels import (
            Kernel, LinearKernel, PolynomialKernel, RBFKernel, SigmoidKernel
        )

        n_features = X.shape[1]

        # Compute gamma for string kernels
        if self.gamma == 'scale':
            self._gamma_actual = 1.0 / (n_features * X.var()) if X.var() > 0 else 1.0
        elif self.gamma == 'auto':
            self._gamma_actual = 1.0 / n_features
        else:
            self._gamma_actual = self.gamma

        # Handle kernel object or string
        if isinstance(self.kernel, Kernel):
            # Use provided kernel object
            self._kernel_obj = self.kernel
            self._kernel_obj.build(X)
            self._use_precomputed = True
        elif isinstance(self.kernel, str):
            # Create kernel from string
            if self.kernel == 'linear':
                self._kernel_obj = LinearKernel()
            elif self.kernel == 'poly':
                self._kernel_obj = PolynomialKernel(
                    degree=self.degree,
                    gamma=self._gamma_actual,
                    coef0=self.coef0
                )
            elif self.kernel == 'rbf':
                self._kernel_obj = RBFKernel(gamma=self._gamma_actual)
            elif self.kernel == 'sigmoid':
                self._kernel_obj = SigmoidKernel(
                    gamma=self._gamma_actual,
                    coef0=self.coef0
                )
            else:
                raise ValueError(f"Unknown kernel: {self.kernel}")

            self._kernel_obj.build(X)

            # String kernels with a native type use the C++ solver directly
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        """Fit the SVC classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : SVC
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)

        # Set up kernel
        self._setup_kernel(X)

        # Store training data
        self._X_train = X

        # Map labels to integer codes for C++
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([label_map[yi] for yi in y], dtype=np.intc)

        if self._use_precomputed:
            # Custom kernel (PUK, String, etc.) — compute kernel matrix in Python
            K = self._kernel_obj.compute_matrix_cross(X, X)
            K = np.ascontiguousarray(K, dtype=np.float64)
            self._cpp_model = _cpp_svm.svc_train_precomputed(
                K, y_int, self.C, self.tol, self.max_iter
            )
        else:
            # Native kernel — use C++ solver directly
            X_c = np.ascontiguousarray(X, dtype=np.float64)
            kt = self._get_kernel_type_int()
            self._cpp_model = _cpp_svm.svc_train(
                X_c, y_int, kt, self.C,
                self._gamma_actual, self.degree, self.coef0,
                self.tol, self.max_iter
            )

        # Extract fitted attributes from C++ model
        model = self._cpp_model
        self.support_ = np.array(model.all_sv_indices, dtype=np.int32)
        self.support_vectors_ = X[self.support_]

        n_sv = len(self.support_)
        n_classes = len(self.classes_)

        if n_sv > 0 and n_classes > 1:
            self.dual_coef_ = np.array(model.all_dual_coef).reshape(n_classes - 1, n_sv)
        else:
            self.dual_coef_ = np.array(model.all_dual_coef).reshape(1, -1)

        self.intercept_ = np.array(model.intercept)
        self.n_support_ = np.array(model.n_support, dtype=int)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self._use_precomputed:
            K_test = self._kernel_obj.compute_matrix_cross(X, self._X_train)
            K_test = np.ascontiguousarray(K_test, dtype=np.float64)
            int_labels = _cpp_svm.svc_predict_precomputed(self._cpp_model, K_test)
        else:
            X_c = np.ascontiguousarray(X, dtype=np.float64)
            X_train_c = np.ascontiguousarray(self._X_train, dtype=np.float64)
            kt = self._get_kernel_type_int()
            int_labels = _cpp_svm.svc_predict(
                self._cpp_model, X_train_c, X_c,
                kt, self._gamma_actual, self.degree, self.coef0
            )

        # Map integer labels back to original class labels
        return self.classes_[np.asarray(int_labels)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Uses Platt scaling on decision function values for binary
        classification, and OvO voting proportions for multiclass.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        dec = self.decision_function(X)
        n_classes = len(self.classes_)

        if n_classes == 2:
            # Platt scaling: sigmoid on decision values
            prob_pos = 1.0 / (1.0 + np.exp(-dec))
            proba = np.column_stack([1 - prob_pos, prob_pos])
        else:
            # OvO voting proportions
            n_samples = X.shape[0]
            n_models = n_classes * (n_classes - 1) // 2
            votes = np.zeros((n_samples, n_classes))
            m = 0
            for a in range(n_classes):
                for b in range(a + 1, n_classes):
                    mask = dec[:, m] > 0
                    votes[mask, a] += 1
                    votes[~mask, b] += 1
                    m += 1
            # Normalize to probabilities
            totals = votes.sum(axis=1, keepdims=True)
            totals = np.maximum(totals, 1)
            proba = votes / totals

        return proba

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the decision function values for input samples.

        For binary classification, returns signed distances to the hyperplane.
        For multiclass, returns one decision value per one-vs-one classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        decision : np.ndarray of shape (n_samples,) or (n_samples, n_classifiers)
            Decision function values. For binary, shape is ``(n_samples,)``.
            For multiclass one-vs-one, shape is ``(n_samples, n_classifiers)``.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self._use_precomputed:
            K_test = self._kernel_obj.compute_matrix_cross(X, self._X_train)
            K_test = np.ascontiguousarray(K_test, dtype=np.float64)
            return np.asarray(_cpp_svm.svc_decision_function_precomputed(
                self._cpp_model, K_test
            ))
        else:
            X_c = np.ascontiguousarray(X, dtype=np.float64)
            X_train_c = np.ascontiguousarray(self._X_train, dtype=np.float64)
            kt = self._get_kernel_type_int()
            return np.asarray(_cpp_svm.svc_decision_function(
                self._cpp_model, X_train_c, X_c,
                kt, self._gamma_actual, self.degree, self.coef0
            ))

    @property
    def kernel_(self):
        """Return the actual kernel object used."""
        return self._kernel_obj

    def __repr__(self) -> str:
        """String representation."""
        kernel_name = (self.kernel if isinstance(self.kernel, str)
                      else self.kernel.__class__.__name__)
        if self._is_fitted:
            if self.support_vectors_ is not None:
                n_sv = len(self.support_vectors_)
            else:
                n_sv = 0
            return f"SVC(kernel={kernel_name}, C={self.C}, n_support_vectors={n_sv})"
        return f"SVC(kernel={kernel_name}, C={self.C})"
