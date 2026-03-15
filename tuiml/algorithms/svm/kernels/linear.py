"""Linear Kernel implementation."""

import numpy as np
from typing import Dict, Any

from tuiml.base.kernels import Kernel, kernel

@kernel(tags=["linear", "basic"], version="1.0.0")
class LinearKernel(Kernel):
    """Linear Kernel computing the standard **dot product** between two vectors.

    The Linear Kernel is the simplest kernel function, equivalent to operating
    in the **original input space** without any non-linear mapping. It is
    well suited for **linearly separable** data and high-dimensional sparse
    feature spaces.

    Overview
    --------
    The linear kernel evaluation proceeds as follows:

    1. Accept two input vectors :math:`x` and :math:`y`
    2. Compute their **inner product** (dot product)
    3. Return the scalar result --- no feature-space transformation is applied

    Theory
    ------
    The linear kernel function is defined as:

    .. math::
        K(x, y) = x^T y = \\sum_{i=1}^{p} x_i y_i

    where :math:`p` is the number of features. The implicit feature map is
    the **identity**: :math:`\\phi(x) = x`, so the kernel corresponds to a
    linear decision boundary in the original input space.

    Parameters
    ----------
    (No parameters.)

    Attributes
    ----------
    X\\_ : np.ndarray
        Training data stored after ``build()``.

    n_samples\\_ : int
        Number of training samples.

    n_features\\_ : int
        Number of features observed during ``build()``.

    Notes
    -----
    **Complexity:**

    - Evaluation: :math:`O(p)` per pair, where :math:`p` = number of features
    - Matrix computation: :math:`O(n^2 p)` for :math:`n` samples

    **When to use LinearKernel:**

    - Text classification with high-dimensional TF-IDF or bag-of-words features
    - Linearly separable data or when a simple baseline is needed
    - Very high-dimensional data where non-linear kernels are too expensive
    - When interpretability of the weight vector is important

    References
    ----------
    .. [Vapnik1995] Vapnik, V.N. (1995).
           **The Nature of Statistical Learning Theory.**
           *Springer-Verlag, New York*.
           DOI: `10.1007/978-1-4757-2440-0 <https://doi.org/10.1007/978-1-4757-2440-0>`_

    See Also
    --------
    :class:`~tuiml.algorithms.svm.kernels.RBFKernel` : Gaussian RBF kernel for non-linear boundaries.
    :class:`~tuiml.algorithms.svm.kernels.PolynomialKernel` : Polynomial kernel including linear as degree 1.

    Examples
    --------
    Basic usage for computing the kernel matrix:

    >>> from tuiml.algorithms.svm.kernels import LinearKernel
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> kernel = LinearKernel()
    >>> kernel.build(X_train)
    >>> K = kernel.compute_matrix()
    >>> print(K.shape)
    (3, 3)
    """

    _libsvm_kernel_type = 0  # LINEAR

    def __init__(self):
        """Initialize linear kernel."""
        super().__init__()

    def _libsvm_params(self) -> str:
        """Return libsvm parameter string."""
        return ""

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {}

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate linear kernel: K(x, y) = x^T * y.

        Parameters
        ----------
        x1 : np.ndarray
            First vector.
        x2 : np.ndarray
            Second vector.

        Returns
        -------
        val : float
            Dot product of x1 and x2.
        """
        return float(np.dot(x1, x2))

    def compute_matrix(self) -> np.ndarray:
        """Compute the linear kernel matrix :math:`K = X X^T` efficiently.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            The kernel (Gram) matrix.
        """
        self._check_is_built()
        return self.X_ @ self.X_.T

    def compute_matrix_cross(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the linear kernel matrix between two sample sets.

        Parameters
        ----------
        X1 : np.ndarray of shape (n1, n_features)
            First set of samples.
        X2 : np.ndarray of shape (n2, n_features)
            Second set of samples.

        Returns
        -------
        K : np.ndarray of shape (n1, n2)
            Kernel matrix :math:`K = X_1 X_2^T`.
        """
        from tuiml.base.kernels import _xnp
        xnp = _xnp()
        X1 = xnp.asarray(X1, dtype=float)
        X2 = xnp.asarray(X2, dtype=float)
        K = X1 @ X2.T
        return np.asarray(K)

    def __repr__(self) -> str:
        """String representation."""
        return "LinearKernel()"
