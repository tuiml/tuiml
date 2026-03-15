"""Precomputed Kernel Matrix implementation."""

import numpy as np
from typing import Dict, Any

from tuiml.base.kernels import Kernel, kernel

@kernel(tags=["precomputed", "matrix"], version="1.0.0")
class PrecomputedKernel(Kernel):
    """Precomputed Kernel Matrix for supplying a pre-calculated **Gram matrix**.

    The Precomputed Kernel allows users to provide their own kernel (Gram) matrix
    rather than having it computed internally. This is useful when the kernel is
    **expensive to compute**, when using a **custom kernel** not available in the
    library, or when the kernel matrix comes from an **external source**.

    Overview
    --------
    The usage workflow is:

    1. Compute a kernel matrix :math:`K` externally (e.g., a custom domain-specific kernel)
    2. Pass the matrix to ``PrecomputedKernel`` (either at construction or via ``set_kernel_matrix``)
    3. Call ``build(X)`` to associate training indices with the matrix
    4. Use ``compute(i, j)`` or ``compute_matrix()`` to retrieve kernel values

    Theory
    ------
    A valid kernel matrix :math:`K \\in \\mathbb{R}^{n \\times n}` must be
    **symmetric** and **positive semi-definite** (Mercer's condition):

    .. math::
        K_{ij} = K(x_i, x_j), \\quad K = K^T, \\quad \\forall c: c^T K c \\geq 0

    Any function :math:`K: \\mathcal{X} \\times \\mathcal{X} \\to \\mathbb{R}`
    satisfying Mercer's theorem can be represented as a precomputed matrix.

    Parameters
    ----------
    kernel_matrix : np.ndarray or None, default=None
        Precomputed kernel matrix of shape ``(n_samples, n_samples)``.
        If ``None``, the matrix passed to ``build()`` is treated as the kernel
        matrix itself.

    Attributes
    ----------
    n_samples\\_ : int
        Number of samples in the kernel matrix.

    n_features\\_ : int
        Number of features (or ``n_samples`` when the matrix is used directly).

    Notes
    -----
    **Complexity:**

    - Lookup: :math:`O(1)` per pair (direct matrix indexing)
    - Storage: :math:`O(n^2)` for the full Gram matrix

    **When to use PrecomputedKernel:**

    - When using a domain-specific or custom kernel function
    - When the kernel computation is expensive and should be done once
    - Graph kernels, Wasserstein kernels, or other specialized similarity measures
    - When leveraging external libraries to compute the kernel matrix

    References
    ----------
    .. [Scholkopf2002] Schoelkopf, B. and Smola, A.J. (2002).
           **Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond.**
           *MIT Press*.

    See Also
    --------
    :class:`~tuiml.algorithms.svm.kernels.RBFKernel` : Gaussian RBF kernel computed on-the-fly.
    :class:`~tuiml.algorithms.svm.kernels.LinearKernel` : Linear kernel for dot-product similarity.

    Examples
    --------
    Basic usage with an externally computed kernel matrix:

    >>> from tuiml.algorithms.svm.kernels import PrecomputedKernel
    >>> import numpy as np
    >>>
    >>> # Create a simple kernel matrix (linear kernel)
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> K = X @ X.T
    >>> kernel = PrecomputedKernel(kernel_matrix=K)
    >>> kernel.build(X)
    PrecomputedKernel(...)
    >>> value = kernel.compute(0, 1)
    """

    def __init__(self, kernel_matrix: np.ndarray = None):
        """Initialize precomputed kernel.

        Parameters
        ----------
        kernel_matrix : np.ndarray or None, default=None
            Precomputed kernel (Gram) matrix of shape
            ``(n_samples, n_samples)``.
        """
        super().__init__()
        self._kernel_matrix = kernel_matrix

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "kernel_matrix": {
                "type": "array",
                "description": "Precomputed kernel matrix"
            }
        }

    def build(self, X: np.ndarray) -> "PrecomputedKernel":
        """Build with training data.

        If kernel_matrix was not provided, X is treated as the kernel matrix.

        Parameters
        ----------
        X : np.ndarray
            Training data or kernel matrix.

        Returns
        -------
        self : PrecomputedKernel
            Returns the built instance.
        """
        X = np.asarray(X, dtype=float)

        if self._kernel_matrix is None:
            # Assume X is the kernel matrix
            if X.ndim != 2 or X.shape[0] != X.shape[1]:
                raise ValueError(
                    "If kernel_matrix not provided, X must be a square kernel matrix"
                )
            self._kernel_matrix = X
            self.n_samples_ = X.shape[0]
            self.n_features_ = X.shape[0]  # Not really applicable
        else:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.n_samples_ = X.shape[0]
            self.n_features_ = X.shape[1]

            if self._kernel_matrix.shape[0] != self.n_samples_:
                raise ValueError(
                    f"Kernel matrix size ({self._kernel_matrix.shape[0]}) "
                    f"doesn't match data size ({self.n_samples_})"
                )

        self.X_ = X
        self._is_built = True
        return self

    def set_kernel_matrix(self, K: np.ndarray) -> "PrecomputedKernel":
        """Set or update the kernel matrix.

        Parameters
        ----------
        K : np.ndarray of shape (n_samples, n_samples)
            Kernel matrix.

        Returns
        -------
        self : PrecomputedKernel
            Returns the updated instance.
        """
        K = np.asarray(K, dtype=float)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("Kernel matrix must be square")
        self._kernel_matrix = K
        return self

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate is not supported for precomputed kernels.

        Use ``compute(i, j)`` with integer indices instead.

        Parameters
        ----------
        x1 : np.ndarray
            First vector (unused).
        x2 : np.ndarray
            Second vector (unused).

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Always raised; use ``compute(i, j)`` instead.
        """
        raise NotImplementedError(
            "PrecomputedKernel doesn't support evaluate(). "
            "Use compute(i, j) with indices instead."
        )

    def compute(self, i: int, j: int) -> float:
        """Get precomputed kernel value.

        Parameters
        ----------
        i : int
            Index of first instance.
        j : int
            Index of second instance.

        Returns
        -------
        val : float
            K[i, j] from precomputed matrix.
        """
        self._check_is_built()
        return float(self._kernel_matrix[i, j])

    def compute_matrix(self) -> np.ndarray:
        """Return a copy of the precomputed kernel matrix.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            A copy of the stored Gram matrix.
        """
        self._check_is_built()
        return self._kernel_matrix.copy()

    def compute_matrix_cross(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Retrieve sub-matrix from the precomputed kernel matrix by index.

        Parameters
        ----------
        X1 : np.ndarray of shape (n1,) or (n1, n_features)
            If 1-D integer array, treated as row indices into the stored
            kernel matrix. Otherwise falls back to the base class.
        X2 : np.ndarray of shape (n2,) or (n2, n_features)
            If 1-D integer array, treated as column indices into the
            stored kernel matrix. Otherwise falls back to the base class.

        Returns
        -------
        K : np.ndarray of shape (n1, n2)
            Sub-matrix of the precomputed kernel.
        """
        self._check_is_built()
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        if X1.ndim == 1 and np.issubdtype(X1.dtype, np.integer):
            idx1 = X1
        else:
            idx1 = np.arange(X1.shape[0])
        if X2.ndim == 1 and np.issubdtype(X2.dtype, np.integer):
            idx2 = X2
        else:
            idx2 = np.arange(X2.shape[0])
        return self._kernel_matrix[np.ix_(idx1, idx2)].copy()

    def __repr__(self) -> str:
        """String representation."""
        if self._is_built:
            return f"PrecomputedKernel(n_samples={self.n_samples_})"
        return "PrecomputedKernel(not built)"

# Alias for WEKA compatibility
PrecomputedKernelMatrixKernel = PrecomputedKernel
