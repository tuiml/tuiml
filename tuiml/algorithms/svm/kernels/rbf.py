"""RBF (Radial Basis Function) Kernel implementation."""

import numpy as np
from typing import Dict, Any

from tuiml.base.kernels import CachedKernel, kernel

@kernel(tags=["rbf", "gaussian", "nonlinear"], version="1.0.0")
class RBFKernel(CachedKernel):
    """RBF (Radial Basis Function) Kernel, also known as the **Gaussian kernel**.

    The RBF kernel maps data into an **infinite-dimensional** feature space,
    enabling SVMs to learn arbitrarily complex non-linear decision boundaries.
    It is the most widely used kernel for general-purpose classification and
    regression tasks.

    Overview
    --------
    The kernel evaluation proceeds as follows:

    1. Compute the **squared Euclidean distance** :math:`\\|x - y\\|^2` between two vectors
    2. Scale by the negative kernel coefficient :math:`-\\gamma`
    3. Apply the **exponential function** to produce a similarity in :math:`(0, 1]`

    During ``build()``, squared norms are **precomputed** for efficient
    matrix-level evaluation.

    Theory
    ------
    The RBF kernel function is defined as:

    .. math::
        K(x, y) = \\exp\\bigl(-\\gamma \\|x - y\\|^2\\bigr)

    Equivalently, with bandwidth parameter :math:`\\sigma`:

    .. math::
        K(x, y) = \\exp\\!\\left(-\\frac{\\|x - y\\|^2}{2\\sigma^2}\\right)

    where :math:`\\gamma = 1 / (2\\sigma^2)`.

    **Properties:**

    - :math:`K(x, x) = 1` for all :math:`x` (unit self-similarity)
    - :math:`K(x, y) \\to 0` as :math:`\\|x - y\\| \\to \\infty`
    - The kernel is **positive semi-definite** for all :math:`\\gamma > 0`

    Parameters
    ----------
    gamma : Union[str, float], default=0.01
        Kernel coefficient controlling the width of the Gaussian:

        - ``'scale'`` --- Uses ``1 / (n_features * X.var())``
        - ``'auto'`` --- Uses ``1 / n_features``
        - ``float`` --- User-defined positive coefficient

    cache_size : int, default=250007
        Maximum number of kernel evaluations to cache for repeated lookups.

    Attributes
    ----------
    gamma\\_ : float
        Actual gamma value used (computed during ``build()`` when
        ``'scale'`` or ``'auto'`` is specified).

    Notes
    -----
    **Complexity:**

    - Single evaluation: :math:`O(p)` where :math:`p` = number of features
    - Matrix computation: :math:`O(n^2 p)` for :math:`n` samples (vectorized)

    **When to use RBFKernel:**

    - Default choice when no domain knowledge suggests a specific kernel
    - Non-linearly separable data of moderate dimensionality
    - When a smooth, radially symmetric similarity measure is appropriate
    - Classification and regression problems with continuous features

    References
    ----------
    .. [Scholkopf2002] Schoelkopf, B. and Smola, A.J. (2002).
           **Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond.**
           *MIT Press*.

    .. [Chang2010] Chang, Y.W., Hsieh, C.J., Chang, K.W., Ringgaard, M. and Lin, C.J. (2010).
           **Training and Testing Low-degree Polynomial Data Mappings via Linear SVM.**
           *Journal of Machine Learning Research*, 11, pp. 1471-1490.

    See Also
    --------
    :class:`~tuiml.algorithms.svm.kernels.LinearKernel` : Linear kernel for linearly separable data.
    :class:`~tuiml.algorithms.svm.kernels.PolynomialKernel` : Polynomial kernel for finite-order feature maps.
    :class:`~tuiml.algorithms.svm.kernels.PearsonUniversalKernel` : Universal kernel that can approximate RBF.

    Examples
    --------
    Basic usage with an explicit gamma value:

    >>> from tuiml.algorithms.svm.kernels import RBFKernel
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> kernel = RBFKernel(gamma=0.1)
    >>> kernel.build(X_train)
    RBFKernel(...)
    >>> K = kernel.compute_matrix()
    >>> print(K.shape)
    (3, 3)
    """

    _libsvm_kernel_type = 2  # RBF

    def __init__(self, gamma: float = 0.01, cache_size: int = 250007):
        """Initialize RBF kernel.

        Parameters
        ----------
        gamma : Union[str, float], default=0.01
            Kernel coefficient or ``'scale'``/``'auto'`` strategy.
        cache_size : int, default=250007
            Maximum number of cached kernel evaluations.
        """
        super().__init__(cache_size=cache_size)
        self.gamma = gamma
        self.gamma_ = None
        self._dot_precalc = None

    def _libsvm_params(self) -> str:
        """Return libsvm parameter string."""
        g = self.gamma_ if self.gamma_ is not None else self.gamma
        return f"-g {g}"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "gamma": {
                "type": "number",
                "default": 0.01,
                "minimum": 0,
                "description": "Kernel coefficient (1/(2*sigma^2))"
            }
        }

    def build(self, X: np.ndarray) -> "RBFKernel":
        """Build kernel and precompute squared norms.

        Parameters
        ----------
        X : np.ndarray
            Training data.

        Returns
        -------
        self : RBFKernel
            Returns the built instance.
        """
        super().build(X)

        # Handle gamma='scale' or 'auto'
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                X_var = X.var()
                self.gamma_ = 1.0 / (self.n_features_ * X_var) if X_var > 0 else 1.0
            elif self.gamma == 'auto':
                self.gamma_ = 1.0 / self.n_features_
            else:
                raise ValueError(f"Unknown gamma: {self.gamma}")
        else:
            self.gamma_ = self.gamma

        # Precompute squared norms for efficiency
        self._dot_precalc = np.sum(X ** 2, axis=1)

        return self

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2).

        Parameters
        ----------
        x1 : np.ndarray
            First vector.
        x2 : np.ndarray
            Second vector.

        Returns
        -------
        val : float
            RBF kernel value in (0, 1].
        """
        gamma = self.gamma_ if self.gamma_ is not None else self.gamma
        diff = x1 - x2
        squared_dist = np.dot(diff, diff)
        return np.exp(-gamma * squared_dist)

    def compute(self, i: int, j: int) -> float:
        """Compute RBF kernel efficiently using precomputed norms.

        Parameters
        ----------
        i : int
            Index of first instance.
        j : int
            Index of second instance.

        Returns
        -------
        val : float
            Kernel value.
        """
        self._check_is_built()

        if self.cache_size != -1:
            # Check cache
            key = (min(i, j), max(i, j))
            if key in self._cache:
                self._cache_hits += 1
                return self._cache[key]
            self._cache_misses += 1

        # Efficient computation using precomputed squared norms
        squared_dist = (self._dot_precalc[i] + self._dot_precalc[j]
                       - 2 * np.dot(self.X_[i], self.X_[j]))
        value = np.exp(-self.gamma_ * max(0, squared_dist))

        if self.cache_size != -1:
            if self.cache_size == 0 or len(self._cache) < self.cache_size:
                self._cache[key] = value

        return value

    def compute_matrix(self) -> np.ndarray:
        """Compute the full RBF kernel matrix using vectorized operations.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            The kernel (Gram) matrix with :math:`K[i,j] = \\exp(-\\gamma \\|x_i - x_j\\|^2)`.
        """
        self._check_is_built()

        # Efficient vectorized computation
        # K[i,j] = exp(-gamma * (||x_i||^2 + ||x_j||^2 - 2*<x_i, x_j>))
        sq_norms = self._dot_precalc
        dot_products = self.X_ @ self.X_.T
        sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * dot_products
        sq_dists = np.maximum(sq_dists, 0)  # Handle numerical errors

        return np.exp(-self.gamma_ * sq_dists)

    def compute_matrix_cross(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the RBF kernel matrix between two sample sets.

        Parameters
        ----------
        X1 : np.ndarray of shape (n1, n_features)
            First set of samples.
        X2 : np.ndarray of shape (n2, n_features)
            Second set of samples.

        Returns
        -------
        K : np.ndarray of shape (n1, n2)
            Kernel matrix with :math:`K[i,j] = \\exp(-\\gamma \\|X_1[i] - X_2[j]\\|^2)`.
        """
        from tuiml.base.kernels import _xnp
        xnp = _xnp()
        X1 = xnp.asarray(X1, dtype=float)
        X2 = xnp.asarray(X2, dtype=float)
        gamma = self.gamma_ if self.gamma_ is not None else self.gamma

        sq_norms_1 = xnp.sum(X1 ** 2, axis=1)
        sq_norms_2 = xnp.sum(X2 ** 2, axis=1)
        dot_products = X1 @ X2.T
        sq_dists = sq_norms_1[:, None] + sq_norms_2[None, :] - 2 * dot_products
        sq_dists = xnp.maximum(sq_dists, 0)

        K = xnp.exp(-gamma * sq_dists)
        return np.asarray(K)

    def __repr__(self) -> str:
        """String representation."""
        gamma_str = self.gamma_ if self.gamma_ is not None else self.gamma
        return f"RBFKernel(gamma={gamma_str})"

# Alias for common name
GaussianKernel = RBFKernel
