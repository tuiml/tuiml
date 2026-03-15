"""Pearson Universal Kernel implementation."""

import numpy as np
from typing import Dict, Any

from tuiml.base.kernels import CachedKernel, kernel

@kernel(tags=["puk", "universal", "pearson"], version="1.0.0")
class PearsonUniversalKernel(CachedKernel):
    """Pearson Universal Kernel (PUK) based on the **Pearson VII function**.

    A **universal kernel** that can approximate other common kernels --- including
    the polynomial, RBF, and sigmoid kernels --- by adjusting its two parameters
    :math:`\\omega` (tailing factor) and :math:`\\sigma` (peak width). This makes
    PUK a powerful choice for **hyper-parameter search** when the best kernel
    family is unknown.

    Overview
    --------
    The kernel evaluation proceeds as follows:

    1. Compute the **squared Euclidean distance** :math:`\\|x - y\\|^2`
    2. Multiply by a precomputed scaling factor derived from :math:`\\omega` and :math:`\\sigma`
    3. Apply the **Pearson VII** function to produce a similarity in :math:`(0, 1]`

    During ``build()``, the scaling factor and squared norms are **precomputed**
    for efficient pairwise evaluation.

    Theory
    ------
    The Pearson VII function-based kernel is defined as:

    .. math::
        K(x, y) = \\frac{1}{\\left[1 + \\left(\\frac{2\\sqrt{\\|x - y\\|^2} \\cdot \\sqrt{2^{1/\\omega} - 1}}{\\sigma}\\right)^2\\right]^{\\omega}}

    where:

    - :math:`\\omega` --- Tailing factor controlling the shape of the peak (higher = heavier tails)
    - :math:`\\sigma` --- Peak width parameter (higher = wider peak)

    **Kernel approximations:**

    - :math:`\\omega \\approx 0.5, \\sigma \\approx 1` approximates the **RBF kernel**
    - Large :math:`\\omega` values approximate **polynomial-like** behavior
    - The kernel always satisfies :math:`K(x, x) = 1` and :math:`K(x, y) \\in (0, 1]`

    Parameters
    ----------
    omega : float, default=1.0
        Tailing factor controlling the curve shape. Must be positive.

    sigma : float, default=1.0
        Peak width parameter. Must be positive.

    cache_size : int, default=250007
        Maximum number of cached kernel evaluations.

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

    - Single evaluation: :math:`O(p)` where :math:`p` = number of features
    - Matrix computation: :math:`O(n^2 p)` for :math:`n` samples

    **When to use PearsonUniversalKernel:**

    - When the best kernel family (RBF, polynomial, sigmoid) is unknown
    - Automated model selection or hyper-parameter search
    - When a single kernel with two tunable parameters is preferred over multiple kernel types
    - Chemometrics, spectroscopy, and regression tasks where PUK has shown strong results

    References
    ----------
    .. [Uestuen2006] Uestuen, B., Melssen, W.J. and Buydens, L.M.C. (2006).
           **Facilitating the Application of Support Vector Regression by Using a Universal Pearson VII Function Based Kernel.**
           *Chemometrics and Intelligent Laboratory Systems*, 81(1), pp. 29-40.
           DOI: `10.1016/j.chemolab.2005.09.003 <https://doi.org/10.1016/j.chemolab.2005.09.003>`_

    See Also
    --------
    :class:`~tuiml.algorithms.svm.kernels.RBFKernel` : RBF kernel (approximated by PUK with small omega).
    :class:`~tuiml.algorithms.svm.kernels.PolynomialKernel` : Polynomial kernel (approximated by PUK with large omega).

    Examples
    --------
    Basic usage with default parameters:

    >>> from tuiml.algorithms.svm.kernels import PearsonUniversalKernel
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> kernel = PearsonUniversalKernel(omega=1.0, sigma=1.0)
    >>> kernel.build(X_train)
    PearsonUniversalKernel(...)
    >>> value = kernel.evaluate(X_train[0], X_train[1])
    """

    def __init__(self, omega: float = 1.0,
                 sigma: float = 1.0,
                 cache_size: int = 250007):
        """Initialize Pearson Universal Kernel.

        Parameters
        ----------
        omega : float, default=1.0
            Tailing factor controlling the curve shape.
        sigma : float, default=1.0
            Peak width parameter.
        cache_size : int, default=250007
            Maximum number of cached kernel evaluations.
        """
        super().__init__(cache_size=cache_size)
        self.omega = omega
        self.sigma = sigma
        self._factor = None
        self._dot_precalc = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "omega": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.01,
                "description": "Tailing factor (controls curve shape)"
            },
            "sigma": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.01,
                "description": "Peak width parameter"
            }
        }

    @classmethod
    def get_references(cls):
        """Return academic references."""
        return [
            "Uestuen, B., Melssen, W.J., & Buydens, L.M.C. (2006). "
            "Facilitating the application of Support Vector Regression by using "
            "a universal Pearson VII function based kernel. "
            "Chemometrics and Intelligent Laboratory Systems, 81, 29-40."
        ]

    def build(self, X: np.ndarray) -> "PearsonUniversalKernel":
        """Build kernel and precompute factor.

        Parameters
        ----------
        X : np.ndarray
            Training data.

        Returns
        -------
        self : PearsonUniversalKernel
            Returns the built instance.
        """
        super().build(X)

        # Precompute the factor: 2 * sqrt(2^(1/omega) - 1) / sigma
        self._factor = 2.0 * np.sqrt(np.power(2.0, 1.0 / self.omega) - 1) / self.sigma

        # Precompute squared norms
        self._dot_precalc = np.sum(X ** 2, axis=1)

        return self

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate PearsonUniversalKernel kernel.

        K(x, y) = 1 / (1 + (factor * sqrt(||x-y||^2))^2)^omega

        Parameters
        ----------
        x1 : np.ndarray
            First vector.
        x2 : np.ndarray
            Second vector.

        Returns
        -------
        val : float
            PearsonUniversalKernel kernel value in (0, 1].
        """
        diff = x1 - x2
        squared_dist = np.dot(diff, diff)

        factor = self._factor if self._factor is not None else (
            2.0 * np.sqrt(np.power(2.0, 1.0 / self.omega) - 1) / self.sigma
        )

        # K(x,y) = 1 / (1 + (factor * dist)^2)^omega
        intermediate = factor * np.sqrt(max(0, squared_dist))
        return 1.0 / np.power(1.0 + intermediate * intermediate, self.omega)

    def compute_matrix(self) -> np.ndarray:
        """Compute the PUK kernel matrix using vectorized operations.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            The kernel (Gram) matrix.
        """
        self._check_is_built()
        from tuiml.base.kernels import _xnp
        xnp = _xnp()
        X = xnp.asarray(self.X_, dtype=float)

        sq_norms = xnp.sum(X ** 2, axis=1)
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * (X @ X.T)
        sq_dists = xnp.maximum(sq_dists, 0)

        intermediate = self._factor * xnp.sqrt(sq_dists)
        K = 1.0 / xnp.power(1.0 + intermediate * intermediate, self.omega)
        return np.asarray(K)

    def compute_matrix_cross(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the PUK kernel matrix between two sample sets.

        Parameters
        ----------
        X1 : np.ndarray of shape (n1, n_features)
            First set of samples.
        X2 : np.ndarray of shape (n2, n_features)
            Second set of samples.

        Returns
        -------
        K : np.ndarray of shape (n1, n2)
            Kernel matrix.
        """
        from tuiml.base.kernels import _xnp
        xnp = _xnp()
        X1 = xnp.asarray(X1, dtype=float)
        X2 = xnp.asarray(X2, dtype=float)

        factor = self._factor if self._factor is not None else (
            2.0 * np.sqrt(np.power(2.0, 1.0 / self.omega) - 1) / self.sigma
        )

        sq_norms_1 = xnp.sum(X1 ** 2, axis=1)
        sq_norms_2 = xnp.sum(X2 ** 2, axis=1)
        sq_dists = sq_norms_1[:, None] + sq_norms_2[None, :] - 2 * (X1 @ X2.T)
        sq_dists = xnp.maximum(sq_dists, 0)

        intermediate = factor * xnp.sqrt(sq_dists)
        K = 1.0 / xnp.power(1.0 + intermediate * intermediate, self.omega)
        return np.asarray(K)

    def compute(self, i: int, j: int) -> float:
        """Compute PearsonUniversalKernel kernel efficiently using precomputed values.

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
            key = (min(i, j), max(i, j))
            if key in self._cache:
                self._cache_hits += 1
                return self._cache[key]
            self._cache_misses += 1

        # Efficient computation using precomputed squared norms
        squared_dist = (self._dot_precalc[i] + self._dot_precalc[j]
                       - 2 * np.dot(self.X_[i], self.X_[j]))
        squared_dist = max(0, squared_dist)

        intermediate = self._factor * np.sqrt(squared_dist)
        value = 1.0 / np.power(1.0 + intermediate * intermediate, self.omega)

        if self.cache_size != -1:
            if self.cache_size == 0 or len(self._cache) < self.cache_size:
                self._cache[key] = value

        return value

    def __repr__(self) -> str:
        """String representation."""
        return f"PearsonUniversalKernel(omega={self.omega}, sigma={self.sigma})"

# Alias for WEKA compatibility
Puk = PearsonUniversalKernel
PearsonUniversalKernel = PearsonUniversalKernel
