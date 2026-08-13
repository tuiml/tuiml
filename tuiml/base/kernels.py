"""
Base class for SVM Kernel functions.

Kernels compute similarity between instances in a (possibly infinite)
feature space without explicitly computing the feature mapping.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Large prime used as the default kernel cache size. A prime size spreads the
# symmetric (i, j) cache keys more evenly across hash buckets, keeping
# collisions low as the cache fills.
_DEFAULT_CACHE_SIZE = 250007

def kernel(tags: List[str] = None, version: str = "1.0.0"):
    """Class decorator that registers a kernel with the component registry.

    Parameters
    ----------
    tags : list of str, default=None
        Tags for categorization in the registry.
    version : str, default="1.0.0"
        Version string for the kernel.

    Returns
    -------
    decorator : callable
        Decorator that attaches registry metadata to the kernel class.
    """
    def decorator(cls):
        """Attach registry metadata to the kernel class."""
        cls._tags = tags or []
        cls._version = version
        cls._component_type = "kernel"
        return cls
    return decorator

class Kernel(ABC):
    """Abstract base class for kernel functions.

    A kernel function :math:`K(x, y)` computes the dot product of two
    instances in a (possibly high-dimensional or infinite) feature space
    without explicitly computing the feature mapping :math:`\\phi`:

    .. math::
        K(x, y) = \\langle \\phi(x), \\phi(y) \\rangle

    Kernels must satisfy Mercer's condition (positive semi-definite)
    to ensure valid behavior with SVMs.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_samples, n_features)
        Training data (set by ``build()``).
    n_samples_ : int
        Number of training samples.
    n_features_ : int
        Number of features.
    """

    # libsvm kernel type: 0=linear, 1=poly, 2=rbf, 3=sigmoid, None=precomputed
    _libsvm_kernel_type: Optional[int] = None

    def __init__(self):
        """Initialize the kernel."""
        self._is_built = False
        self.X_ = None
        self.n_samples_ = None
        self.n_features_ = None

    def build(self, X: np.ndarray) -> "Kernel":
        """Build the kernel with training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : Kernel
            The built kernel, for method chaining.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_ = X
        self.n_samples_, self.n_features_ = X.shape
        self._is_built = True

        return self

    @abstractmethod
    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate the kernel function :math:`K(x_1, x_2)`.

        Parameters
        ----------
        x1 : np.ndarray of shape (n_features,)
            First instance.
        x2 : np.ndarray of shape (n_features,)
            Second instance.

        Returns
        -------
        value : float
            Kernel value.
        """
        pass

    def compute(self, i: int, j: int) -> float:
        """Compute kernel value between training instances ``i`` and ``j``.

        Parameters
        ----------
        i : int
            Index of first instance.
        j : int
            Index of second instance.

        Returns
        -------
        value : float
            Kernel value ``K(X[i], X[j])``.
        """
        self._check_is_built()
        return self.evaluate(self.X_[i], self.X_[j])

    def compute_row(self, i: int) -> np.ndarray:
        """Compute kernel values between instance ``i`` and all training instances.

        Parameters
        ----------
        i : int
            Index of query instance.

        Returns
        -------
        row : np.ndarray of shape (n_samples,)
            Kernel values ``[K(X[i], X[0]), K(X[i], X[1]), ...]``.
        """
        self._check_is_built()
        return np.array([self.evaluate(self.X_[i], self.X_[j])
                        for j in range(self.n_samples_)])

    def compute_matrix(self) -> np.ndarray:
        """Compute the full kernel (Gram) matrix.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            Kernel matrix where ``K[i, j] = K(X[i], X[j])``.
        """
        self._check_is_built()
        K = np.zeros((self.n_samples_, self.n_samples_))

        for i in range(self.n_samples_):
            for j in range(i, self.n_samples_):
                K[i, j] = self.evaluate(self.X_[i], self.X_[j])
                K[j, i] = K[i, j]  # Symmetric

        return K

    def compute_matrix_cross(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the kernel matrix between two different sets of samples.

        Parameters
        ----------
        X1 : np.ndarray of shape (n1, n_features)
            First set of samples.
        X2 : np.ndarray of shape (n2, n_features)
            Second set of samples.

        Returns
        -------
        K : np.ndarray of shape (n1, n2)
            Kernel matrix where ``K[i, j] = K(X1[i], X2[j])``.
        """
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.evaluate(X1[i], X2[j])
        return K

    def compute_with_point(self, x: np.ndarray) -> np.ndarray:
        """Compute kernel values between a point and all training instances.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            Query point.

        Returns
        -------
        values : np.ndarray of shape (n_samples,)
            Kernel values between ``x`` and each training instance.
        """
        self._check_is_built()
        x = np.asarray(x, dtype=float)
        return np.array([self.evaluate(x, self.X_[j])
                        for j in range(self.n_samples_)])

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema for the kernel.

        Returns
        -------
        schema : dict of str to dict
            Mapping of constructor parameter names to JSON-Schema-style
            descriptions. Empty for kernels without hyperparameters.
        """
        return {}

    def _check_is_built(self):
        """Raise ``RuntimeError`` if the kernel has not been built with training data."""
        if not self._is_built:
            raise RuntimeError("Kernel not built. Call build() first.")

    def _libsvm_params(self) -> str:
        """Return libsvm parameter string for this kernel's hyperparameters.

        Returns
        -------
        params : str
            libsvm option flags (e.g. ``'-g 0.1 -d 3 -r 1.0'``).
            Empty string for kernels with no extra params.
        """
        return ""

    def __repr__(self) -> str:
        """Return string representation of the kernel."""
        name = self.__class__.__name__
        if self._is_built:
            return f"{name}(n_samples={self.n_samples_})"
        return f"{name}(not built)"

class CachedKernel(Kernel):
    """Kernel with caching for repeated evaluations.

    Stores computed kernel values to avoid redundant calculations.

    Parameters
    ----------
    cache_size : int, default=250007
        Size of the cache, in number of stored kernel entries. Use ``0`` for
        an unbounded (full) cache and ``-1`` to disable caching entirely. The
        default is a large prime so the symmetric ``(i, j)`` cache keys hash
        evenly across buckets.
    """

    def __init__(self, cache_size: int = _DEFAULT_CACHE_SIZE):
        """Initialize cached kernel.

        Parameters
        ----------
        cache_size : int, default=250007
            Number of stored kernel entries (a prime is recommended so keys
            spread evenly across hash buckets).
        """
        super().__init__()
        self.cache_size = cache_size
        self._cache: Dict[tuple, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def build(self, X: np.ndarray) -> "CachedKernel":
        """Build kernel with training data and initialize an empty cache.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : CachedKernel
            The built kernel, for method chaining.
        """
        super().build(X)
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        return self

    def compute(self, i: int, j: int) -> float:
        """Compute kernel value between training instances ``i`` and ``j`` with caching.

        Parameters
        ----------
        i : int
            Index of first instance.
        j : int
            Index of second instance.

        Returns
        -------
        value : float
            Kernel value ``K(X[i], X[j])``, served from the cache when available.
        """
        self._check_is_built()

        if self.cache_size == -1:
            # No caching
            return self.evaluate(self.X_[i], self.X_[j])

        # Ensure symmetric lookup
        key = (min(i, j), max(i, j))

        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        self._cache_misses += 1
        value = self.evaluate(self.X_[i], self.X_[j])

        # Add to cache if not full
        if self.cache_size == 0 or len(self._cache) < self.cache_size:
            self._cache[key] = value

        return value

    def clear_cache(self) -> None:
        """Clear the kernel cache and reset hit/miss counters.

        Returns
        -------
        None
        """
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns
        -------
        stats : dict of str to int
            Dictionary with ``"hits"``, ``"misses"``, and ``"size"`` counts.
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache)
        }
