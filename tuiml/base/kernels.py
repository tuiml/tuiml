"""
Base class for SVM Kernel functions.

Kernels compute similarity between instances in a (possibly infinite)
feature space without explicitly computing the feature mapping.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

def kernel(tags: List[str] = None, version: str = "1.0.0"):
    """
    Decorator for kernel registration.

    Args:
        tags: List of tags for categorization
        version: Version string
    """
    def decorator(cls):
        cls._tags = tags or []
        cls._version = version
        cls._component_type = "kernel"
        return cls
    return decorator

class Kernel(ABC):
    """
    Abstract base class for kernel functions.

    A kernel function K(x, y) computes the dot product of two instances
    in a (possibly high-dimensional or infinite) feature space without
    explicitly computing the feature mapping phi:
    ``K(x, y) = <phi(x), phi(y)>``

    Kernels must satisfy Mercer's condition (positive semi-definite)
    to ensure valid behavior with SVMs.

    Attributes:
        X_: Training data (set by build())
        n_samples_: Number of training samples
        n_features_: Number of features
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
        """
        Build the kernel with training data.

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            Self for method chaining
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
        """
        Evaluate the kernel function K(x1, x2).

        Args:
            x1: First instance
            x2: Second instance

        Returns:
            Kernel value
        """
        pass

    def compute(self, i: int, j: int) -> float:
        """
        Compute kernel value between training instances i and j.

        Args:
            i: Index of first instance
            j: Index of second instance

        Returns:
            K(X[i], X[j])
        """
        self._check_is_built()
        return self.evaluate(self.X_[i], self.X_[j])

    def compute_row(self, i: int) -> np.ndarray:
        """
        Compute kernel values between instance i and all training instances.

        Args:
            i: Index of query instance

        Returns:
            Array of kernel values [K(X[i], X[0]), K(X[i], X[1]), ...]
        """
        self._check_is_built()
        return np.array([self.evaluate(self.X_[i], self.X_[j])
                        for j in range(self.n_samples_)])

    def compute_matrix(self) -> np.ndarray:
        """
        Compute the full kernel (Gram) matrix.

        Returns:
            Kernel matrix K where K[i,j] = K(X[i], X[j])
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
        """
        Compute kernel values between a point and all training instances.

        Args:
            x: Query point (n_features,)

        Returns:
            Array of kernel values
        """
        self._check_is_built()
        x = np.asarray(x, dtype=float)
        return np.array([self.evaluate(x, self.X_[j])
                        for j in range(self.n_samples_)])

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema for the kernel."""
        return {}

    def _check_is_built(self):
        """Check if kernel has been built with training data."""
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
        """String representation."""
        name = self.__class__.__name__
        if self._is_built:
            return f"{name}(n_samples={self.n_samples_})"
        return f"{name}(not built)"

class CachedKernel(Kernel):
    """
    Kernel with caching for repeated evaluations.

    Stores computed kernel values to avoid redundant calculations.

    Parameters:
        cache_size: Size of the cache (0 for full cache, -1 for no cache)
    """

    def __init__(self, cache_size: int = 250007):
        """
        Initialize cached kernel.

        Args:
            cache_size: Size of cache (prime number recommended)
        """
        super().__init__()
        self.cache_size = cache_size
        self._cache: Dict[tuple, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def build(self, X: np.ndarray) -> "CachedKernel":
        """Build kernel and initialize cache."""
        super().build(X)
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        return self

    def compute(self, i: int, j: int) -> float:
        """Compute kernel value with caching."""
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
        """Clear the kernel cache."""
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache)
        }
