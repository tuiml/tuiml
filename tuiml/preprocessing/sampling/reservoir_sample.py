"""
ReservoirSampler filter.

Reservoir sampling for streaming/large datasets.
"""

from typing import Optional, Tuple
import numpy as np

from tuiml.base.preprocessing import InstanceTransformer, transformer

@transformer(tags=["sampling", "reservoir", "streaming"], version="1.0.0")
class ReservoirSampler(InstanceTransformer):
    """
    Reservoir sampling for random sampling from large datasets.

    Implements Algorithm R (Vitter, 1985) for uniform random sampling
    without replacement. Useful for streaming data or when dataset
    is too large to fit in memory.

    Parameters
    ----------
    sample_size : int, default=100
        Number of instances to sample.
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.preprocessing.sampling import ReservoirSampler

    >>> X = np.arange(1000).reshape(-1, 1)
    >>> y = np.zeros(1000)

    >>> # Sample 100 instances
    >>> sampler = ReservoirSampler(sample_size=100, random_state=42)
    >>> X_sample, y_sample = sampler.fit_transform(X, y)
    >>> len(X_sample)
    100

    Notes
    -----
    This implementation is deterministic given a random_state.
    For true streaming applications, you would typically process
    data one instance at a time.

    References
    ----------
    Vitter, J. S. (1985). Random sampling with a reservoir.
    ACM Transactions on Mathematical Software, 11(1), 37-57.
    """

    def __init__(
        self,
        sample_size: int = 100,
        random_state: Optional[int] = None,
    ):
        """
        Initialize ReservoirSampler.

        Args:
            sample_size: Number of instances to sample
            random_state: Random seed
        """
        super().__init__()
        self.sample_size = sample_size
        self.random_state = random_state

        if sample_size < 1:
            raise ValueError(f"sample_size must be >= 1, got {sample_size}")

    @classmethod
    def get_parameter_schema(cls):
        return {
            "sample_size": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Number of instances to sample",
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed",
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "ReservoirSampler":
        """
        Fit the sampler (no-op).

        Args:
            X: Input data
            y: Target values (optional)

        Returns:
            Self
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._n_features_in = X.shape[1]
        self._is_fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Perform reservoir sampling.

        Args:
            X: Input data
            y: Target values (optional)

        Returns:
            (X_sampled, y_sampled) tuple
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        # If dataset is smaller than sample size, return all
        if n_samples <= self.sample_size:
            return X.copy(), y.copy() if y is not None else None

        # Algorithm R: Reservoir Sampling
        # Initialize reservoir with first k items
        reservoir_indices = list(range(self.sample_size))

        # Process remaining items
        for i in range(self.sample_size, n_samples):
            # Pick random index in [0, i]
            j = rng.randint(0, i + 1)
            # If j < k, replace reservoir[j] with item i
            if j < self.sample_size:
                reservoir_indices[j] = i

        # Sort indices to maintain some order
        reservoir_indices = sorted(reservoir_indices)

        X_sampled = X[reservoir_indices]
        y_sampled = y[reservoir_indices] if y is not None else None

        return X_sampled, y_sampled

    def __repr__(self) -> str:
        return f"ReservoirSampler(sample_size={self.sample_size})"
