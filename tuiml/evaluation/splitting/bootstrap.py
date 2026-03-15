"""
Bootstrap sampling splitters.
"""

import numpy as np
from typing import Iterator, Optional, Tuple
from tuiml.base.splitting import BaseSplitter

class BootstrapSplit(BaseSplitter):
    """
    Bootstrap validation (.632 method).

    Samples with replacement for training, out-of-bag samples for testing.

    Parameters
    ----------
    n_iterations : int, default=100
        Number of bootstrap iterations.
    sample_size : float, default=1.0
        Size of bootstrap sample relative to original dataset.
    random_state : int, optional
        Random seed.

    Notes
    -----
    On average, ~63.2% of samples appear in training set (bootstrap sample)
    and ~36.8% are out-of-bag (test set).

    Examples
    --------
    >>> from tuiml.evaluation.splitting import BootstrapSplit
    >>> import numpy as np
    >>> X = np.arange(100).reshape(-1, 1)
    >>> bs = BootstrapSplit(n_iterations=10)
    >>> for train_idx, test_idx in bs.split(X):
    ...     print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    """

    def __init__(
        self,
        n_iterations: int = 100,
        sample_size: float = 1.0,
        random_state: Optional[int] = None
    ):
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
        if not 0 < sample_size <= 2.0:
            raise ValueError("sample_size must be between 0 and 2")

        self.n_iterations = n_iterations
        self.sample_size = sample_size
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "n_iterations": {
                "type": "integer",
                "default": 100,
                "description": "Number of bootstrap iterations"
            },
            "sample_size": {
                "type": "number",
                "default": 1.0,
                "description": "Size of bootstrap sample relative to original dataset (between 0 and 2)"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate bootstrap split indices."""
        X, y = self._validate_input(X, y)
        n_samples = len(X)
        n_bootstrap = int(n_samples * self.sample_size)

        rng = np.random.RandomState(self.random_state)
        all_indices = set(range(n_samples))

        for _ in range(self.n_iterations):
            # Sample with replacement
            train_idx = rng.choice(n_samples, size=n_bootstrap, replace=True)

            # Out-of-bag samples are the test set
            train_set = set(train_idx)
            test_idx = np.array(list(all_indices - train_set))

            if len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits."""
        return self.n_iterations

    def __repr__(self) -> str:
        return f"BootstrapSplit(n_iterations={self.n_iterations}, sample_size={self.sample_size})"
