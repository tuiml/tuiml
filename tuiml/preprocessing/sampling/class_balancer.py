"""
ClassBalanceSampler filter.

Balance class distribution through resampling.
- weka.filters.supervised.instance.ClassBalanceSampler
"""

from typing import Optional, Tuple
import numpy as np
from collections import Counter

from tuiml.base.preprocessing import InstanceTransformer, transformer

@transformer(tags=["sampling", "balance", "class", "oversample", "undersample"], version="1.0.0")
class ClassBalanceSampler(InstanceTransformer):
    """
    Balance class distribution through resampling.

    Resamples instances to achieve balanced class distribution.
    Can oversample minority classes, undersample majority classes, or both.

    Parameters
    ----------
    strategy : str, default="oversample"
        Balancing strategy:
        - "oversample": Duplicate minority class instances
        - "undersample": Remove majority class instances
        - "both": Combination (SMOTE-like without synthetic generation)
    target_ratio : float, default=1.0
        Target ratio of minority to majority class (1.0 = equal).
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.preprocessing.sampling import ClassBalanceSampler

    >>> # Imbalanced dataset: 90 class 0, 10 class 1
    >>> X = np.arange(100).reshape(-1, 1)
    >>> y = np.array([0]*90 + [1]*10)

    >>> # Oversample minority class
    >>> balancer = ClassBalanceSampler(strategy="oversample", random_state=42)
    >>> X_bal, y_bal = balancer.fit_transform(X, y)
    >>> Counter(y_bal)
    Counter({0: 90, 1: 90})

    >>> # Undersample majority class
    >>> balancer = ClassBalanceSampler(strategy="undersample", random_state=42)
    >>> X_bal, y_bal = balancer.fit_transform(X, y)
    >>> Counter(y_bal)
    Counter({0: 10, 1: 10})
    """

    def __init__(
        self,
        strategy: str = "oversample",
        target_ratio: float = 1.0,
        random_state: Optional[int] = None,
    ):
        """
        Initialize ClassBalanceSampler.

        Args:
            strategy: Balancing strategy
            target_ratio: Target minority/majority ratio
            random_state: Random seed
        """
        super().__init__()
        self.strategy = strategy
        self.target_ratio = target_ratio
        self.random_state = random_state

        if strategy not in ("oversample", "undersample", "both"):
            raise ValueError(
                f"strategy must be 'oversample', 'undersample', or 'both', "
                f"got {strategy!r}"
            )
        if target_ratio <= 0 or target_ratio > 1:
            raise ValueError(f"target_ratio must be in (0, 1], got {target_ratio}")

    @classmethod
    def get_parameter_schema(cls):
        return {
            "strategy": {
                "type": "string",
                "default": "oversample",
                "enum": ["oversample", "undersample", "both"],
                "description": "Balancing strategy",
            },
            "target_ratio": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "maximum": 1,
                "exclusiveMinimum": True,
                "description": "Target minority/majority ratio",
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
    ) -> "ClassBalanceSampler":
        """
        Fit the balancer.

        Args:
            X: Input data
            y: Target values (required)

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
        Balance class distribution.

        Args:
            X: Input data
            y: Target values (required)

        Returns:
            (X_balanced, y_balanced) tuple
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y is None:
            raise ValueError("y is required for class balancing")

        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)

        # Get class counts
        class_counts = Counter(y)
        classes = list(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        max_count = max(counts)
        min_count = min(counts)

        # Determine target counts based on strategy
        if self.strategy == "oversample":
            target_count = int(max_count * self.target_ratio)
        elif self.strategy == "undersample":
            target_count = int(min_count / self.target_ratio)
        else:  # both
            target_count = int((max_count + min_count) / 2 * self.target_ratio)

        # Resample each class
        new_indices = []

        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            n_cls = len(cls_indices)

            if n_cls < target_count:
                # Need to oversample
                if self.strategy in ("oversample", "both"):
                    # Repeat and add random samples
                    n_extra = target_count - n_cls
                    extra_indices = rng.choice(cls_indices, size=n_extra, replace=True)
                    new_indices.extend(cls_indices)
                    new_indices.extend(extra_indices)
                else:
                    new_indices.extend(cls_indices)
            elif n_cls > target_count:
                # Need to undersample
                if self.strategy in ("undersample", "both"):
                    sampled = rng.choice(cls_indices, size=target_count, replace=False)
                    new_indices.extend(sampled)
                else:
                    new_indices.extend(cls_indices)
            else:
                new_indices.extend(cls_indices)

        new_indices = np.array(new_indices)
        rng.shuffle(new_indices)

        X_balanced = X[new_indices]
        y_balanced = y[new_indices]

        return X_balanced, y_balanced

    def __repr__(self) -> str:
        return (
            f"ClassBalanceSampler(strategy={self.strategy!r}, "
            f"target_ratio={self.target_ratio})"
        )
