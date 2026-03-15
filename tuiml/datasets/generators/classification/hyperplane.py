"""
Hyperplane data generator.

Generates data separated by a random hyperplane.
"""

import numpy as np
from typing import Optional, Dict, Any

from tuiml.base.generators import ClassificationGenerator, GeneratedData

class Hyperplane(ClassificationGenerator):
    """
    Hyperplane data generator.

    Generates data points in n-dimensional space separated by a random
    hyperplane. Points are classified based on which side of the
    hyperplane they fall on.

    The hyperplane is defined by:

    .. math::
        w_1 x_1 + w_2 x_2 + \\dots + w_n x_n = \\text{threshold}

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    n_features : int, default=10
        Number of features (dimensions).
    n_drift_features : int, default=0
        Number of features with changing weights (concept drift).
    noise : float, default=0.0
        Fraction of labels to flip (noise).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = Hyperplane(n_samples=1000, n_features=10)
    >>> data = gen.generate()
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 10,
        n_drift_features: int = 0,
        noise: float = 0.0,
        random_state: Optional[int] = None
    ):
        """Initialize Hyperplane generator.

        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate.
        n_features : int, default=10
            Number of features (dimensions).
        n_drift_features : int, default=0
            Number of features with changing weights for concept drift.
        noise : float, default=0.0
            Fraction of labels to flip.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__(n_samples, n_features, n_classes=2, random_state=random_state)
        self.n_drift_features = min(n_drift_features, n_features)
        self.noise = noise

    def generate(self) -> GeneratedData:
        """Generate hyperplane-separated data.

        Returns
        -------
        GeneratedData
            Generated dataset with feature array X of shape
            (n_samples, n_features) and binary class labels y.
        """
        rng = self._rng

        # Generate random hyperplane weights
        weights = rng.uniform(-1, 1, self.n_features)

        # Normalize weights
        weights /= np.linalg.norm(weights)

        # Generate random points in [0, 1]^n
        X = rng.uniform(0, 1, (self.n_samples, self.n_features))

        # Compute threshold (use sum of weights * 0.5 for balanced classes)
        threshold = np.sum(weights) * 0.5

        # Classify points
        scores = X @ weights
        y = (scores >= threshold).astype(int)

        # Add noise (flip labels)
        if self.noise > 0:
            flip_mask = rng.uniform(size=self.n_samples) < self.noise
            y[flip_mask] = 1 - y[flip_mask]

        return GeneratedData(
            X=X,
            y=y,
            feature_names=[f"x{i}" for i in range(self.n_features)],
            target_names=['class0', 'class1']
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        schema = super().get_parameter_schema()
        schema["n_drift_features"] = {
            "type": "integer",
            "default": 0,
            "minimum": 0,
            "description": "Number of features with changing weights"
        }
        schema["noise"] = {
            "type": "number",
            "default": 0.0,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Fraction of labels to flip"
        }
        return schema
