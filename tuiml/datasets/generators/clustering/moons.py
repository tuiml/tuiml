"""
Two Moons data generator.

Generates two interleaving half circles (moons).
"""

import numpy as np
from typing import Optional, Dict, Any

from tuiml.base.generators import ClusteringGenerator, GeneratedData

class Moons(ClusteringGenerator):
    """
    Two Moons data generator.

    Generates two interleaving half circles (moons).
    This is a classic non-linearly separable dataset.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    noise : float, default=0.1
        Standard deviation of Gaussian noise.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = Moons(n_samples=1000, noise=0.1)
    >>> data = gen.generate()
    """

    def __init__(
        self,
        n_samples: int = 100,
        noise: float = 0.1,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        """Initialize Moons generator.

        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate.
        noise : float, default=0.1
            Standard deviation of Gaussian noise added to the data.
        shuffle : bool, default=True
            Whether to shuffle the samples.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__(n_samples, n_features=2, n_clusters=2, random_state=random_state)
        self.noise = noise
        self.shuffle = shuffle

    def generate(self) -> GeneratedData:
        """Generate two moons data.

        Returns
        -------
        GeneratedData
            Generated dataset with feature array X of shape
            (n_samples, 2) and moon labels y (0 or 1).
        """
        rng = self._rng

        n_samples_out = self.n_samples // 2
        n_samples_in = self.n_samples - n_samples_out

        # Outer moon
        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))

        # Inner moon (shifted)
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
        inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

        X = np.vstack([
            np.column_stack([outer_circ_x, outer_circ_y]),
            np.column_stack([inner_circ_x, inner_circ_y])
        ])

        y = np.hstack([
            np.zeros(n_samples_out, dtype=int),
            np.ones(n_samples_in, dtype=int)
        ])

        # Add noise
        if self.noise > 0:
            X += rng.normal(0, self.noise, X.shape)

        # Shuffle
        if self.shuffle:
            indices = rng.permutation(len(X))
            X = X[indices]
            y = y[indices]

        return GeneratedData(
            X=X,
            y=y,
            feature_names=['x0', 'x1'],
            target_names=['moon0', 'moon1']
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        schema = ClusteringGenerator.get_parameter_schema()
        schema["noise"] = {
            "type": "number",
            "default": 0.1,
            "minimum": 0.0,
            "description": "Standard deviation of Gaussian noise"
        }
        schema["shuffle"] = {
            "type": "boolean",
            "default": True,
            "description": "Whether to shuffle the samples"
        }
        del schema["n_features"]
        del schema["n_clusters"]
        return schema
