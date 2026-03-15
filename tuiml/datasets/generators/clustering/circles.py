"""
Concentric Circles data generator.

Generates two concentric circles.
"""

import numpy as np
from typing import Optional, Dict, Any

from tuiml.base.generators import ClusteringGenerator, GeneratedData

class Circles(ClusteringGenerator):
    """
    Concentric Circles data generator.

    Generates two concentric circles (inner and outer).
    This is a classic non-linearly separable dataset.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    noise : float, default=0.05
        Standard deviation of Gaussian noise.
    factor : float, default=0.5
        Scale factor between inner and outer circle (0 < factor < 1).
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = Circles(n_samples=1000, noise=0.05, factor=0.5)
    >>> data = gen.generate()
    """

    def __init__(
        self,
        n_samples: int = 100,
        noise: float = 0.05,
        factor: float = 0.5,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        """Initialize Circles generator.

        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate.
        noise : float, default=0.05
            Standard deviation of Gaussian noise added to the data.
        factor : float, default=0.5
            Scale factor between inner and outer circle (0 < factor < 1).
        shuffle : bool, default=True
            Whether to shuffle the samples.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__(n_samples, n_features=2, n_clusters=2, random_state=random_state)
        self.noise = noise
        self.factor = factor
        self.shuffle = shuffle

    def generate(self) -> GeneratedData:
        """Generate concentric circles data.

        Returns
        -------
        GeneratedData
            Generated dataset with feature array X of shape
            (n_samples, 2) and circle labels y (0=outer, 1=inner).
        """
        rng = self._rng

        n_samples_out = self.n_samples // 2
        n_samples_in = self.n_samples - n_samples_out

        # Outer circle
        linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
        outer_circ_x = np.cos(linspace_out)
        outer_circ_y = np.sin(linspace_out)

        # Inner circle (scaled by factor)
        linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
        inner_circ_x = np.cos(linspace_in) * self.factor
        inner_circ_y = np.sin(linspace_in) * self.factor

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
            target_names=['outer', 'inner']
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        schema = ClusteringGenerator.get_parameter_schema()
        schema["noise"] = {
            "type": "number",
            "default": 0.05,
            "minimum": 0.0,
            "description": "Standard deviation of Gaussian noise"
        }
        schema["factor"] = {
            "type": "number",
            "default": 0.5,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Scale factor for inner circle"
        }
        del schema["n_features"]
        del schema["n_clusters"]
        return schema
