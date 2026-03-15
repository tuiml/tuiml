"""
Swiss Roll data generator.

Generates the classic Swiss Roll 3D manifold.
"""

import numpy as np
from typing import Optional, Dict, Any

from tuiml.base.generators import ClusteringGenerator, GeneratedData

class SwissRoll(ClusteringGenerator):
    """
    Swiss Roll data generator.

    Generates the classic Swiss Roll 3D manifold dataset.
    This is commonly used to test dimensionality reduction algorithms.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    hole : bool, default=False
        Whether to include a hole in the middle.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = SwissRoll(n_samples=1000, noise=0.1)
    >>> data = gen.generate()
    """

    def __init__(
        self,
        n_samples: int = 100,
        noise: float = 0.0,
        hole: bool = False,
        random_state: Optional[int] = None
    ):
        """Initialize SwissRoll generator.

        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate.
        noise : float, default=0.0
            Standard deviation of Gaussian noise added to the data.
        hole : bool, default=False
            Whether to include a hole in the middle of the roll.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__(n_samples, n_features=3, n_clusters=1, random_state=random_state)
        self.noise = noise
        self.hole = hole

    def generate(self) -> GeneratedData:
        """Generate Swiss Roll data.

        Returns
        -------
        GeneratedData
            Generated dataset with feature array X of shape
            (n_samples, 3) and continuous position values y.
        """
        rng = self._rng

        if self.hole:
            # With hole: t in [3*pi/2, 3*pi]
            t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=self.n_samples))
        else:
            # Without hole: t in [pi/2, 4*pi]
            t = 0.5 * np.pi + 3.5 * np.pi * rng.uniform(size=self.n_samples)

        # Height
        y_coord = 21 * rng.uniform(size=self.n_samples)

        # Swiss roll coordinates
        x = t * np.cos(t)
        z = t * np.sin(t)

        X = np.column_stack([x, y_coord, z])

        # The "color" / position along the roll
        y = t

        # Add noise
        if self.noise > 0:
            X += rng.normal(0, self.noise, X.shape)

        return GeneratedData(
            X=X,
            y=y,
            feature_names=['x', 'y', 'z']
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        schema = ClusteringGenerator.get_parameter_schema()
        schema["noise"] = {
            "type": "number",
            "default": 0.0,
            "minimum": 0.0,
            "description": "Standard deviation of Gaussian noise"
        }
        schema["hole"] = {
            "type": "boolean",
            "default": False,
            "description": "Whether to include a hole in the middle"
        }
        del schema["n_features"]
        del schema["n_clusters"]
        return schema
