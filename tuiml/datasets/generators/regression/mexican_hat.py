"""
Mexican Hat (Ricker wavelet) data generator.
"""

import numpy as np
from typing import Optional, Dict, Any

from tuiml.base.generators import RegressionGenerator, GeneratedData

class MexicanHat(RegressionGenerator):
    """
    Mexican Hat (Ricker wavelet) data generator.

    Generates data using the Mexican Hat function, which is the
    negative second derivative of a Gaussian:

    .. math::
        y = \\left(1 - \\frac{r^2}{\\sigma^2}\\right) \\exp\\left(-\\frac{r^2}{2\\sigma^2}\\right)

    where :math:`r` is the distance from the center.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    n_features : int, default=2
        Number of features (default: 2 for visualization).
    amplitude : float, default=1.0
        Amplitude of the hat.
    sigma : float, default=1.0
        Width parameter of the Gaussian.
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = MexicanHat(n_samples=1000, sigma=1.0)
    >>> data = gen.generate()
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 2,
        amplitude: float = 1.0,
        sigma: float = 1.0,
        noise: float = 0.0,
        random_state: Optional[int] = None
    ):
        """Initialize MexicanHat generator.

        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate.
        n_features : int, default=2
            Number of input features.
        amplitude : float, default=1.0
            Amplitude of the Mexican Hat function.
        sigma : float, default=1.0
            Width parameter of the underlying Gaussian.
        noise : float, default=0.0
            Standard deviation of Gaussian noise added to the target.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__(n_samples, n_features, noise, random_state)
        self.amplitude = amplitude
        self.sigma = sigma

    def generate(self) -> GeneratedData:
        """Generate Mexican Hat data.

        Returns
        -------
        GeneratedData
            Generated dataset with feature array X of shape
            (n_samples, n_features) and continuous target values y.
        """
        rng = self._rng

        # Generate uniform points in [-4*sigma, 4*sigma]
        range_val = 4 * self.sigma
        X = rng.uniform(-range_val, range_val, (self.n_samples, self.n_features))

        # Compute radial distance from origin
        r_squared = np.sum(X ** 2, axis=1)
        sigma_squared = self.sigma ** 2

        # Mexican Hat function
        y = self.amplitude * (1 - r_squared / sigma_squared) * np.exp(-r_squared / (2 * sigma_squared))

        # Add noise
        if self.noise > 0:
            y += rng.normal(0, self.noise, self.n_samples)

        return GeneratedData(
            X=X,
            y=y,
            feature_names=[f"x{i}" for i in range(self.n_features)]
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        schema = super().get_parameter_schema()
        schema["amplitude"] = {
            "type": "number",
            "default": 1.0,
            "description": "Amplitude of the Mexican Hat"
        }
        schema["sigma"] = {
            "type": "number",
            "default": 1.0,
            "minimum": 0.01,
            "description": "Width parameter"
        }
        return schema
