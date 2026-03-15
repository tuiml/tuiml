"""
Sine wave data generator.

Generates data using sine functions with optional noise.
"""

import numpy as np
from typing import Optional, Dict, Any

from tuiml.base.generators import RegressionGenerator, GeneratedData

class Sine(RegressionGenerator):
    """
    Sine wave data generator.

    Generates data using a sine function:

    .. math::
        y = A \\sin(2\\pi f x + \\phi) + c

    where :math:`A` is amplitude, :math:`f` is frequency, :math:`\\phi` is phase,
    and :math:`c` is offset.

    For multiple features, the output is the sum of sine waves.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    n_features : int, default=1
        Number of features.
    amplitude : float, default=1.0
        Amplitude of sine wave.
    frequency : float, default=1.0
        Frequency of sine wave.
    phase : float, default=0.0
        Phase shift (radians).
    offset : float, default=0.0
        Vertical offset.
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = Sine(n_samples=1000, frequency=2.0, noise=0.1)
    >>> data = gen.generate()
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 1,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        phase: float = 0.0,
        offset: float = 0.0,
        noise: float = 0.0,
        random_state: Optional[int] = None
    ):
        """Initialize Sine generator.

        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate.
        n_features : int, default=1
            Number of input features.
        amplitude : float, default=1.0
            Amplitude of the sine wave.
        frequency : float, default=1.0
            Frequency of the sine wave.
        phase : float, default=0.0
            Phase shift in radians.
        offset : float, default=0.0
            Vertical offset added to the output.
        noise : float, default=0.0
            Standard deviation of Gaussian noise added to the target.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__(n_samples, n_features, noise, random_state)
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset

    def generate(self) -> GeneratedData:
        """Generate sine wave data.

        Returns
        -------
        GeneratedData
            Generated dataset with feature array X of shape
            (n_samples, n_features) and continuous target values y.
        """
        rng = self._rng

        # Generate uniform points in [0, 1]
        X = rng.uniform(0, 1, (self.n_samples, self.n_features))

        # Compute sine for each feature and sum
        y = np.zeros(self.n_samples)
        for j in range(self.n_features):
            y += self.amplitude * np.sin(2 * np.pi * self.frequency * X[:, j] + self.phase)

        y = y / self.n_features + self.offset

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
            "description": "Amplitude of sine wave"
        }
        schema["frequency"] = {
            "type": "number",
            "default": 1.0,
            "description": "Frequency of sine wave"
        }
        schema["phase"] = {
            "type": "number",
            "default": 0.0,
            "description": "Phase shift (radians)"
        }
        schema["offset"] = {
            "type": "number",
            "default": 0.0,
            "description": "Vertical offset"
        }
        return schema
