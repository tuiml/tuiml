"""
LED (Light Emitting Diode) data generator.

Generates data simulating a seven-segment LED display.
"""

import numpy as np
from typing import Optional, Dict, Any

from tuiml.base.generators import ClassificationGenerator, GeneratedData

class LED(ClassificationGenerator):
    """
    LED (Light Emitting Diode) data generator.

    Generates data simulating a seven-segment LED display that shows
    digits 0-9. Each segment has a probability of being inverted (noise).

    The seven segments are arranged as:
        _0_
       |   |
       1   2
       |_3_|
       |   |
       4   5
       |_6_|

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    noise : float, default=0.1
        Probability of segment inversion (0.0-1.0).
    n_irrelevant : int, default=17
        Number of irrelevant attributes to add.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = LED(n_samples=1000, noise=0.1)
    >>> data = gen.generate()
    """

    # LED patterns for digits 0-9 (7 segments)
    # 1 = segment on, 0 = segment off
    PATTERNS = np.array([
        [1, 1, 1, 0, 1, 1, 1],  # 0
        [0, 0, 1, 0, 0, 1, 0],  # 1
        [1, 0, 1, 1, 1, 0, 1],  # 2
        [1, 0, 1, 1, 0, 1, 1],  # 3
        [0, 1, 1, 1, 0, 1, 0],  # 4
        [1, 1, 0, 1, 0, 1, 1],  # 5
        [1, 1, 0, 1, 1, 1, 1],  # 6
        [1, 0, 1, 0, 0, 1, 0],  # 7
        [1, 1, 1, 1, 1, 1, 1],  # 8
        [1, 1, 1, 1, 0, 1, 1],  # 9
    ])

    def __init__(
        self,
        n_samples: int = 100,
        noise: float = 0.1,
        n_irrelevant: int = 17,
        random_state: Optional[int] = None
    ):
        """Initialize LED generator.

        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate.
        noise : float, default=0.1
            Probability of inverting each LED segment.
        n_irrelevant : int, default=17
            Number of irrelevant binary attributes to append.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        n_features = 7 + n_irrelevant
        super().__init__(n_samples, n_features, n_classes=10, random_state=random_state)
        self.noise = noise
        self.n_irrelevant = n_irrelevant

    def generate(self) -> GeneratedData:
        """Generate LED data.

        Returns
        -------
        GeneratedData
            Generated dataset with feature array X of shape
            (n_samples, 7 + n_irrelevant) and digit labels y (0-9).
        """
        rng = self._rng

        n_relevant = 7
        X = np.zeros((self.n_samples, self.n_features))
        y = np.zeros(self.n_samples, dtype=int)

        for i in range(self.n_samples):
            # Choose random digit
            digit = rng.integers(0, 10)
            y[i] = digit

            # Get pattern and apply noise
            pattern = self.PATTERNS[digit].copy().astype(float)

            for j in range(n_relevant):
                if rng.uniform() < self.noise:
                    pattern[j] = 1 - pattern[j]

            X[i, :n_relevant] = pattern

            # Add irrelevant attributes (random binary)
            if self.n_irrelevant > 0:
                X[i, n_relevant:] = rng.integers(0, 2, self.n_irrelevant)

        feature_names = [f"seg{i}" for i in range(7)]
        feature_names += [f"irr{i}" for i in range(self.n_irrelevant)]

        return GeneratedData(
            X=X,
            y=y,
            feature_names=feature_names,
            target_names=[str(i) for i in range(10)]
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        schema = ClassificationGenerator.get_parameter_schema()
        schema["noise"] = {
            "type": "number",
            "default": 0.1,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Probability of segment inversion"
        }
        schema["n_irrelevant"] = {
            "type": "integer",
            "default": 17,
            "minimum": 0,
            "description": "Number of irrelevant attributes"
        }
        del schema["n_features"]  # Fixed based on n_irrelevant
        return schema
