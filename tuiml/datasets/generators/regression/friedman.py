"""
Friedman data generators.

Classic benchmark functions for regression.
"""

import numpy as np
from typing import Optional, Dict, Any

from tuiml.base.generators import RegressionGenerator, GeneratedData

class Friedman(RegressionGenerator):
    """
    Friedman regression data generator.

    Generates data using one of the Friedman benchmark functions.
    These are standard test functions for regression algorithms.

    Functions:

    - 1: :math:`y = 10 \\sin(\\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10 x_4 + 5 x_5`
    - 2: :math:`y = \\sqrt{x_1^2 + (x_2 x_3 - 1/(x_2 x_4))^2}`
    - 3: :math:`y = \\arctan((x_2 x_3 - 1/(x_2 x_4)) / x_1)`

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    n_features : int, default=10
        Number of features (minimum 5 for function 1, 4 for 2&3).
    function : int, default=1
        Friedman function (1, 2, or 3).
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = Friedman(n_samples=1000, function=1, noise=1.0)
    >>> data = gen.generate()
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 10,
        function: int = 1,
        noise: float = 0.0,
        random_state: Optional[int] = None
    ):
        """Initialize Friedman generator.

        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate.
        n_features : int, default=10
            Number of features (minimum 5 for function 1, 4 for 2 and 3).
        function : int, default=1
            Friedman function index (1, 2, or 3).
        noise : float, default=0.0
            Standard deviation of Gaussian noise added to the target.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        min_features = 5 if function == 1 else 4
        n_features = max(n_features, min_features)
        super().__init__(n_samples, n_features, noise, random_state)
        self.function = function

    def generate(self) -> GeneratedData:
        """Generate Friedman data.

        Returns
        -------
        GeneratedData
            Generated dataset with feature array X of shape
            (n_samples, n_features) and continuous target values y.
        """
        rng = self._rng

        X = rng.uniform(0, 1, (self.n_samples, self.n_features))

        if self.function == 1:
            y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) +
                 20 * (X[:, 2] - 0.5) ** 2 +
                 10 * X[:, 3] +
                 5 * X[:, 4])
        elif self.function == 2:
            y = np.sqrt(X[:, 0] ** 2 +
                       (X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3] + 1e-10)) ** 2)
        else:  # function == 3
            y = np.arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3] + 1e-10)) /
                         (X[:, 0] + 1e-10))

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
        schema["function"] = {
            "type": "integer",
            "default": 1,
            "minimum": 1,
            "maximum": 3,
            "description": "Friedman function (1, 2, or 3)"
        }
        return schema
