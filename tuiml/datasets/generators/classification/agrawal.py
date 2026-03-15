"""
Agrawal data generator.

Generates data using the Agrawal function generator.
"""

import numpy as np
from typing import Optional, Dict, Any

from tuiml.base.generators import ClassificationGenerator, GeneratedData

class Agrawal(ClassificationGenerator):
    """
    Agrawal data generator.

    Generates data using various classification functions defined by Agrawal.
    The generator creates 9 numeric attributes (salary, commission, age, etc.)
    and assigns a binary class based on one of 10 classification functions.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    function : int, default=1
        Classification function (1-10).
    perturbation : float, default=0.05
        Amount of noise to add (0.0-1.0).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes generated:
        - salary: 20000-150000
        - commission: 0-75000
        - age: 20-80
        - education_level: 0-4
        - car: 1-20
        - zipcode: 9 values
        - house_value: 50000-1000000
        - years_house: 0-30
        - loan: 0-500000

    Examples
    --------
    >>> gen = Agrawal(n_samples=1000, function=1)
    >>> data = gen.generate()
    """

    FEATURE_NAMES = [
        'salary', 'commission', 'age', 'education_level', 'car',
        'zipcode', 'house_value', 'years_house', 'loan'
    ]

    def __init__(
        self,
        n_samples: int = 100,
        function: int = 1,
        perturbation: float = 0.05,
        random_state: Optional[int] = None
    ):
        """Initialize Agrawal generator.

        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate.
        function : int, default=1
            Classification function index (1-10).
        perturbation : float, default=0.05
            Probability of flipping the class label.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__(n_samples, n_features=9, n_classes=2, random_state=random_state)
        self.function = max(1, min(10, function))
        self.perturbation = perturbation

    def generate(self) -> GeneratedData:
        """Generate Agrawal data.

        Returns
        -------
        GeneratedData
            Generated dataset with feature array X of shape
            (n_samples, 9) and binary class labels y.
        """
        rng = self._rng

        X = np.zeros((self.n_samples, 9))
        y = np.zeros(self.n_samples, dtype=int)

        for i in range(self.n_samples):
            # Generate attributes
            salary = rng.uniform(20000, 150000)
            commission = 0 if salary >= 75000 else rng.uniform(10000, 75000)
            age = rng.integers(20, 81)
            education = rng.integers(0, 5)
            car = rng.integers(1, 21)
            zipcode = rng.integers(0, 9)
            house_value = rng.uniform(50000, 1000000) * (zipcode + 1) / 5
            years_house = rng.integers(0, 31)
            loan = rng.uniform(0, 500000)

            X[i] = [salary, commission, age, education, car,
                    zipcode, house_value, years_house, loan]

            # Determine class based on function
            y[i] = self._classify(X[i])

            # Add perturbation (flip label with probability)
            if rng.uniform() < self.perturbation:
                y[i] = 1 - y[i]

        return GeneratedData(
            X=X,
            y=y,
            feature_names=self.FEATURE_NAMES.copy(),
            target_names=['group_A', 'group_B']
        )

    def _classify(self, x: np.ndarray) -> int:
        """Apply the selected classification function to a single instance.

        Parameters
        ----------
        x : np.ndarray
            Feature vector of length 9.

        Returns
        -------
        int
            Binary class label (0 or 1).
        """
        salary, commission, age, education, car, zipcode, house_value, years_house, loan = x

        if self.function == 1:
            return 1 if age < 40 or age >= 60 else 0
        elif self.function == 2:
            return 1 if age < 40 and salary < 100000 else 0
        elif self.function == 3:
            return 1 if age < 40 or salary < 100000 else 0
        elif self.function == 4:
            return 1 if (age < 40 and salary < 100000) or (age >= 40 and salary >= 100000) else 0
        elif self.function == 5:
            return 1 if loan > house_value else 0
        elif self.function == 6:
            return 1 if salary - loan < 50000 else 0
        elif self.function == 7:
            disposable = 2 * (salary + commission) / 3 - loan / 5 - 20000
            return 1 if disposable > 0 else 0
        elif self.function == 8:
            return 1 if age >= 40 and loan > 200000 else 0
        elif self.function == 9:
            return 1 if years_house > 20 and loan > house_value / 2 else 0
        else:  # function == 10
            return 1 if salary + commission > 100000 and house_value > 500000 else 0

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        schema = ClassificationGenerator.get_parameter_schema()
        schema["function"] = {
            "type": "integer",
            "default": 1,
            "minimum": 1,
            "maximum": 10,
            "description": "Classification function (1-10)"
        }
        schema["perturbation"] = {
            "type": "number",
            "default": 0.05,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Noise level (probability of label flip)"
        }
        # Remove n_features as it's fixed
        del schema["n_features"]
        return schema
