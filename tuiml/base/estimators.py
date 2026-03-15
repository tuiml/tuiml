"""
Base class for probability estimators.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, List

class Estimator(ABC):
    """
    Abstract base class for all estimators.

    Estimators are used to model the probability distribution of a variable.
    """

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def add_value(self, value: float, weight: float = 1.0) -> None:
        """
        Add a value to the estimator.

        Args:
            value: The value to add
            weight: The weight of the value (default 1.0)
        """
        pass

    @abstractmethod
    def get_probability(self, value: float) -> float:
        """
        Get the probability of a value given the estimated distribution.

        Args:
            value: The value to get probability for

        Returns:
            Probability density or mass
        """
        pass

    def add_values(self, values: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """
        Add multiple values to the estimator.

        Args:
            values: Array of values
            weights: Optional array of weights
        """
        if weights is None:
            weights = np.ones(len(values))

        for v, w in zip(values, weights):
            self.add_value(v, w)

    def fit(self, values: np.ndarray, weights: Optional[np.ndarray] = None) -> "Estimator":
        """
        Fit the estimator to data (batch mode).

        Args:
            values: Array of values
            weights: Optional array of weights

        Returns:
            Self for method chaining
        """
        self.add_values(values, weights)
        self._is_fitted = True
        return self
