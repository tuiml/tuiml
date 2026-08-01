"""Base class for probability estimators.

Estimators model the probability distribution of a single variable and
support incremental (``add_value``) as well as batch (``fit``) updates.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, List

class Estimator(ABC):
    """Abstract base class for all probability estimators.

    Estimators are used to model the probability distribution of a variable.
    """

    def __init__(self):
        """Initialize the estimator."""
        self._is_fitted = False

    @abstractmethod
    def add_value(self, value: float, weight: float = 1.0) -> None:
        """Add a value to the estimator.

        Parameters
        ----------
        value : float
            The value to add.
        weight : float, default=1.0
            The weight of the value.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def get_probability(self, value: float) -> float:
        """Get the probability of a value given the estimated distribution.

        Parameters
        ----------
        value : float
            The value to get the probability for.

        Returns
        -------
        probability : float
            Probability density or mass at ``value``.
        """
        pass

    def add_values(self, values: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """Add multiple values to the estimator.

        Parameters
        ----------
        values : np.ndarray of shape (n_values,)
            Array of values to add.
        weights : np.ndarray of shape (n_values,), optional
            Weight for each value. Defaults to uniform weights of 1.0.

        Returns
        -------
        None
        """
        if weights is None:
            weights = np.ones(len(values))

        for v, w in zip(values, weights):
            self.add_value(v, w)

    def fit(self, values: np.ndarray, weights: Optional[np.ndarray] = None) -> "Estimator":
        """Fit the estimator to data (batch mode).

        Parameters
        ----------
        values : np.ndarray of shape (n_values,)
            Array of values to fit on.
        weights : np.ndarray of shape (n_values,), optional
            Weight for each value. Defaults to uniform weights of 1.0.

        Returns
        -------
        self : Estimator
            The fitted estimator, for method chaining.
        """
        self.add_values(values, weights)
        self._is_fitted = True
        return self
