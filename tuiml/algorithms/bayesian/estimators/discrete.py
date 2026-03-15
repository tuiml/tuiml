"""Discrete probability estimator."""

import numpy as np
from typing import Dict, Optional, Union
from tuiml.base.estimators import Estimator

class DiscreteEstimator(Estimator):
    """Discrete probability estimator for **categorical** data.

    Estimates a **discrete probability distribution** (PMF) by counting the
    frequency of occurrences. Supports **additive (Laplace) smoothing** to
    handle new or rare symbols.

    Overview
    --------
    The estimator works as follows:

    1. Maintain a **count array** of size ``num_symbols``
    2. On each call to :meth:`add_value`, increment the count for the
       given symbol index by the specified weight
    3. On each call to :meth:`get_probability`, return the (optionally
       smoothed) relative frequency of the queried symbol

    Theory
    ------
    Without smoothing, the probability of symbol :math:`k` is the
    **maximum likelihood estimate**:

    .. math::

        \\hat{P}(k) = \\frac{n_k}{N}

    With **Laplace smoothing**, the estimate becomes:

    .. math::

        \\hat{P}(k) = \\frac{n_k + 1}{N + K}

    where :math:`n_k` is the count for symbol :math:`k`, :math:`N` is the
    total count, and :math:`K` is the number of possible symbols.

    Parameters
    ----------
    num_symbols : int
        The number of possible symbols or discrete states in the distribution.
    laplace : bool, default=True
        Whether to use Laplace smoothing (add-one smoothing). If True,
        probabilities are computed as ``(count + 1) / (total + num_symbols)``.

    Attributes
    ----------
    counts : np.ndarray of shape (num_symbols,)
        The raw counts (frequency) for each symbol index.
    total_count : float
        The aggregate count (sum of weights) of all samples added.

    Notes
    -----
    **Complexity:**

    - ``add_value``: :math:`O(1)` per observation
    - ``get_probability``: :math:`O(1)` per query

    **When to use DiscreteEstimator:**

    - Features are categorical or have been discretised into bins
    - Estimating conditional probability tables in Bayesian networks
    - A simple, fast frequency-based estimator is sufficient

    References
    ----------
    .. [Cestnik1990] Cestnik, B. (1990).
           **Estimating Probabilities: A Crucial Task in Machine Learning.**
           *Proceedings of the 9th European Conference on Artificial Intelligence*,
           pp. 147-149.

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.estimators.NormalEstimator` : Parametric Gaussian estimator for continuous data.
    :class:`~tuiml.algorithms.bayesian.estimators.KernelEstimator` : Non-parametric kernel density estimator.

    Examples
    --------
    Frequency-based probability estimation with Laplace smoothing:

    >>> from tuiml.algorithms.bayesian.estimators import DiscreteEstimator
    >>>
    >>> # Estimator for 3 possible symbols
    >>> est = DiscreteEstimator(num_symbols=3, laplace=True)
    >>> est.add_value(0)
    >>> est.add_value(0)
    >>> est.add_value(1)
    >>>
    >>> # Query probabilities (smoothed)
    >>> est.get_probability(0)  # doctest: +SKIP
    0.5
    """
    
    def __init__(self, num_symbols: int, laplace: bool = True):
        """Initialize the discrete estimator.

        Parameters
        ----------
        num_symbols : int
            Number of possible symbols/values.
        laplace : bool, default=True
            Whether to use Laplace smoothing (add-1 smoothing).
        """
        super().__init__()
        self.num_symbols = num_symbols
        self.laplace = laplace
        
        # Initialize counts
        # If laplace, we effectively start with count 1 for each symbol
        self.counts = np.zeros(num_symbols)
        self.total_count = 0.0
        
    def add_value(self, value: float, weight: float = 1.0) -> None:
        """Add a new value observation to the distribution.

        Parameters
        ----------
        value : float
            The index of the symbol to add (must be between ``0`` and 
            ``num_symbols - 1``). Non-integer values will be truncated.
        weight : float, default=1.0
            The weight or frequency count of the observation.
        """
        idx = int(value)
        if idx < 0 or idx >= self.num_symbols:
            # Silently ignore out of bounds (or could raise error)
            return
            
        self.counts[idx] += weight
        self.total_count += weight
        
    def get_probability(self, value: float) -> float:
        """Estimate the probability of a specific symbol indicator.

        Parameters
        ----------
        value : float
            The index of the symbol to query.

        Returns
        -------
        prob : float
            The estimated probability mass of the symbol. Returns 0.0 if 
            the index is out of bounds.
        """
        idx = int(value)
        if idx < 0 or idx >= self.num_symbols:
            return 0.0
            
        current_count = self.counts[idx]
        total = self.total_count
        
        if self.laplace:
            # (count + 1) / (total + num_symbols)
            return (current_count + 1.0) / (total + self.num_symbols)
        else:
            if total == 0:
                return 0.0
            return current_count / total
            
    def get_count(self, value: int) -> float:
        """Get the raw count for a symbol."""
        if 0 <= value < self.num_symbols:
            return self.counts[value]
        return 0.0
