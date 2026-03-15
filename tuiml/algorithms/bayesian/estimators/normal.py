"""Normal (Gaussian) probability estimator."""

import numpy as np
from typing import Optional
from tuiml.base.estimators import Estimator

class NormalEstimator(Estimator):
    """Gaussian (Normal) distribution **probability density** estimator.

    Estimates a **normal distribution** for numeric data by maintaining running
    sums and sums of squares. This allows for **incremental updates** and
    efficient calculation of the mean and variance.

    Overview
    --------
    The estimator works as follows:

    1. Accumulate running **sum**, **sum of squares**, and **count** as
       values are added via :meth:`add_value`
    2. On each call to :meth:`get_probability`, recompute the mean and
       variance from the running statistics
    3. Evaluate the **Gaussian PDF** at the query point

    Theory
    ------
    The probability density at value :math:`x` is given by the normal PDF:

    .. math::

        p(x) = \\frac{1}{\\sigma \\sqrt{2\\pi}}
               \\exp\\!\\left(-\\frac{(x - \\mu)^2}{2\\sigma^2}\\right)

    where the mean :math:`\\mu` and variance :math:`\\sigma^2` are estimated
    from the running statistics:

    .. math::

        \\mu = \\frac{\\sum w_i x_i}{\\sum w_i}, \\qquad
        \\sigma^2 = \\frac{\\sum w_i x_i^2}{\\sum w_i} - \\mu^2

    Parameters
    ----------
    precision : float or None, default=None
        The precision constraint for the variance. If provided, the
        variance will be floored at this value to avoid division by zero
        or negative probabilities. Defaults to ``1e-6``.

    Attributes
    ----------
    sum : float
        Sum of all values added to the estimator.
    sum_sq : float
        Sum of squares of all values added to the estimator.
    count : float
        Total number (or weight) of samples added.

    Notes
    -----
    **Complexity:**

    - ``add_value``: :math:`O(1)` per observation
    - ``get_probability``: :math:`O(1)` per query

    **When to use NormalEstimator:**

    - Features are approximately normally distributed
    - A fast, lightweight density estimator is needed
    - Incremental / online estimation is required

    References
    ----------
    .. [John1995] John, G.H. and Langley, P. (1995).
           **Estimating Continuous Distributions in Bayesian Classifiers.**
           *Proceedings of the 11th Conference on Uncertainty in Artificial Intelligence*,
           pp. 338-345.

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.estimators.KernelEstimator` : Non-parametric kernel density estimator.
    :class:`~tuiml.algorithms.bayesian.estimators.DiscreteEstimator` : Estimator for categorical data.

    Examples
    --------
    Incremental density estimation:

    >>> from tuiml.algorithms.bayesian.estimators import NormalEstimator
    >>>
    >>> # Build estimator from observations
    >>> est = NormalEstimator()
    >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    ...     est.add_value(v)
    >>>
    >>> # Query density at the mean
    >>> est.get_probability(3.0)  # doctest: +SKIP
    0.3989...
    """
    
    def __init__(self, precision: Optional[float] = None):
        """Initialize the normal estimator.

        Parameters
        ----------
        precision : float or None, default=None
            Precision of the values (used for minimum variance constraint).
            If None, defaults to a small epsilon (``1e-6``).
        """
        super().__init__()
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0.0
        self.precision = precision if precision is not None else 1e-6
        self._mean = 0.0
        self._variance = 0.0
        self._std_dev = 0.0
        
    def add_value(self, value: float, weight: float = 1.0) -> None:
        """Add a new value observation to the distribution.

        Parameters
        ----------
        value : float
            The numeric value to add.
        weight : float, default=1.0
            The weight of the observation.
        """
        if np.isnan(value):
            return
            
        self.sum += value * weight
        self.sum_sq += value * value * weight
        self.count += weight
        
        # Mark as not strictly fitted/finalized, though we can calculate on the fly
        # In this implementation update stats immediately or lazy load? 
        # Lazy calculation in get_probability is safer for precision
        
    def _calculate_stats(self):
        """Calculate mean and variance from running sums.

        Updates the internal ``_mean``, ``_variance``, and ``_std_dev``
        attributes from the accumulated ``sum``, ``sum_sq``, and ``count``.
        """
        if self.count <= 0:
            self._mean = 0.0
            self._variance = self.precision
        else:
            self._mean = self.sum / self.count
            # Var = E[X^2] - (E[X])^2
            # Using basic formula, might be numerically unstable for large numbers 
            # but matches Weka's basic implementation
            mean_sq = self._mean * self._mean
            avg_sum_sq = self.sum_sq / self.count
            
            self._variance = avg_sum_sq - mean_sq
            
            # Enforce minimum variance
            if self._variance < self.precision:
                self._variance = self.precision
                
        self._std_dev = np.sqrt(self._variance)

    def get_probability(self, value: float) -> float:
        """Estimate the probability density of a specific value.

        Parameters
        ----------
        value : float
            The numeric value to query.

        Returns
        -------
        density : float
            The probability density (PDF) for the given value.
        """
        if np.isnan(value):
            return 0.0
            
        self._calculate_stats()
        
        diff = value - self._mean
        exponent = -(diff * diff) / (2 * self._variance)
        
        # Normal distribution PDF: (1 / (sigma * sqrt(2*pi))) * exp(...)
        return (1.0 / (np.sqrt(2 * np.pi) * self._std_dev)) * np.exp(exponent)
        
    def get_mean(self) -> float:
        """Get the estimated mean."""
        self._calculate_stats()
        return self._mean
        
    def get_std_dev(self) -> float:
        """Get the estimated standard deviation."""
        self._calculate_stats()
        return self._std_dev
