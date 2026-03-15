"""Kernel density estimator."""

import numpy as np
from typing import List, Optional
from tuiml.base.estimators import Estimator

class KernelEstimator(Estimator):
    """Gaussian **Kernel Density Estimator** (KDE) for non-parametric density estimation.

    Provides a **non-parametric** estimation of the probability density function
    using a sum of **Gaussian kernels** centred at each data point. This is
    advantageous for **multimodal** or non-normal distributions where a simple
    parametric model (like Gaussian) would fail.

    Overview
    --------
    The estimator works as follows:

    1. Store every observed value along with its weight via :meth:`add_value`
    2. Compute the **bandwidth** (kernel width) using Silverman's rule of thumb
    3. At query time, sum the Gaussian kernel contributions from all stored
       points and normalise by total weight and bandwidth

    Theory
    ------
    The kernel density estimate at point :math:`x` is:

    .. math::

        \\hat{f}(x) = \\frac{1}{n\\,h} \\sum_{i=1}^{n} w_i \\, K\\!\\left(\\frac{x - x_i}{h}\\right)

    where :math:`K` is the Gaussian kernel:

    .. math::

        K(u) = \\frac{1}{\\sqrt{2\\pi}} \\exp\\!\\left(-\\tfrac{1}{2} u^2\\right)

    The **bandwidth** :math:`h` is selected using Silverman's rule of thumb:

    .. math::

        h = \\hat{\\sigma} \\cdot n^{-1/5}

    where :math:`\\hat{\\sigma}` is the sample standard deviation and :math:`n`
    is the number of observations.

    Parameters
    ----------
    precision : float or None, default=None
        The precision constraint for bandwidth calculation and rounding.
        Defaults to ``1e-6``.

    Attributes
    ----------
    values : list of float
        The list of values added to the estimator.
    weights : list of float
        The weights corresponding to each value.
    total_weight : float
        Sum of all weights added.
    standard_deviation : float
        The calculated bandwidth (h) for the kernels. A value of ``-1.0``
        indicates it needs to be recalculated.

    Notes
    -----
    **Complexity:**

    - ``add_value``: :math:`O(1)` per observation
    - ``get_probability``: :math:`O(n)` per query where :math:`n` is the number of stored values

    **When to use KernelEstimator:**

    - Feature distributions are multimodal or strongly non-Gaussian
    - More flexible density estimation is needed at the cost of speed
    - The number of training samples per class is relatively small

    References
    ----------
    .. [Silverman1986] Silverman, B.W. (1986).
           **Density Estimation for Statistics and Data Analysis.**
           *Chapman and Hall*, London.

    .. [John1995] John, G.H. and Langley, P. (1995).
           **Estimating Continuous Distributions in Bayesian Classifiers.**
           *Proceedings of the 11th Conference on Uncertainty in Artificial Intelligence*,
           pp. 338-345.

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.estimators.NormalEstimator` : Parametric Gaussian estimator (faster, less flexible).
    :class:`~tuiml.algorithms.bayesian.estimators.DiscreteEstimator` : Estimator for categorical data.

    Examples
    --------
    Non-parametric density estimation:

    >>> from tuiml.algorithms.bayesian.estimators import KernelEstimator
    >>>
    >>> # Build estimator from bimodal data
    >>> est = KernelEstimator()
    >>> for v in [1.0, 1.1, 1.2, 5.0, 5.1, 5.2]:
    ...     est.add_value(v)
    >>>
    >>> # Query density at each mode
    >>> est.get_probability(1.1)  # doctest: +SKIP
    0.25...
    """
    
    def __init__(self, precision: Optional[float] = None):
        """Initialize the kernel estimator.

        Parameters
        ----------
        precision : float or None, default=None
            Precision for rounding (grid resolution). Defaults to ``1e-6``.
        """
        super().__init__()
        self.values: List[float] = []
        self.weights: List[float] = []
        self.total_weight = 0.0
        self.precision = precision if precision is not None else 1e-6
        self.standard_deviation = -1.0 # Calculated on demand
        self.all_weights_one = True
        
        # Max number of points to keep (Weka optimization default is usually 50?? No, Weka keeps all unless merging)
        # Weka's KernelEstimator actually does a smart merge to keep size manageable.
        # For simplicity here, we keep all points (exact KDE) but this may be slow for large datasets.
        # Improvement: Add simple merging strategy if list grows too large.
        
    def add_value(self, value: float, weight: float = 1.0) -> None:
        """Add a new value observation to the distribution.

        Parameters
        ----------
        value : float
            The numeric value to add.
        weight : float, default=1.0
            The weight or frequency of the observation.
        """
        if np.isnan(value):
            return
            
        self.values.append(value)
        self.weights.append(weight)
        self.total_weight += weight
        
        if weight != 1.0:
            self.all_weights_one = False
            
        # Invalidate standard deviation cache
        self.standard_deviation = -1.0
        
    def _calculate_std_dev(self):
        """Calculate the bandwidth (standard deviation) for the kernels.

        Uses Silverman's rule of thumb: :math:`h = \\hat{\\sigma} \\cdot n^{-1/5}`.
        Updates the ``standard_deviation`` attribute in-place.
        """
        if self.total_weight == 0:
            self.standard_deviation = self.precision
            return

        # Calculate weighted mean
        mean = sum(v * w for v, w in zip(self.values, self.weights)) / self.total_weight
        
        # Calculate weighted variance
        variance = sum(w * (v - mean)**2 for v, w in zip(self.values, self.weights)) / self.total_weight
        
        # Heuristic for bandwidth: ~ n^(-1/5) * sigma
        # Weka uses: 1.06 * min(sigma, IQR/1.34) * n^(-1/5) 
        # Simplified here:
        population_std = np.sqrt(variance)
        if population_std < self.precision:
            self.standard_deviation = self.precision
        else:
            # Bandwidth selection (Silverman's rule of thumb variant)
            n = len(self.values)
            self.standard_deviation = max(population_std * (n ** -0.2), self.precision)

    def get_probability(self, value: float) -> float:
        """Estimate the probability density at a given value.

        Calculated as the sum of Gaussian kernels centered at each training 
        point, normalized by the bandwidth and total weight.

        Parameters
        ----------
        value : float
            The numeric value for which to calculate the density.

        Returns
        -------
        density : float
            The estimated probability density (PDF) at the given point.
        """
        if self.total_weight <= 0:
            return 0.0
            
        if self.standard_deviation < 0:
            self._calculate_std_dev()
            
        # P(x) = (1/n) * sum( K((x - xi)/h) / h )
        # where K is Gaussian kernel: (1/sqrt(2pi)) * exp(-0.5 * u^2)
        
        prob_sum = 0.0
        inverse_bandwidth = 1.0 / self.standard_deviation
        normalization = 1.0 / (np.sqrt(2 * np.pi) * self.total_weight * self.standard_deviation)
        
        for v, w in zip(self.values, self.weights):
            diff = (value - v) * inverse_bandwidth
            # Optimization: ignore contributions from far away points (e.g. > 3 std devs)
            # if abs(diff) > 3: continue
            
            kernel_val = np.exp(-0.5 * diff * diff)
            prob_sum += w * kernel_val
            
        return prob_sum * normalization
