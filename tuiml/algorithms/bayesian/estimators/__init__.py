"""Probability estimators for Naive Bayes.

A Naive Bayes classifier needs :math:`P(x_i \\mid c)` for every feature.
These estimators are the pluggable pieces that supply it, which is what lets
one classifier handle numeric and categorical features by swapping the
estimator rather than the algorithm.

Estimators
----------
- **NormalEstimator:** Fits a Gaussian per feature and class. The default for
  numeric data, and the right choice when a feature is roughly bell-shaped.
- **KernelEstimator:** Kernel density estimation. Slower, but assumes no
  particular distribution — use it for skewed or multi-modal features.
- **DiscreteEstimator:** Smoothed counts, for categorical or already
  discretised features.
- **Estimator:** The base class to subclass for a new one.

See Also
--------
:mod:`tuiml.algorithms.bayesian` : The classifiers these plug into.
"""

from tuiml.base.estimators import Estimator
from .normal import NormalEstimator
from .discrete import DiscreteEstimator
from .kernel import KernelEstimator

__all__ = [
    "Estimator",
    "NormalEstimator",
    "DiscreteEstimator",
    "KernelEstimator"
]
