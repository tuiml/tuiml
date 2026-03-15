"""
Discretization transformers for binning continuous features.

Available:
    - EqualWidthDiscretizer: Equal-width binning (WEKA: Discretize)
    - QuantileDiscretizer: Equal-frequency/quantile binning (WEKA: Discretize)
    - MDLDiscretizer: MDL-based discretization (WEKA: Discretize)
"""

from tuiml.preprocessing.discretization.equal_width import EqualWidthDiscretizer
from tuiml.preprocessing.discretization.equal_frequency import QuantileDiscretizer
from tuiml.preprocessing.discretization.mdl import MDLDiscretizer

__all__ = [
    "EqualWidthDiscretizer",
    "QuantileDiscretizer",
    "MDLDiscretizer",
]
