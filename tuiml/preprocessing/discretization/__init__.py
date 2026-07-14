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

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

if SKLEARN_AVAILABLE:
    from tuiml.preprocessing.discretization.sklearn_kbins import SklearnKBinsDiscretizer

__all__ = [
    "EqualWidthDiscretizer",
    "QuantileDiscretizer",
    "MDLDiscretizer",
]

if SKLEARN_AVAILABLE:
    __all__.extend([
        "SklearnKBinsDiscretizer",
    ])

