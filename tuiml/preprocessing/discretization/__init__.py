"""Binning continuous features into discrete intervals.

Turning a numeric column into a small number of bins. This is what algorithms
that expect categorical input need, and it can help elsewhere by letting a
model express a non-linear relationship it could not otherwise fit.

Transformers
------------
- **EqualWidthDiscretizer:** Bins of equal range. Simple, but a skewed
  feature leaves most bins nearly empty.
- **QuantileDiscretizer:** Bins holding equal numbers of samples. Robust to
  skew, at the cost of uneven bin widths.
- **MDLDiscretizer:** Chooses cut points by minimum description length,
  using the target to place boundaries where the class actually changes, and
  deciding the number of bins for you.

Notes
-----
MDL is supervised: it reads ``y``. Fit it on the training split only —
fitting on everything chooses boundaries informed by the test labels and
inflates the score.

Discretising always discards information. It is worth it when the algorithm
requires it or the relationship is genuinely non-monotonic, not by default.
"""

from tuiml.preprocessing.discretization.equal_width import EqualWidthDiscretizer
from tuiml.preprocessing.discretization.equal_frequency import QuantileDiscretizer
from tuiml.preprocessing.discretization.mdl import MDLDiscretizer

__all__ = [
    "EqualWidthDiscretizer",
    "QuantileDiscretizer",
    "MDLDiscretizer",
]
