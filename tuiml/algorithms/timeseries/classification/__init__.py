"""Time-series classification.

Classifying a *whole series* by its shape, rather than forecasting its next
value. Input is a panel of shape ``(n_samples, n_channels, n_timepoints)``;
a 2-D array is read as univariate.

This is a different task from everything in the parent
:mod:`tuiml.algorithms.timeseries` package, which forecasts. It is also
different from ordinary classification: flattening a series into columns and
handing it to a feature-matrix classifier discards the time ordering that
carries the signal.

Algorithms
----------
- **MiniRocketClassifier:** 84 fixed dilated kernels summarised by proportion
  of positive values, then a linear head. Near state-of-the-art accuracy at a
  tiny fraction of the cost, and the right default at any real scale.
- **HIVECOTEClassifier:** Combines all of the below, weighting each by its own
  cross-validated accuracy. The most accurate and by far the most expensive.
- **TimeSeriesForestClassifier:** Mean, standard deviation and slope of random
  intervals, in a forest. The temporally localised option: it says *where* in
  the series the difference lives.
- **BOSSClassifier:** Turns each series into a bag of symbolic words built
  from low-frequency Fourier coefficients. The noise-tolerant option: the
  low-pass step discards exactly the detail noise lives in.
- **ShapeletTransformClassifier:** Finds the short subsequences that separate
  the classes, and represents each series by its distance to them. The
  interpretable option: the fitted shapelets are real subsequences you can
  plot and read.
- **DTWNeighborsClassifier:** Nearest neighbour under Dynamic Time Warping.
  The standard baseline of the field, and still hard to beat — but its cost
  grows with the training set, which MINIROCKET's does not.

Transforms
----------
- **MiniRocketTransformer:** The MINIROCKET features without a classifier, for
  pipelines and non-classification uses.

Distances
---------
- **dtw_distance / dtw_pairwise:** Elastic alignment that tolerates stretching
  and compression of the time axis.
- **lb_keogh / lb_keogh_envelope:** The cheap lower bound that makes
  nearest-neighbour search under DTW affordable.

Notes
-----
Z-normalise each series before using an elastic distance unless the absolute
level is genuinely meaningful — otherwise DTW largely measures offset. A
Sakoe-Chiba band of roughly 10% of the series length is the usual starting
point: it speeds the computation up and typically improves accuracy by
forbidding degenerate alignments.
"""

from tuiml.algorithms.timeseries.classification._base import (
    TimeSeriesClassifier,
    as_panel,
)
from tuiml.algorithms.timeseries.classification.distance import (
    dtw_distance,
    dtw_pairwise,
    lb_keogh,
    lb_keogh_envelope,
)
from tuiml.algorithms.timeseries.classification.knn import DTWNeighborsClassifier
from tuiml.algorithms.timeseries.classification.dictionary import BOSSClassifier
from tuiml.algorithms.timeseries.classification.hive_cote import HIVECOTEClassifier
from tuiml.algorithms.timeseries.classification.interval import (
    TimeSeriesForestClassifier,
)
from tuiml.algorithms.timeseries.classification.shapelets import (
    ShapeletTransformClassifier,
)
from tuiml.algorithms.timeseries.classification.rocket import (
    MiniRocketClassifier,
    MiniRocketTransformer,
)

__all__ = [
    "TimeSeriesClassifier",
    "as_panel",
    "DTWNeighborsClassifier",
    "MiniRocketClassifier",
    "MiniRocketTransformer",
    "ShapeletTransformClassifier",
    "BOSSClassifier",
    "TimeSeriesForestClassifier",
    "HIVECOTEClassifier",
    "dtw_distance",
    "dtw_pairwise",
    "lb_keogh",
    "lb_keogh_envelope",
]
