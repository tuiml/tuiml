"""Time series analysis and forecasting algorithms.

Models for data where order matters and observations are not independent.
Unlike the rest of :mod:`tuiml.algorithms`, these learn from a sequence's own
history rather than from a feature matrix.

Algorithms
----------
- **AR:** Autoregressive — the next value from its own past values.
- **MA:** Moving average — the next value from past forecast errors.
- **ARMA:** Autoregressive moving average, for stationary series.
- **ARIMA:** ARMA plus differencing, for series with a trend.
- **ExponentialSmoothing:** Weighted average with exponentially decaying
  weights; handles level, trend and seasonality.
- **Prophet:** Additive model with trend, seasonality and holiday terms.
- **STLDecomposition:** Splits a series into trend, seasonal and residual
  components.

Classification
--------------
:mod:`tuiml.algorithms.timeseries.classification` is a separate task: it
labels a *whole series* by its shape rather than forecasting its next value.
See :class:`~tuiml.algorithms.timeseries.classification.DTWNeighborsClassifier`.

Notes
-----
These subclass :class:`~tuiml.base.algorithms.Regressor`, not ``Classifier``:
forecasting predicts a continuous value. Ordinary cross-validation shuffles
observations and leaks the future into the past, so evaluate with
:class:`~tuiml.evaluation.splitting.TimeSeriesSplit`, which only ever trains
on data preceding the test fold.
"""

# Base classes (single source of truth)
from tuiml.base.algorithms import Regressor, Classifier, regressor, classifier

# Classical ARIMA family
from tuiml.algorithms.timeseries.ar import AR
from tuiml.algorithms.timeseries.ma import MA
from tuiml.algorithms.timeseries.arma import ARMA
from tuiml.algorithms.timeseries.arima import ARIMA

# Other forecasting methods
from tuiml.algorithms.timeseries.exponential_smoothing import ExponentialSmoothing
from tuiml.algorithms.timeseries.prophet import Prophet

# Decomposition
from tuiml.algorithms.timeseries.stl_decomposition import STLDecomposition

# Whole-series classification. Imported here so the registry, which discovers
# components by importing tuiml.algorithms, sees these too.
from tuiml.algorithms.timeseries.classification import (
    DTWNeighborsClassifier,
    MiniRocketClassifier,
    MiniRocketTransformer,
    ShapeletTransformClassifier,
    TimeSeriesClassifier,
)

__all__ = [
    # Base classes
    "Regressor",
    "Classifier",
    "regressor",
    "classifier",
    # ARIMA family
    "AR",
    "MA",
    "ARMA",
    "ARIMA",
    # Other forecasting
    "ExponentialSmoothing",
    "Prophet",
    # Decomposition
    "STLDecomposition",
    "TimeSeriesClassifier",
    "DTWNeighborsClassifier",
    "MiniRocketClassifier",
    "MiniRocketTransformer",
    "ShapeletTransformClassifier",
]
