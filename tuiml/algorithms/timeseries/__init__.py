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
- **SARIMAX:** Seasonal ARIMA with exogenous regressors, estimated by exact
  Gaussian maximum likelihood through a Kalman filter. Unlike ``ARIMA`` it
  really does fit the seasonal and moving-average terms, and it supports
  regressors and forecast intervals.
- **VAR:** Vector autoregression — several series predicted jointly from the
  lagged history of all of them.
- **ThetaForecaster:** The Theta method; equivalent to simple exponential
  smoothing with half the series' OLS drift, with optional deseasonalisation.
- **TBATS:** Exponential smoothing with trigonometric seasonality, which is
  what lets it handle high-frequency and *non-integer* seasonal periods
  (365.25) that seasonal ARIMA cannot.
- **CrostonForecaster:** Intermittent demand — smooths demand sizes and
  inter-arrival intervals separately.
- **STLDecomposition:** Splits a series into trend, seasonal and residual
  components.

Deep forecasting
----------------
:mod:`tuiml.algorithms.timeseries.deep` holds ``NBEATSForecaster``,
``NHITSForecaster`` and ``PatchTSTForecaster``. They are native TuiML
implementations, but they need a tensor library with autograd, so fitting one
requires ``pip install 'tuiml[torch]'``. Importing them never imports torch and
constructing one never requires it — only ``fit`` does, and it raises a clear
``ImportError`` naming the install command.

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

# State-space and specialised forecasters
from tuiml.algorithms.timeseries.sarimax import SARIMAX
from tuiml.algorithms.timeseries.var import VAR
from tuiml.algorithms.timeseries.theta import ThetaForecaster
from tuiml.algorithms.timeseries.tbats import TBATS
from tuiml.algorithms.timeseries.croston import CrostonForecaster

# Deep forecasters. These need ``pip install 'tuiml[torch]'`` to *fit*, but the
# import below never pulls torch in: the classes are defined and registered on
# every install so the catalog is identical everywhere, and the dependency is
# only checked when ``fit`` is called.
from tuiml.algorithms.timeseries.deep import (
    NBEATSForecaster,
    NHITSForecaster,
    PatchTSTForecaster,
)

# Whole-series classification. Imported here so the registry, which discovers
# components by importing tuiml.algorithms, sees these too.
from tuiml.algorithms.timeseries.classification import (
    BOSSClassifier,
    DTWNeighborsClassifier,
    HIVECOTEClassifier,
    TimeSeriesForestClassifier,
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
    # State-space and specialised forecasting
    "SARIMAX",
    "VAR",
    "ThetaForecaster",
    "TBATS",
    "CrostonForecaster",
    # Deep forecasting (needs tuiml[torch] to fit)
    "NBEATSForecaster",
    "NHITSForecaster",
    "PatchTSTForecaster",
    # Decomposition
    "STLDecomposition",
    "TimeSeriesClassifier",
    "DTWNeighborsClassifier",
    "MiniRocketClassifier",
    "MiniRocketTransformer",
    "ShapeletTransformClassifier",
    "BOSSClassifier",
    "TimeSeriesForestClassifier",
    "HIVECOTEClassifier",
]
