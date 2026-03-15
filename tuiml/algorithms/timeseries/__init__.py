"""Time series analysis and forecasting algorithms."""

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
]
