"""
Time series preprocessing module for tuiml.

Provides transformers for time series data:
- LagTransformer: Create lagged features (WEKA TimeSeriesTranslate equivalent)
- DifferenceTransformer: Create difference features (WEKA TimeSeriesDelta equivalent)
- weka.filters.unsupervised.attribute.TimeSeriesTranslate -> LagTransformer
- weka.filters.unsupervised.attribute.TimeSeriesDelta -> DifferenceTransformer

Usage:
    >>> from tuiml.preprocessing.timeseries import LagTransformer, DifferenceTransformer
    >>>
    >>> # Create lagged features
    >>> lag = LagTransformer(lag=1, columns=[0, 1])
    >>> X_lagged = lag.fit_transform(X)
    >>>
    >>> # Create difference features
    >>> delta = DifferenceTransformer(lag=1, columns=[0])
    >>> X_diff = delta.fit_transform(X)
"""

from tuiml.preprocessing.timeseries.lag import LagTransformer
from tuiml.preprocessing.timeseries.delta import DifferenceTransformer

__all__ = [
    "LagTransformer",
    "DifferenceTransformer",
]
