"""Turning a time series into features a model can learn from.

A forecasting model needs the past available as columns: what the value was
some steps ago, and how much it has changed since. These transformers shift
the selected columns in place, which is what lets an ordinary regressor be
applied to sequential data.

Transformers
------------
- **LagTransformer:** Replaces a column with its value ``lag`` steps away, so
  each row carries history instead of only the present.
- **DifferenceTransformer:** Replaces it with the change over ``lag`` steps.
  Differencing also removes a trend, which is what makes a drifting series
  stationary enough to model.

Notes
-----
Mind the sign: ``lag=-1`` looks **backwards**, at the previous value, and is
almost always what you want. A positive ``lag`` looks *forward*, feeding the
model a value from the future — which trains beautifully and predicts
nothing. Both transformers default to ``lag=-1``.

Shifting leaves boundary rows with no value to draw on. By default those are
filled with ``NaN`` and the row count is preserved; pass
``fill_with_missing=False`` to drop them instead.

Evaluate with :class:`~tuiml.evaluation.splitting.TimeSeriesSplit`. Lagged
features make a shuffled split leak outright: a training row can hold the
very value a test row is being asked to predict.

Examples
--------
>>> import numpy as np
>>> from tuiml.preprocessing.timeseries import LagTransformer
>>> X = np.arange(5, dtype=float).reshape(-1, 1)
>>> LagTransformer(lag=-1, columns=[0], fill_with_missing=False).fit_transform(X).ravel().tolist()
[0.0, 1.0, 2.0, 3.0]

See Also
--------
:mod:`tuiml.algorithms.timeseries` : Forecasting models for this data.
"""

from tuiml.preprocessing.timeseries.lag import LagTransformer
from tuiml.preprocessing.timeseries.delta import DifferenceTransformer

__all__ = [
    "LagTransformer",
    "DifferenceTransformer",
]
