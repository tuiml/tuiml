"""Deep, torch-backed forecasters.

Three neural window forecasters that learn a mapping from a lookback window to
a horizon, rather than from a hand-specified process as the classical models
in :mod:`tuiml.algorithms.timeseries` do.

Algorithms
----------
- **NBEATSForecaster:** doubly-residual stacks of fully connected blocks, with
  an optional interpretable trend/seasonality decomposition.
- **NHITSForecaster:** N-BEATS plus multi-rate max pooling and hierarchical
  interpolation, so each stack specialises in one frequency band.
- **PatchTSTForecaster:** a Transformer over patch tokens, with channel
  independence and reversible instance normalisation.

Notes
-----
These models require **PyTorch**, which TuiML treats as an optional extra::

    pip install 'tuiml[torch]'

Importing this package does *not* import torch, and the classes construct,
register and report their parameter schema without it, so ``list_algorithms()``
shows the same catalog on every install. The dependency is demanded by
:meth:`fit`, which raises an ``ImportError`` naming the install command.

They all follow the forecasting convention used by the classical models:
``fit(y)`` takes the series itself rather than a design matrix, and
``predict(steps)`` returns the next ``steps`` values.

Examples
--------
>>> from tuiml.algorithms.timeseries.deep import NBEATSForecaster
>>> model = NBEATSForecaster(random_state=0)
>>> model.lookback, model.horizon
(24, 8)
"""

from tuiml.algorithms.timeseries.deep.nbeats import NBEATSForecaster
from tuiml.algorithms.timeseries.deep.nhits import NHITSForecaster
from tuiml.algorithms.timeseries.deep.patchtst import PatchTSTForecaster

__all__ = [
    "NBEATSForecaster",
    "NHITSForecaster",
    "PatchTSTForecaster",
]
