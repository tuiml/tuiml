"""Data splitting strategies for model evaluation.

How you split decides whether a score means anything. The strategies here
differ in what they hold constant — class balance, group membership, or time
order — and picking the wrong one leaks information from test into training
and reports a score the model will not reproduce in production.

Cross-validation
----------------
- **KFold / StratifiedKFold:** K-fold, the stratified variant preserving
  class proportions in every fold. Prefer stratified for classification,
  especially with imbalanced classes.
- **RepeatedKFold / RepeatedStratifiedKFold:** K-fold repeated with different
  shuffles, for a more stable estimate.
- **LeaveOneOut / LeavePOut:** One (or p) samples held out per fold.
  Near-unbiased but expensive; for small datasets.

Holdout and resampling
----------------------
- **train_test_split:** One-line holdout split.
- **HoldoutSplit / StratifiedHoldoutSplit:** The same as a splitter object.
- **ShuffleSplit / StratifiedShuffleSplit:** Repeated random splits, with
  test sizes independent of the number of iterations.
- **BootstrapSplit:** Sampling with replacement.

Constrained splits
------------------
- **GroupKFold / StratifiedGroupKFold:** Keeps rows sharing a group id in the
  same fold. Use when several rows describe one subject, patient or session:
  splitting them apart lets the model recognise the subject rather than learn
  the task.
- **TimeSeriesSplit:** Trains only on data preceding each test fold. The
  right choice for anything ordered in time, where a shuffled split would
  train on the future.

Scoring
-------
- **cross_val_score:** Fit and score across a splitter in one call.

Examples
--------
>>> from tuiml.evaluation.splitting import train_test_split
>>> from tuiml.datasets import load_iris
>>> data = load_iris()
>>> X_train, X_test, y_train, y_test = train_test_split(
...     data.X, data.y, test_size=0.2, random_state=0)
>>> X_train.shape[0], X_test.shape[0]
(120, 30)
"""

from .kfold import (
    cross_val_score,
    KFold,
    StratifiedKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
)
from .holdout import (
    train_test_split,
    HoldoutSplit,
    StratifiedHoldoutSplit,
)
from .leave_one_out import LeaveOneOut, LeavePOut
from .bootstrap import BootstrapSplit
from .timeseries import TimeSeriesSplit
from .group import GroupKFold, StratifiedGroupKFold
from .shuffle import ShuffleSplit, StratifiedShuffleSplit
from tuiml.base.splitting import BaseSplitter

__all__ = [
    # Base
    "BaseSplitter",
    # Cross-validation
    "cross_val_score",
    # K-Fold
    "KFold",
    "StratifiedKFold",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    # Holdout
    "train_test_split",
    "HoldoutSplit",
    "StratifiedHoldoutSplit",
    # Leave-out
    "LeaveOneOut",
    "LeavePOut",
    # Bootstrap
    "BootstrapSplit",
    # Time Series
    "TimeSeriesSplit",
    # Group
    "GroupKFold",
    "StratifiedGroupKFold",
    # Shuffle
    "ShuffleSplit",
    "StratifiedShuffleSplit",
]
