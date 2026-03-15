"""
Data splitting utilities for model evaluation.
- CrossValidationResultProducer.java
- RandomSplitResultProducer.java

This module provides various strategies for splitting data:
- KFold / StratifiedKFold: K-fold cross-validation
- train_test_split: Simple holdout split
- LeaveOneOut: Leave-one-out cross-validation
- BootstrapSplit: Bootstrap sampling
- TimeSeriesSplit: Time series cross-validation
- GroupKFold: K-fold with groups
- ShuffleSplit: Random permutation splits
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
