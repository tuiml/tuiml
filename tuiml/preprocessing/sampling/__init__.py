"""Resampling to fix class imbalance.

When one class dominates, a model can score well by ignoring the rare one —
and the rare one is usually the class you care about: the fraud, the fault,
the diagnosis. These transformers rebalance the training set so the minority
carries real weight.

Oversampling
------------
Adds minority examples, keeping every majority row.

- **RandomOverSampler:** Duplicates minority rows, optionally jittered.
- **SMOTESampler:** Interpolates *new* points between neighbouring minority
  rows, so the model sees fresh points rather than repeats.
- **BorderlineSMOTESampler:** SMOTE concentrated near the decision boundary,
  where the errors actually happen.
- **ADASYNSampler:** Generates more where the minority is hardest to learn.
- **SVMSMOTESampler / KMeansSMOTESampler:** SMOTE guided by an SVM boundary
  or by clusters.
- **ClusterOverSampler:** Cluster-aware oversampling.

Undersampling
-------------
Drops majority examples. Faster to train, at the cost of discarding data.

- **RandomUnderSampler:** Removes majority rows at random.
- **TomekLinksSampler:** Removes majority rows that sit right against a
  minority row, cleaning the boundary rather than thinning everywhere.
- **ENNSampler:** Edited Nearest Neighbours cleaning.
- **CNNSampler:** Condensed Nearest Neighbour condensing.
- **NearMissSampler:** Distance-based selection.
- **HardnessThresholdSampler:** Drops the rows a classifier finds easy.

Other
-----
- **ClassBalanceSampler:** Balance the class distribution directly.
- **ReservoirSampler:** Fixed-size random sample from a stream.

Notes
-----
Resample the **training split only**. Synthetic or duplicated rows in a test
set produce a score that cannot be reproduced on real data — and with
oversampling, near-copies of training rows end up in test, which inflates it
outright. Passing these as ``pipeline`` steps to :func:`tuiml.train` handles
this: they are applied inside each fold, to the training half.

Accuracy stays misleading after resampling. Judge on the original
distribution, with a metric that respects the minority class.
"""

# Class balancing
from tuiml.preprocessing.sampling.reservoir_sample import ReservoirSampler
from tuiml.preprocessing.sampling.class_balancer import ClassBalanceSampler

# SMOTE family
from tuiml.preprocessing.sampling.smote import (
    SMOTESampler,
    BorderlineSMOTESampler,
    ADASYNSampler,
    SVMSMOTESampler,
    KMeansSMOTESampler,
)

# Oversampling
from tuiml.preprocessing.sampling.oversampling import (
    RandomOverSampler,
    ClusterOverSampler,
)

# Undersampling
from tuiml.preprocessing.sampling.undersampling import (
    RandomUnderSampler,
    TomekLinksSampler,
    ENNSampler,
    CNNSampler,
    NearMissSampler,
    HardnessThresholdSampler,
)

__all__ = [
    # Class balancing
    "ReservoirSampler",
    "ClassBalanceSampler",
    # SMOTE family
    "SMOTESampler",
    "BorderlineSMOTESampler",
    "ADASYNSampler",
    "SVMSMOTESampler",
    "KMeansSMOTESampler",
    # Oversampling
    "RandomOverSampler",
    "ClusterOverSampler",
    # Undersampling
    "RandomUnderSampler",
    "TomekLinksSampler",
    "ENNSampler",
    "CNNSampler",
    "NearMissSampler",
    "HardnessThresholdSampler",
]
