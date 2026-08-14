"""Anomaly detection algorithms.

Finding the rare rows: fraud, faults, intrusions. These fit on what normal
looks like and score how far each point departs from it, which is what makes
them work when anomalies are too few — or too varied — to learn as a class.

Algorithms
----------
- **IsolationForestDetector:** Isolates points with random splits; anomalies
  need fewer splits to separate. Fast, scales well, and the usual first
  choice.
- **LocalOutlierFactorDetector:** Compares a point's local density to its
  neighbours'. Catches anomalies that are only unusual for their region,
  which a global method misses.
- **OneClassSVMDetector:** Learns a boundary enclosing the normal data.
- **EllipticEnvelopeDetector:** Fits a Gaussian and flags low-probability
  points. Strong when the data really is roughly normal, poor when it is not.
- **ECODDetector:** Per-dimension empirical tail probabilities. Parameter-free,
  scales to high dimension, and reports which feature caused each flag.
- **COPODDetector:** The same tail idea framed through the empirical copula;
  the widely cited baseline.
- **HBOSDetector:** One histogram per feature. The fastest of the family, with
  a fitted model whose size does not depend on the training set.

- **KNNDetector:** Distance to the k-th nearest neighbour. Works on the joint
  distribution, so it finds points that are ordinary in every single feature
  yet sit in an empty region.
- **ABODDetector:** Variance of the angles subtended at a point. Angles resist
  the dimensionality curse better than distances do.
- **LSCPDetector:** An ensemble that picks the best base detector separately
  for each point's own neighbourhood, rather than assuming one detector wins
  everywhere.

ECOD, COPOD and HBOS assume outlyingness shows up in individual features. They are
fast and dimension-scalable but blind to anomalies that are only unusual in
the joint distribution — for those use the isolation or density methods above,
which see feature interactions but struggle in very high
dimensions, where distances concentrate and become uninformative.

Notes
-----
These take a ``contamination`` parameter — the expected proportion of
anomalies — which sets the threshold. It is a prior, not something learned,
so it is the parameter worth getting right.

Accuracy is the wrong metric here: with 1% anomalies, calling everything
normal scores 99%. Use precision, recall or ROC AUC on the anomaly class.

See Also
--------
:mod:`tuiml.preprocessing.outliers` : When extreme values are noise to
    remove before modelling, rather than the thing to predict.
"""

# Base classes (single source of truth)
from tuiml.base.algorithms import Classifier, anomaly_detector

# Anomaly detection algorithms
from tuiml.algorithms.anomaly.isolation_forest import IsolationForestDetector
from tuiml.algorithms.anomaly.local_outlier_factor import LocalOutlierFactorDetector
from tuiml.algorithms.anomaly.elliptic_envelope import EllipticEnvelopeDetector
from tuiml.algorithms.anomaly.one_class_svm import OneClassSVMDetector
from tuiml.algorithms.anomaly.ecod import ECODDetector
from tuiml.algorithms.anomaly.copod import COPODDetector
from tuiml.algorithms.anomaly.hbos import HBOSDetector
from tuiml.algorithms.anomaly.knn_detector import KNNDetector
from tuiml.algorithms.anomaly.abod import ABODDetector
from tuiml.algorithms.anomaly.lscp import LSCPDetector

__all__ = [
    # Base classes
    "Classifier",
    "anomaly_detector",
    # Algorithms
    "IsolationForestDetector",
    "LocalOutlierFactorDetector",
    "EllipticEnvelopeDetector",
    "OneClassSVMDetector",
    "ECODDetector",
    "COPODDetector",
    "HBOSDetector",
    "KNNDetector",
    "ABODDetector",
    "LSCPDetector",
]
