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
from tuiml.base.algorithms import Classifier, classifier

# Anomaly detection algorithms
from tuiml.algorithms.anomaly.isolation_forest import IsolationForestDetector
from tuiml.algorithms.anomaly.local_outlier_factor import LocalOutlierFactorDetector
from tuiml.algorithms.anomaly.elliptic_envelope import EllipticEnvelopeDetector
from tuiml.algorithms.anomaly.one_class_svm import OneClassSVMDetector

__all__ = [
    # Base classes
    "Classifier",
    "classifier",
    # Algorithms
    "IsolationForestDetector",
    "LocalOutlierFactorDetector",
    "EllipticEnvelopeDetector",
    "OneClassSVMDetector",
]
