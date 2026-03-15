"""Anomaly detection algorithms.

This module provides algorithms for detecting anomalies and outliers in data,
including isolation-based, density-based, geometric, and SVM-based methods.
"""

# Base classes (single source of truth)
from tuiml.base.algorithms import Classifier, classifier

# Anomaly detection algorithms
from tuiml.algorithms.anomaly.isolation_forest import IsolationForestDetector
from tuiml.algorithms.anomaly.local_outlier_factor import LocalOutlierFactorDetector
from tuiml.algorithms.anomaly.elliptic_envelope import EllipticEnvelopeDetector
from tuiml.algorithms.anomaly.one_class_svm import OneClassSVMDetector
from tuiml.algorithms.anomaly.abod import ABODDetector

__all__ = [
    # Base classes
    "Classifier",
    "classifier",
    # Algorithms
    "IsolationForestDetector",
    "LocalOutlierFactorDetector",
    "EllipticEnvelopeDetector",
    "OneClassSVMDetector",
    "ABODDetector",
]
