"""
Sampling transformers for instance-level operations.

This module provides:

Class Balancing:
    - ClassBalanceSampler: Balance class distribution
    - ReservoirSampler: Reservoir sampling algorithm

SMOTE Family (smote.py):
    - SMOTESampler: Synthetic Minority Over-sampling Technique
    - BorderlineSMOTESampler: SMOTE for borderline instances
    - ADASYNSampler: Adaptive Synthetic Sampling
    - SVMSMOTESampler: SVM-based SMOTE
    - KMeansSMOTESampler: K-Means clustering SMOTE

Oversampling (oversampling.py):
    - RandomOverSampler: Random duplication of minority samples
    - ClusterOverSampler: Cluster-aware oversampling

Undersampling (undersampling.py):
    - RandomUnderSampler: Random removal of majority samples
    - TomekLinksSampler: Remove Tomek links
    - ENNSampler: Edited Nearest Neighbours cleaning
    - CNNSampler: Condensed Nearest Neighbour condensing
    - NearMissSampler: Distance-based undersampling
    - HardnessThresholdSampler: Classifier-based undersampling
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
