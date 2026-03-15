"""Base classes and utility functions for evaluation metrics.

Metrics in TuiML provide a unified interface for assessing model 
performance across classification, regression, and clustering tasks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import numpy as np

class MetricType(Enum):
    """Enumeration of machine learning task categories.
    
    Used to validate if a metric is appropriate for a given model type.
    """
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RANKING = "ranking"

class AverageType(Enum):
    """Strategies for aggregating multi-class performance.

    - **MICRO**: Total true positives, false negatives and false positives.
    - **MACRO**: Unweighted mean per class (treats all classes equally).
    - **WEIGHTED**: Average weighted by class support (accounts for imbalance).
    - **BINARY**: Specific to problems with only two classes.
    """
    MICRO = "micro"      # Global averaging
    MACRO = "macro"      # Per-class average
    WEIGHTED = "weighted"  # Class-size weighted average
    BINARY = "binary"    # Only for binary classification
    SAMPLES = "samples"  # For multilabel

class Metric(ABC):
    """Abstract base class for all performance evaluators.

    Metrics are callable objects that calculate a score comparing the ground 
    truth (:math:`y_{true}`) with the model predictions (:math:`y_{pred}`).
    """

    def __init__(self, name: str, metric_type: MetricType):
        """
        Initialize a metric.

        Args:
            name: Name of the metric
            metric_type: Type of metric (classification, regression, etc.)
        """
        self.name = name
        self.metric_type = metric_type

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Compute the metric value.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels or probabilities
            **kwargs: Additional metric-specific parameters

        Returns:
            The computed metric value
        """
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Allow calling the metric directly."""
        return self.compute(y_true, y_pred, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

def check_consistent_length(*arrays) -> None:
    """
    Check that all arrays have consistent first dimensions.

    Args:
        *arrays: Arrays to check

    Raises:
        ValueError: If arrays have inconsistent lengths
    """
    lengths = [len(arr) for arr in arrays if arr is not None]
    if len(set(lengths)) > 1:
        raise ValueError(f"Found input arrays with inconsistent numbers of samples: {lengths}")

def check_classification_targets(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Check that y_true and y_pred are valid classification targets.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Raises:
        ValueError: If inputs are invalid
    """
    check_consistent_length(y_true, y_pred)

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred cannot be empty")

def get_num_classes(y_true: np.ndarray, y_pred: Optional[np.ndarray] = None) -> int:
    """
    Get the number of unique classes in the data.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (optional)

    Returns:
        Number of unique classes
    """
    if y_pred is not None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        classes = np.unique(y_true)
    return len(classes)

def get_class_labels(y_true: np.ndarray, y_pred: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Get sorted unique class labels from the data.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (optional)

    Returns:
        Sorted array of unique class labels
    """
    if y_pred is not None:
        return np.unique(np.concatenate([y_true, y_pred]))
    return np.unique(y_true)

def is_binary(y_true: np.ndarray, y_pred: Optional[np.ndarray] = None) -> bool:
    """
    Check if this is a binary classification problem.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (optional)

    Returns:
        True if binary classification
    """
    return get_num_classes(y_true, y_pred) == 2

def weighted_sum(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted sum of values.

    Args:
        values: Values to sum
        weights: Weights for each value

    Returns:
        Weighted sum
    """
    return np.sum(values * weights) / np.sum(weights)

def safe_divide(numerator: Union[float, np.ndarray],
                denominator: Union[float, np.ndarray],
                zero_division: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safely divide, handling division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        zero_division: Value to return when denominator is zero

    Returns:
        Result of division
    """
    if isinstance(denominator, np.ndarray):
        # Use np.divide with out and where to avoid RuntimeWarning
        numerator = np.asarray(numerator)
        result = np.full_like(denominator, zero_division, dtype=np.float64)
        mask = denominator != 0
        np.divide(numerator, denominator, out=result, where=mask)
        return result
    else:
        if denominator == 0:
            return zero_division
        return numerator / denominator
