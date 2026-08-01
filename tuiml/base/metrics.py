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
        """Initialize a metric.

        Parameters
        ----------
        name : str
            Name of the metric.
        metric_type : MetricType
            Task category the metric applies to (classification, regression, etc.).
        """
        self.name = name
        self.metric_type = metric_type

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Compute the metric value.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_samples,)
            Ground truth labels.
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels or probabilities.
        **kwargs : dict
            Additional metric-specific parameters.

        Returns
        -------
        score : float
            The computed metric value.
        """
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Compute the metric by calling the object directly.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_samples,)
            Ground truth labels.
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels or probabilities.
        **kwargs : dict
            Additional metric-specific parameters.

        Returns
        -------
        score : float
            The computed metric value.
        """
        return self.compute(y_true, y_pred, **kwargs)

    def __repr__(self) -> str:
        """Return string representation of the metric."""
        return f"{self.__class__.__name__}(name='{self.name}')"

def check_consistent_length(*arrays) -> None:
    """Check that all arrays have consistent first dimensions.

    Parameters
    ----------
    *arrays : sequence of array-like
        Arrays to check. ``None`` entries are ignored.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If arrays have inconsistent lengths.
    """
    lengths = [len(arr) for arr in arrays if arr is not None]
    if len(set(lengths)) > 1:
        raise ValueError(f"Found input arrays with inconsistent numbers of samples: {lengths}")

def check_classification_targets(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Check that ``y_true`` and ``y_pred`` are valid classification targets.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If inputs are empty or have inconsistent lengths.
    """
    check_consistent_length(y_true, y_pred)

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred cannot be empty")

def get_num_classes(y_true: np.ndarray, y_pred: Optional[np.ndarray] = None) -> int:
    """Get the number of unique classes in the data.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : np.ndarray of shape (n_samples,), optional
        Predicted labels. If given, classes from both arrays are counted.

    Returns
    -------
    n_classes : int
        Number of unique classes.
    """
    if y_pred is not None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        classes = np.unique(y_true)
    return len(classes)

def get_class_labels(y_true: np.ndarray, y_pred: Optional[np.ndarray] = None) -> np.ndarray:
    """Get sorted unique class labels from the data.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : np.ndarray of shape (n_samples,), optional
        Predicted labels. If given, labels from both arrays are included.

    Returns
    -------
    labels : np.ndarray
        Sorted array of unique class labels.
    """
    if y_pred is not None:
        return np.unique(np.concatenate([y_true, y_pred]))
    return np.unique(y_true)

def is_binary(y_true: np.ndarray, y_pred: Optional[np.ndarray] = None) -> bool:
    """Check if this is a binary classification problem.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : np.ndarray of shape (n_samples,), optional
        Predicted labels.

    Returns
    -------
    binary : bool
        True if the data contains exactly two classes.
    """
    return get_num_classes(y_true, y_pred) == 2

def weighted_sum(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted average of values.

    Parameters
    ----------
    values : np.ndarray
        Values to aggregate.
    weights : np.ndarray
        Weight for each value.

    Returns
    -------
    result : float
        Sum of ``values * weights`` normalized by the total weight.
    """
    return np.sum(values * weights) / np.sum(weights)

def safe_divide(numerator: Union[float, np.ndarray],
                denominator: Union[float, np.ndarray],
                zero_division: float = 0.0) -> Union[float, np.ndarray]:
    """Safely divide, handling division by zero.

    Parameters
    ----------
    numerator : float or np.ndarray
        Numerator.
    denominator : float or np.ndarray
        Denominator.
    zero_division : float, default=0.0
        Value to return where the denominator is zero.

    Returns
    -------
    result : float or np.ndarray
        Element-wise result of the division, with ``zero_division``
        substituted wherever the denominator is zero.
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
