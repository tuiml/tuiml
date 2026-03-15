"""
Base classes for experiment framework.
- Experiment.java
- ResultProducer interface
- SplitEvaluator interface
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import time
from datetime import datetime

class ExperimentType(Enum):
    """Type of experiment."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"

class ValidationMethod(Enum):
    """Validation method for experiments."""
    CROSS_VALIDATION = "cross_validation"
    HOLDOUT = "holdout"
    REPEATED_CV = "repeated_cv"
    LEAVE_ONE_OUT = "leave_one_out"
    BOOTSTRAP = "bootstrap"

@dataclass
class ExperimentConfig:
    """
    Configuration for an experiment.

    Parameters
    ----------
    name : str
        Name of the experiment.
    validation_method : ValidationMethod
        Validation method to use.
    n_folds : int
        Number of folds for cross-validation.
    n_repeats : int
        Number of repetitions.
    test_size : float
        Test set size for holdout validation.
    random_state : int, optional
        Random seed for reproducibility.
    stratify : bool
        Whether to stratify splits (for classification).
    shuffle : bool
        Whether to shuffle before splitting.
    n_jobs : int
        Number of parallel jobs (-1 for all cores).
    verbose : int
        Verbosity level.
    """
    name: str = "Experiment"
    validation_method: ValidationMethod = ValidationMethod.CROSS_VALIDATION
    n_folds: int = 10
    n_repeats: int = 1
    test_size: float = 0.2
    random_state: Optional[int] = 42
    stratify: bool = True
    shuffle: bool = True
    n_jobs: int = 1
    verbose: int = 0

@dataclass
class FoldResult:
    """
    Results from a single fold evaluation.
    """
    fold_idx: int
    repeat_idx: int = 0
    train_indices: np.ndarray = None
    test_indices: np.ndarray = None
    y_true: np.ndarray = None
    y_pred: np.ndarray = None
    y_prob: Optional[np.ndarray] = None
    train_time: float = 0.0
    test_time: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

@dataclass
class ModelResult:
    """
    Results for a single model across all folds.
    """
    model_name: str
    model: Any
    fold_results: List[FoldResult] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.fold_results)

    def get_metric_values(self, metric_name: str) -> np.ndarray:
        """Get values of a metric across all folds."""
        return np.array([f.metrics.get(metric_name, np.nan) for f in self.fold_results])

    def get_metric_mean(self, metric_name: str) -> float:
        """Get mean of a metric across folds."""
        values = self.get_metric_values(metric_name)
        return np.nanmean(values)

    def get_metric_std(self, metric_name: str) -> float:
        """Get std of a metric across folds."""
        values = self.get_metric_values(metric_name)
        return np.nanstd(values)

    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = self.get_metric_values(metric_name)
        return {
            "mean": np.nanmean(values),
            "std": np.nanstd(values),
            "min": np.nanmin(values),
            "max": np.nanmax(values),
            "median": np.nanmedian(values)
        }

    @property
    def total_train_time(self) -> float:
        return sum(f.train_time for f in self.fold_results)

    @property
    def total_test_time(self) -> float:
        return sum(f.test_time for f in self.fold_results)

@dataclass
class DatasetResult:
    """
    Results for all models on a single dataset.
    """
    dataset_name: str
    n_samples: int
    n_features: int
    model_results: Dict[str, ModelResult] = field(default_factory=dict)

    def add_model_result(self, result: ModelResult):
        """Add a model's results."""
        self.model_results[result.model_name] = result

    def get_model_names(self) -> List[str]:
        """Get list of model names."""
        return list(self.model_results.keys())

    def get_metric_comparison(self, metric_name: str) -> Dict[str, Dict[str, float]]:
        """Get metric comparison across all models."""
        return {
            name: result.get_metric_stats(metric_name)
            for name, result in self.model_results.items()
        }

@dataclass
class ExperimentResults:
    """
    Complete results from an experiment.
    """
    config: ExperimentConfig
    experiment_type: ExperimentType
    dataset_results: Dict[str, DatasetResult] = field(default_factory=dict)
    start_time: datetime = None
    end_time: datetime = None

    def add_dataset_result(self, result: DatasetResult):
        """Add dataset results."""
        self.dataset_results[result.dataset_name] = result

    def get_dataset_names(self) -> List[str]:
        """Get list of dataset names."""
        return list(self.dataset_results.keys())

    def get_model_names(self) -> List[str]:
        """Get list of all model names."""
        models = set()
        for dr in self.dataset_results.values():
            models.update(dr.get_model_names())
        return sorted(models)

    @property
    def duration(self) -> float:
        """Get experiment duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

class BaseValidator(ABC):
    """
    Base class for validation strategies.
    """

    @abstractmethod
    def split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.

        Parameters
        ----------
        X : ndarray
            Feature matrix.
        y : ndarray
            Target values.

        Returns
        -------
        splits : list of (train_indices, test_indices)
        """
        pass

    @abstractmethod
    def get_n_splits(self) -> int:
        """Get number of splits."""
        pass

class BaseExperiment(ABC):
    """
    Base class for experiments.
    """

    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.results: Optional[ExperimentResults] = None

    @abstractmethod
    def run(
        self,
        models: Dict[str, Any],
        datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        metrics: List[str] = None
    ) -> ExperimentResults:
        """
        Run the experiment.

        Parameters
        ----------
        models : dict
            Dictionary of {name: model} pairs.
        datasets : dict
            Dictionary of {name: (X, y)} pairs.
        metrics : list of str, optional
            Metrics to compute.

        Returns
        -------
        results : ExperimentResults
        """
        pass

__all__ = [
    "ExperimentType",
    "ValidationMethod",
    "ExperimentConfig",
    "FoldResult",
    "ModelResult",
    "DatasetResult",
    "ExperimentResults",
    "BaseValidator",
    "BaseExperiment",
    "_ensure_numpy",
]

def _ensure_numpy(arr) -> np.ndarray:
    """Convert to numpy array."""
    if hasattr(arr, 'values'):
        return arr.values
    return np.asarray(arr)
