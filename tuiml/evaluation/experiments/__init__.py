"""
Experiment framework for model comparison.
"""

from tuiml.base.experiments import (
    ExperimentType,
    ValidationMethod,
    ExperimentConfig,
    FoldResult,
    ModelResult,
    DatasetResult,
    ExperimentResults,
    BaseExperiment,
)

from .experiment import (
    Experiment,
    run_experiment,
)

__all__ = [
    "ExperimentType",
    "ValidationMethod",
    "ExperimentConfig",
    "FoldResult",
    "ModelResult",
    "DatasetResult",
    "ExperimentResults",
    "BaseExperiment",
    "Experiment",
    "run_experiment",
]
