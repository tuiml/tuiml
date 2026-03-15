"""
Hyperparameter tuning utilities.
- CVParameterSelection.java
- GridSearch.java

This module provides:
- GridSearchCV: Exhaustive search over parameter grid
- RandomSearchCV: Random search over parameter distributions
- BayesianSearchCV: Bayesian optimization using Gaussian Processes
"""

from .grid_search import GridSearchCV
from .random_search import RandomSearchCV
from .bayesian_search import BayesianSearchCV
from tuiml.base.tuning import (
    BaseTuner,
    TuningResult,
    ParameterGrid,
    ParameterDistribution,
)

__all__ = [
    "BaseTuner",
    "TuningResult",
    "ParameterGrid",
    "ParameterDistribution",
    "GridSearchCV",
    "RandomSearchCV",
    "BayesianSearchCV",
]
