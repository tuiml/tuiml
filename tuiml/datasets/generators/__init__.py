"""
Data Generators for synthetic dataset creation.

Organized by task type:
- classification: Generators for classification problems
- regression: Generators for regression problems
- clustering: Generators for clustering problems
"""

# Base classes
from tuiml.base.generators import (
    DataGenerator,
    ClassificationGenerator,
    RegressionGenerator,
    ClusteringGenerator,
    GeneratedData,
)

# Classification generators
from tuiml.datasets.generators.classification import (
    RandomRBF,
    Agrawal,
    LED,
    Hyperplane,
)

# Regression generators
from tuiml.datasets.generators.regression import (
    Friedman,
    MexicanHat,
    Sine,
)

# Clustering generators
from tuiml.datasets.generators.clustering import (
    Blobs,
    Moons,
    Circles,
    SwissRoll,
)

__all__ = [
    # Base classes
    "DataGenerator",
    "ClassificationGenerator",
    "RegressionGenerator",
    "ClusteringGenerator",
    "GeneratedData",
    # Classification
    "RandomRBF",
    "Agrawal",
    "LED",
    "Hyperplane",
    # Regression
    "Friedman",
    "MexicanHat",
    "Sine",
    # Clustering
    "Blobs",
    "Moons",
    "Circles",
    "SwissRoll",
]
