"""
Classification data generators.

Generate synthetic data for testing classification algorithms.
"""

from tuiml.datasets.generators.classification.random_rbf import RandomRBF
from tuiml.datasets.generators.classification.agrawal import Agrawal
from tuiml.datasets.generators.classification.led import LED
from tuiml.datasets.generators.classification.hyperplane import Hyperplane

__all__ = [
    "RandomRBF",
    "Agrawal",
    "LED",
    "Hyperplane",
]
