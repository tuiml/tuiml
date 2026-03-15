"""
Regression data generators.

Generate synthetic data for testing regression algorithms.
"""

from tuiml.datasets.generators.regression.friedman import Friedman
from tuiml.datasets.generators.regression.mexican_hat import MexicanHat
from tuiml.datasets.generators.regression.sine import Sine

__all__ = [
    "Friedman",
    "MexicanHat",
    "Sine",
]
