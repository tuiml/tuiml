"""
Clustering data generators.

Generate synthetic data for testing clustering algorithms.
"""

from tuiml.datasets.generators.clustering.blobs import Blobs
from tuiml.datasets.generators.clustering.moons import Moons
from tuiml.datasets.generators.clustering.circles import Circles
from tuiml.datasets.generators.clustering.swiss_roll import SwissRoll

__all__ = [
    "Blobs",
    "Moons",
    "Circles",
    "SwissRoll",
]
