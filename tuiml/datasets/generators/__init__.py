"""Synthetic datasets, generated to order.

Data whose ground truth you control: the exact shape, size, noise level and
class structure you want. Useful for testing an algorithm against a known
answer, reproducing a result without shipping data, and building streams with
concept drift, which real datasets rarely provide on demand.

Classification
--------------
- **Agrawal:** Loan-application data with ten selectable functions; a
  standard stream-mining benchmark.
- **Hyperplane:** Rotating hyperplane, for gradual concept drift.
- **RandomRBF:** Radial basis clusters, optionally drifting.
- **LED:** Seven-segment display digits with configurable noise.

Regression
----------
- **Friedman:** The Friedman benchmarks, mixing non-linear and irrelevant
  features.
- **MexicanHat:** The radially symmetric ``sinc`` surface.
- **Sine:** Sine wave with noise.

Clustering
----------
- **Blobs:** Isotropic Gaussian clusters.
- **Moons / Circles:** Two interleaved shapes — not linearly separable, so
  they show where a centroid-based clusterer fails.
- **SwissRoll:** A 2-D manifold rolled through 3-D, for manifold learning.

Notes
-----
Every generator takes ``random_state``: set it and the dataset is
reproducible. All return a :class:`GeneratedData` carrying ``X`` and ``y``.

Examples
--------
>>> from tuiml.datasets.generators import Blobs
>>> data = Blobs(n_samples=300, n_features=2, n_clusters=3,
...              random_state=0).generate()
>>> data.X.shape
(300, 2)
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
