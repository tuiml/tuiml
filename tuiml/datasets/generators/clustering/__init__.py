"""Synthetic clustering datasets.

Unlabelled data whose true grouping you know, which is the only way to check
what a clusterer actually found. The shapes are chosen to separate algorithms
that assume round clusters from those that do not.

Generators
----------
- **Blobs:** Isotropic Gaussian clusters. The easy case, and what k-means
  assumes — every algorithm should handle it.
- **Moons:** Two interleaved crescents. Not linearly separable, so k-means
  splits them the wrong way while a density-based clusterer gets them right.
- **Circles:** One ring inside another. Same lesson, concentric.
- **SwissRoll:** A 2-D sheet rolled through 3-D. Points near each other in
  space can be far apart along the sheet, which is the problem manifold
  learning exists to solve.

Notes
-----
These return ``y`` as well: the true cluster assignment. Do not fit on it —
it is there so you can score the result with a metric like
:func:`~tuiml.evaluation.metrics.adjusted_rand_score`.

Examples
--------
>>> from tuiml.datasets.generators.clustering import Moons
>>> data = Moons(n_samples=200, noise=0.05, random_state=0).generate()
>>> data.X.shape
(200, 2)
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
