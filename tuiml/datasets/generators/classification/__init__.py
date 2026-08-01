"""Synthetic classification datasets.

Labelled data with a known decision boundary, so you can check what an
algorithm recovers. Several also generate *concept drift* — the boundary
moving partway through the stream — which is what streaming learners have to
detect and adapt to.

Generators
----------
- **Agrawal:** Loan applications over nine attributes, with ten selectable
  target functions. A standard stream-mining benchmark; switching function
  mid-stream produces abrupt drift.
- **Hyperplane:** Points labelled by which side of a rotating hyperplane they
  fall on. Rotating it slowly gives gradual drift.
- **RandomRBF:** Radial basis clusters with random centroids, optionally
  drifting.
- **LED:** The seven-segment digits, with a tunable proportion of segments
  flipped — a controlled way to vary label noise.

Examples
--------
>>> from tuiml.datasets.generators.classification import Agrawal
>>> data = Agrawal(n_samples=1000, function=1, random_state=0).generate()
>>> data.X.shape
(1000, 9)
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
