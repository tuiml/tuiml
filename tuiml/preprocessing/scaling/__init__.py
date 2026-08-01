"""Putting numeric features on a comparable scale.

Any algorithm that measures distance or follows a gradient — k-NN, SVMs,
k-means, neural networks, the SGD linear models — is dominated by whichever
column happens to have the largest numbers. Scaling removes that accident.
Tree-based algorithms split one feature at a time and are unaffected.

Transformers
------------
- **StandardScaler:** Centre at zero, scale to unit variance. The usual
  default, and what most algorithms assume.
- **MinMaxScaler:** Squash into a fixed range, by default ``[0, 1]``. Keeps
  zeros zero and bounds the output, but a single extreme value compresses
  everything else.
- **CenterScaler:** Subtract the mean only, leaving the spread alone.

Notes
-----
Fit on the training split and apply that fit to the test split — never fit on
both. Fitting on all the data lets the test set's mean and variance influence
the transform, which inflates the score. Passing these as ``pipeline`` steps
to :func:`tuiml.train` handles the ordering for you, refitting the scaler
inside each cross-validation fold.

Examples
--------
>>> from tuiml.preprocessing.scaling import StandardScaler
>>> from tuiml.datasets import load_iris
>>> data = load_iris()
>>> X_scaled = StandardScaler().fit_transform(data.X)
>>> bool(abs(X_scaled.mean()) < 1e-9)
True
"""

from tuiml.preprocessing.scaling.normalize import MinMaxScaler
from tuiml.preprocessing.scaling.standardize import StandardScaler
from tuiml.preprocessing.scaling.center import CenterScaler

__all__ = [
    "MinMaxScaler",
    "StandardScaler",
    "CenterScaler",
]
