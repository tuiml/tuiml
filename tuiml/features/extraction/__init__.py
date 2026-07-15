"""Feature extraction and dimensionality reduction methods.

Feature extraction transforms the original feature space into a new, 
lower-dimensional space that preserves as much information as possible. 
This is essential for visualization, noise reduction, and improving 
computational efficiency.

Algorithms
----------
- :class:`~tuiml.features.extraction.PCAExtractor`: Projects data onto the axes of maximum variance.
- :class:`~tuiml.features.extraction.RandomProjectionExtractor`: Projects data onto a random subspace 
  (preserves distances).
- :class:`~tuiml.features.extraction.SparseRandomProjectionExtractor`: Optimized random projection 
  using sparse matrices.

Examples
--------
Reducing dimensionality with PCA:

>>> from tuiml.features.extraction import PCAExtractor
>>> import numpy as np
>>> X = np.random.randn(50, 20)
>>> pca = PCAExtractor(n_components=5)
>>> X_reduced = pca.fit_transform(X)

Using Random Projection for fast reduction:

>>> from tuiml.features.extraction import RandomProjectionExtractor
>>> rp = RandomProjectionExtractor(n_components='auto')
>>> X_rp = rp.fit_transform(X)
"""

from tuiml.features.extraction.pca import PCAExtractor
from tuiml.features.extraction.random_projection import (
    RandomProjectionExtractor,
    SparseRandomProjectionExtractor,
)

__all__ = [
    "PCAExtractor",
    "RandomProjectionExtractor",
    "SparseRandomProjectionExtractor",
]
