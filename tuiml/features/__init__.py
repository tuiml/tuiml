"""Feature engineering and dimensionality reduction module.

The `tuiml.features` module provides a comprehensive set of tools for 
transforming, selecting, and extracting meaningful features from raw data. 
It follows the scikit-learn API pattern for transformations.

Module Overview
---------------
This module is organized into three main functional areas:

1. **Selection**: Methods for identifying the most predictive subset of features.
   - *Filter methods*: :class:`~tuiml.features.selection.VarianceThresholdSelector`, 
     :class:`~tuiml.features.selection.CFSSelector`, etc.
   - *Wrapper methods*: :class:`~tuiml.features.selection.SequentialFeatureSelector`, 
     :class:`~tuiml.features.selection.WrapperSelector`, etc.

2. **Extraction**: Methods for projecting data into a new, lower-dimensional space.
   - :class:`~tuiml.features.extraction.PCAExtractor`: Principal Component Analysis.
   - :class:`~tuiml.features.extraction.RandomProjectionExtractor`: Johnson-Lindenstrauss based reduction.

3. **Generation**: Methods for constructing new features through mathematical combinations.
   - :class:`~tuiml.features.generation.PolynomialFeaturesGenerator`: Feature interactions.
   - :class:`~tuiml.features.generation.BinningFeaturesGenerator`: Discretization of continuous values.
   - :class:`~tuiml.features.generation.MathematicalFeaturesGenerator`: Functional transformations.

Examples
--------
Selecting top-k features using Correlation-based Feature Selection (CFS):

>>> from tuiml.features.selection import CFSSelector
>>> import numpy as np
>>> X, y = np.random.randn(50, 20), np.random.randint(0, 2, 50)
>>> selector = CFSSelector()
>>> X_selected = selector.fit_transform(X, y)
>>> print(f"Reduced to {X_selected.shape[1]} features")

Extracting principal components:

>>> from tuiml.features.extraction import PCAExtractor
>>> pca = PCAExtractor(n_components=0.95)
>>> X_pca = pca.fit_transform(X)
"""

from tuiml.base.features import (
    FeatureMethod,
    FeatureSelector,
    FeatureExtractor,
    FeatureConstructor,
    feature_selector,
    feature_extractor,
    feature_constructor,
)

# Import submodules for easy access
from tuiml.features import selection
from tuiml.features import extraction
from tuiml.features import generation

__all__ = [
    # Base classes
    "FeatureMethod",
    "FeatureSelector",
    "FeatureExtractor",
    "FeatureConstructor",
    # Decorators
    "feature_selector",
    "feature_extractor",
    "feature_constructor",
    # Submodules
    "selection",
    "extraction",
    "generation",
]
