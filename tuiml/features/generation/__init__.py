"""Feature generation and construction module.

This module provides tools for creating new features from existing ones 
through domain-agnostic mathematical operations, interactions, and 
discretization.

Overview
--------
The generators are designed to augment the feature space with non-linear 
relationships:

- :class:`~tuiml.features.generation.PolynomialFeaturesGenerator`: Higher-order terms.
- :class:`~tuiml.features.generation.InteractionFeaturesGenerator`: Cross-feature products.
- :class:`~tuiml.features.generation.MathematicalFeaturesGenerator`: Functional mappings (log, exp).
- :class:`~tuiml.features.generation.BinningFeaturesGenerator`: Continuous to categorical mappings.

Examples
--------
Creating interaction features:

>>> from tuiml.features.generation import InteractionFeaturesGenerator
>>> import numpy as np
>>> X = np.array([[1, 2], [3, 4]])
>>> gen = InteractionFeaturesGenerator()
>>> X_new = gen.fit_transform(X)

Discretizing continuous features into bins:

>>> from tuiml.features.generation import BinningFeaturesGenerator
>>> binner = BinningFeaturesGenerator(n_bins=5, strategy='uniform')
>>> X_binned = binner.fit_transform(X)
"""

from tuiml.features.generation.polynomial import (
    PolynomialFeaturesGenerator,
    InteractionFeaturesGenerator,
)

from tuiml.features.generation.mathematical import (
    MathematicalFeaturesGenerator,
    BinningFeaturesGenerator,
)

__all__ = [
    "PolynomialFeaturesGenerator",
    "InteractionFeaturesGenerator",
    "MathematicalFeaturesGenerator",
    "BinningFeaturesGenerator",
]
