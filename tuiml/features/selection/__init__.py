"""Feature selection methods for tuiml.

This module provides a wide range of feature selection algorithms that follow the 
scikit-learn fit/transform pattern. These filters and wrappers help in reducing 
overfitting, improving accuracy, and reducing training time by identifying the 
most relevant features.

Overview
--------
The selectors are categorized into several strategies:

Univariate Selectors
~~~~~~~~~~~~~~~~~~~~
These score each feature independently using statistical tests:
- :class:`~tuiml.features.selection.SelectKBestSelector`: Keep top :math:`k` features.
- :class:`~tuiml.features.selection.SelectPercentileSelector`: Keep top percentage.
- :class:`~tuiml.features.selection.SelectThresholdSelector`: Keep by raw score.
- :class:`~tuiml.features.selection.SelectFprSelector`: Keep by signicance (p-value).

Subset Selection (Filter & Wrapper)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These evaluate groups of features together:
- :class:`~tuiml.features.selection.CFSSelector`: Correlation-based Subset Selection.
- :class:`~tuiml.features.selection.SequentialFeatureSelector`: Greedy forward/backward.
- :class:`~tuiml.features.selection.BestFirstSelector`: Search with backtracking.
- :class:`~tuiml.features.selection.WrapperSelector`: Evaluates subsets with an estimator.

Unsupervised & Stochastic
~~~~~~~~~~~~~~~~~~~~~~~~~
- :class:`~tuiml.features.selection.VarianceThresholdSelector`: Removes constant features.
- :class:`~tuiml.features.selection.RandomSubsetSelector`: Stochastic subsampling.
- :class:`~tuiml.features.selection.BootstrapFeaturesSelector`: Sample with replacement.

Examples
--------
Univariate selection with Information Gain:

>>> from tuiml.features.selection import SelectKBestSelector
>>> from tuiml.evaluation.metrics import information_gain
>>> import numpy as np
>>> X, y = np.random.randn(20, 10), np.random.randint(0, 2, 20)
>>> selector = SelectKBestSelector(score_func=information_gain, k=5)
>>> X_new = selector.fit_transform(X, y)

Automatic subset selection using CFS:

>>> from tuiml.features.selection import CFSSelector
>>> cfs = CFSSelector()
>>> X_new = cfs.fit_transform(X, y)
>>> print(f"Selected: {cfs.get_support(indices=True)}")
"""

# Base classes and utilities
from tuiml.features.selection._base import (
    SelectorMixin,
    GenericUnivariateSelector,
)

# Univariate selectors
from tuiml.features.selection.univariate import (
    SelectKBestSelector,
    SelectPercentileSelector,
    SelectThresholdSelector,
    SelectFprSelector,
)

# Variance-based selection
from tuiml.features.selection.variance import (
    VarianceThresholdSelector,
)

# Sequential/wrapper selectors
from tuiml.features.selection.sequential import (
    SequentialFeatureSelector,
    BestFirstSelector,
)

# Subset evaluators
from tuiml.features.selection.subset import (
    CFSSelector,
    WrapperSelector,
)

# Random selection
from tuiml.features.selection.random_subset import (
    RandomSubsetSelector,
    BootstrapFeaturesSelector,
)

__all__ = [
    # Base
    "SelectorMixin",
    "GenericUnivariateSelector",
    # Univariate
    "SelectKBestSelector",
    "SelectPercentileSelector",
    "SelectThresholdSelector",
    "SelectFprSelector",
    # Variance
    "VarianceThresholdSelector",
    # Sequential
    "SequentialFeatureSelector",
    "BestFirstSelector",
    # Subset
    "CFSSelector",
    "WrapperSelector",
    # Random
    "RandomSubsetSelector",
    "BootstrapFeaturesSelector",
]
