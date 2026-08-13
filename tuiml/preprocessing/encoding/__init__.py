"""
Encoding transformers for categorical feature encoding.

This module provides various encoding techniques for converting between
categorical (nominal/string) and numeric representations.

Available transformers:
    - OneHotEncoder: One column per category, indicator-coded.
    - OrdinalEncoder: Categories mapped to integer codes.
    - LabelEncoder: String labels mapped to a categorical index.
    - RareCategoryEncoder: Infrequent categories merged into a single bucket.

Examples
--------
>>> from tuiml.preprocessing.encoding import OneHotEncoder
>>> import numpy as np

>>> # One-hot encode categorical features
>>> X = np.array([[0, 1], [1, 2], [2, 0]])  # Categorical indices
>>> encoder = OneHotEncoder(categories=[[0, 1, 2], [0, 1, 2]])
>>> X_encoded = encoder.fit_transform(X)
"""

from tuiml.preprocessing.encoding.one_hot import OneHotEncoder
from tuiml.preprocessing.encoding.ordinal import OrdinalEncoder
from tuiml.preprocessing.encoding.label import LabelEncoder
from tuiml.preprocessing.encoding.rare_category import RareCategoryEncoder

__all__ = [
    "OneHotEncoder",
    "OrdinalEncoder",
    "LabelEncoder",
    "RareCategoryEncoder",
]
