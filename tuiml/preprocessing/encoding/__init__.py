"""
Encoding transformers for categorical feature encoding.

This module provides various encoding techniques for converting between
categorical (nominal/string) and numeric representations,
based on WEKA's unsupervised attribute filters.

Available transformers:
    - OneHotEncoder: One-hot encoding (WEKA: NominalToBinary)
    - OrdinalEncoder: Ordinal encoding (WEKA: OrdinalToNumeric)
    - LabelEncoder: String to categorical (WEKA: StringToNominal)
    - RareCategoryEncoder: Merge rare categories (WEKA: MergeInfrequentNominalValues)

Examples
--------
>>> from tuiml.preprocessing.encoding import OneHotEncoder
>>> import numpy as np

>>> # One-hot encode categorical features
>>> X = np.array([[0, 1], [1, 2], [2, 0]])  # Categorical indices
>>> encoder = OneHotEncoder(categories=[[0, 1, 2], [0, 1, 2]])
>>> X_encoded = encoder.fit_transform(X)
"""

from tuiml.preprocessing.encoding.nominal_to_binary import OneHotEncoder
from tuiml.preprocessing.encoding.ordinal_to_numeric import OrdinalEncoder
from tuiml.preprocessing.encoding.string_to_nominal import LabelEncoder
from tuiml.preprocessing.encoding.merge_infrequent import RareCategoryEncoder

__all__ = [
    "OneHotEncoder",
    "OrdinalEncoder",
    "LabelEncoder",
    "RareCategoryEncoder",
]
