"""Filling in missing values.

Most algorithms cannot fit on ``NaN`` at all, so missing data has to be
resolved before anything else in a pipeline. These transformers learn what to
fill from the training data and apply the same values later, which is what
keeps a test split from influencing its own imputation.

Transformers
------------
- **SimpleImputer:** Replace with the column's mean, median, most frequent
  value, or a constant. Fast, and the sensible default.
- **KNNImputer:** Replace using the k most similar rows. More faithful when
  features are correlated, and considerably slower — it searches neighbours
  for every missing entry.

Notes
-----
Impute before scaling: a scaler fitted on data still containing ``NaN``
propagates it into the mean and variance. The ``"standard"`` and ``"full"``
pipeline presets already order the two correctly.

Missingness is sometimes informative — a blank field can mean "not
applicable" rather than "unknown". Where that is true, record it as its own
feature before imputing, or the signal is erased.
"""

from tuiml.preprocessing.imputation.simple_imputer import SimpleImputer
from tuiml.preprocessing.imputation.knn_imputer import KNNImputer

__all__ = [
    "SimpleImputer",
    "KNNImputer",
]
