"""
RareCategoryEncoder filter.

Merge infrequent (rare) nominal values into a single category.
"""

from typing import Optional, List, Dict
import numpy as np
from collections import Counter

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["nominal", "merge", "infrequent", "rare"], version="1.0.0")
class RareCategoryEncoder(Transformer):
    """
    Merge infrequent nominal values into a single category.

    Values that appear below a frequency threshold are merged into
    a single "other" category.

    Parameters
    ----------
    min_frequency : int or float, default=5
        Minimum frequency for a value to be kept.
        - int: Absolute count threshold
        - float (0-1): Proportion threshold
    columns : list of int, optional
        Indices of columns to process. If None, processes all.
    merged_value : str or int, default=-1
        Value to use for merged categories.
        - For numeric encoding: typically -1 or max+1
        - For string: "other", "rare", etc.

    Attributes
    ----------
    value_maps_ : dict
        Mapping of original values to merged values for each column.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.preprocessing.encoding import RareCategoryEncoder

    >>> # Values: 0 appears 5 times, 1 appears 3 times, 2 appears 1 time
    >>> X = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [2]])

    >>> # Merge values appearing less than 3 times
    >>> merger = RareCategoryEncoder(min_frequency=3)
    >>> X_merged = merger.fit_transform(X)
    >>> # Value 2 is merged into -1
    """

    def __init__(
        self,
        min_frequency: float = 5,
        columns: Optional[List[int]] = None,
        merged_value: int = -1,
    ):
        """
        Initialize RareCategoryEncoder.

        Args:
            min_frequency: Minimum frequency threshold
            columns: Columns to process (None for all)
            merged_value: Value to use for merged categories
        """
        super().__init__()
        self.min_frequency = min_frequency
        self.columns = columns
        self.merged_value = merged_value

    @classmethod
    def get_parameter_schema(cls):
        return {
            "min_frequency": {
                "type": "number",
                "default": 5,
                "minimum": 0,
                "description": "Minimum frequency threshold (int=count, float=proportion)",
            },
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "default": None,
                "description": "Columns to process (None for all)",
            },
            "merged_value": {
                "type": "integer",
                "default": -1,
                "description": "Value to use for merged categories",
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "RareCategoryEncoder":
        """
        Learn which values to merge.

        Args:
            X: Input data
            y: Ignored
            feature_names: Optional feature names

        Returns:
            Self
        """
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names
        n_samples = X.shape[0]

        # Determine columns
        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Calculate threshold
        if 0 < self.min_frequency < 1:
            threshold = int(self.min_frequency * n_samples)
        else:
            threshold = int(self.min_frequency)

        # Build value maps for each column
        self._value_maps = {}

        for col in self._columns:
            col_data = X[:, col]
            # Count frequencies (excluding NaN)
            valid_data = col_data[~np.isnan(col_data)]
            counts = Counter(valid_data)

            # Create mapping
            value_map = {}
            for value, count in counts.items():
                if count >= threshold:
                    value_map[value] = value  # Keep original
                else:
                    value_map[value] = self.merged_value  # Merge

            self._value_maps[col] = value_map

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the value merging.

        Args:
            X: Input data

        Returns:
            Data with infrequent values merged
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self._n_features_in}"
            )

        result = X.copy()

        for col in self._columns:
            value_map = self._value_maps[col]
            col_data = result[:, col]

            for i, val in enumerate(col_data):
                if np.isnan(val):
                    continue
                # Map value, defaulting to merged_value for unseen values
                result[i, col] = value_map.get(val, self.merged_value)

        return result

    @property
    def value_maps_(self) -> Dict[int, Dict]:
        """Mapping of original values to merged values for each column."""
        self._check_is_fitted()
        return dict(self._value_maps)

    def __repr__(self) -> str:
        return (
            f"RareCategoryEncoder(min_frequency={self.min_frequency}, "
            f"merged_value={self.merged_value})"
        )
