"""
LabelEncoder transformer.

Convert string attributes to nominal (integer-encoded categorical).
"""

from typing import Optional, List
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["encoding", "string", "nominal"], version="1.0.0")
class LabelEncoder(Transformer):
    """Convert string attributes to nominal (integer-encoded categorical).

    Encodes categorical string values into distinct integers. This is a common 
    preprocessing step for algorithms that require numerical input.

    Parameters
    ----------
    columns : list of int, optional
        Indices of columns to convert. If ``None``, automatically detects and 
        converts all non-numeric (string/object) columns.

    Attributes
    ----------
    categories_ : dict
        Mapping of column index to the list of unique categories found.

    Notes
    -----
    - Unlike ``OrdinalEncoder``, this does not assume any particular order.
    - It can handle unseen categories by mapping them to -1 during transform.

    See Also
    --------
    :class:`~tuiml.preprocessing.encoding.OneHotEncoder` : Binary vector representation.
    :class:`~tuiml.preprocessing.encoding.OrdinalEncoder` : Ordered integer encoding.

    Examples
    --------
    Encode string categories to integers:

    >>> from tuiml.preprocessing.encoding import LabelEncoder
    >>> import numpy as np
    >>> X = np.array([['cat'], ['dog'], ['cat']], dtype=object)
    >>> encoder = LabelEncoder()
    >>> X_encoded = encoder.fit_transform(X)
    >>> print(X_encoded.flatten())
    [0. 1. 0.]
    """

    def __init__(
        self,
        columns: Optional[List[int]] = None,
    ):
        super().__init__()
        self.columns = columns

    @classmethod
    def get_parameter_schema(cls):
        return {
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "default": None,
                "description": "Columns to convert (None for auto-detect)",
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "LabelEncoder":
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        # Determine columns to convert
        if self.columns is not None:
            self._columns = self.columns
        else:
            # Auto-detect string/object columns
            self._columns = []
            for i in range(self._n_features_in):
                try:
                    X[:, i].astype(float)
                except (ValueError, TypeError):
                    self._columns.append(i)

        # Build category mappings
        self._categories = {}
        self._category_maps = {}

        for col in self._columns:
            unique_vals = []
            for val in X[:, col]:
                if val not in unique_vals and val is not None:
                    try:
                        if not np.isnan(float(val)):
                            unique_vals.append(val)
                    except (ValueError, TypeError):
                        unique_vals.append(val)

            self._categories[col] = unique_vals
            self._category_maps[col] = {v: i for i, v in enumerate(unique_vals)}

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self._n_features_in}"
            )

        result = np.zeros(X.shape, dtype=float)

        for i in range(self._n_features_in):
            if i in self._columns:
                cat_map = self._category_maps[i]
                for j in range(X.shape[0]):
                    val = X[j, i]
                    if val is None:
                        result[j, i] = np.nan
                    else:
                        result[j, i] = cat_map.get(val, -1)
            else:
                result[:, i] = X[:, i].astype(float)

        return result

    @property
    def categories_(self):
        self._check_is_fitted()
        return dict(self._categories)

    def __repr__(self) -> str:
        return f"LabelEncoder(columns={self.columns})"
