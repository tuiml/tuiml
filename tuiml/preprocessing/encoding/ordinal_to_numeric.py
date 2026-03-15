"""
OrdinalEncoder transformer.

Convert ordinal (ordered categorical) to numeric values.
"""

from typing import Optional, List, Dict
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["encoding", "ordinal", "numeric"], version="1.0.0")
class OrdinalEncoder(Transformer):
    """Convert ordinal (ordered categorical) attributes to numeric.

    Maps categories to integer values based on a specified or inferred order. 
    This is suitable for features where the order carries meaning (e.g., "low", "medium", "high").

    Parameters
    ----------
    categories : dict or list, optional
        The order of categories for each column.
        - ``Dict``: ``{col_idx: [cat1, cat2, ...]}``
        - ``List``: ``[[cats_col0], [cats_col1], ...]``
        If ``None``, the order is inferred from the first occurrence in the data.

    columns : list of int, optional
        Indices of columns to encode. If ``None``, all columns are encoded.

    Attributes
    ----------
    category_maps_ : dict
        Mapping of categories to integers for each column.

    See Also
    --------
    :class:`~tuiml.preprocessing.encoding.LabelEncoder` : For non-ordered categories.

    Examples
    --------
    Encode ordered categories:

    >>> from tuiml.preprocessing.encoding import OrdinalEncoder
    >>> import numpy as np
    >>> X = np.array([['low'], ['medium'], ['high']], dtype=object)
    >>> encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
    >>> X_encoded = encoder.fit_transform(X)
    >>> print(X_encoded.flatten())
    [0. 1. 2.]
    """

    def __init__(
        self,
        categories: Optional[Dict[int, List]] = None,
        columns: Optional[List[int]] = None,
    ):
        super().__init__()
        self.categories = categories
        self.columns = columns

    @classmethod
    def get_parameter_schema(cls):
        return {
            "categories": {
                "type": ["object", "array", "null"],
                "description": "Ordered categories for each column",
            },
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "default": None,
                "description": "Columns to encode",
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "OrdinalEncoder":
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Build category mappings
        self._category_maps = {}

        for i, col in enumerate(self._columns):
            if self.categories is not None:
                if isinstance(self.categories, dict):
                    cats = self.categories.get(col, None)
                else:
                    cats = self.categories[i] if i < len(self.categories) else None
            else:
                cats = None

            if cats is None:
                # Infer order from first occurrence
                seen = []
                for val in X[:, col]:
                    if val not in seen and not (isinstance(val, float) and np.isnan(val)):
                        seen.append(val)
                cats = seen

            self._category_maps[col] = {cat: idx for idx, cat in enumerate(cats)}

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

        result = np.zeros_like(X, dtype=float)

        for i in range(self._n_features_in):
            if i in self._columns:
                cat_map = self._category_maps[i]
                for j in range(X.shape[0]):
                    val = X[j, i]
                    if isinstance(val, float) and np.isnan(val):
                        result[j, i] = np.nan
                    else:
                        result[j, i] = cat_map.get(val, -1)
            else:
                result[:, i] = X[:, i].astype(float)

        return result

    def __repr__(self) -> str:
        return f"OrdinalEncoder(columns={self.columns})"
