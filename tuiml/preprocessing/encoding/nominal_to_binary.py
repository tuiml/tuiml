"""
OneHotEncoder transformer.

One-hot encoding for categorical features.
"""

from typing import Optional, List
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["encoding", "nominal", "binary", "one-hot"], version="1.0.0")
class OneHotEncoder(Transformer):
    """One-hot encoding for categorical (nominal) features.

    Transforms categorical features into a binary vector representation where 
    each category is represented by a separate column.

    Overview
    --------
    One-hot encoding is essential for algorithms that cannot handle categorical 
    data directly. It creates :math:`N` binary columns for a feature with 
    :math:`N` unique categories (or :math:`N-1` if dropping a category).

    Parameters
    ----------
    categories : list of list, optional
        Manually specified categories for each column. If ``None``, categories 
        are inferred from the training data.

    drop : {"first", "if_binary"}, optional
        Strategy to drop one category per feature to avoid multicollinearity:
        - ``None``: Keep all categories.
        - ``"first"``: Drop the first category.
        - ``"if_binary"``: Drop the first category only if the feature is binary.

    columns : list of int, optional
        Indices of columns to encode. If ``None``, all columns are encoded.

    Attributes
    ----------
    categories_ : list of np.ndarray
        The categories determined during fitting for each encoded column.

    See Also
    --------
    :class:`~tuiml.preprocessing.encoding.LabelEncoder` : Integer encoding.
    :class:`~tuiml.preprocessing.encoding.OrdinalEncoder` : Ordered integer encoding.

    Examples
    --------
    Encode a single categorical feature:

    >>> from tuiml.preprocessing.encoding import OneHotEncoder
    >>> import numpy as np
    >>> X = np.array([[0], [1], [2], [0]])
    >>> encoder = OneHotEncoder()
    >>> X_encoded = encoder.fit_transform(X)
    >>> print(X_encoded.shape)
    (4, 3)
    """

    def __init__(
        self,
        categories: Optional[List[List]] = None,
        drop: Optional[str] = None,
        columns: Optional[List[int]] = None,
    ):
        super().__init__()
        self.categories = categories
        self.drop = drop
        self.columns = columns

    @classmethod
    def get_parameter_schema(cls):
        return {
            "categories": {
                "type": ["array", "null"],
                "description": "Categories for each column",
            },
            "drop": {
                "type": ["string", "null"],
                "default": None,
                "enum": [None, "first", "if_binary"],
                "description": "Drop one category per feature",
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
    ) -> "OneHotEncoder":
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Learn categories
        if self.categories is not None:
            self._categories = self.categories
        else:
            self._categories = []
            for col in self._columns:
                unique_vals = np.unique(X[:, col][~np.isnan(X[:, col])])
                self._categories.append(sorted(unique_vals.tolist()))

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_input(X)

        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self._n_features_in}"
            )

        # Build output
        output_parts = []
        col_idx = 0

        for i in range(self._n_features_in):
            if i in self._columns:
                cats = self._categories[col_idx]
                n_cats = len(cats)

                # Determine how many columns to create
                if self.drop == "first":
                    start_idx = 1
                elif self.drop == "if_binary" and n_cats == 2:
                    start_idx = 1
                else:
                    start_idx = 0

                # One-hot encode
                for cat_idx in range(start_idx, n_cats):
                    cat = cats[cat_idx]
                    col_encoded = (X[:, i] == cat).astype(float)
                    output_parts.append(col_encoded.reshape(-1, 1))

                col_idx += 1
            else:
                output_parts.append(X[:, i:i+1])

        return np.hstack(output_parts)

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        self._check_is_fitted()

        if input_features is not None:
            names = list(input_features)
        elif self._feature_names_in is not None:
            names = list(self._feature_names_in)
        else:
            names = [f"x{i}" for i in range(self._n_features_in)]

        output_names = []
        col_idx = 0

        for i in range(self._n_features_in):
            if i in self._columns:
                cats = self._categories[col_idx]
                n_cats = len(cats)

                if self.drop == "first":
                    start_idx = 1
                elif self.drop == "if_binary" and n_cats == 2:
                    start_idx = 1
                else:
                    start_idx = 0

                for cat_idx in range(start_idx, n_cats):
                    output_names.append(f"{names[i]}_{cats[cat_idx]}")

                col_idx += 1
            else:
                output_names.append(names[i])

        return output_names

    def __repr__(self) -> str:
        return f"OneHotEncoder(drop={self.drop!r})"
