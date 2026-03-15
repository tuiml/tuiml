"""
SimpleImputer transformer.

Simple imputation strategies for missing values.
"""

from typing import Optional, List, Union
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["imputation", "missing", "imputer"], version="1.0.0")
class SimpleImputer(Transformer):
    """Impute missing values using simple statistical strategies.

    Provides basic univariate strategies for completing missing values, 
    either by using mean/median/mode of columns or a constant value.

    Parameters
    ----------
    strategy : {"mean", "median", "most_frequent", "constant"}, default="mean"
        The imputation strategy:
        - ``"mean"``: Replace with the average value (numeric only).
        - ``"median"``: Replace with the middle value (numeric only).
        - ``"most_frequent"``: Replace with the mode (categorical or numeric).
        - ``"constant"``: Replace with ``fill_value``.

    fill_value : float or str, optional
        The value to use when ``strategy="constant"``.

    columns : list of int, optional
        Indices of columns to impute. If ``None``, all columns are processed.

    Attributes
    ----------
    statistics_ : dict
        Mapping of column index to the fill value used for that column.

    Notes
    -----
    **Strategies:**
    - **Mean/Median:** Robust to some outliers but can distort the 
      distribution variance.
    - **Most Frequent:** Suitable for categorical data.
    - **Constant:** Useful for marking missingness as a deliberate category 
      (e.g., "Missing" or -1).

    See Also
    --------
    :class:`~tuiml.preprocessing.imputation.KNNImputer` : Multivariate imputation.

    Examples
    --------
    Impute missing values with column means:

    >>> from tuiml.preprocessing.imputation import SimpleImputer
    >>> import numpy as np
    >>> X = np.array([[1, 2], [np.nan, 3], [7, 6]])
    >>> imputer = SimpleImputer(strategy="mean")
    >>> X_imputed = imputer.fit_transform(X)
    >>> print(X_imputed[1, 0])
    4.0
    """

    def __init__(
        self,
        strategy: str = "mean",
        fill_value: Optional[Union[float, str]] = None,
        columns: Optional[List[int]] = None,
    ):
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns

        valid_strategies = {"mean", "median", "most_frequent", "constant"}
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")

    @classmethod
    def get_parameter_schema(cls):
        return {
            "strategy": {
                "type": "string",
                "default": "mean",
                "enum": ["mean", "median", "most_frequent", "constant"],
                "description": "Imputation strategy",
            },
            "fill_value": {
                "type": ["number", "string", "null"],
                "default": None,
                "description": "Value for constant strategy",
            },
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "default": None,
                "description": "Columns to impute",
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "SimpleImputer":
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Compute fill values
        self._fill_values = {}

        for col in self._columns:
            col_data = X[:, col]
            valid_data = col_data[~np.isnan(col_data)]

            if len(valid_data) == 0:
                self._fill_values[col] = 0.0
                continue

            if self.strategy == "mean":
                self._fill_values[col] = np.mean(valid_data)
            elif self.strategy == "median":
                self._fill_values[col] = np.median(valid_data)
            elif self.strategy == "most_frequent":
                values, counts = np.unique(valid_data, return_counts=True)
                self._fill_values[col] = values[np.argmax(counts)]
            elif self.strategy == "constant":
                if self.fill_value is None:
                    raise ValueError("fill_value required for constant strategy")
                self._fill_values[col] = self.fill_value

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_input(X)

        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self._n_features_in}"
            )

        result = X.copy()

        for col in self._columns:
            mask = np.isnan(result[:, col])
            result[mask, col] = self._fill_values[col]

        return result

    def __repr__(self) -> str:
        return f"SimpleImputer(strategy={self.strategy!r})"
