"""
QuantileDiscretizer transformer.

Equal-frequency binning for continuous features.
"""

from typing import Optional, List
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["discretization", "binning", "equal-frequency", "quantile"], version="1.0.0")
class QuantileDiscretizer(Transformer):
    """Discretize continuous features into equal-frequency bins.

    Partition continuous data into bins such that each bin contains 
    approximately the same number of samples.

    Overview
    --------
    Quantile binning (equal-frequency) is more robust to skewed distributions 
    than equal-width binning, as it adapts the bin widths to the density of 
    the data.

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins to create.

    columns : list of int, optional
        Indices of columns to discretize. If ``None``, all columns are processed.

    Attributes
    ----------
    bin_edges_ : dict
        Mapping of column index to the calculated quantile-based edges.

    See Also
    --------
    :class:`~tuiml.preprocessing.discretization.EqualWidthDiscretizer` : Equal-width binning.

    Examples
    --------
    Discretize values into 5 equal-frequency bins:

    >>> from tuiml.preprocessing.discretization import QuantileDiscretizer
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [10], [11], [12], [20], [21]])
    >>> discretizer = QuantileDiscretizer(n_bins=4)
    >>> X_binned = discretizer.fit_transform(X)
    """

    def __init__(
        self,
        n_bins: int = 10,
        columns: Optional[List[int]] = None,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.columns = columns

    @classmethod
    def get_parameter_schema(cls):
        return {
            "n_bins": {
                "type": "integer",
                "default": 10,
                "minimum": 2,
                "description": "Number of bins",
            },
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "default": None,
                "description": "Columns to discretize",
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "QuantileDiscretizer":
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Compute bin edges for each column using quantiles
        self._bin_edges = {}

        for col in self._columns:
            col_data = X[:, col]
            valid_data = col_data[~np.isnan(col_data)]

            if len(valid_data) == 0:
                self._bin_edges[col] = np.array([0, 1])
                continue

            # Compute quantile-based edges
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(valid_data, quantiles)

            # Ensure unique edges
            edges = np.unique(edges)
            self._bin_edges[col] = edges

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
            edges = self._bin_edges[col]
            if len(edges) > 2:
                binned = np.digitize(X[:, col], edges[1:-1])
            else:
                binned = np.zeros(X.shape[0])
            result[:, col] = binned

        return result

    @property
    def bin_edges_(self):
        self._check_is_fitted()
        return dict(self._bin_edges)

    def __repr__(self) -> str:
        return f"QuantileDiscretizer(n_bins={self.n_bins})"
