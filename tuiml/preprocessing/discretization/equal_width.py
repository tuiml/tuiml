"""
EqualWidthDiscretizer transformer.

Equal-width binning for continuous features.
"""

from typing import Optional, List
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["discretization", "binning", "equal-width"], version="1.0.0")
class EqualWidthDiscretizer(Transformer):
    """Discretize continuous features into equal-width bins.

    Partition consecutive values of numerical features into a set of bins of 
    identical width.

    Overview
    --------
    Equal-width discretization divides the range of a feature into :math:`N` 
    intervals of the same size. This can help in reducing the influence of 
    outliers (once binned) and simplifying complex continuous relationships.

    Theory
    ------
    The width :math:`w` of each bin for a feature with range 
    :math:`[x_{min}, x_{max}]` is:

    .. math::
        w = \frac{x_{max} - x_{min}}{N}

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins to create.

    columns : list of int, optional
        Indices of columns to discretize. If ``None``, all columns are processed.

    Attributes
    ----------
    bin_edges_ : dict
        Mapping of column index to the calculated bin edges.

    See Also
    --------
    :class:`~tuiml.preprocessing.discretization.QuantileDiscretizer` : Equal-frequency binning.
    :class:`~tuiml.preprocessing.discretization.MDLDiscretizer` : Entropy-based binning.

    Examples
    --------
    Discretize values from 1 to 10 into 5 bins:

    >>> from tuiml.preprocessing.discretization import EqualWidthDiscretizer
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    >>> discretizer = EqualWidthDiscretizer(n_bins=5)
    >>> X_binned = discretizer.fit_transform(X)
    >>> print(X_binned.flatten())
    [0. 0. 1. 1. 2. 2. 3. 3. 4. 4.]
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
    ) -> "EqualWidthDiscretizer":
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Compute bin edges for each column
        self._bin_edges = {}

        for col in self._columns:
            col_data = X[:, col]
            valid_data = col_data[~np.isnan(col_data)]

            if len(valid_data) == 0:
                self._bin_edges[col] = np.array([0, 1])
                continue

            min_val = np.min(valid_data)
            max_val = np.max(valid_data)

            # Create equal-width bins
            edges = np.linspace(min_val, max_val, self.n_bins + 1)
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
            # Digitize returns bin indices (1 to n_bins)
            # We want 0 to n_bins-1
            binned = np.digitize(X[:, col], edges[1:-1])
            result[:, col] = binned

        return result

    @property
    def bin_edges_(self):
        self._check_is_fitted()
        return dict(self._bin_edges)

    def __repr__(self) -> str:
        return f"EqualWidthDiscretizer(n_bins={self.n_bins})"
