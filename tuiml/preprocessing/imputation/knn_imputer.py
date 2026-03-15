"""
KNNImputer transformer.

K-Nearest Neighbors imputation for missing values.
"""

from typing import Optional, List
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["imputation", "missing", "knn", "imputer"], version="1.0.0")
class KNNImputer(Transformer):
    """Impute missing values using K-Nearest Neighbors.

    Each sample's missing values are imputed using the mean value from its 
    :math:`k` nearest neighbors in the training set.

    Overview
    --------
    KNN imputation is a multivariate strategy that uses the similarity between 
    samples to estimate missing values. It is generally more accurate than 
    simple univariate imputation when features are correlated.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.

    weights : {"uniform", "distance"}, default="uniform"
        Weight function used in prediction:
        - ``"uniform"``: All neighbors are weighted equally.
        - ``"distance"``: Weights neighbors by the inverse of their distance.

    columns : list of int, optional
        Indices of columns to impute. If ``None``, all columns are processed.

    See Also
    --------
    :class:`~tuiml.preprocessing.imputation.SimpleImputer` : Univariate imputation.

    Examples
    --------
    Impute using 2 nearest neighbors:

    >>> from tuiml.preprocessing.imputation import KNNImputer
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [np.nan, 6], [8, 8]])
    >>> imputer = KNNImputer(n_neighbors=2)
    >>> X_imputed = imputer.fit_transform(X)
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        columns: Optional[List[int]] = None,
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.columns = columns

        if weights not in ("uniform", "distance"):
            raise ValueError(f"weights must be 'uniform' or 'distance'")

    @classmethod
    def get_parameter_schema(cls):
        return {
            "n_neighbors": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Number of neighbors",
            },
            "weights": {
                "type": "string",
                "default": "uniform",
                "enum": ["uniform", "distance"],
                "description": "Weight function",
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
    ) -> "KNNImputer":
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Store training data for neighbor computation
        self._X_fit = X.copy()

        self._is_fitted = True
        return self

    def _compute_distances(self, x1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances handling NaN values."""
        distances = np.zeros(X2.shape[0])

        for i in range(X2.shape[0]):
            # Only use features where both have values
            mask = ~(np.isnan(x1) | np.isnan(X2[i]))
            if mask.sum() == 0:
                distances[i] = np.inf
            else:
                diff = x1[mask] - X2[i][mask]
                distances[i] = np.sqrt(np.sum(diff ** 2))

        return distances

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_input(X)

        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self._n_features_in}"
            )

        result = X.copy()

        for i in range(X.shape[0]):
            row = result[i]
            missing_cols = [c for c in self._columns if np.isnan(row[c])]

            if not missing_cols:
                continue

            # Find neighbors
            distances = self._compute_distances(row, self._X_fit)

            # Get k nearest neighbors with valid values
            for col in missing_cols:
                # Only consider neighbors with non-missing values in this column
                valid_mask = ~np.isnan(self._X_fit[:, col])
                valid_distances = distances.copy()
                valid_distances[~valid_mask] = np.inf

                k = min(self.n_neighbors, valid_mask.sum())
                if k == 0:
                    # Fall back to column mean
                    col_data = self._X_fit[:, col]
                    result[i, col] = np.nanmean(col_data)
                    continue

                neighbor_idx = np.argsort(valid_distances)[:k]
                neighbor_values = self._X_fit[neighbor_idx, col]
                neighbor_dists = valid_distances[neighbor_idx]

                if self.weights == "uniform":
                    result[i, col] = np.mean(neighbor_values)
                else:  # distance
                    # Handle zero distances
                    neighbor_dists = np.maximum(neighbor_dists, 1e-10)
                    weights = 1.0 / neighbor_dists
                    weights /= weights.sum()
                    result[i, col] = np.sum(weights * neighbor_values)

        return result

    def __repr__(self) -> str:
        return f"KNNImputer(n_neighbors={self.n_neighbors}, weights={self.weights!r})"
