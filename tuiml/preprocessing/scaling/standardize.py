"""
StandardScaler transformer.

Z-score standardization (zero mean, unit variance).
"""

from typing import Optional, List
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["scaling", "standardization", "z-score"], version="1.0.0")
class StandardScaler(Transformer):
    """Zero-mean and unit-variance standardization.

    Standardizes features by removing the mean and scaling to unit variance. 
    This is often a prerequisite for many machine learning estimators.

    Overview
    --------
    Standardization (or Z-score normalization) transforms each feature such that 
    it has a mean of 0 and a standard deviation of 1.

    Theory
    ------
    The standard score of a sample :math:`x` is calculated as:

    .. math::
        z = \\frac{x - \\mu}{\\sigma}

    where :math:`\\mu` is the mean of the training samples and :math:`\\sigma` is 
    the standard deviation.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling by subtracting the mean.

    with_std : bool, default=True
        If True, scale the data to unit variance (standard deviation of 1).

    columns : list of int, optional
        Indices of columns to transform. If ``None``, all numerical columns 
        are transformed.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_selected_columns,)
        The mean value for each feature in the training set.

    std_ : np.ndarray of shape (n_selected_columns,)
        The standard deviation for each feature in the training set.

    Notes
    -----
    **When to use:**
    - For algorithms sensitive to the scale of features (e.g., SVM, k-NN, Logistic Regression).
    - When features follow a Gaussian-like distribution.
    - Before dimensionality reduction (PCA).

    **Implementation Details:**
    - Uses `np.nanmean` and `np.nanstd` to be robust to missing values.
    - Handles zero variance by setting the scale to 1.0 to avoid division by zero.

    See Also
    --------
    :class:`~tuiml.preprocessing.scaling.MinMaxScaler` : Scaling to a fixed range.
    :class:`~tuiml.preprocessing.scaling.CenterScaler` : Vertical translation only.

    Examples
    --------
    Standardize a simple 2D array:

    >>> from tuiml.preprocessing.scaling import StandardScaler
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> scaler = StandardScaler()
    >>> X_std = scaler.fit_transform(X)
    >>> print(np.round(X_std.mean(axis=0), 2))
    [0. 0.]
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        columns: Optional[List[int]] = None,
    ):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.columns = columns

    @classmethod
    def get_parameter_schema(cls):
        return {
            "with_mean": {
                "type": "boolean",
                "default": True,
                "description": "Center data before scaling",
            },
            "with_std": {
                "type": "boolean",
                "default": True,
                "description": "Scale to unit variance",
            },
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "default": None,
                "description": "Columns to transform (None for all)",
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "StandardScaler":
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Compute mean and std for each column
        self._mean = np.nanmean(X[:, self._columns], axis=0)
        self._std = np.nanstd(X[:, self._columns], axis=0)

        # Handle zero std
        self._std[self._std == 0] = 1.0

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

        for i, col in enumerate(self._columns):
            if self.with_mean:
                result[:, col] = result[:, col] - self._mean[i]
            if self.with_std:
                result[:, col] = result[:, col] / self._std[i]

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_input(X)

        result = X.copy()

        for i, col in enumerate(self._columns):
            if self.with_std:
                result[:, col] = result[:, col] * self._std[i]
            if self.with_mean:
                result[:, col] = result[:, col] + self._mean[i]

        return result

    def __repr__(self) -> str:
        return f"StandardScaler(with_mean={self.with_mean}, with_std={self.with_std})"
