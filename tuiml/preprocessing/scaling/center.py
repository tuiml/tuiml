"""
CenterScaler transformer.

Mean centering (subtract mean from features).
"""

from typing import Optional, List
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["scaling", "centering", "mean"], version="1.0.0")
class CenterScaler(Transformer):
    """Mean centering of categorical or numerical features.

    Transforms features by subtracting the mean of each column, effectively 
    resulting in a distribution centered at zero.

    Theory
    ------
    The centered value of a sample :math:`x` is calculated as:

    .. math::
        x_{centered} = x - \\mu

    where :math:`\\mu` is the mean of the training samples.

    Parameters
    ----------
    columns : list of int, optional
        Indices of columns to transform. If ``None``, all columns are transformed.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_selected_columns,)
        The per-column mean observed in the training data.

    See Also
    --------
    :class:`~tuiml.preprocessing.scaling.StandardScaler` : Centering and scaling.

    Examples
    --------
    Center a simple 2D array:

    >>> from tuiml.preprocessing.scaling import CenterScaler
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> center = CenterScaler()
    >>> X_centered = center.fit_transform(X)
    >>> print(np.round(X_centered.mean(axis=0), 2))
    [0. 0.]
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
                "description": "Columns to transform (None for all)",
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "CenterScaler":
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Compute mean for each column
        self._mean = np.nanmean(X[:, self._columns], axis=0)

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
            result[:, col] = result[:, col] - self._mean[i]

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_input(X)

        result = X.copy()

        for i, col in enumerate(self._columns):
            result[:, col] = result[:, col] + self._mean[i]

        return result

    def __repr__(self) -> str:
        return f"CenterScaler(columns={self.columns})"
