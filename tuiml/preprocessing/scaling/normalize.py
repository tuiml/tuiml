"""
MinMaxScaler transformer.

Min-max normalization to scale features to a specified range.
"""

from typing import Optional, List
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["scaling", "normalization", "min-max"], version="1.0.0")
class MinMaxScaler(Transformer):
    """Min-max normalization to scale features to a specified range.

    Transforms features by scaling each feature to a given range, typically [0, 1].

    Overview
    --------
    Min-max scaling is a common normalization technique that preserves the relative 
    distances between values while mapping them into a bounded interval.

    Theory
    ------
    The transformation is given by:

    .. math::
        x_{scaled} = \\frac{x - \\min(x_{train})}{\\max(x_{train}) - \\min(x_{train})} \\cdot S + T

    where :math:`S` is the scale (default 1.0) and :math:`T` is the 
    translation (default 0.0).

    Parameters
    ----------
    scale : float, default=1.0
        The scaling factor (width) of the output range.

    translation : float, default=0.0
        The lower bound (minimum value) of the output range.

    columns : list of int, optional
        Indices of columns to transform. If ``None``, transforms all columns.

    Attributes
    ----------
    min_ : np.ndarray of shape (n_selected_columns,)
        Per-column minimum observed in the training data.

    max_ : np.ndarray of shape (n_selected_columns,)
        Per-column maximum observed in the training data.

    range_ : np.ndarray of shape (n_selected_columns,)
        Per-column range (:math:`max - min`) observed in the training data.

    Notes
    -----
    **When to use:**
    - When you need features to be in a specific range (e.g., [0, 1] for neural networks).
    - When you want to preserve zero entries in sparse data.
    - When features do not follow a Gaussian distribution.

    **Limitations:**
    - Highly sensitive to outliers, as they can significantly squash the inliers.

    See Also
    --------
    :class:`~tuiml.preprocessing.scaling.StandardScaler` : Scaling to zero mean and unit variance.

    Examples
    --------
    Scale features to the [0, 1] range:

    >>> from tuiml.preprocessing.scaling import MinMaxScaler
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> normalizer = MinMaxScaler()
    >>> X_norm = normalizer.fit_transform(X)
    >>> print(X_norm.min(axis=0), X_norm.max(axis=0))
    [0. 0.] [1. 1.]
    """

    def __init__(
        self,
        scale: float = 1.0,
        translation: float = 0.0,
        columns: Optional[List[int]] = None,
    ):
        super().__init__()
        self.scale = scale
        self.translation = translation
        self.columns = columns

    @classmethod
    def get_parameter_schema(cls):
        return {
            "scale": {
                "type": "number",
                "default": 1.0,
                "description": "Scaling factor for the output range",
            },
            "translation": {
                "type": "number",
                "default": 0.0,
                "description": "Translation of the output range",
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
    ) -> "MinMaxScaler":
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Compute min and max for each column
        self._min = np.nanmin(X[:, self._columns], axis=0)
        self._max = np.nanmax(X[:, self._columns], axis=0)
        self._range = self._max - self._min

        # Handle zero range
        self._range[self._range == 0] = 1.0

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
            result[:, col] = (
                (X[:, col] - self._min[i]) / self._range[i] * self.scale
                + self.translation
            )

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_input(X)

        result = X.copy()

        for i, col in enumerate(self._columns):
            result[:, col] = (
                (X[:, col] - self.translation) / self.scale * self._range[i]
                + self._min[i]
            )

        return result

    def __repr__(self) -> str:
        return f"MinMaxScaler(scale={self.scale}, translation={self.translation})"
