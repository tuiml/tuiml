"""
ValueClipper transformer.

ValueClipper values to a specified range.
"""

from typing import Optional, List, Union
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["outlier", "clip", "range"], version="1.0.0")
class ValueClipper(Transformer):
    """Clip feature values to a specified range or percentile.

    Constrains the numerical range of features by capping values at the 
    specified lower and upper thresholds.

    Parameters
    ----------
    lower : float, optional
        Fixed lower bound. Values below this are set to ``lower``.

    upper : float, optional
        Fixed upper bound. Values above this are set to ``upper``.

    percentile : tuple of float, optional
        Percentile boundaries expressed as ``(lower_pct, upper_pct)``. 
        If provided, the absolute bounds are calculated from the training 
        data (e.g., ``(1, 99)`` for winsorization).

    columns : list of int, optional
        Indices of columns to clip. If ``None``, all columns are processed.

    Attributes
    ----------
    bounds_ : dict
        Calculated (lower, upper) boundaries used for each column.

    See Also
    --------
    :class:`~tuiml.preprocessing.outliers.IQROutlierDetector` : Clipping based on IQR.

    Examples
    --------
    Clip values to the [0, 10] range:

    >>> from tuiml.preprocessing.outliers import ValueClipper
    >>> import numpy as np
    >>> X = np.array([[-10], [5], [20]])
    >>> clipper = ValueClipper(lower=0, upper=10)
    >>> X_clipped = clipper.fit_transform(X)
    >>> print(X_clipped.flatten())
    [ 0.  5. 10.]
    """

    def __init__(
        self,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        percentile: Optional[tuple] = None,
        columns: Optional[List[int]] = None,
    ):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.percentile = percentile
        self.columns = columns

    @classmethod
    def get_parameter_schema(cls):
        return {
            "lower": {
                "type": ["number", "null"],
                "default": None,
                "description": "Lower bound",
            },
            "upper": {
                "type": ["number", "null"],
                "default": None,
                "description": "Upper bound",
            },
            "percentile": {
                "type": ["array", "null"],
                "items": {"type": "number"},
                "default": None,
                "description": "Percentile bounds (lower_pct, upper_pct)",
            },
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "default": None,
                "description": "Columns to clip",
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "ValueClipper":
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Compute bounds
        self._bounds = {}

        for col in self._columns:
            col_data = X[:, col]
            valid_data = col_data[~np.isnan(col_data)]

            if self.percentile is not None:
                lower_pct, upper_pct = self.percentile
                if len(valid_data) > 0:
                    lower = np.percentile(valid_data, lower_pct)
                    upper = np.percentile(valid_data, upper_pct)
                else:
                    lower, upper = -np.inf, np.inf
            else:
                lower = self.lower if self.lower is not None else -np.inf
                upper = self.upper if self.upper is not None else np.inf

            self._bounds[col] = (lower, upper)

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
            lower, upper = self._bounds[col]
            result[:, col] = np.clip(result[:, col], lower, upper)

        return result

    def __repr__(self) -> str:
        if self.percentile is not None:
            return f"ValueClipper(percentile={self.percentile})"
        return f"ValueClipper(lower={self.lower}, upper={self.upper})"
