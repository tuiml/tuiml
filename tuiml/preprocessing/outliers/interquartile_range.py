"""
IQROutlierDetector transformer.

IQR-based outlier detection and handling.
"""

from typing import Optional, List
import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["outlier", "iqr", "detection"], version="1.0.0")
class IQROutlierDetector(Transformer):
    """Detect and handle outliers using the Interquartile Range (IQR).

    Identifies "extreme" values that fall far outside the central 50% of the 
    data distribution.

    Overview
    --------
    The IQR method is a non-parametric outlier detection technique. It defines 
    a "normal" range based on the distance between the first and third quartiles.

    Theory
    ------
    The Interquartile Range (IQR) is :math:`IQR = Q3 - Q1`. A value :math:`x` is 
    considered an outlier if:

    .. math::
        x < Q1 - k \\cdot IQR \\quad \\text{or} \\quad x > Q3 + k \\cdot IQR

    where :math:`k` is the multiplier (typically 1.5).

    Parameters
    ----------
    factor : float, default=1.5
        The multiplier :math:`k`.
        - ``1.5``: Detects "mild" outliers (Tukey's standard).
        - ``3.0``: Detects "extreme" outliers.

    action : {"clip", "nan", "remove"}, default="clip"
        Strategy to handle detected outliers:
        - ``"clip"``: Replace outliers with the nearest boundary value.
        - ``"nan"``: Replace outliers with ``np.nan``.
        - ``"remove"``: Delete rows containing outliers (use with caution in pipelines).

    columns : list of int, optional
        Indices of columns to process. If ``None``, all columns are checked.

    Attributes
    ----------
    bounds_ : dict
        Mapping of column index to the calculated (lower, upper) boundaries.

    Notes
    -----
    **Robustness:**
    - Since it uses quartiles, this method is less sensitive to outliers than 
      mean/sigma-based methods.
    - It assumes a unimodal distribution but not necessarily normality.

    Examples
    --------
    Remove outliers from a distribution:

    >>> from tuiml.preprocessing.outliers import IQROutlierDetector
    >>> import numpy as np
    >>> X = np.array([[10], [12], [11], [10.5], [100.0]])
    >>> detector = IQROutlierDetector(action="clip")
    >>> X_clean = detector.fit_transform(X)
    """

    def __init__(
        self,
        factor: float = 1.5,
        action: str = "clip",
        columns: Optional[List[int]] = None,
    ):
        super().__init__()
        self.factor = factor
        self.action = action
        self.columns = columns

        if action not in ("clip", "nan", "remove"):
            raise ValueError(f"action must be 'clip', 'nan', or 'remove'")

    @classmethod
    def get_parameter_schema(cls):
        return {
            "factor": {
                "type": "number",
                "default": 1.5,
                "minimum": 0,
                "description": "IQR multiplier",
            },
            "action": {
                "type": "string",
                "default": "clip",
                "enum": ["clip", "nan", "remove"],
                "description": "How to handle outliers",
            },
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "default": None,
                "description": "Columns to check",
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "IQROutlierDetector":
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Compute IQR bounds for each column
        self._bounds = {}

        for col in self._columns:
            col_data = X[:, col]
            valid_data = col_data[~np.isnan(col_data)]

            if len(valid_data) == 0:
                self._bounds[col] = (-np.inf, np.inf)
                continue

            q1 = np.percentile(valid_data, 25)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1

            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr
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

        if self.action == "remove":
            # Find rows with outliers
            outlier_rows = np.zeros(X.shape[0], dtype=bool)
            for col in self._columns:
                lower, upper = self._bounds[col]
                outlier_rows |= (X[:, col] < lower) | (X[:, col] > upper)
            return result[~outlier_rows]

        # Handle outliers in place
        for col in self._columns:
            lower, upper = self._bounds[col]

            outlier_low = result[:, col] < lower
            outlier_high = result[:, col] > upper

            if self.action == "clip":
                result[outlier_low, col] = lower
                result[outlier_high, col] = upper
            elif self.action == "nan":
                result[outlier_low | outlier_high, col] = np.nan

        return result

    @property
    def bounds_(self):
        self._check_is_fitted()
        return dict(self._bounds)

    def __repr__(self) -> str:
        return f"IQROutlierDetector(factor={self.factor}, action={self.action!r})"
