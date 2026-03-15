"""
Time series lag (translate) transformer.

Creates lagged (shifted) features from time series data.
"""

import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["timeseries", "lag", "shift", "translate"], version="1.0.0")
class LagTransformer(Transformer):
    """Create lagged features from time-ordered data.

    Shifts feature values by a specified number of periods. Useful for 
    transforming time series data into a supervised learning format where 
    past values are used to predict future ones.

    Parameters
    ----------
    lag : int, default=-1
        The number of periods to shift.
        - Negative values (e.g., ``-1``): Use past values (most common).
        - Positive values (e.g., ``1``): Use future values (look-ahead).

    columns : list of int, optional
        Indices of columns to shift. If ``None``, all columns are lagged.

    fill_with_missing : bool, default=True
        - If ``True``: Keeps the original number of rows and fills boundary 
          indices with ``np.nan``.
        - If ``False``: Removes the rows that would contain ``np.nan`` results.

    invert_selection : bool, default=False
        If ``True``, applies the lag to all columns *except* those specified 
        in ``columns``.

    Attributes
    ----------
    feature_names_out_ : list of str
        The generated names for the shifted features (e.g., "x-1").

    See Also
    --------
    :class:`~tuiml.preprocessing.timeseries.DifferenceTransformer` : Computes the change between periods.

    Examples
    --------
    Lag a feature by 1 period to use the previous day's value:

    >>> from tuiml.preprocessing.timeseries import LagTransformer
    >>> import numpy as np
    >>> X = np.array([[10], [20], [30], [40]])
    >>> lagger = LagTransformer(lag=-1)
    >>> X_lagged = lagger.fit_transform(X)
    >>> print(X_lagged.flatten())
    [nan 10. 20. 30.]
    """

    def __init__(
        self,
        lag: int = -1,
        columns: list[int] | None = None,
        fill_with_missing: bool = True,
        invert_selection: bool = False,
    ):
        super().__init__()
        self.lag = lag
        self.columns = columns
        self.fill_with_missing = fill_with_missing
        self.invert_selection = invert_selection

    @classmethod
    def get_parameter_schema(cls):
        return {
            "lag": {
                "type": "integer",
                "default": -1,
                "description": "Number of instances to shift (negative=past, positive=future)",
            },
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "default": None,
                "description": "Columns to apply lag to (None for all)",
            },
            "fill_with_missing": {
                "type": "boolean",
                "default": True,
                "description": "Use missing values for boundaries (False removes instances)",
            },
            "invert_selection": {
                "type": "boolean",
                "default": False,
                "description": "Apply to non-specified columns instead",
            },
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "LagTransformer":
        """
        Fit the transformer.

        Args:
            X: Input data (n_samples, n_features)
            y: Ignored
            feature_names: Optional feature names

        Returns:
            Self
        """
        X = self._validate_input(X)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        # Determine which columns to transform
        if self.columns is not None:
            selected = set(self.columns)
        else:
            selected = set(range(self._n_features_in))

        if self.invert_selection:
            self._columns = [i for i in range(self._n_features_in) if i not in selected]
        else:
            self._columns = [i for i in range(self._n_features_in) if i in selected]

        # Generate feature names with suffix
        self._generate_feature_names()

        self._is_fitted = True
        return self

    def _generate_feature_names(self):
        """Generate output feature names."""
        if self._feature_names_in is not None:
            base_names = list(self._feature_names_in)
        else:
            base_names = [f"x{i}" for i in range(self._n_features_in)]

        self.feature_names_out_ = []
        sign = "+" if self.lag > 0 else ""

        for i in range(self._n_features_in):
            if i in self._columns:
                self.feature_names_out_.append(f"{base_names[i]}{sign}{self.lag}")
            else:
                self.feature_names_out_.append(base_names[i])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by shifting values.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Transformed data with lagged values
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self._n_features_in}"
            )

        result = X.copy()
        abs_lag = abs(self.lag)

        # Apply lag to selected columns
        for col in self._columns:
            result[:, col] = self._shift_column(X[:, col], self.lag)

        # Handle boundary instances
        if not self.fill_with_missing:
            if self.lag < 0:
                result = result[abs_lag:]
            elif self.lag > 0:
                result = result[:-abs_lag] if abs_lag > 0 else result

        return result

    def _shift_column(self, col: np.ndarray, lag: int) -> np.ndarray:
        """Shift a single column by lag periods."""
        n = len(col)
        result = np.empty(n)
        result[:] = np.nan
        abs_lag = abs(lag)

        if lag < 0:
            # Past values: shift forward
            if abs_lag < n:
                result[abs_lag:] = col[:-abs_lag]
        elif lag > 0:
            # Future values: shift backward
            if abs_lag < n:
                result[:-abs_lag] = col[abs_lag:]
        else:
            result = col.copy()

        return result

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()
        return self.feature_names_out_

    def __repr__(self) -> str:
        return (
            f"LagTransformer(lag={self.lag}, columns={self.columns}, "
            f"fill_with_missing={self.fill_with_missing})"
        )
