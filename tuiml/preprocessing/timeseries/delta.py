"""
Time series delta (difference) transformer.

Creates difference features from time series data.
"""

import numpy as np

from tuiml.base.preprocessing import Transformer, transformer

@transformer(tags=["timeseries", "delta", "difference"], version="1.0.0")
class DifferenceTransformer(Transformer):
    """Compute the difference between periods in time-ordered data.

    Calculates the change in feature values between the current instance and 
    a lagged instance. This is a standard technique for making a non-stationary 
    time series stationary.

    Theory
    ------
    The differenced value :math:`\\Delta x_t` is calculated as:

    .. math::
        \\Delta x_t = x_t - x_{t-k}

    where :math:`k` is the lag period.

    Parameters
    ----------
    lag : int, default=-1
        The number of periods to shift for differencing.
        - Negative values (e.g., ``-1``): Subtract the previous value 
          (``current - past``).
        - Positive values (e.g., ``1``): Subtract the next value 
          (``current - future``).

    columns : list of int, optional
        Indices of numeric columns to difference. If ``None``, all numeric 
        columns are processed.

    fill_with_missing : bool, default=True
        - If ``True``: Keeps the original number of rows and fills boundary 
          indices with ``np.nan``.
        - If ``False``: Removes the rows that would contain ``np.nan`` results.

    invert_selection : bool, default=False
        If ``True``, applies the difference to all columns *except* those 
        specified in ``columns``.

    Attributes
    ----------
    feature_names_out_ : list of str
        The generated names for the difference features (e.g., "x d-1").

    See Also
    --------
    :class:`~tuiml.preprocessing.timeseries.LagTransformer` : Simple value shifting.

    Examples
    --------
    Calculate day-over-day changes:

    >>> from tuiml.preprocessing.timeseries import DifferenceTransformer
    >>> import numpy as np
    >>> X = np.array([[10], [12], [11], [15]])
    >>> differencer = DifferenceTransformer(lag=-1)
    >>> X_diff = differencer.fit_transform(X)
    >>> print(X_diff.flatten())
    [nan  2. -1.  4.]
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
                "description": "Number of instances for difference (negative=past, positive=future)",
            },
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "default": None,
                "description": "Columns to compute differences for (None for all)",
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
    ) -> "DifferenceTransformer":
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
                self.feature_names_out_.append(f"{base_names[i]} d{sign}{self.lag}")
            else:
                self.feature_names_out_.append(base_names[i])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by computing differences.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Transformed data with difference values
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self._n_features_in}"
            )

        result = X.copy()
        abs_lag = abs(self.lag)

        # Apply delta to selected columns
        for col in self._columns:
            result[:, col] = self._compute_delta(X[:, col], self.lag)

        # Handle boundary instances
        if not self.fill_with_missing:
            if self.lag < 0:
                result = result[abs_lag:]
            elif self.lag > 0:
                result = result[:-abs_lag] if abs_lag > 0 else result

        return result

    def _compute_delta(self, col: np.ndarray, lag: int) -> np.ndarray:
        """Compute difference for a single column."""
        n = len(col)
        result = np.empty(n)
        result[:] = np.nan
        abs_lag = abs(lag)

        if lag < 0:
            # Difference from past: current - previous
            if abs_lag < n:
                for i in range(abs_lag, n):
                    if not np.isnan(col[i]) and not np.isnan(col[i - abs_lag]):
                        result[i] = col[i] - col[i - abs_lag]
        elif lag > 0:
            # Difference from future: current - next
            if abs_lag < n:
                for i in range(n - abs_lag):
                    if not np.isnan(col[i]) and not np.isnan(col[i + abs_lag]):
                        result[i] = col[i] - col[i + abs_lag]
        else:
            result = np.zeros(n)

        return result

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()
        return self.feature_names_out_

    def __repr__(self) -> str:
        return (
            f"DifferenceTransformer(lag={self.lag}, columns={self.columns}, "
            f"fill_with_missing={self.fill_with_missing})"
        )
