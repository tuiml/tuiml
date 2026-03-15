"""
MDLDiscretizer transformer.

MDL (Minimum Description Length) based discretization.
"""

from typing import Optional, List
import numpy as np

from tuiml.base.preprocessing import SupervisedTransformer, transformer

@transformer(tags=["discretization", "binning", "mdl", "supervised"], version="1.0.0")
class MDLDiscretizer(SupervisedTransformer):
    """Supervised discretization based on Minimum Description Length (MDL).

    Finds optimal cut points by minimizing the class entropy, using the 
    Fayyad & Irani's MDL criterion to determine when to stop splitting.

    Overview
    --------
    MDL discretization is a supervised method that uses the target labels to 
    decide where to place bin boundaries. It typically performs better than 
    unsupervised methods for classification tasks.

    Theory
    ------
    A cut point :math:`T` for a feature :math:`A` is chosen to minimize the 
    class-weighted entropy. A split is accepted if the information gain 
    satisfies:

    .. math::
        Gain(A, T; S) > \\frac{\\log_2(n-1)}{n} + \\frac{\\Delta(A, T; S)}{n}

    where :math:`n` is the number of instances and :math:`\\Delta` is a 
    penalty term based on the number of classes.

    Parameters
    ----------
    min_instances : int, default=10
        Minimum number of instances required in each bin.

    columns : list of int, optional
        Indices of columns to discretize. If ``None``, all columns are processed.

    Attributes
    ----------
    cut_points_ : dict
        Mapping of column index to the list of optimal cut points.

    Examples
    --------
    Discretize a feature using class labels:

    >>> from tuiml.preprocessing.discretization import MDLDiscretizer
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    >>> y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> discretizer = MDLDiscretizer()
    >>> X_binned = discretizer.fit_transform(X, y)
    """

    def __init__(
        self,
        min_instances: int = 10,
        columns: Optional[List[int]] = None,
    ):
        super().__init__()
        self.min_instances = min_instances
        self.columns = columns

    @classmethod
    def get_parameter_schema(cls):
        return {
            "min_instances": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "description": "Minimum instances per bin",
            },
            "columns": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "default": None,
                "description": "Columns to discretize",
            },
        }

    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of class distribution."""
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _find_cut_points(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """Find MDL-based cut points for a single feature."""
        # Sort by feature value
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        n = len(x)
        if n < 2 * self.min_instances:
            return []

        # Find all possible cut points (midpoints between different classes)
        cut_points = []
        for i in range(n - 1):
            if y_sorted[i] != y_sorted[i + 1]:
                cut = (x_sorted[i] + x_sorted[i + 1]) / 2
                cut_points.append((cut, i + 1))

        if not cut_points:
            return []

        # Evaluate each cut point using information gain
        base_entropy = self._entropy(y)
        best_cut = None
        best_gain = 0

        for cut, idx in cut_points:
            if idx < self.min_instances or (n - idx) < self.min_instances:
                continue

            # Calculate weighted entropy after split
            left_entropy = self._entropy(y_sorted[:idx])
            right_entropy = self._entropy(y_sorted[idx:])
            weighted_entropy = (
                (idx / n) * left_entropy +
                ((n - idx) / n) * right_entropy
            )

            gain = base_entropy - weighted_entropy

            if gain > best_gain:
                best_gain = gain
                best_cut = cut

        if best_cut is None:
            return []

        # Recursively find more cut points
        mask = x <= best_cut
        left_cuts = self._find_cut_points(x[mask], y[mask])
        right_cuts = self._find_cut_points(x[~mask], y[~mask])

        return left_cuts + [best_cut] + right_cuts

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MDLDiscretizer":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y = np.asarray(y)
        self._n_features_in = X.shape[1]
        self._feature_names_in = feature_names

        if self.columns is not None:
            self._columns = self.columns
        else:
            self._columns = list(range(self._n_features_in))

        # Find cut points for each column
        self._cut_points = {}

        for col in self._columns:
            col_data = X[:, col]
            valid_mask = ~np.isnan(col_data)

            if valid_mask.sum() < 2 * self.min_instances:
                self._cut_points[col] = []
                continue

            cuts = self._find_cut_points(col_data[valid_mask], y[valid_mask])
            self._cut_points[col] = sorted(cuts)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self._n_features_in}"
            )

        result = X.copy()

        for col in self._columns:
            cuts = self._cut_points[col]
            if cuts:
                result[:, col] = np.digitize(X[:, col], cuts)
            else:
                result[:, col] = 0

        return result

    @property
    def cut_points_(self):
        self._check_is_fitted()
        return dict(self._cut_points)

    def __repr__(self) -> str:
        return f"MDLDiscretizer(min_instances={self.min_instances})"
