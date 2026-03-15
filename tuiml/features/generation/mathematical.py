"""
Mathematical feature transformations.

This module provides mathematical transformations for feature construction.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union
import warnings

from tuiml.base.features import FeatureConstructor, feature_constructor

def _ensure_numpy(X) -> np.ndarray:
    """Convert input to NumPy array."""
    if hasattr(X, 'values'):
        return X.values
    return np.asarray(X, dtype=np.float64)

# Available transformation functions
TRANSFORMS = {
    'log': np.log,
    'log1p': np.log1p,      # log(1 + x), safe for x >= 0
    'log10': np.log10,
    'log2': np.log2,
    'sqrt': np.sqrt,
    'cbrt': np.cbrt,        # cube root
    'exp': np.exp,
    'expm1': np.expm1,      # exp(x) - 1
    'square': np.square,
    'reciprocal': np.reciprocal,  # 1/x
    'abs': np.abs,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'tanh': np.tanh,        # hyperbolic tangent (bounded)
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
}

@feature_constructor(tags=["mathematical", "transformation", "nonlinear"], version="1.0.0")
class MathematicalFeaturesGenerator(FeatureConstructor):
    """Apply mathematical transformations to create new features.

    Applies one or more mathematical functions (log, sqrt, square, etc.) to each 
    input feature, creating derived non-linear features.

    Overview
    --------
    Mathematical transformations are often used to:
    - Linearize non-linear relationships.
    - Normalize distribution of skewed features (e.g., via log transform).
    - Create interaction-like terms for a single feature (e.g., polynomial).

    Parameters
    ----------
    transformations : list of str, default=['log1p', 'sqrt', 'square']
        List of transformations to apply. Supported options:
        - ``"log"``, ``"log1p"``, ``"log10"``, ``"log2"``
        - ``"sqrt"``, ``"cbrt"`` (cube root)
        - ``"exp"``, ``"expm1"``
        - ``"square"``, ``"reciprocal"``, ``"abs"``
        - ``"sin"``, ``"cos"``, ``"tan"``, ``"tanh"``, ``"sigmoid"``

    include_original : bool, default=True
        If True, the original features are preserved in the output.

    handle_invalid : {"raise", "nan", "clip"}, default="nan"
        Strategy to handle math errors (like log of negative):
        - ``"raise"``: Stop and raise ValueError.
        - ``"nan"``: Produce NaN for invalid values.
        - ``"clip"``: Clip input to a valid range (e.g., :math:`x > 0` for log).

    Attributes
    ----------
    n_input_features_ : int
        Number of features in the input data.

    n_output_features_ : int
        Number of features in the transformed data.

    See Also
    --------
    :class:`~tuiml.features.generation.PolynomialFeaturesGenerator` : Cross-feature polynomials.
    :class:`~tuiml.features.generation.BinningFeaturesGenerator` : Continuous to categorical.

    Examples
    --------
    Construct square and square-root features:

    >>> from tuiml.features.generation import MathematicalFeaturesGenerator
    >>> import numpy as np
    >>> X = np.array([[1, 4], [9, 16]])
    >>> math_feat = MathematicalFeaturesGenerator(transformations=['sqrt', 'square'])
    >>> X_new = math_feat.fit_transform(X)
    >>> print(X_new.shape)
    (2, 6)
    """

    def __init__(
        self,
        transformations: List[str] = None,
        include_original: bool = True,
        handle_invalid: str = 'nan'
    ):
        super().__init__()
        self.transformations = transformations or ['log1p', 'sqrt', 'square']
        self.include_original = include_original
        self.handle_invalid = handle_invalid

        self.n_input_features_: Optional[int] = None
        self.n_output_features_: Optional[int] = None
        self._transform_funcs: Optional[List[Callable]] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MathematicalFeaturesGenerator":
        """
        Validate transformations and compute output shape.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : MathematicalFeaturesGenerator
            The fitted transformer.
        """
        X = _ensure_numpy(X)
        n_samples, n_features = X.shape

        self.n_input_features_ = n_features
        self._n_features_in = n_features

        # Validate transformations
        self._transform_funcs = []
        for t in self.transformations:
            if t not in TRANSFORMS:
                raise ValueError(
                    f"Unknown transformation '{t}'. "
                    f"Available: {list(TRANSFORMS.keys())}"
                )
            self._transform_funcs.append((t, TRANSFORMS[t]))

        # Calculate output features
        n_transforms = len(self.transformations)
        if self.include_original:
            self.n_output_features_ = n_features * (1 + n_transforms)
        else:
            self.n_output_features_ = n_features * n_transforms

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply mathematical transformations.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_output_features)
            Transformed data with new features.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)

        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError(
                f"X has {n_features} features, but MathematicalFeaturesGenerator "
                f"is expecting {self.n_input_features_} features."
            )

        result_parts = []

        if self.include_original:
            result_parts.append(X.copy())

        for name, func in self._transform_funcs:
            transformed = self._apply_transform(X, name, func)
            result_parts.append(transformed)

        return np.hstack(result_parts)

    def _apply_transform(self, X: np.ndarray, name: str, func: Callable) -> np.ndarray:
        """Apply a single transformation with error handling.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        name : str
            Name of the transformation.
        func : callable
            Transformation function to apply.

        Returns
        -------
        result : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        X_copy = X.copy()

        if self.handle_invalid == 'clip':
            X_copy = self._clip_for_transform(X_copy, name)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(X_copy)

        if self.handle_invalid == 'nan':
            # Replace inf with nan
            result = np.where(np.isinf(result), np.nan, result)
        elif self.handle_invalid == 'raise':
            if np.any(~np.isfinite(result)):
                raise ValueError(
                    f"Invalid values produced by '{name}' transformation. "
                    f"Use handle_invalid='nan' or 'clip' to handle this."
                )

        return result

    def _clip_for_transform(self, X: np.ndarray, name: str) -> np.ndarray:
        """Clip values to valid range for a given transformation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        name : str
            Name of the transformation.

        Returns
        -------
        X_clipped : ndarray of shape (n_samples, n_features)
            Data with values clipped to valid domain.
        """
        eps = 1e-10

        if name in ['log', 'log10', 'log2']:
            X = np.clip(X, eps, None)
        elif name == 'log1p':
            X = np.clip(X, -1 + eps, None)
        elif name == 'sqrt':
            X = np.clip(X, 0, None)
        elif name == 'reciprocal':
            X = np.where(np.abs(X) < eps, eps * np.sign(X + eps), X)

        return X

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Get output feature names for mathematical transformations.

        Parameters
        ----------
        input_features : list of str, optional
            Input feature names. If None, uses x0, x1, etc.

        Returns
        -------
        feature_names : ndarray of str
            Output feature names.
        """
        self._check_is_fitted()

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_input_features_)]

        names = []

        if self.include_original:
            names.extend(input_features)

        for transform_name, _ in self._transform_funcs:
            for feat in input_features:
                names.append(f"{transform_name}({feat})")

        return np.array(names)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "transformations": {
                "type": "array",
                "items": {"type": "string", "enum": list(TRANSFORMS.keys())},
                "default": ["log1p", "sqrt", "square"],
                "description": "List of transformations to apply"
            },
            "include_original": {
                "type": "boolean",
                "default": True,
                "description": "Include original features in output"
            },
            "handle_invalid": {
                "type": "string",
                "enum": ["raise", "nan", "clip"],
                "default": "nan",
                "description": "How to handle invalid values"
            }
        }

@feature_constructor(tags=["binning", "discretization"], version="1.0.0")
class BinningFeaturesGenerator(FeatureConstructor):
    """Create binned (discretized) versions of continuous features.

    Converts continuous features into categorical bins, which can capture non-linear 
    relationships that a linear model might miss.

    Overview
    --------
    Binning divides the range of a continuous variable into intervals (bins). 
    It can be configured to use equal-width bins, equal-frequency bins (quantiles), 
    or bins determined by k-means clustering.

    Parameters
    ----------
    n_bins : int, default=5
        Number of bins for each feature.

    strategy : {"uniform", "quantile", "kmeans"}, default="quantile"
        Binning strategy:
        - ``"uniform"``: All bins have identical widths.
        - ``"quantile"``: All bins have approximately same number of samples.
        - ``"kmeans"``: Bin edges are determined by 1D k-means clustering.

    encode : {"onehot", "ordinal"}, default="ordinal"
        How to represent binned features:
        - ``"ordinal"``: Integer representing the bin index (0, 1, 2, ...).
        - ``"onehot"``: Vector representing bin membership.

    include_original : bool, default=False
        If True, include original continuous features in output along with binned ones.

    Attributes
    ----------
    n_input_features_ : int
        Number of input features observed.

    bin_edges_ : list of np.ndarray
        Array of bin edges for each feature.

    Notes
    -----
    **When to use:**
    - To linearize non-linear relationships in generalized linear models.
    - To handle outliers (by grouping them into the first or last bin).
    - To simplify a complex distribution into categories.

    See Also
    --------
    :class:`~tuiml.features.generation.MathematicalFeaturesGenerator` : Functional transformations.

    Examples
    --------
    Discretize into 4 equal-frequency bins:

    >>> from tuiml.features.generation import BinningFeaturesGenerator
    >>> import numpy as np
    >>> X = np.random.randn(100, 2)
    >>> binner = BinningFeaturesGenerator(n_bins=4, strategy='quantile')
    >>> X_binned = binner.fit_transform(X)
    >>> print(np.unique(X_binned))
    [0. 1. 2. 3.]
    """

    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = 'quantile',
        encode: str = 'ordinal',
        include_original: bool = False
    ):
        super().__init__()
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.include_original = include_original

        self.n_input_features_: Optional[int] = None
        self.bin_edges_: Optional[List[np.ndarray]] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BinningFeaturesGenerator":
        """
        Compute bin edges for each feature.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used.

        Returns
        -------
        self : BinningFeaturesGenerator
            The fitted transformer.
        """
        X = _ensure_numpy(X)
        n_samples, n_features = X.shape

        self.n_input_features_ = n_features
        self._n_features_in = n_features
        self.bin_edges_ = []

        for i in range(n_features):
            col = X[:, i]
            col = col[~np.isnan(col)]  # Remove NaN

            if self.strategy == 'uniform':
                edges = np.linspace(col.min(), col.max(), self.n_bins + 1)
            elif self.strategy == 'quantile':
                percentiles = np.linspace(0, 100, self.n_bins + 1)
                edges = np.percentile(col, percentiles)
                # Handle duplicate edges
                edges = np.unique(edges)
            elif self.strategy == 'kmeans':
                edges = self._kmeans_binning(col)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            self.bin_edges_.append(edges)

        self._is_fitted = True
        return self

    def _kmeans_binning(self, col: np.ndarray) -> np.ndarray:
        """Compute bin edges using 1-D k-means clustering.

        Parameters
        ----------
        col : ndarray of shape (n_samples,)
            Single feature column to bin.

        Returns
        -------
        edges : ndarray
            Sorted unique bin edges including min and max values.
        """
        # Simple k-means implementation for 1D
        col_sorted = np.sort(col)

        # Initialize centroids
        centroids = np.percentile(col, np.linspace(0, 100, self.n_bins + 1)[1:-1])

        for _ in range(10):  # iterations
            # Assign to nearest centroid
            distances = np.abs(col[:, np.newaxis] - centroids)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = []
            for k in range(len(centroids)):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids.append(np.mean(col[mask]))
                else:
                    new_centroids.append(centroids[k])
            centroids = np.array(new_centroids)

        # Convert centroids to bin edges
        edges = np.concatenate([[col.min()], np.sort(centroids), [col.max()]])
        return np.unique(edges)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply binning transformation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_binned : ndarray
            Binned features.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)

        n_samples, n_features = X.shape

        # Compute bin indices
        binned = np.zeros((n_samples, n_features), dtype=int)
        for i in range(n_features):
            binned[:, i] = np.digitize(X[:, i], self.bin_edges_[i][1:-1])

        if self.encode == 'onehot':
            # One-hot encoding
            onehot_parts = []
            for i in range(n_features):
                n_bins_i = len(self.bin_edges_[i]) - 1
                onehot = np.zeros((n_samples, n_bins_i))
                for j in range(n_samples):
                    bin_idx = min(binned[j, i], n_bins_i - 1)
                    onehot[j, bin_idx] = 1
                onehot_parts.append(onehot)
            result = np.hstack(onehot_parts)
        else:
            result = binned.astype(float)

        if self.include_original:
            return np.hstack([X, result])
        return result

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Get output feature names for binned features.

        Parameters
        ----------
        input_features : list of str, optional
            Input feature names. If None, uses x0, x1, etc.

        Returns
        -------
        feature_names : ndarray of str
            Output feature names.
        """
        self._check_is_fitted()

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_input_features_)]

        names = []

        if self.include_original:
            names.extend(input_features)

        if self.encode == 'onehot':
            for i, feat in enumerate(input_features):
                n_bins_i = len(self.bin_edges_[i]) - 1
                for b in range(n_bins_i):
                    names.append(f"{feat}_bin{b}")
        else:
            for feat in input_features:
                names.append(f"{feat}_binned")

        return np.array(names)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_bins": {
                "type": "integer",
                "default": 5,
                "minimum": 2,
                "description": "Number of bins"
            },
            "strategy": {
                "type": "string",
                "enum": ["uniform", "quantile", "kmeans"],
                "default": "quantile",
                "description": "Binning strategy"
            },
            "encode": {
                "type": "string",
                "enum": ["ordinal", "onehot"],
                "default": "ordinal",
                "description": "Encoding method"
            },
            "include_original": {
                "type": "boolean",
                "default": False,
                "description": "Include original features"
            }
        }
