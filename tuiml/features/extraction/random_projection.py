"""
Random Projection for dimensionality reduction.

This module provides Random Projection, which reduces dimensionality
using a random matrix while preserving distances (Johnson-Lindenstrauss lemma).
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union

from tuiml.base.features import FeatureExtractor, feature_extractor

def _ensure_numpy(X) -> np.ndarray:
    """Convert input to NumPy array.

    Parameters
    ----------
    X : array-like
        Input data to convert.

    Returns
    -------
    result : np.ndarray
        Data as a NumPy array.
    """
    if hasattr(X, 'values'):
        return X.values
    return np.asarray(X, dtype=np.float64)

@feature_extractor(tags=["dimensionality_reduction", "random", "projection"], version="1.0.0")
class RandomProjectionExtractor(FeatureExtractor):
    """Random Projection for dimensionality reduction.

    Reduces the dimensionality of the data by projecting it onto a lower dimensional 
    subspace using a random matrix.

    Overview
    --------
    Random Projection is a computationally efficient way to reduce dimensionality. 
    It is based on the **Johnson-Lindenstrauss lemma**, which states that if points 
    in a high-dimensional space are projected onto a randomly chosen subspace of 
    suitable dimension, then the distances between the points are approximately 
    preserved.

    Theory
    ------
    For a data matrix :math:`X_{n \\times p}`, the projected matrix is:

    .. math::
        X_{new} = X_{n \\times p} \\cdot R_{p \\times k}^T

    where :math:`R` is a random matrix. The minimum dimension :math:`k` to 
    preserve distances within a factor of :math:`1 \\pm \\epsilon` is:

    .. math::
        k \\ge \\frac{4 \\ln(n)}{\\epsilon^2 / 2 - \\epsilon^3 / 3}

    Parameters
    ----------
    n_components : int, float, or "auto", default=10
        Target dimensionality:
        - ``int >= 1``: absolute number of components.
        - ``float < 1``: percentage of original features.
        - ``"auto"``: Use the Johnson-Lindenstrauss formula to determine :math:`k`.

    distribution : {"gaussian", "sparse1", "sparse2"}, default="gaussian"
        Distribution used for the random matrix:
        - ``"gaussian"``: Normal distribution :math:`N(0, 1/k)`.
        - ``"sparse1"``: :math:`\\sqrt{3} \\times \\{-1, 0, 1\\}` with probabilities :math:`\\{1/6, 2/3, 1/6\\}`.
        - ``"sparse2"``: :math:`\\{-1, 1\\}` with probabilities :math:`\\{1/2, 1/2\\}`.

    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    n_components_ : int
        Actual number of components used for projection.

    components_ : np.ndarray of shape (n_components, n_features)
        The generated random projection matrix.

    Notes
    -----
    **Complexity:**
    - :math:`O(n \\cdot p \\cdot k)` for projection.
    - Much faster than PCA as it doesn't require eigenvalue decomposition.

    **When to use:**
    - Very high-dimensional datasets where PCA is computationally prohibitive.
    - When preserving pairwise distances is more important than maximizing variance.
    - As a fast preprocessing step for distance-based algorithms (k-NN, clustering).

    **Limitations:**
    - Only approximately preserves distances.
    - Projection matrix is not optimized for the specific data (data-independent).

    References
    ----------
    .. [Fradkin2003] Fradkin, D. and Madigan, D. (2003). **Experiments with random 
           projections for machine learning.** *KDD '03*, pp. 517-522.

    .. [JL1984] Johnson, W. B. and Lindenstrauss, J. (1984). **Extensions of 
           Lipschitz mappings into a Hilbert space.**

    See Also
    --------
    :class:`~tuiml.features.extraction.PCAExtractor` : Variance-maximizing projection.
    """

    def __init__(
        self,
        n_components: Union[int, float, str] = 10,
        distribution: str = "gaussian",
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_components = n_components
        self.distribution = distribution
        self.random_state = random_state

        self.n_components_: Optional[int] = None
        self.components_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RandomProjectionExtractor":
        """
        Fit the random projection matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : RandomProjectionExtractor
            The fitted transformer.
        """
        X = _ensure_numpy(X)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self._n_features_in = n_features

        # Determine number of components
        if isinstance(self.n_components, str) and self.n_components == "auto":
            # Johnson-Lindenstrauss minimum dimension
            # eps = 0.1 by default
            eps = 0.1
            self.n_components_ = self._johnson_lindenstrauss_min_dim(n_samples, eps)
        elif isinstance(self.n_components, float) and self.n_components < 1:
            # Percentage of features
            self.n_components_ = max(1, int(n_features * self.n_components))
        else:
            self.n_components_ = int(self.n_components)

        # Ensure n_components doesn't exceed n_features
        self.n_components_ = min(self.n_components_, n_features)

        # Generate random projection matrix
        rng = np.random.RandomState(self.random_state)
        self.components_ = self._generate_random_matrix(
            self.n_components_, n_features, rng
        )

        self._is_fitted = True
        return self

    def _johnson_lindenstrauss_min_dim(self, n_samples: int, eps: float = 0.1) -> int:
        """
        Find minimum dimension using Johnson-Lindenstrauss lemma.

        k >= 4 * ln(n) / (eps^2 / 2 - eps^3 / 3)
        """
        denominator = (eps ** 2) / 2 - (eps ** 3) / 3
        return int(4 * np.log(n_samples) / denominator) + 1

    def _generate_random_matrix(
        self, n_components: int, n_features: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Generate random projection matrix based on distribution.

        Parameters
        ----------
        n_components : int
            Number of components (rows in the matrix).
        n_features : int
            Number of features (columns in the matrix).
        rng : np.random.RandomState
            Random number generator instance.

        Returns
        -------
        components : np.ndarray of shape (n_components, n_features)
            The random projection matrix.
        """

        if self.distribution == "gaussian":
            # Gaussian random projection
            # Scale by 1/sqrt(n_components) for unit length columns
            components = rng.randn(n_components, n_features)
            components /= np.sqrt(n_components)

        elif self.distribution == "sparse1":
            # WEKA sparse1: sqrt(3) * {-1 with prob 1/6, 0 with prob 2/3, +1 with prob 1/6}
            sqrt3 = np.sqrt(3)
            components = np.zeros((n_components, n_features))

            for i in range(n_components):
                for j in range(n_features):
                    r = rng.random()
                    if r < 1/6:
                        components[i, j] = -sqrt3
                    elif r < 1/3:  # 1/6 + 1/6 = 1/3
                        components[i, j] = sqrt3
                    # else: 0 (with prob 2/3)

        elif self.distribution == "sparse2":
            # WEKA sparse2: {-1 with prob 1/2, +1 with prob 1/2}
            components = np.where(
                rng.random((n_components, n_features)) < 0.5, -1.0, 1.0
            )

        else:
            raise ValueError(
                f"Unknown distribution '{self.distribution}'. "
                f"Use 'gaussian', 'sparse1', or 'sparse2'."
            )

        return components

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply random projection to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projected data.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but RandomProjectionExtractor "
                f"is expecting {self.n_features_in_} features."
            )

        # Project: X_new = X @ components.T
        return X @ self.components_.T

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Approximate inverse transform using pseudo-inverse.

        Note: This is an approximation since random projection
        is not exactly invertible.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_components)
            Projected data.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Approximately reconstructed data.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)

        # Use pseudo-inverse for approximate reconstruction
        components_pinv = np.linalg.pinv(self.components_)
        return X @ components_pinv.T

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : list of str, optional
            Ignored, output names are always rp0, rp1, etc.

        Returns
        -------
        feature_names_out : np.ndarray of str
            Names of the output features.
        """
        self._check_is_fitted()
        return np.array([f"rp{i}" for i in range(self.n_components_)])

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_components": {
                "type": ["integer", "number", "string"],
                "default": 10,
                "description": "Number of components (int), percentage (float < 1), or 'auto'"
            },
            "distribution": {
                "type": "string",
                "enum": ["gaussian", "sparse1", "sparse2"],
                "default": "gaussian",
                "description": "Distribution for random matrix"
            },
            "random_state": {
                "type": "integer",
                "description": "Random seed for reproducibility"
            }
        }

@feature_extractor(tags=["dimensionality_reduction", "random", "sparse"], version="1.0.0")
class SparseRandomProjectionExtractor(RandomProjectionExtractor):
    """Sparse Random Projection for dimensionality reduction.

    A convenience class that uses a sparse random matrix by default, which is more 
    computationally efficient than Gaussian projection for high-dimensional data.

    Overview
    --------
    Sparse random projection reduces memory requirements and speeds up the 
    transformation by using a matrix where most entries are zero. It still 
    preserves distances according to the Johnson-Lindenstrauss lemma.

    This is equivalent to ``RandomProjectionExtractor`` with ``distribution="sparse1"``.

    Parameters
    ----------
    n_components : int, float, or "auto", default=10
        Target dimensionality.

    density : float, default=1/3
        Ratio of non-zero elements in the random matrix. 
        The default value 1/3 corresponds to the "sparse1" distribution.

    random_state : int, optional
        Random seed for reproducibility.

    See Also
    --------
    :class:`~tuiml.features.extraction.RandomProjectionExtractor` : Base random projection class.

    Examples
    --------
    Project high-dimensional data using sparse matrix:

    >>> from tuiml.features.extraction import SparseRandomProjectionExtractor
    >>> import numpy as np
    >>> X = np.random.randn(100, 1000)
    >>> srp = SparseRandomProjectionExtractor(n_components="auto")
    >>> X_projected = srp.fit_transform(X)
    >>> print(f"Projected to {srp.n_components_} dimensions")
    """

    def __init__(
        self,
        n_components: Union[int, float, str] = 10,
        density: float = 1/3,
        random_state: Optional[int] = None
    ):
        super().__init__(
            n_components=n_components,
            distribution="sparse1",
            random_state=random_state
        )
        self.density = density

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_components": {
                "type": ["integer", "number", "string"],
                "default": 10,
                "description": "Number of components"
            },
            "density": {
                "type": "number",
                "default": 0.333,
                "description": "Ratio of non-zero elements"
            },
            "random_state": {
                "type": "integer",
                "description": "Random seed"
            }
        }
