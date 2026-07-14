"""Scikit-Learn PCA dimensionality reduction wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional, Union

from tuiml.base.features import FeatureExtractor, feature_extractor

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    PCA = None
    SKLEARN_AVAILABLE = False

@feature_extractor(tags=["extraction", "dimensionality_reduction", "pca", "sklearn"], version="1.0.0")
class SklearnPCAExtractor(FeatureExtractor):
    """
    Principal Component Analysis (PCA) using Scikit-Learn.

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    Parameters
    ----------
    n_components : int, float or None, default=None
        Number of components to keep.
    whiten : bool, default=False
        When True, the components are divided by the singular values to
        ensure uncorrelated outputs with unit component-wise variances.
    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        Solver to use for SVD.
    random_state : int, optional
        Used when the svd_solver is 'arpack' or 'randomized'.

    Attributes
    ----------
    components_ : np.ndarray
        Principal axes in feature space.
    explained_variance_ : np.ndarray
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_ : np.ndarray
        Percentage of variance explained by each of the selected components.
    singular_values_ : np.ndarray
        The singular values corresponding to each of the selected components.
    mean_ : np.ndarray
        Per-feature empirical mean.
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        whiten: bool = False,
        svd_solver: str = 'auto',
        random_state: Optional[int] = None
    ):
        super().__init__(n_components=n_components)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.random_state = random_state

        self.model_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_features_in_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_components": {
                "type": ["integer", "number", "null"],
                "default": None,
                "description": "Number of components to keep (int), variance proportion (float), or None for all"
            },
            "whiten": {
                "type": "boolean",
                "default": False,
                "description": "If True, whiten the components"
            },
            "svd_solver": {
                "type": "string",
                "default": "auto",
                "enum": ["auto", "full", "arpack", "randomized"],
                "description": "Solver to use for SVD"
            }
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnPCAExtractor":
        """Fit the PCA model."""
        X = self._ensure_numpy(X)
        self.n_features_in_ = X.shape[1]
        
        self.model_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            random_state=self.random_state
        )
        self.model_.fit(X)
        
        self.components_ = self.model_.components_
        self.explained_variance_ = self.model_.explained_variance_
        self.explained_variance_ratio_ = self.model_.explained_variance_ratio_
        self.singular_values_ = self.model_.singular_values_
        self.mean_ = self.model_.mean_
        self.n_components_ = self.model_.n_components_
        
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to X."""
        self._check_is_fitted()
        X = self._ensure_numpy(X)
        return self.model_.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data back to its original space."""
        self._check_is_fitted()
        X = self._ensure_numpy(X)
        return self.model_.inverse_transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the model with X and apply dimensionality reduction on X."""
        self.fit(X, y)
        return self.transform(X)
        
    def _ensure_numpy(self, X) -> np.ndarray:
        """Convert input to NumPy array."""
        if hasattr(X, 'values'):
            return X.values
        return np.asarray(X)

    def get_feature_names_out(self) -> Optional[List[str]]:
        """Get output feature names."""
        self._check_is_fitted()
        if self.n_components_ is None:
            return None
        return [f"PCA{i+1}" for i in range(self.n_components_)]

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SklearnPCAExtractor(n_components={self.n_components}, "
            f"whiten={self.whiten}, svd_solver='{self.svd_solver}')"
        )
