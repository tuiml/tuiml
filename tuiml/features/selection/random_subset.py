"""
Random Subset feature selection.

This module provides random feature selection, which randomly selects
a subset of features. Useful for ensemble methods and feature bagging.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union

from tuiml.base.features import FeatureSelector, feature_selector
from ._base import SelectorMixin, _ensure_numpy

@feature_selector(tags=["random", "subset", "unsupervised"], version="1.0.0")
class RandomSubsetSelector(FeatureSelector, SelectorMixin):
    """Randomly select a subset of features.

    Chooses a random subset of features, either an absolute number or a percentage. 
    Useful for ensemble methods, feature bagging, and reducing dimensionality when 
    labels are unavailable.

    Overview
    --------
    This selector samples a set of feature indices without replacement. It is a 
    purely stochastic method and does not use any information from the data values 
    or target labels during selection.

    Parameters
    ----------
    n_features : int or float, default=0.5
        Number of features to select:
        - If ``int >= 1``: absolute number of features.
        - If ``float < 1``: fraction of total features.

    invert : bool, default=False
        If True, randomly removes features instead of selecting them (retains 
        all features NOT in the random sample).

    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    selected_features_ : np.ndarray
        Indices of the randomly selected features.

    n_features_selected_ : int
        Total number of features selected.

    Notes
    -----
    **When to use:**
    - For creating diverse ensembles via feature bagging.
    - To benchmark other feature selection methods against a random baseline.
    - When you want to reduce dimensionality quickly without any statistical assumptions.

    **Limitations:**
    - Highly likely to keep irrelevant or redundant features.
    - Results depend entirely on the random seed.

    See Also
    --------
    :class:`~tuiml.features.selection.BootstrapFeaturesSelector` : Sample with replacement.
    :class:`~tuiml.features.selection.VarianceThresholdSelector` : Remove constant features.

    Examples
    --------
    Randomly select half of the features:

    >>> from tuiml.features.selection import RandomSubsetSelector
    >>> import numpy as np
    >>> X = np.random.randn(10, 20)
    >>> selector = RandomSubsetSelector(n_features=0.5, random_state=42)
    >>> X_new = selector.fit_transform(X)
    >>> print(X_new.shape[1])
    10
    """

    def __init__(
        self,
        n_features: Union[int, float] = 0.5,
        invert: bool = False,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_features = n_features
        self.invert = invert
        self.random_state = random_state

        self.selected_features_: Optional[np.ndarray] = None
        self.n_features_selected_: Optional[int] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RandomSubsetSelector":
        """
        Fit the random subset selector.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
            RandomSubsetSelector is unsupervised.

        Returns
        -------
        self : RandomSubsetSelector
            The fitted selector.
        """
        X = _ensure_numpy(X)
        n_samples, n_features = X.shape

        self._n_features_in = n_features

        # Determine number of features to select
        if isinstance(self.n_features, float) and self.n_features < 1:
            # Fraction of features
            n_select = max(1, int(round(n_features * self.n_features)))
        else:
            # Absolute number
            n_select = int(self.n_features)

        # Ensure valid range
        n_select = max(1, min(n_select, n_features))

        # Random selection
        rng = np.random.RandomState(self.random_state)
        all_indices = np.arange(n_features)

        # Sample without replacement
        selected = rng.choice(all_indices, size=n_select, replace=False)

        if self.invert:
            # Invert: keep features NOT in the random sample
            selected = np.setdiff1d(all_indices, selected)

        # Sort indices to maintain original order
        self._selected_indices = np.sort(selected)
        self.selected_features_ = self._selected_indices
        self.n_features_selected_ = len(self._selected_indices)

        # Create dummy scores (random selection has no meaningful scores)
        self._feature_scores = np.zeros(n_features)
        self._feature_scores[self._selected_indices] = 1.0

        self._is_fitted = True
        return self

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return binary feature scores indicating randomly selected features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused, scores are precomputed).
        y : ndarray of shape (n_samples,)
            Target values (unused, unsupervised selector).

        Returns
        -------
        scores : ndarray of shape (n_features,)
            Binary scores (1.0 for selected, 0.0 for unselected).
        """
        return self._feature_scores

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X by keeping only the randomly selected features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_selected_features)
            Data with selected features.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)

        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features, but RandomSubsetSelector "
                f"is expecting {self._n_features_in} features."
            )

        if len(self._selected_indices) == 0:
            return X[:, :0]
        return X[:, self._selected_indices]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_features": {
                "type": ["integer", "number"],
                "default": 0.5,
                "description": "Number of features (int >= 1) or fraction (float < 1)"
            },
            "invert": {
                "type": "boolean",
                "default": False,
                "description": "Randomly remove instead of select (WEKA -V option)"
            },
            "random_state": {
                "type": "integer",
                "description": "Random seed for reproducibility"
            }
        }

@feature_selector(tags=["random", "bootstrap", "ensemble"], version="1.0.0")
class BootstrapFeaturesSelector(FeatureSelector, SelectorMixin):
    """Bootstrap feature selection for ensemble methods.

    Selects features using bootstrap sampling (sampling with replacement to create 
    a sample of the same size, then taking unique indices), commonly used in Random 
    Forest and other bagging-based ensembles.

    Overview
    --------
    Bootstrap feature selection introduces diversity by allowing some features to be 
    sampled multiple times while others are omitted in a single draw. This increases 
    the randomness and robustness of ensemble models.

    Parameters
    ----------
    n_features : int, float, or {"sqrt", "log2"}, default="sqrt"
        Number of features to draw in the bootstrap sample:
        - ``int >= 1``: absolute number.
        - ``float < 1``: fraction of total.
        - ``"sqrt"``: :math:`\\sqrt{n_{features}}`.
        - ``"log2"``: :math:`log_2(n_{features})`.

    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    selected_features_ : np.ndarray
        Unique indices selected during the bootstrap process.

    n_features_selected_ : int
        Actual number of unique features selected.

    Notes
    -----
    **When to use:**
    - Specifically for ensemble methods like Random Forest.
    - When you want to simulate feature bagging.

    See Also
    --------
    :class:`~tuiml.features.selection.RandomSubsetSelector` : Sample without replacement.

    Examples
    --------
    Select square root of features using bootstrap:

    >>> from tuiml.features.selection import BootstrapFeaturesSelector
    >>> import numpy as np
    >>> X = np.random.randn(10, 100)
    >>> selector = BootstrapFeaturesSelector(n_features="sqrt", random_state=42)
    >>> X_new = selector.fit_transform(X)
    >>> print(X_new.shape[1])
    10
    """

    def __init__(
        self,
        n_features: Union[int, float, str] = "sqrt",
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_features = n_features
        self.random_state = random_state

        self.selected_features_: Optional[np.ndarray] = None
        self.n_features_selected_: Optional[int] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BootstrapFeaturesSelector":
        """
        Fit the bootstrap feature selector.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used.

        Returns
        -------
        self : BootstrapFeaturesSelector
            The fitted selector.
        """
        X = _ensure_numpy(X)
        n_samples, n_features = X.shape

        self._n_features_in = n_features

        # Determine number of features
        if isinstance(self.n_features, str):
            if self.n_features == "sqrt":
                n_select = max(1, int(np.sqrt(n_features)))
            elif self.n_features == "log2":
                n_select = max(1, int(np.log2(n_features)))
            else:
                raise ValueError(
                    f"Unknown n_features string '{self.n_features}'. "
                    f"Use 'sqrt', 'log2', int, or float."
                )
        elif isinstance(self.n_features, float) and self.n_features < 1:
            n_select = max(1, int(round(n_features * self.n_features)))
        else:
            n_select = int(self.n_features)

        n_select = max(1, min(n_select, n_features))

        # Bootstrap sample (with replacement, then unique)
        rng = np.random.RandomState(self.random_state)
        sampled = rng.choice(n_features, size=n_select, replace=True)

        # Get unique indices (bootstrap may sample same feature multiple times)
        self._selected_indices = np.sort(np.unique(sampled))
        self.selected_features_ = self._selected_indices
        self.n_features_selected_ = len(self._selected_indices)

        # Dummy scores
        self._feature_scores = np.zeros(n_features)
        self._feature_scores[self._selected_indices] = 1.0

        self._is_fitted = True
        return self

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return binary feature scores indicating bootstrap-selected features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused, scores are precomputed).
        y : ndarray of shape (n_samples,)
            Target values (unused, unsupervised selector).

        Returns
        -------
        scores : ndarray of shape (n_features,)
            Binary scores (1.0 for selected, 0.0 for unselected).
        """
        return self._feature_scores

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X by keeping only the bootstrap-sampled features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_selected_features)
            Data with only bootstrap-selected features.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)

        if len(self._selected_indices) == 0:
            return X[:, :0]
        return X[:, self._selected_indices]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_features": {
                "type": ["integer", "number", "string"],
                "default": "sqrt",
                "description": "Number of features: int, float < 1, 'sqrt', or 'log2'"
            },
            "random_state": {
                "type": "integer",
                "description": "Random seed"
            }
        }
