"""
Polynomial feature construction.

This module provides polynomial and interaction feature generation.
"""

import numpy as np
from itertools import combinations_with_replacement, combinations
from typing import Any, Dict, List, Optional, Tuple

from tuiml.base.features import FeatureConstructor, feature_constructor

def _ensure_numpy(X) -> np.ndarray:
    """Convert input to NumPy array."""
    if hasattr(X, 'values'):
        return X.values
    return np.asarray(X)

@feature_constructor(tags=["polynomial", "interactions", "nonlinear"], version="1.0.0")
class PolynomialFeaturesGenerator(FeatureConstructor):
    r"""Generate polynomial and interaction features.

    Creates a new feature matrix consisting of all polynomial combinations of the 
    features with degree less than or equal to the specified degree.

    Overview
    --------
    Polynomial features are used to model non-linear relationships by including 
    higher-order terms and interactions between features. 

    If an input sample is :math:`[a, b]`, the degree-2 polynomial features are:
    :math:`[1, a, b, a^2, ab, b^2]`.

    Theory
    ------
    For a set of variables :math:`X_1, X_2, \dots, X_p`, a polynomial feature of 
    degree :math:`d` is given by:

    .. math::
        X_1^{e_1} X_2^{e_2} \dots X_p^{e_p} \quad \\text{where} \quad \sum_{i=1}^p e_i \le d

    Parameters
    ----------
    degree : int, default=2
        Maximum degree of the polynomial features.

    interaction_only : bool, default=False
        If True, only interaction features are produced: features that are products 
        of at most ``degree`` *distinct* input features (e.g., :math:`ab` is kept, 
        but :math:`a^2` is not).

    include_bias : bool, default=True
        If True, include a bias column (all 1s), which corresponds to the 
        degree 0 polynomial.

    order : {"C", "F"}, default="C"
        Order of output array in the dense case. 'F' order is faster to compute 
        but may slow down subsequent estimators.

    Attributes
    ----------
    n_input_features_ : int
        Number of input features.

    n_output_features_ : int
        Number of output features.

    powers_ : np.ndarray of shape (n_output_features, n_input_features)
        Exponent for each input feature in each output feature.

    Notes
    -----
    **Complexity:**
    - The number of features grows combinatorially with the degree: 
      :math:`\\binom{n+d}{d}`. Be cautious with high degrees and many features.

    See Also
    --------
    :class:`~tuiml.features.generation.InteractionFeaturesGenerator` : Pairwise interactions only.

    Examples
    --------
    Generate degree-2 features:

    >>> from tuiml.features.generation import PolynomialFeaturesGenerator
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4]])
    >>> poly = PolynomialFeaturesGenerator(degree=2)
    >>> X_poly = poly.fit_transform(X)
    >>> print(poly.get_feature_names_out(['a', 'b']))
    ['1' 'a' 'b' 'a^2' 'a*b' 'b^2']
    """

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = True,
        order: str = 'C'
    ):
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order

        self.n_input_features_: Optional[int] = None
        self.n_output_features_: Optional[int] = None
        self.powers_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PolynomialFeaturesGenerator":
        """
        Compute the polynomial feature combinations.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : PolynomialFeaturesGenerator
            The fitted transformer.
        """
        X = _ensure_numpy(X)
        n_samples, n_features = X.shape

        self.n_input_features_ = n_features
        self._n_features_in = n_features

        # Generate power combinations
        self.powers_ = self._combinations(n_features, self.degree,
                                          self.interaction_only,
                                          self.include_bias)
        self.n_output_features_ = len(self.powers_)
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to polynomial features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_poly : ndarray of shape (n_samples, n_output_features)
            Transformed data with polynomial features.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)

        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError(
                f"X has {n_features} features, but PolynomialFeaturesGenerator "
                f"is expecting {self.n_input_features_} features."
            )

        # Compute polynomial features
        X_poly = np.empty((n_samples, self.n_output_features_),
                          dtype=X.dtype, order=self.order)

        for i, powers in enumerate(self.powers_):
            X_poly[:, i] = np.prod(X ** powers, axis=1)

        return X_poly

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """
        Get output feature names.

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

        feature_names = []
        for powers in self.powers_:
            if np.sum(powers) == 0:
                feature_names.append("1")
            else:
                terms = []
                for i, p in enumerate(powers):
                    if p == 1:
                        terms.append(input_features[i])
                    elif p > 1:
                        terms.append(f"{input_features[i]}^{int(p)}")
                feature_names.append("*".join(terms) if terms else "1")

        return np.array(feature_names)

    @staticmethod
    def _combinations(n_features: int, degree: int,
                      interaction_only: bool, include_bias: bool) -> np.ndarray:
        """Generate power combinations for polynomial features.

        Parameters
        ----------
        n_features : int
            Number of input features.
        degree : int
            Maximum polynomial degree.
        interaction_only : bool
            If True, only produce interaction features.
        include_bias : bool
            If True, include a bias (all-zeros) row.

        Returns
        -------
        combinations : ndarray of shape (n_combinations, n_features)
            Array of power vectors for each output feature.
        """
        combinations_list = []

        if include_bias:
            combinations_list.append(np.zeros(n_features, dtype=int))

        for d in range(1, degree + 1):
            if interaction_only:
                # Only distinct feature combinations
                for combo in combinations(range(n_features), d):
                    powers = np.zeros(n_features, dtype=int)
                    for idx in combo:
                        powers[idx] = 1
                    combinations_list.append(powers)
            else:
                # All combinations with replacement
                for combo in combinations_with_replacement(range(n_features), d):
                    powers = np.zeros(n_features, dtype=int)
                    for idx in combo:
                        powers[idx] += 1
                    combinations_list.append(powers)

        return np.array(combinations_list)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "degree": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Maximum polynomial degree"
            },
            "interaction_only": {
                "type": "boolean",
                "default": False,
                "description": "If True, only produce interaction features"
            },
            "include_bias": {
                "type": "boolean",
                "default": True,
                "description": "If True, include a bias column of ones"
            }
        }

@feature_constructor(tags=["interactions", "pairwise"], version="1.0.0")
class InteractionFeaturesGenerator(FeatureConstructor):
    r"""Generate pairwise interaction features only.

    Creates interaction terms between all unique pairs of features. This is a 
    specialized version of ``PolynomialFeaturesGenerator`` with ``degree=2`` 
    and ``interaction_only=True``.

    Overview
    --------
    Interaction features allow models to capture dependencies between two 
    different variables. For features :math:`X_i` and :math:`X_j`, the 
    interaction term is :math:`X_i \cdot X_j`.

    Parameters
    ----------
    include_original : bool, default=True
        If True, include original input features in the output matrix.

    Attributes
    ----------
    n_input_features_ : int
        Number of input features.

    n_output_features_ : int
        Number of output features.

    interaction_pairs_ : list of tuple
        Indices of feature pairs used for interactions.

    See Also
    --------
    :class:`~tuiml.features.generation.PolynomialFeaturesGenerator` : General polynomial features.

    Examples
    --------
    Generate interaction terms for 3 features:

    >>> from tuiml.features.generation import InteractionFeaturesGenerator
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> inter = InteractionFeaturesGenerator(include_original=True)
    >>> X_inter = inter.fit_transform(X)
    >>> print(X_inter.shape[1])
    6
    """

    def __init__(self, include_original: bool = True):
        super().__init__()
        self.include_original = include_original

        self.n_input_features_: Optional[int] = None
        self.n_output_features_: Optional[int] = None
        self.interaction_pairs_: Optional[List[Tuple[int, int]]] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "InteractionFeaturesGenerator":
        """
        Compute interaction pairs.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : InteractionFeaturesGenerator
            The fitted transformer.
        """
        X = _ensure_numpy(X)
        n_samples, n_features = X.shape

        self.n_input_features_ = n_features
        self._n_features_in = n_features

        # Generate interaction pairs
        self.interaction_pairs_ = list(combinations(range(n_features), 2))

        if self.include_original:
            self.n_output_features_ = n_features + len(self.interaction_pairs_)
        else:
            self.n_output_features_ = len(self.interaction_pairs_)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Generate interaction features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_inter : ndarray of shape (n_samples, n_output_features)
            Data with interaction features.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)

        n_samples = X.shape[0]

        # Compute interactions
        interactions = np.empty((n_samples, len(self.interaction_pairs_)))
        for idx, (i, j) in enumerate(self.interaction_pairs_):
            interactions[:, idx] = X[:, i] * X[:, j]

        if self.include_original:
            return np.hstack([X, interactions])
        else:
            return interactions

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Get output feature names for interaction features.

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

        for i, j in self.interaction_pairs_:
            names.append(f"{input_features[i]}*{input_features[j]}")

        return np.array(names)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "include_original": {
                "type": "boolean",
                "default": True,
                "description": "Include original features in output"
            }
        }
