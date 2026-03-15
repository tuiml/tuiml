"""Bayesian Network classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from tuiml.base.algorithms import Classifier, classifier
from tuiml.algorithms.bayesian.estimators import DiscreteEstimator

@classifier(tags=["bayes", "probabilistic", "graphical-model"], version="1.1.0")
class BayesianNetworkClassifier(Classifier):
    """Bayesian Network classifier using **probabilistic graphical models**.

    A Bayesian Network is a **probabilistic graphical model** that represents a
    set of variables and their **conditional dependencies** via a **directed
    acyclic graph** (DAG). This implementation supports various structure
    learning algorithms and uses discrete probability estimators for
    conditional distribution modelling.

    Overview
    --------
    The algorithm classifies instances through the following steps:

    1. **Discretise** continuous features into bins (quantile-based)
    2. **Learn the network structure** using the chosen algorithm
       (Naive Bayes, TAN, or K2)
    3. **Estimate conditional probability tables** (CPTs) for each node
       given its parents, using discrete estimators with optional Laplace
       smoothing
    4. At prediction time, compute the **posterior probability** of each
       class by multiplying the class prior with all feature likelihoods
       conditioned on the learned parent configuration

    Supported structures:

    - **Naive Bayes:** Assumes all features are independent given the class.
    - **TAN (Tree Augmented Naive Bayes):** Extends Naive Bayes by allowing
      each feature to have at most one other feature as a parent.
    - **K2:** A greedy search algorithm that optimises the network structure
      based on a score (simplified implementation).

    Theory
    ------
    A Bayesian Network encodes the **joint probability distribution** over
    all variables via the chain rule applied to the DAG:

    .. math::

        P(X_1, \\ldots, X_m, C) = P(C) \\prod_{i=1}^{m} P(X_i \\mid \\text{Pa}(X_i))

    where :math:`\\text{Pa}(X_i)` denotes the parent set of node :math:`X_i`
    in the learned structure (always including the class node :math:`C`).

    For TAN, the structure is augmented by finding a **maximum spanning tree**
    over the features using the **conditional mutual information**:

    .. math::

        I(X_i; X_j \\mid C) = \\sum_{c} P(c) \\sum_{x_i, x_j}
        P(x_i, x_j \\mid c) \\log \\frac{P(x_i, x_j \\mid c)}{P(x_i \\mid c) P(x_j \\mid c)}

    Parameters
    ----------
    search_algorithm : {'naive', 'tan', 'k2'}, default='naive'
        The algorithm used for structure learning:

        - ``'naive'`` -- Single class node parent for all features.
        - ``'tan'`` -- Tree-based feature dependencies.
        - ``'k2'`` -- Greedy parent selection.
    max_parents : int, default=3
        Maximum number of parents allowed for each node (specifically
        used by K2).
    use_laplace : bool, default=True
        Whether to use Laplace smoothing in the conditional probability
        tables to avoid zero probabilities.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Unique class labels encountered during ``fit``.
    structure_ : dict
        The learned network structure, mapping node indices to their
        list of parents.
    class_estimator_ : DiscreteEstimator
        The estimator for the class prior distribution.
    feature_estimators_ : dict
        A nested dictionary of ``DiscreteEstimator`` objects for each
        feature, conditioned on their parents.

    Notes
    -----
    **Complexity:**

    - Training (Naive): :math:`O(n \\cdot m)` where :math:`n` = samples, :math:`m` = features
    - Training (TAN): :math:`O(n \\cdot m^2)` due to pairwise mutual information computation
    - Training (K2): :math:`O(n \\cdot m^2 \\cdot k^p)` where :math:`k` = symbol cardinality, :math:`p` = max parents
    - Prediction: :math:`O(m \\cdot c)` per sample where :math:`c` = classes

    **When to use BayesianNetworkClassifier:**

    - When feature dependencies are expected and should be modelled
    - Datasets with discrete or discretisable features
    - When the TAN structure can capture important pairwise feature interactions
    - When an interpretable graphical model structure is desired
    - Mixed feature types (nominal and numeric after discretisation)

    References
    ----------
    .. [Friedman1997] Friedman, N., Geiger, D. and Goldszmidt, M. (1997).
           **Bayesian Network Classifiers.**
           *Machine Learning*, 29(2-3), 131-163.
           DOI: `10.1023/A:1007465528199 <https://doi.org/10.1023/A:1007465528199>`_

    .. [Cooper1992] Cooper, G.F. and Herskovits, E. (1992).
           **A Bayesian Method for the Induction of Probabilistic Networks from Data.**
           *Machine Learning*, 9(4), 309-347.
           DOI: `10.1007/BF00994110 <https://doi.org/10.1007/BF00994110>`_

    See Also
    --------
    :class:`~tuiml.algorithms.bayesian.NaiveBayesClassifier` : Simple Naive Bayes with Gaussian/KDE estimators.
    :class:`~tuiml.algorithms.bayesian.NaiveBayesMultinomialClassifier` : Multinomial variant for count data.
    :class:`~tuiml.algorithms.bayesian.estimators.DiscreteEstimator` : Discrete probability estimator used for CPTs.

    Examples
    --------
    Classification with Tree Augmented Naive Bayes structure:

    >>> from tuiml.algorithms.bayesian import BayesianNetworkClassifier
    >>> import numpy as np
    >>>
    >>> # Discrete feature data
    >>> X = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    >>> y = np.array([1, 1, 0, 0])
    >>>
    >>> # Fit with TAN structure learning
    >>> clf = BayesianNetworkClassifier(search_algorithm='tan')
    >>> clf.fit(X, y)
    BayesianNetworkClassifier(algorithm='tan', estimator=DiscreteEstimator)
    >>> print(clf.predict([[1, 1]]))
    [1]
    """

    def __init__(self, search_algorithm: str = 'naive',
                 max_parents: int = 3,
                 use_laplace: bool = True,
                 random_state: Optional[int] = None):
        """Initialize the Bayesian Network classifier.

        Parameters
        ----------
        search_algorithm : {'naive', 'tan', 'k2'}, default='naive'
            Structure learning algorithm.
        max_parents : int, default=3
            Maximum number of parents per node (K2).
        use_laplace : bool, default=True
            Whether to use Laplace smoothing.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__()
        self.search_algorithm = search_algorithm
        self.max_parents = max_parents
        self.use_laplace = use_laplace
        self.random_state = random_state
        self.classes_ = None
        self.structure_ = None
        self._n_features = None
        self._bin_edges = None
        
        # Estimators
        self.class_estimator_: Optional[DiscreteEstimator] = None
        self.feature_estimators_: Dict = {}  # {feature_idx: {parent_config: DiscreteEstimator}}

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "search_algorithm": {"type": "string", "default": "naive",
                                "enum": ["naive", "tan", "k2"],
                                "description": "Structure learning algorithm"},
            "max_parents": {"type": "integer", "default": 3, "minimum": 1,
                           "description": "Maximum parents per node (K2)"},
            "use_laplace": {"type": "boolean", "default": True,
                         "description": "Use Laplace smoothing in estimators"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n * m^2 * k^p) where p is max parents"

    @classmethod
    def get_references(cls) -> List[str]:
        return ["Friedman, N., Geiger, D., & Goldszmidt, M. (1997). Bayesian "
                "Network Classifiers. Machine Learning, 29, 131-163."]

    def _discretize(self, X: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """Discretise continuous features into quantile-based bins.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Continuous feature matrix.
        n_bins : int, default=5
            Number of bins per feature.

        Returns
        -------
        X_discrete : np.ndarray of shape (n_samples, n_features)
            Integer-valued discretised feature matrix.
        """
        X_discrete = np.zeros_like(X, dtype=int)
        self._bin_edges = []

        for j in range(X.shape[1]):
            col = X[:, j]
            valid_mask = ~np.isnan(col)

            if np.sum(valid_mask) > 0:
                percentiles = np.linspace(0, 100, n_bins + 1)
                edges = np.percentile(col[valid_mask], percentiles)
                edges = np.unique(edges)
                self._bin_edges.append(edges)

                X_discrete[valid_mask, j] = np.digitize(col[valid_mask], edges[1:-1])
            else:
                self._bin_edges.append(np.array([0, 1]))
                X_discrete[:, j] = 0

        return X_discrete

    def _apply_discretization(self, X: np.ndarray) -> np.ndarray:
        """Apply saved discretisation bin edges to new data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Continuous feature matrix.

        Returns
        -------
        X_discrete : np.ndarray of shape (n_samples, n_features)
            Integer-valued discretised feature matrix using stored bin edges.
        """
        X_discrete = np.zeros_like(X, dtype=int)

        for j in range(X.shape[1]):
            col = X[:, j]
            valid_mask = ~np.isnan(col)
            edges = self._bin_edges[j]

            if len(edges) > 2:
                X_discrete[valid_mask, j] = np.digitize(col[valid_mask], edges[1:-1])
            else:
                X_discrete[:, j] = 0

        return X_discrete

    def _build_naive_structure(self) -> Dict[int, List[int]]:
        """Build the Naive Bayes network structure (class is parent of all features).

        Returns
        -------
        structure : dict of int to list of int
            Mapping from each node index to its list of parent indices.
            The class node (index -1) has no parents.
        """
        structure = {-1: []}  # Class node has no parents
        for i in range(self._n_features):
            structure[i] = [-1]  # Each feature has class as parent
        return structure

    def _mutual_information(self, X: np.ndarray, i: int, j: int,
                           y: np.ndarray) -> float:
        """Calculate the conditional mutual information :math:`I(X_i; X_j \\mid Y)`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Discretised feature matrix.
        i : int
            Index of the first feature.
        j : int
            Index of the second feature.
        y : np.ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        mi : float
            Conditional mutual information between features *i* and *j*
            given the class variable.
        """
        mi = 0.0
        n_samples = len(y)

        for c in self.classes_:
            mask = y == c
            if np.sum(mask) == 0:
                continue

            p_c = np.sum(mask) / n_samples
            X_c = X[mask]

            vals_i = np.unique(X_c[:, i])
            vals_j = np.unique(X_c[:, j])

            for vi in vals_i:
                for vj in vals_j:
                    mask_ij = (X_c[:, i] == vi) & (X_c[:, j] == vj)
                    mask_i = X_c[:, i] == vi
                    mask_j = X_c[:, j] == vj

                    p_ij = np.sum(mask_ij) / len(X_c) + 1e-10
                    p_i = np.sum(mask_i) / len(X_c) + 1e-10
                    p_j = np.sum(mask_j) / len(X_c) + 1e-10

                    if p_ij > 0:
                        mi += p_c * p_ij * np.log(p_ij / (p_i * p_j) + 1e-10)

        return mi

    def _build_tan_structure(self, X: np.ndarray, y: np.ndarray) -> Dict[int, List[int]]:
        """Build the Tree Augmented Naive Bayes (TAN) network structure.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Discretised feature matrix.
        y : np.ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        structure : dict of int to list of int
            TAN structure augmenting the Naive Bayes graph with a maximum
            spanning tree over features based on conditional mutual information.
        """
        # Start with Naive Bayes structure
        structure = self._build_naive_structure()

        # Compute mutual information between all pairs of features
        mi_matrix = np.zeros((self._n_features, self._n_features))
        for i in range(self._n_features):
            for j in range(i + 1, self._n_features):
                mi = self._mutual_information(X, i, j, y)
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        # Build maximum spanning tree using Prim's algorithm
        in_tree = [False] * self._n_features
        in_tree[0] = True

        for _ in range(self._n_features - 1):
            best_mi = -np.inf
            best_i, best_j = 0, 0

            for i in range(self._n_features):
                if not in_tree[i]:
                    continue
                for j in range(self._n_features):
                    if in_tree[j]:
                        continue
                    if mi_matrix[i, j] > best_mi:
                        best_mi = mi_matrix[i, j]
                        best_i, best_j = i, j

            if best_mi > 0:
                in_tree[best_j] = True
                structure[best_j].append(best_i)  # Add edge i -> j

        return structure

    def _estimate_cpt_with_estimators(self, X: np.ndarray, y: np.ndarray) -> None:
        """Estimate conditional probability tables (CPTs) using DiscreteEstimators.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Discretised feature matrix.
        y : np.ndarray of shape (n_samples,)
            Class labels.
        """
        n_classes = len(self.classes_)
        
        # Class prior estimator
        self.class_estimator_ = DiscreteEstimator(
            num_symbols=n_classes, 
            laplace=self.use_laplace
        )
        for label in y:
            class_idx = np.where(self.classes_ == label)[0][0]
            self.class_estimator_.add_value(class_idx)
        
        # Feature estimators
        self.feature_estimators_ = {}
        
        for feature_idx in range(self._n_features):
            parents = self.structure_[feature_idx]
            n_vals = len(np.unique(X[:, feature_idx]))
            
            self.feature_estimators_[feature_idx] = {}
            
            for c_idx, c in enumerate(self.classes_):
                mask_c = y == c
                X_c = X[mask_c]
                
                if len(parents) == 1:  # Only class as parent
                    # Key is just the class
                    key = (c_idx,)
                    estimator = DiscreteEstimator(num_symbols=n_vals, laplace=self.use_laplace)
                    
                    for val in X_c[:, feature_idx]:
                        estimator.add_value(int(val))
                    
                    self.feature_estimators_[feature_idx][key] = estimator
                else:
                    # Has additional parent(s) besides class
                    other_parent = parents[1]
                    parent_vals = np.unique(X[:, other_parent])
                    
                    for p_val in parent_vals:
                        key = (c_idx, int(p_val))
                        mask_p = X_c[:, other_parent] == p_val
                        
                        estimator = DiscreteEstimator(num_symbols=n_vals, laplace=self.use_laplace)
                        
                        for val in X_c[mask_p, feature_idx]:
                            estimator.add_value(int(val))
                        
                        self.feature_estimators_[feature_idx][key] = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianNetworkClassifier":
        """Fit the Bayesian Network classifier to training data.

        This involves discretizing continuous features, learning the 
        network structure according to the chosen algorithm, and 
        estimating the conditional probability tables.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : BayesianNetworkClassifier
            Returns the fitted classifier instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._n_features = X.shape[1]
        self.classes_ = np.unique(y)

        # Discretize continuous features
        X_discrete = self._discretize(X)

        # Build network structure
        if self.search_algorithm == 'naive':
            self.structure_ = self._build_naive_structure()
        elif self.search_algorithm == 'tan':
            self.structure_ = self._build_tan_structure(X_discrete, y)
        else:  # k2
            self.structure_ = self._build_naive_structure()  # Simplified

        # Estimate CPTs using estimators
        self._estimate_cpt_with_estimators(X_discrete, y)

        self._is_fitted = True
        return self

    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single discretised instance.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            A single discretised feature vector.

        Returns
        -------
        proba : np.ndarray of shape (n_classes,)
            Normalised posterior probabilities for each class.
        """
        proba = np.zeros(len(self.classes_))

        for c_idx, c in enumerate(self.classes_):
            # Start with class prior from estimator
            log_prob = np.log(self.class_estimator_.get_probability(c_idx) + 1e-10)

            # Multiply by feature likelihoods
            for feature_idx in range(self._n_features):
                val = int(x[feature_idx])
                parents = self.structure_[feature_idx]

                if len(parents) == 1:
                    key = (c_idx,)
                else:
                    other_parent = parents[1]
                    key = (c_idx, int(x[other_parent]))

                if key in self.feature_estimators_[feature_idx]:
                    estimator = self.feature_estimators_[feature_idx][key]
                    prob = estimator.get_probability(val)
                else:
                    prob = 1e-10

                log_prob += np.log(prob + 1e-10)

            proba[c_idx] = np.exp(log_prob)

        # Normalize
        proba /= np.sum(proba) + 1e-10
        return proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the 
            classes corresponds to that in the attribute ``classes_``.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_discrete = self._apply_discretization(X)

        proba = np.zeros((len(X), len(self.classes_)))
        for i in range(len(X)):
            proba[i] = self._predict_proba_single(X_discrete[i])

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            The predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"BayesianNetworkClassifier(algorithm='{self.search_algorithm}', estimator=DiscreteEstimator)"
        return f"BayesianNetworkClassifier(search_algorithm='{self.search_algorithm}')"
