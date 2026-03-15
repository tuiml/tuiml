"""Incremental conceptual clustering using category utility."""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from tuiml.base.algorithms import UpdateableClusterer, clusterer

@dataclass
class CobwebClustererNode:
    """Node in the COBWEB classification tree."""
    # Statistics for numeric attributes
    count: int = 0
    mean: Optional[np.ndarray] = None
    sum_sq: Optional[np.ndarray] = None  # Sum of squared values

    # Children nodes
    children: List["CobwebClustererNode"] = field(default_factory=list)

    # Parent reference
    parent: Optional["CobwebClustererNode"] = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def std(self) -> np.ndarray:
        """Compute standard deviation for each attribute."""
        if self.count == 0:
            return np.zeros_like(self.mean)
        variance = (self.sum_sq / self.count) - (self.mean ** 2)
        variance = np.maximum(variance, 0)  # Handle numerical errors
        return np.sqrt(variance)

@clusterer(tags=["incremental", "hierarchical", "conceptual"], version="1.0.0")
class CobwebClusterer(UpdateableClusterer):
    r"""
    COBWEB/CLASSIT incremental conceptual clustering.

    Builds a **hierarchical clustering tree** incrementally by processing one
    instance at a time. The algorithm uses **Category Utility (CU)** to
    guide tree construction, measuring the increase in expected correct
    attribute predictions given the clustering.

    Overview
    --------
    For each new instance, the algorithm decides among four operators:

    1. **Incorporate** the instance into the best existing child node
    2. **Create** a new child node for the instance
    3. **Merge** the two best children into a single node
    4. **Split** the best child, promoting its children to the current level

    The operator that maximizes Category Utility is chosen at each step.

    Theory
    ------
    For categorical attributes, Category Utility is defined as:

    .. math::
        CU = \\frac{\sum_{k=1}^K P(C_k) \sum_i \sum_j P(A_i = V_{ij} | C_k)^2 - \sum_i \sum_j P(A_i = V_{ij})^2}{K}

    For numeric attributes (CLASSIT extension), it uses the reduction in
    variance:

    .. math::
        CU = \\frac{1}{K} \sum_{k=1}^K P(C_k) \sum_i \left( \\frac{1}{\sigma_{ik}} - \\frac{1}{\sigma_{i}} \\right)

    where :math:`\sigma_{ik}` is the standard deviation of attribute :math:`i`
    in cluster :math:`k`, and :math:`\sigma_i` is the global standard deviation.

    Parameters
    ----------
    acuity : float, default=1.0
        Minimum standard deviation for numeric attributes. It prevents
        infinite category utility when a cluster has only one point.
    cutoff : float, default=0.01
        Category utility threshold for splitting an existing node.

    Attributes
    ----------
    root_ : CobwebClustererNode
        Root node of the classification tree.
    n_nodes_ : int
        Total number of nodes in the generated tree.
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels assigned during the fit process.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(B \cdot \log n \cdot m)` per instance, where :math:`B`
      is the branching factor and :math:`m` is the number of features.
    - Space: :math:`O(\\text{n\_nodes} \cdot m)`.

    **When to use CobwebClusterer:**

    - Streaming or incremental data where instances arrive one at a time
    - When you need a hierarchical conceptual clustering
    - Exploratory analysis where the number of clusters is unknown
    - When interpretability of the cluster hierarchy is important

    References
    ----------
    .. [Fisher1987] Fisher, D. (1987).
           **Knowledge acquisition via incremental conceptual clustering.**
           *Machine Learning*, 2(2), pp. 139-172.

    .. [Gennari1990] Gennari, J.H., Langley, P., & Fisher, D. (1990).
           **Models of incremental concept formation.**
           *Artificial Intelligence*, 40(1-3), pp. 11-61.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.AgglomerativeClusterer` : Batch hierarchical clustering using agglomerative merging.
    :class:`~tuiml.algorithms.clustering.KMeansClusterer` : Partitional clustering when cluster count is known.

    Examples
    --------
    Basic incremental clustering:

    >>> import numpy as np
    >>> from tuiml.algorithms.clustering import CobwebClusterer
    >>> X = np.array([[1, 2], [1.1, 2.1], [10, 10], [10.1, 10.1]])
    >>> cobweb = CobwebClusterer(acuity=0.1)
    >>> cobweb.fit(X)
    >>> cobweb.n_clusters_
    2
    >>> # Online update
    >>> cobweb.update([[1.2, 2.2]])
    CobwebClusterer(n_clusters=2, n_nodes=9)
    """

    def __init__(self, acuity: float = 1.0, cutoff: float = 0.01):
        """Initialize COBWEB clusterer.

        Parameters
        ----------
        acuity : float, default=1.0
            Minimum standard deviation.
        cutoff : float, default=0.01
            Category utility threshold.
        """
        super().__init__()
        self.acuity = acuity
        self.cutoff = cutoff
        self.root_ = None
        self.n_nodes_ = 0
        self._n_features = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "acuity": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "description": "Minimum standard deviation"
            },
            "cutoff": {
                "type": "number",
                "default": 0.01,
                "minimum": 0,
                "description": "Category utility threshold"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "incremental"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(B * log(n)) per instance, where B is branching factor"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Fisher, D. (1987). Knowledge acquisition via incremental "
            "conceptual clustering. Machine Learning, 2(2), 139-172.",
            "Gennari, J.H., Langley, P., & Fisher, D. (1990). Models of "
            "incremental concept formation. Artificial Intelligence, 40, 11-61."
        ]

    def _create_node(self) -> CobwebClustererNode:
        """Create a new node with proper initialization.

        Returns
        -------
        node : CobwebClustererNode
            A new node with zeroed mean and sum-of-squares arrays.
        """
        node = CobwebClustererNode()
        node.mean = np.zeros(self._n_features)
        node.sum_sq = np.zeros(self._n_features)
        self.n_nodes_ += 1
        return node

    def _update_node(self, node: CobwebClustererNode, instance: np.ndarray) -> None:
        """Update node statistics with a new instance.

        Parameters
        ----------
        node : CobwebClustererNode
            The node whose statistics are updated.
        instance : np.ndarray of shape (n_features,)
            The new data point to incorporate.
        """
        node.count += 1
        if node.mean is None:
            node.mean = instance.copy()
            node.sum_sq = instance ** 2
        else:
            # Online update of mean and sum of squares
            node.mean = node.mean + (instance - node.mean) / node.count
            node.sum_sq = node.sum_sq + instance ** 2

    def _category_utility(self, node: CobwebClustererNode) -> float:
        """Compute category utility for a node.

        CU measures the increase in expected correct attribute predictions
        given the clustering.

        Parameters
        ----------
        node : CobwebClustererNode
            The parent node whose children define the partition.

        Returns
        -------
        cu : float
            Category utility value for the current partition.
        """
        if node.count == 0 or len(node.children) == 0:
            return 0.0

        # For numeric attributes, use expected reduction in variance
        parent_std = node.std()
        parent_std = np.maximum(parent_std, self.acuity)

        cu = 0.0
        for child in node.children:
            if child.count == 0:
                continue
            p_child = child.count / node.count
            child_std = child.std()
            child_std = np.maximum(child_std, self.acuity)

            # Variance reduction contribution
            for i in range(self._n_features):
                cu += p_child * (1.0 / child_std[i] - 1.0 / parent_std[i])

        return cu / len(node.children)

    def _best_host(self, node: CobwebClustererNode, instance: np.ndarray) -> tuple:
        """Find the best child to host an instance.

        Parameters
        ----------
        node : CobwebClustererNode
            The parent node whose children are evaluated.
        instance : np.ndarray of shape (n_features,)
            The data point to place.

        Returns
        -------
        best_child : CobwebClustererNode or None
            The child that maximizes category utility, or None if no children.
        best_cu : float
            The category utility achieved by placing the instance in best_child.
        """
        if len(node.children) == 0:
            return None, -np.inf

        best_child = None
        best_cu = -np.inf

        for child in node.children:
            # Temporarily add instance to child
            old_count = child.count
            old_mean = child.mean.copy() if child.mean is not None else None
            old_sum_sq = child.sum_sq.copy() if child.sum_sq is not None else None

            self._update_node(child, instance)

            # Compute category utility
            cu = self._category_utility(node)

            # Restore child
            child.count = old_count
            child.mean = old_mean
            child.sum_sq = old_sum_sq

            if cu > best_cu:
                best_cu = cu
                best_child = child

        return best_child, best_cu

    def _incorporate(self, node: CobwebClustererNode, instance: np.ndarray) -> CobwebClustererNode:
        """Incorporate an instance into the tree starting at node.

        Parameters
        ----------
        node : CobwebClustererNode
            The current node in the tree to incorporate the instance into.
        instance : np.ndarray of shape (n_features,)
            The data point to incorporate.

        Returns
        -------
        leaf : CobwebClustererNode
            The leaf node where the instance was ultimately placed.
        """
        # Update node statistics
        self._update_node(node, instance)

        if node.is_leaf():
            # Create two children: one for existing, one for new
            if node.count > 1:
                # Split the leaf
                existing_child = self._create_node()
                existing_child.count = node.count - 1
                existing_child.mean = (node.mean * node.count - instance) / (node.count - 1)
                existing_child.sum_sq = node.sum_sq - instance ** 2
                existing_child.parent = node

                new_child = self._create_node()
                self._update_node(new_child, instance)
                new_child.parent = node

                node.children = [existing_child, new_child]

            return node

        # Find best and second-best hosts
        best_child, best_cu = self._best_host(node, instance)

        # Consider creating a new child
        new_child = self._create_node()
        self._update_node(new_child, instance)
        new_child.parent = node

        # Temporarily add new child
        node.children.append(new_child)
        new_cu = self._category_utility(node)
        node.children.pop()

        # Decide: add to best host or create new child
        if best_cu >= new_cu and best_child is not None:
            # Add to existing child
            return self._incorporate(best_child, instance)
        else:
            # Create new child
            node.children.append(new_child)
            return new_child

    def fit(self, X: np.ndarray) -> "CobwebClusterer":
        """Build a COBWEB tree from the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        self : CobwebClusterer
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, self._n_features = X.shape

        # Initialize root
        self.root_ = self._create_node()
        self.n_nodes_ = 1

        # Incorporate each instance
        for instance in X:
            self._incorporate(self.root_, instance)

        # Assign labels
        self.labels_ = self._assign_labels(X)
        self.n_clusters_ = len(self.root_.children) if self.root_.children else 1
        self._is_fitted = True

        return self

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        """Assign cluster labels based on tree structure.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points to assign labels to.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster label for each point based on the top-level children.
        """
        labels = np.zeros(X.shape[0], dtype=int)

        if self.root_ is None or len(self.root_.children) == 0:
            return labels

        for i, instance in enumerate(X):
            # Find closest leaf
            node = self.root_
            while not node.is_leaf():
                best_child = None
                best_dist = np.inf
                for j, child in enumerate(node.children):
                    if child.mean is not None:
                        dist = np.sum((instance - child.mean) ** 2)
                        if dist < best_dist:
                            best_dist = dist
                            best_child = child
                            labels[i] = j
                if best_child is None:
                    break
                node = best_child

        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to cluster.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self._assign_labels(X)

    def update(self, X: np.ndarray) -> "CobwebClusterer":
        """Incrementally update the model with new instances.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to incorporate into the classification tree.

        Returns
        -------
        self : CobwebClusterer
            Updated estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self._is_fitted:
            return self.fit(X)

        for instance in X:
            self._incorporate(self.root_, instance)

        self.n_clusters_ = len(self.root_.children) if self.root_.children else 1
        return self

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"CobwebClusterer(n_clusters={self.n_clusters_}, "
                   f"n_nodes={self.n_nodes_})")
        return f"CobwebClusterer(acuity={self.acuity}, cutoff={self.cutoff})"
