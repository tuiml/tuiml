"""
Base classes and registry for machine learning algorithms.

This module provides the foundation for the plugin-based algorithm system.
All base classes (Classifier, Clusterer, Regressor, Associator) are defined here
with proper hub integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, FrozenSet
from dataclasses import dataclass
import numpy as np
import threading
import time
import asyncio

from tuiml.hub import registry, ComponentType, Registrable

# =============================================================================
# Algorithm Base Class
# =============================================================================

class Algorithm(Registrable, ABC):
    """Abstract base class for all machine learning algorithms.

    Provides a unified interface for model lifecycle management, including 
    training, prediction, serving, and hub registration.

    Overview
    --------
    This class serves as the foundation for specialized thinkers like 
    Classifiers and Regressors. It defines standard methods for metadata 
    retrieval, parameter handling, and REST serving.

    Attributes
    ----------
    _is_fitted : bool
        Internal flag indicating if the model has been trained.

    Notes
    -----
    Subclasses MUST implement :meth:`fit` and :meth:`predict`.
    """

    _component_type = ComponentType.ALGORITHM

    def __init__(self):
        """Initialize algorithm."""
        self._is_fitted = False

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """
        Return algorithm metadata for registration.

        Returns:
            Dictionary with algorithm information
        """
        return {
            "name": cls.__name__,
            "type": getattr(cls, "_algorithm_type", "unknown"),
            "description": cls.__doc__ or "No description available",
            "parameters": cls.get_parameter_schema(),
            "capabilities": cls.get_capabilities(),
            "complexity": cls.get_complexity(),
            "references": cls.get_references(),
        }

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """
        Return JSON Schema for algorithm parameters.

        Returns:
            Dictionary mapping parameter names to their schemas

        Example::

            {
                "n_trees": {
                    "type": "integer",
                    "default": 100,
                    "minimum": 1,
                    "maximum": 1000,
                    "description": "Number of trees in the forest"
                }
            }
        """
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """
        Return list of algorithm capabilities.

        Returns:
            List of capability strings

        Example::

            ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]
        """
        return []

    @classmethod
    def get_complexity(cls) -> str:
        """
        Return time/space complexity.

        Returns:
            String describing complexity

        Example::

            "O(n * m * log(n))"
        """
        return "Not specified"

    @classmethod
    def get_references(cls) -> List[str]:
        """
        Return list of academic references.

        Returns:
            List of citation strings

        Example::

            ["Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32."]
        """
        return []

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        """
        Train the algorithm on data.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) - optional for unsupervised

        Returns:
            Self (for method chaining)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Test features (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        pass

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and predict in one step (useful for clustering).

        Args:
            X: Training features
            y: Training labels (optional)

        Returns:
            Predictions
        """
        self.fit(X, y)
        return self.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """
        Get algorithm parameters.

        Returns:
            Dictionary of current parameters
        """
        # Get all attributes that don't start with underscore
        params = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                params[key] = value
        return params

    def set_params(self, **params) -> "Algorithm":
        """
        Set algorithm parameters.

        Args:
            **params: Parameters to set

        Returns:
            Self (for method chaining)
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def _check_is_fitted(self):
        """Check if algorithm has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling predict"
            )

    def serve(
        self,
        port: int = 8000,
        host: str = "127.0.0.1",
        model_id: Optional[str] = None,
        background: bool = False,
        **kwargs
    ):
        """
        Serve this model via REST API.

        Starts a FastAPI server to serve predictions from this model.
        The model must be fitted before calling serve().

        Args:
            port: Port to listen on (default: 8000)
            host: Host to bind to (default: '127.0.0.1')
            model_id: Identifier for the model (default: class name)
            background: If True, run in a background thread (default: False)
            **kwargs: Additional arguments passed to uvicorn

        Example::

            nb = NaiveBayesClassifier()
            nb.fit(X_train, y_train)
            nb.serve(port=8000)
            # Server running at http://127.0.0.1:8000
            # API docs at http://127.0.0.1:8000/docs

        Endpoints:
            - POST /predict - Make predictions
            - POST /predict_proba - Get class probabilities (if supported)
            - GET /health - Health check
            - GET /models/{model_id} - Model info
        """
        self._check_is_fitted()

        try:
            from tuiml.serving import ModelServer
        except ImportError:
            raise ImportError(
                "FastAPI is required for model serving. "
                "Install with: pip install fastapi uvicorn"
            )

        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required for model serving. "
                "Install with: pip install uvicorn"
            )

        # Use class name as default model_id
        if model_id is None:
            model_id = self.__class__.__name__

        # Create server and add this model directly
        server = ModelServer()

        # Directly add model to manager (bypassing file load)
        from datetime import datetime
        server.manager._models[model_id] = {
            "model": self,
            "path": "<in-memory>",
            "info": {
                "model_class": self.__class__.__name__,
                "model_module": self.__class__.__module__,
                "params": self.get_params(),
            },
            "metadata": {},
            "loaded_at": datetime.now().isoformat(),
            "prediction_count": 0,
        }

        app = server.create_app()

        # Handle existing event loops (e.g. Jupyter Notebooks)
        try:
            asyncio.get_running_loop()
            in_running_loop = True
        except RuntimeError:
            in_running_loop = False

        if in_running_loop and not background:
            background = True
            print("Detected running event loop (e.g., Jupyter). Running server in background thread.")

        print(f"Starting TuiML Model Server...")
        print(f"  Model: {self.__class__.__name__}")
        print(f"  Model ID: {model_id}")
        print(f"  Server: http://{host}:{port}")
        print(f"  API Docs: http://{host}:{port}/docs")
        print()

        if background:
            config = uvicorn.Config(app, host=host, port=port, log_level="warning", **kwargs)
            uvicorn_server = uvicorn.Server(config)
            
            def run_server():
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(uvicorn_server.serve())
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            
            # Wait briefly for server to start
            time.sleep(0.5)
            return {
                "server_id": f"{host}:{port}",
                "url": f"http://{host}:{port}",
                "thread": thread,
                "server": uvicorn_server
            }
        else:
            uvicorn.run(app, host=host, port=port, **kwargs)


# =============================================================================
# Classifier Base Class
# =============================================================================

class Classifier(Algorithm):
    """Base class for supervised classification algorithms.

    Classifiers learn to assign categorical labels (classes) to instances 
    based on training data.

    See Also
    --------
    :class:`~tuiml.base.algorithms.Regressor` : For continuous output.
    """

    _component_type = ComponentType.CLASSIFIER

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Test features

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support probability prediction"
        )

# =============================================================================
# Clusterer Base Classes
# =============================================================================

class Clusterer(Algorithm):
    """Base class for unsupervised clustering algorithms.

    Clustering groups similar instances together without the need for 
    pre-defined labels, discovering the underlying structure of the data.

    Overview
    --------
    Algorithms in this category typically handle partition-based (e.g., K-Means) 
    or hierarchical grouping.

    Attributes
    ----------
    n_clusters_ : int, optional
        The number of resulting clusters (if applicable).
    labels_ : np.ndarray
        The labels assigned to each instance in the training set.
    cluster_centers_ : np.ndarray, optional
        Coordinates of the cluster centroids.
    """

    _component_type = ComponentType.CLUSTERER

    def __init__(self):
        """Initialize the clusterer."""
        super().__init__()
        self.n_clusters_ = None
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Clusterer":
        """
        Build the clustering model from training data.

        Args:
            X: Training data (n_samples, n_features)
            y: Ignored (unsupervised learning)

        Returns:
            Self for method chaining
        """
        pass

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the model and return cluster labels.

        Args:
            X: Training data (n_samples, n_features)
            y: Ignored

        Returns:
            Cluster labels (n_samples,)
        """
        self.fit(X)
        return self.labels_

    def __repr__(self) -> str:
        """String representation."""
        name = self.__class__.__name__
        if self._is_fitted:
            return f"{name}(n_clusters={self.n_clusters_})"
        return f"{name}(not fitted)"

class DensityBasedClusterer(Clusterer):
    """
    Base class for density-based clusterers.

    Density-based clusterers can estimate the probability of
    cluster membership for each instance.
    """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster membership probabilities.

        Args:
            X: Data (n_samples, n_features)

        Returns:
            Probability matrix (n_samples, n_clusters)
        """
        pass

    def log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of data under the model.

        Args:
            X: Data (n_samples, n_features)

        Returns:
            Log-likelihood value
        """
        self._check_is_fitted()
        proba = self.predict_proba(X)
        return np.sum(np.log(np.sum(proba, axis=1) + 1e-10))

class UpdateableClusterer(Clusterer):
    """
    Base class for clusterers that support incremental updates.
    """

    @abstractmethod
    def update(self, X: np.ndarray) -> "UpdateableClusterer":
        """
        Update the model with new instances.

        Args:
            X: New data (n_samples, n_features)

        Returns:
            Self for method chaining
        """
        pass

# =============================================================================
# Regressor Base Class
# =============================================================================

class Regressor(Algorithm):
    """Base class for regression algorithms."""

    _component_type = ComponentType.REGRESSOR

# =============================================================================
# Associator Base Class and Data Structures
# =============================================================================

@dataclass
class FrequentItemset:
    """Represents a set of items that appear together frequently.

    Parameters
    ----------
    items : frozenset of int
        The set of item identifiers included in the itemset.
    support : float
        The proportion of transactions containing this itemset: 
        :math:`P(items)`.
    count : int, default=0
        The absolute frequency count of the itemset.
    """
    items: FrozenSet[int]
    support: float
    count: int = 0

    def __repr__(self) -> str:
        return f"Itemset({set(self.items)}, sup={self.support:.3f})"

    def __len__(self) -> int:
        return len(self.items)

    def __hash__(self) -> int:
        return hash(self.items)

    def __eq__(self, other) -> bool:
        if isinstance(other, FrequentItemset):
            return self.items == other.items
        return False

@dataclass
class AssociationRule:
    """Represents a discovered relationship in the form $A \\Rightarrow C$.

    Overview
    --------
    Association rules quantify how likely the consequent ($C$) is to appear 
    given the presence of the antecedent ($A$).

    Parameters
    ----------
    antecedent : frozenset of int
        The conditional part of the rule ($A$).
    consequent : frozenset of int
        The predicted part of the rule ($C$).
    support : float
        The joint probability: :math:`P(A \\cup C)`.
    confidence : float
        The conditional probability: :math:`P(C|A) = \\frac{P(A \\cup C)}{P(A)}`.
    lift : float, default=1.0
        The ratio of observed support to expected support if independent:
        :math:`\\frac{P(C|A)}{P(C)}`.
    leverage : float, default=0.0
        The difference from independence: :math:`P(A \\cup C) - P(A)P(C)`.
    conviction : float, default=1.0
        Implication strength: :math:`\\frac{1 - P(C)}{1 - \\text{confidence}}`.
    """
    antecedent: FrozenSet[int]
    consequent: FrozenSet[int]
    support: float
    confidence: float
    lift: float = 1.0
    leverage: float = 0.0
    conviction: float = 1.0
    jaccard: float = 0.0
    kulczynski: float = 0.0
    all_confidence: float = 0.0

    def __repr__(self) -> str:
        return (f"{set(self.antecedent)} -> {set(self.consequent)} "
                f"(conf={self.confidence:.3f}, lift={self.lift:.3f})")

    def __hash__(self) -> int:
        return hash((self.antecedent, self.consequent))

    def __eq__(self, other) -> bool:
        if isinstance(other, AssociationRule):
            return (self.antecedent == other.antecedent and
                    self.consequent == other.consequent)
        return False

class Associator(Algorithm):
    """Base class for Association Rule Mining.

    Discovers interesting patterns and relationships between items in 
    large datasets (e.g., market basket analysis).

    Theory
    ------
    Mining typically involves two steps:
    1. Finding all **Frequent Itemsets** that satisfy a minimum support.
    2. Generating **Association Rules** that satisfy a minimum confidence.

    Attributes
    ----------
    frequent_itemsets_ : list of FrequentItemset
        The collection of all itemsets found above the minimum support threshold.
    rules_ : list of AssociationRule
        The collection of all generated rules.
    """

    _component_type = ComponentType.ASSOCIATOR

    def __init__(self):
        """Initialize the associator."""
        super().__init__()
        self.frequent_itemsets_: List[FrequentItemset] = []
        self.rules_: List[AssociationRule] = []
        self.n_transactions_ = 0
        self.n_items_ = 0

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Associator":
        """
        Find frequent itemsets and generate association rules.

        Args:
            X: Transaction data. Can be:
               - Binary matrix (n_transactions, n_items)
               - List of lists containing item indices
            y: Ignored

        Returns:
            Self for method chaining
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Not applicable for association mining.

        Raises:
            NotImplementedError: Association miners don't predict
        """
        raise NotImplementedError(
            "Association rule miners don't support predict(). "
            "Use get_frequent_itemsets() or get_rules() instead."
        )

    def get_frequent_itemsets(self, min_size: int = 1,
                              max_size: Optional[int] = None) -> List[FrequentItemset]:
        """
        Get discovered frequent itemsets.

        Args:
            min_size: Minimum itemset size
            max_size: Maximum itemset size (None for no limit)

        Returns:
            List of frequent itemsets
        """
        self._check_is_fitted()
        result = [fs for fs in self.frequent_itemsets_ if len(fs) >= min_size]
        if max_size is not None:
            result = [fs for fs in result if len(fs) <= max_size]
        return result

    def get_rules(self, min_confidence: Optional[float] = None,
                  min_lift: Optional[float] = None) -> List[AssociationRule]:
        """
        Get discovered association rules.

        Args:
            min_confidence: Minimum confidence threshold
            min_lift: Minimum lift threshold

        Returns:
            List of association rules
        """
        self._check_is_fitted()
        result = self.rules_
        if min_confidence is not None:
            result = [r for r in result if r.confidence >= min_confidence]
        if min_lift is not None:
            result = [r for r in result if r.lift >= min_lift]
        return result

    def _preprocess_transactions(self, X) -> List[FrozenSet[int]]:
        """
        Convert input to list of transaction sets.

        Handles both binary matrix and list-of-lists formats.
        """
        if isinstance(X, np.ndarray):
            # Binary matrix format
            transactions = []
            for row in X:
                items = frozenset(np.where(row > 0)[0])
                if items:
                    transactions.append(items)
            return transactions
        else:
            # List of lists format
            return [frozenset(t) for t in X if t]

    def __repr__(self) -> str:
        """String representation."""
        name = self.__class__.__name__
        if self._is_fitted:
            return (f"{name}(n_itemsets={len(self.frequent_itemsets_)}, "
                    f"n_rules={len(self.rules_)})")
        return f"{name}(not fitted)"

# =============================================================================
# Legacy AlgorithmRegistry (backward compatibility)
# =============================================================================

class AlgorithmRegistry:
    """
    Central registry for all algorithms.

    Provides discovery, registration, and instantiation of algorithms.

    Note: This class now wraps the unified Hub for backward compatibility.
    New code should use the Hub directly.
    """

    @classmethod
    def register(cls, algorithm_class: Type[Algorithm]) -> Type[Algorithm]:
        """
        Register an algorithm.

        Args:
            algorithm_class: Algorithm class to register

        Returns:
            The algorithm class (for decorator usage)
        """
        # Determine component type
        algorithm_type = getattr(algorithm_class, "_algorithm_type", "algorithm")
        type_map = {
            "classifier": ComponentType.CLASSIFIER,
            "clusterer": ComponentType.CLUSTERER,
            "regressor": ComponentType.REGRESSOR,
            "associator": ComponentType.ASSOCIATOR,
        }
        component_type = type_map.get(algorithm_type, ComponentType.ALGORITHM)

        # Register with hub
        registry.register_class(algorithm_class, component_type)
        return algorithm_class

    @classmethod
    def get(cls, name: str) -> Type[Algorithm]:
        """
        Get algorithm class by name.

        Args:
            name: Algorithm name

        Returns:
            Algorithm class

        Raises:
            ValueError: If algorithm not found
        """
        try:
            return registry.get(name)
        except KeyError as e:
            raise ValueError(str(e))

    @classmethod
    def list(cls, type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all algorithms, optionally filtered by type.

        Args:
            type: Filter by type ('classifier', 'clusterer', 'regressor', 'associator')

        Returns:
            List of algorithm metadata dictionaries
        """
        type_map = {
            "classifier": ComponentType.CLASSIFIER,
            "clusterer": ComponentType.CLUSTERER,
            "regressor": ComponentType.REGRESSOR,
            "associator": ComponentType.ASSOCIATOR,
        }

        if type:
            component_type = type_map.get(type)
            return registry.list(component_type)
        else:
            # Return all algorithm types
            results = []
            for ct in [ComponentType.ALGORITHM, ComponentType.CLASSIFIER,
                       ComponentType.CLUSTERER, ComponentType.REGRESSOR,
                       ComponentType.ASSOCIATOR]:
                results.extend(registry.list(ct))
            return results

    @classmethod
    def search(cls, query: str) -> List[Dict[str, Any]]:
        """
        Search algorithms by keyword.

        Args:
            query: Search query

        Returns:
            List of matching algorithm metadata
        """
        return registry.search(query)

    @classmethod
    def get_by_type(cls, algorithm_type: str) -> List[str]:
        """
        Get all algorithm names of a specific type.

        Args:
            algorithm_type: Type to filter by

        Returns:
            List of algorithm names
        """
        type_map = {
            "classifier": ComponentType.CLASSIFIER,
            "clusterer": ComponentType.CLUSTERER,
            "regressor": ComponentType.REGRESSOR,
            "associator": ComponentType.ASSOCIATOR,
        }
        component_type = type_map.get(algorithm_type, ComponentType.ALGORITHM)
        return registry.list_names(component_type)

    @classmethod
    def clear(cls):
        """Clear all registered algorithms (mainly for testing)."""
        registry.clear()

# =============================================================================
# Decorators (with hub registration)
# =============================================================================

def algorithm(type: str = "classifier"):
    """
    Decorator to register an algorithm.

    Args:
        type: Algorithm type ('classifier', 'clusterer', 'regressor', 'associator')

    Returns:
        Decorator function

    Example::

        @algorithm(type="classifier")
        class RandomForest(Classifier):
            pass
    """
    type_map = {
        "classifier": ComponentType.CLASSIFIER,
        "clusterer": ComponentType.CLUSTERER,
        "regressor": ComponentType.REGRESSOR,
        "associator": ComponentType.ASSOCIATOR,
    }
    component_type = type_map.get(type, ComponentType.ALGORITHM)

    def decorator(cls: Type[Algorithm]) -> Type[Algorithm]:
        cls._algorithm_type = type
        return registry.register(component_type)(cls)

    return decorator

def classifier(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """
    Decorator to register a classifier.

    Example::

        @classifier(tags=["ensemble", "tree"])
        class RandomForest(Classifier):
            pass
    """
    return registry.register(
        ComponentType.CLASSIFIER,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

def clusterer(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """
    Decorator to register a clusterer.

    Example::

        @clusterer(tags=["partitioning"])
        class KMeans(Clusterer):
            pass
    """
    return registry.register(
        ComponentType.CLUSTERER,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

def regressor(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """
    Decorator to register a regressor.

    Example::

        @regressor(tags=["linear"])
        class LinearRegression(Regressor):
            pass
    """
    return registry.register(
        ComponentType.REGRESSOR,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

def associator(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """
    Decorator to register an associator.

    Example::

        @associator(tags=["itemset", "frequent"])
        class Apriori(Associator):
            pass
    """
    return registry.register(
        ComponentType.ASSOCIATOR,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

# =============================================================================
# Convenience functions
# =============================================================================

def get_algorithm(name: str) -> Type[Algorithm]:
    """
    Get algorithm class by name.

    Args:
        name: Algorithm name

    Returns:
        Algorithm class
    """
    return registry.get(name)

def list_algorithms(type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available algorithms.

    Args:
        type: Filter by type

    Returns:
        List of algorithm metadata
    """
    return AlgorithmRegistry.list(type)

def search_algorithms(query: str) -> List[Dict[str, Any]]:
    """
    Search algorithms by keyword.

    Args:
        query: Search query

    Returns:
        List of matching algorithms
    """
    return registry.search(query)
