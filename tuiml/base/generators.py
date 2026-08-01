"""Base classes for synthetic data generators.

Data generators create synthetic datasets for testing and benchmarking
machine learning algorithms. Concrete generators subclass one of the
task-specific bases (``ClassificationGenerator``, ``RegressionGenerator``,
``ClusteringGenerator``) and implement ``generate()``.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field

@dataclass
class GeneratedData:
    """Container for generated data.

    Attributes
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Generated feature matrix.
    y : np.ndarray of shape (n_samples,), optional
        Generated targets (labels, values, or cluster assignments).
    feature_names : list of str
        Names of the generated features.
    target_names : list of str, optional
        Names of the target classes or outputs.
    """
    X: np.ndarray
    y: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    target_names: Optional[List[str]] = None

    @property
    def n_samples(self) -> int:
        """Return the number of generated samples."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Return the number of generated features."""
        return self.X.shape[1]

class DataGenerator(ABC):
    """Abstract base class for data generators.

    Data generators create synthetic datasets for testing and
    benchmarking machine learning algorithms.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    n_features : int, default=2
        Number of features.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 2,
        random_state: Optional[int] = None
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    @abstractmethod
    def generate(self) -> GeneratedData:
        """Generate the dataset.

        Returns
        -------
        data : GeneratedData
            Container with ``X``, ``y``, and metadata.
        """
        pass

    def __call__(self, return_X_y: bool = False) -> Union[GeneratedData, Tuple[np.ndarray, np.ndarray]]:
        """Generate data by calling the generator directly.

        Parameters
        ----------
        return_X_y : bool, default=False
            If True, return an ``(X, y)`` tuple instead of ``GeneratedData``.

        Returns
        -------
        data : GeneratedData or tuple of (np.ndarray, np.ndarray)
            The generated dataset, either as a container or as ``(X, y)``.
        """
        data = self.generate()
        if return_X_y:
            return data.X, data.y
        return data

    def reset(self, random_state: Optional[int] = None):
        """Reset the random number generator.

        Parameters
        ----------
        random_state : int, optional
            New random seed. If None, the current seed is reused.

        Returns
        -------
        None
        """
        if random_state is not None:
            self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema for this generator.

        Returns
        -------
        schema : dict of str to dict
            Mapping of constructor parameter names to JSON-Schema-style
            descriptions.
        """
        return {
            "n_samples": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Number of samples to generate"
            },
            "n_features": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Number of features"
            },
            "random_state": {
                "type": "integer",
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return f"{self.__class__.__name__}(n_samples={self.n_samples}, n_features={self.n_features})"

class ClassificationGenerator(DataGenerator):
    """Base class for classification data generators.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    n_features : int, default=2
        Number of features.
    n_classes : int, default=2
        Number of classes.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 2,
        n_classes: int = 2,
        random_state: Optional[int] = None
    ):
        super().__init__(n_samples, n_features, random_state)
        self.n_classes = n_classes

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema for this generator, including ``n_classes``.

        Returns
        -------
        schema : dict of str to dict
            Mapping of constructor parameter names to JSON-Schema-style
            descriptions.
        """
        schema = super().get_parameter_schema()
        schema["n_classes"] = {
            "type": "integer",
            "default": 2,
            "minimum": 2,
            "description": "Number of classes"
        }
        return schema

class RegressionGenerator(DataGenerator):
    """Base class for regression data generators.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    n_features : int, default=2
        Number of features.
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to the targets.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 2,
        noise: float = 0.0,
        random_state: Optional[int] = None
    ):
        super().__init__(n_samples, n_features, random_state)
        self.noise = noise

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema for this generator, including ``noise``.

        Returns
        -------
        schema : dict of str to dict
            Mapping of constructor parameter names to JSON-Schema-style
            descriptions.
        """
        schema = super().get_parameter_schema()
        schema["noise"] = {
            "type": "number",
            "default": 0.0,
            "minimum": 0.0,
            "description": "Standard deviation of Gaussian noise"
        }
        return schema

class ClusteringGenerator(DataGenerator):
    """Base class for clustering data generators.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    n_features : int, default=2
        Number of features.
    n_clusters : int, default=3
        Number of clusters.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 2,
        n_clusters: int = 3,
        random_state: Optional[int] = None
    ):
        super().__init__(n_samples, n_features, random_state)
        self.n_clusters = n_clusters

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema for this generator, including ``n_clusters``.

        Returns
        -------
        schema : dict of str to dict
            Mapping of constructor parameter names to JSON-Schema-style
            descriptions.
        """
        schema = super().get_parameter_schema()
        schema["n_clusters"] = {
            "type": "integer",
            "default": 3,
            "minimum": 1,
            "description": "Number of clusters"
        }
        return schema
