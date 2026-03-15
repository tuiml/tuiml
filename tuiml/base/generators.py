"""
Base class for data generators.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field

@dataclass
class GeneratedData:
    """Container for generated data."""
    X: np.ndarray
    y: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    target_names: Optional[List[str]] = None

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

class DataGenerator(ABC):
    """
    Abstract base class for data generators.

    Data generators create synthetic datasets for testing and
    benchmarking machine learning algorithms.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        random_state: Random seed for reproducibility
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
        """
        Generate the dataset.

        Returns:
            GeneratedData object containing X, y, and metadata
        """
        pass

    def __call__(self, return_X_y: bool = False) -> Union[GeneratedData, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate data (callable interface).

        Args:
            return_X_y: If True, return (X, y) tuple instead of GeneratedData

        Returns:
            GeneratedData or (X, y) tuple
        """
        data = self.generate()
        if return_X_y:
            return data.X, data.y
        return data

    def reset(self, random_state: Optional[int] = None):
        """Reset the random state."""
        if random_state is not None:
            self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema for this generator."""
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
        return f"{self.__class__.__name__}(n_samples={self.n_samples}, n_features={self.n_features})"

class ClassificationGenerator(DataGenerator):
    """
    Base class for classification data generators.
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
        schema = super().get_parameter_schema()
        schema["n_classes"] = {
            "type": "integer",
            "default": 2,
            "minimum": 2,
            "description": "Number of classes"
        }
        return schema

class RegressionGenerator(DataGenerator):
    """
    Base class for regression data generators.
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
        schema = super().get_parameter_schema()
        schema["noise"] = {
            "type": "number",
            "default": 0.0,
            "minimum": 0.0,
            "description": "Standard deviation of Gaussian noise"
        }
        return schema

class ClusteringGenerator(DataGenerator):
    """
    Base class for clustering data generators.
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
        schema = super().get_parameter_schema()
        schema["n_clusters"] = {
            "type": "integer",
            "default": 3,
            "minimum": 1,
            "description": "Number of clusters"
        }
        return schema
