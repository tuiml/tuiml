"""Shared types and enums for the TuiML hub."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass


class ComponentType(Enum):
    """Types of components that can be registered in the hub."""

    ALGORITHM = "algorithm"
    CLASSIFIER = "classifier"
    CLUSTERER = "clusterer"
    REGRESSOR = "regressor"
    ASSOCIATOR = "associator"
    PREPROCESSOR = "preprocessor"
    FILTER = "filter"
    TRANSFORMER = "transformer"
    FEATURE_SELECTOR = "feature_selector"
    FEATURE_EXTRACTOR = "feature_extractor"
    FEATURE_CONSTRUCTOR = "feature_constructor"
    METRIC = "metric"
    EVALUATOR = "evaluator"
    TIMESERIES = "timeseries"


@dataclass
class AlgorithmInfo:
    """Information about a remote algorithm.

    Attributes
    ----------
    id : int
        The unique identifier for the algorithm.
    name : str
        The unique slug/name of the algorithm.
    display_name : str
        The human-readable name of the algorithm.
    description : str
        A brief description of the algorithm.
    algorithm_type : str
        The type of algorithm (e.g., "classifier", "regressor").
    category : str
        The category of the algorithm (e.g., "tree", "function").
    version : str
        The version string of the algorithm.
    author : str or None, default=None
        The author of the algorithm.
    tags : list of str or None, default=None
        List of tags associated with the algorithm.
    downloads : int, default=0
        Number of times the algorithm has been downloaded.
    """
    id: int
    name: str
    display_name: str
    description: str
    algorithm_type: str
    category: str
    version: str
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    downloads: int = 0

    def __repr__(self):
        return f"AlgorithmInfo(name='{self.name}', type='{self.algorithm_type}', version='{self.version}')"


class Registrable(ABC):
    """Base mixin for all registrable components.

    Any class that wants to be registered in the hub should inherit from this.
    """

    _component_type: ComponentType = None
    _component_name: Optional[str] = None

    @classmethod
    def get_component_info(cls) -> Dict[str, Any]:
        """Return component metadata for registration.

        Returns
        -------
        info : dict
            A dictionary containing component information.
        """
        return {
            "name": cls._component_name or cls.__name__,
            "type": cls._component_type.value if cls._component_type else "unknown",
            "description": cls.__doc__ or "No description available",
            "parameters": cls.get_parameter_schema(),
            "version": getattr(cls, "_version", "1.0.0"),
            "author": getattr(cls, "_author", None),
            "tags": getattr(cls, "_tags", []),
        }

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for component parameters.

        Override this method to define custom parameters.

        Returns
        -------
        schema : dict
            A dictionary mapping parameter names to their schemas.
        """
        return {}
