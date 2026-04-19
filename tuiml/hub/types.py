"""Shared types and enums for the local component registry."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from enum import Enum


class ComponentType(Enum):
    """Types of components that can be registered."""

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
