"""
Pydantic Schemas - Request and response models for the REST API.

Defines the data structures for API requests and responses,
providing validation and automatic documentation.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# =============================================================================
# Prediction Schemas
# =============================================================================

class PredictRequest(BaseModel):
    """Request schema for predictions."""

    features: List[List[Union[float, int]]] = Field(
        ...,
        description="2D array of input features. Shape: (n_samples, n_features)",
        examples=[[[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]]
    )


class PredictResponse(BaseModel):
    """Response schema for predictions."""

    predictions: List[Any] = Field(
        ...,
        description="Model predictions for each input sample"
    )
    model_id: str = Field(
        ...,
        description="Identifier of the model used"
    )
    model_class: Optional[str] = Field(
        None,
        description="Class name of the model"
    )


class PredictProbaResponse(BaseModel):
    """Response schema for probability predictions."""

    probabilities: List[List[float]] = Field(
        ...,
        description="Class probabilities. Shape: (n_samples, n_classes)"
    )
    classes: Optional[List[Any]] = Field(
        None,
        description="Class labels corresponding to probability columns"
    )
    model_id: str = Field(
        ...,
        description="Identifier of the model used"
    )


# =============================================================================
# Model Management Schemas
# =============================================================================

class LoadModelRequest(BaseModel):
    """Request schema for loading a model."""

    model_id: str = Field(
        ...,
        description="Unique identifier for the model",
        examples=["my_classifier"]
    )
    path: str = Field(
        ...,
        description="Path to the saved model file",
        examples=["models/classifier.pkl"]
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata to associate with the model"
    )


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""

    model_id: str = Field(..., description="Model identifier")
    model_class: Optional[str] = Field(None, description="Model class name")
    model_module: Optional[str] = Field(None, description="Model module path")
    params: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    path: Optional[str] = Field(None, description="Path to model file")
    loaded_at: Optional[str] = Field(None, description="Timestamp when loaded")
    prediction_count: int = Field(0, description="Number of predictions made")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class ModelListResponse(BaseModel):
    """Response schema for listing models."""

    models: List[str] = Field(
        ...,
        description="List of loaded model identifiers"
    )
    count: int = Field(..., description="Number of loaded models")


# =============================================================================
# Server Status Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field("healthy", description="Server health status")
    version: str = Field(..., description="TuiML version")
    models_loaded: int = Field(..., description="Number of models in memory")


class StatsResponse(BaseModel):
    """Response schema for server statistics."""

    loaded_models: int = Field(..., description="Number of loaded models")
    max_models: int = Field(..., description="Maximum models capacity")
    total_predictions: int = Field(..., description="Total predictions served")
    models: List[Dict[str, Any]] = Field(
        ...,
        description="Per-model statistics"
    )


# =============================================================================
# Error Schemas
# =============================================================================

class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    model_id: Optional[str] = Field(None, description="Related model ID if applicable")


# =============================================================================
# Batch Prediction Schemas
# =============================================================================

class BatchPredictRequest(BaseModel):
    """Request schema for batch predictions across multiple models."""

    requests: List[Dict[str, Any]] = Field(
        ...,
        description="List of prediction requests with model_id and features"
    )


class BatchPredictResponse(BaseModel):
    """Response schema for batch predictions."""

    results: List[Dict[str, Any]] = Field(
        ...,
        description="List of prediction results or errors"
    )
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")
