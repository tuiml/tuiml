"""
TuiML Model Serving - REST API for serving trained models.

This module provides a built-in REST API server for serving trained models
and making predictions via HTTP endpoints.

Usage:
    # Programmatic API
    from tuiml.serving import ModelServer, serve

    # Serve a single model
    serve("model.pkl", port=8000)

    # Or use the server directly
    server = ModelServer()
    server.load_model("my_model", "model.pkl")
    app = server.create_app()

    # CLI
    tuiml serve model.pkl --port 8000
"""

from tuiml.serving.model_manager import ModelManager
from tuiml.serving.server import ModelServer, create_app, serve

__all__ = [
    "ModelManager",
    "ModelServer",
    "create_app",
    "serve",
]
