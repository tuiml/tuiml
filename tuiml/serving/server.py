"""
Model Serving Server - FastAPI REST API for serving trained models.

Provides HTTP endpoints for loading models, making predictions,
and managing the model serving lifecycle.

Usage:
    # Serve a single model
    from tuiml.serving import serve
    serve("model.pkl", port=8000)

    # Or create a custom app
    from tuiml.serving import ModelServer
    server = ModelServer()
    server.load_model("my_model", "model.pkl")
    app = server.create_app()
    # Run with uvicorn: uvicorn app:app
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

from tuiml.serving.model_manager import ModelManager
from tuiml.serving.schemas import (
    PredictRequest,
    PredictResponse,
    PredictProbaResponse,
    LoadModelRequest,
    ModelInfoResponse,
    ModelListResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
)

# FastAPI is optional - check availability
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None


class ModelServer:
    """
    REST API server for serving trained TuiML models.

    Wraps FastAPI to provide endpoints for model management
    and prediction serving.

    Parameters
    ----------
    max_models : int, default=10
        Maximum number of models to keep in memory.
    title : str, default="TuiML Model Server"
        API title for documentation.
    version : str, default="1.0.0"
        API version.

    Examples
    --------
    >>> from tuiml.serving import ModelServer
    >>> server = ModelServer()
    >>> server.load_model("clf", "classifier.pkl")
    >>> app = server.create_app()
    >>> # Run with: uvicorn app:app --port 8000
    """

    def __init__(
        self,
        max_models: int = 10,
        title: str = "TuiML Model Server",
        version: str = "1.0.0",
    ):
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for model serving. "
                "Install with: pip install fastapi uvicorn"
            )

        self.manager = ModelManager(max_models=max_models)
        self.title = title
        self.version = version
        self._app: Optional[FastAPI] = None

    def load_model(
        self,
        model_id: str,
        path: Union[str, Path],
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Load a model into the server.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.
        path : str or Path
            Path to the saved model file.
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        dict
            Model information.
        """
        return self.manager.load(model_id, path, metadata)

    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        return self.manager.unload(model_id)

    def create_app(self) -> "FastAPI":
        """
        Create the FastAPI application with all endpoints.

        Returns
        -------
        FastAPI
            Configured FastAPI application instance.
        """
        from tuiml import __version__ as tuiml_version

        app = FastAPI(
            title=self.title,
            version=self.version,
            description=(
                "REST API for serving trained TuiML machine learning models. "
                "Load models, make predictions, and manage the serving lifecycle."
            ),
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # CORS middleware for cross-origin requests
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        manager = self.manager

        # =====================================================================
        # Root Endpoint
        # =====================================================================

        @app.get("/", tags=["Status"], summary="Server info")
        async def root():
            """Landing page with server info, loaded models, and endpoints."""
            models = manager.list_models()
            model_details = []
            for mid in models:
                try:
                    info = manager.get_model_info(mid)
                    model_details.append({
                        "model_id": mid,
                        "model_class": info.get("model_class"),
                        "model_path": info.get("path"),
                        "predict_url": f"/models/{mid}/predict",
                        "predict_proba_url": f"/models/{mid}/predict_proba",
                    })
                except Exception:
                    model_details.append({"model_id": mid})

            return {
                "service": self.title,
                "version": self.version,
                "tuiml_version": tuiml_version,
                "status": "healthy",
                "models_loaded": len(models),
                "models": model_details,
                "endpoints": {
                    "docs": "/docs",
                    "redoc": "/redoc",
                    "health": "/health",
                    "stats": "/stats",
                    "models": "/models",
                    "predict": "/predict",
                },
            }

        # =====================================================================
        # Health & Status Endpoints
        # =====================================================================

        @app.get(
            "/health",
            response_model=HealthResponse,
            tags=["Status"],
            summary="Health check"
        )
        async def health():
            """Check server health status."""
            return HealthResponse(
                status="healthy",
                version=tuiml_version,
                models_loaded=len(manager.list_models())
            )

        @app.get(
            "/stats",
            response_model=StatsResponse,
            tags=["Status"],
            summary="Server statistics"
        )
        async def stats():
            """Get server statistics including prediction counts."""
            return manager.get_stats()

        # =====================================================================
        # Model Management Endpoints
        # =====================================================================

        @app.get(
            "/models",
            response_model=ModelListResponse,
            tags=["Models"],
            summary="List loaded models"
        )
        async def list_models():
            """List all currently loaded models."""
            models = manager.list_models()
            return ModelListResponse(models=models, count=len(models))

        @app.post(
            "/models",
            response_model=ModelInfoResponse,
            tags=["Models"],
            summary="Load a model"
        )
        async def load_model(request: LoadModelRequest):
            """Load a model from disk into memory."""
            try:
                info = manager.load(
                    request.model_id,
                    request.path,
                    request.metadata
                )
                return info
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get(
            "/models/{model_id}",
            response_model=ModelInfoResponse,
            tags=["Models"],
            summary="Get model info"
        )
        async def get_model_info(model_id: str):
            """Get information about a loaded model."""
            try:
                return manager.get_model_info(model_id)
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_id}' not found"
                )

        @app.delete(
            "/models/{model_id}",
            tags=["Models"],
            summary="Unload a model"
        )
        async def unload_model(model_id: str):
            """Unload a model from memory."""
            if manager.unload(model_id):
                return {"message": f"Model '{model_id}' unloaded"}
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found"
            )

        # =====================================================================
        # Prediction Endpoints
        # =====================================================================

        @app.post(
            "/models/{model_id}/predict",
            response_model=PredictResponse,
            tags=["Predictions"],
            summary="Make predictions"
        )
        async def predict(model_id: str, request: PredictRequest):
            """
            Make predictions using a loaded model.

            Send input features and receive model predictions.
            """
            try:
                predictions = manager.predict(model_id, request.features)
                info = manager.get_model_info(model_id)

                return PredictResponse(
                    predictions=predictions.tolist(),
                    model_id=model_id,
                    model_class=info.get("model_class")
                )
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_id}' not found"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post(
            "/models/{model_id}/predict_proba",
            response_model=PredictProbaResponse,
            tags=["Predictions"],
            summary="Get prediction probabilities"
        )
        async def predict_proba(model_id: str, request: PredictRequest):
            """
            Get class probabilities from a classifier.

            Only works with models that support probability estimation.
            """
            try:
                probabilities = manager.predict_proba(model_id, request.features)

                # Try to get class labels
                model = manager.get_model(model_id)
                classes = None
                if hasattr(model, "classes_"):
                    classes = model.classes_.tolist()

                return PredictProbaResponse(
                    probabilities=probabilities.tolist(),
                    classes=classes,
                    model_id=model_id
                )
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_id}' not found"
                )
            except NotImplementedError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # =====================================================================
        # Convenience Endpoints (when single model loaded)
        # =====================================================================

        @app.post(
            "/predict",
            response_model=PredictResponse,
            tags=["Predictions"],
            summary="Predict (default model)"
        )
        async def predict_default(
            request: PredictRequest,
            model_id: Optional[str] = Query(
                None,
                description="Model ID (uses first loaded if not specified)"
            )
        ):
            """
            Make predictions using the default (first loaded) model.

            Convenient when serving a single model.
            """
            models = manager.list_models()
            if not models:
                raise HTTPException(
                    status_code=400,
                    detail="No models loaded"
                )

            target_model = model_id or models[0]
            try:
                predictions = manager.predict(target_model, request.features)
                info = manager.get_model_info(target_model)

                return PredictResponse(
                    predictions=predictions.tolist(),
                    model_id=target_model,
                    model_class=info.get("model_class")
                )
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{target_model}' not found"
                )

        self._app = app
        return app

    @property
    def app(self) -> "FastAPI":
        """Get or create the FastAPI application."""
        if self._app is None:
            self._app = self.create_app()
        return self._app


def create_app(
    model_path: Optional[Union[str, Path]] = None,
    model_id: str = "default",
    max_models: int = 10,
) -> "FastAPI":
    """
    Create a FastAPI application for model serving.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to a model to load at startup.
    model_id : str, default="default"
        Identifier for the startup model.
    max_models : int, default=10
        Maximum models to cache.

    Returns
    -------
    FastAPI
        Configured FastAPI application.

    Examples
    --------
    >>> from tuiml.serving import create_app
    >>> app = create_app("classifier.pkl")
    >>> # Run with: uvicorn module:app --port 8000
    """
    server = ModelServer(max_models=max_models)

    if model_path:
        server.load_model(model_id, model_path)

    return server.create_app()


# Module-level state for tracking running servers
_SERVERS: Dict[str, Dict] = {}


def serve(
    model_or_path,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_id: str = "default",
    background: bool = True,
    reload: bool = False,
    workers: int = 1,
):
    """Serve a trained model via REST API.

    Accepts a file path, a fitted ``Workflow``, or a model object and starts
    a uvicorn server exposing prediction endpoints.

    Parameters
    ----------
    model_or_path : str, Workflow, or model object
        The model to serve:

        - ``str``: Path to a saved model file
        - ``Workflow``: a fitted pipeline from ``train()`` or ``Workflow.fit()``
        - model object, Any object with a ``predict()`` method

    host : str, default="127.0.0.1"
        Host to bind the server to.

    port : int, default=8000
        Port to listen on.

    model_id : str, default="default"
        Identifier for the model in the server.

    background : bool, default=True
        If True, run the server in a daemon thread and return immediately.
        If False, block until the server is stopped.

    reload : bool, default=False
        Enable auto-reload for development. Only applies when
        ``background=False``.

    workers : int, default=1
        Number of worker processes. Only applies when ``background=False``.

    Returns
    -------
    dict or None
        If ``background=True``, returns a dict with ``server_id``, ``url``,
        and ``endpoints``. If ``background=False``, blocks and returns None.

    Examples
    --------
    Serve the pipeline that ``train()`` returned:

    >>> import tuiml
    >>> model = tuiml.train({
    ...     "model": {"name": "NaiveBayesClassifier"},
    ...     "data": {"source": "iris"},
    ... })                                                # doctest: +SKIP
    >>> info = tuiml.serve(model, port=8000)              # doctest: +SKIP
    >>> info["url"]                                       # doctest: +SKIP
    'http://127.0.0.1:8000'

    Serve any fitted model. Every algorithm works, with or without a
    pipeline around it (models also carry their own ``.serve()`` method):

    >>> from tuiml.algorithms.trees import RandomForestClassifier
    >>> clf = RandomForestClassifier(n_estimators=100)    # doctest: +SKIP
    >>> clf = clf.fit(X_train, y_train)                   # doctest: +SKIP
    >>> info = tuiml.serve(clf, port=8001)                # doctest: +SKIP

    Serve a fitted pipeline built from objects; the fitted preprocessing
    travels with it, so the API accepts raw rows:

    >>> from tuiml import Workflow
    >>> from tuiml.preprocessing import StandardScaler    # doctest: +SKIP
    >>> wf = Workflow([StandardScaler(), RandomForestClassifier()])
    ...                                                   # doctest: +SKIP
    >>> wf = wf.fit(X_train, y_train)                     # doctest: +SKIP
    >>> info = tuiml.serve(wf, port=8002)                 # doctest: +SKIP

    Serve a model saved earlier with ``model.save(path)``:

    >>> info = tuiml.serve("model.pkl", port=8003)        # doctest: +SKIP

    Then manage the running servers:

    >>> tuiml.server_status()                             # doctest: +SKIP
    >>> tuiml.stop_server("127.0.0.1:8001")               # doctest: +SKIP
    >>> tuiml.stop_server()                               # doctest: +SKIP
    """
    import tempfile

    from tuiml.utils.serialization import save_model
    from tuiml.workflow import Workflow

    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for model serving. "
            "Install with: pip install fastapi uvicorn"
        )

    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required for serving. "
            "Install with: pip install uvicorn"
        )

    server = ModelServer()

    if isinstance(model_or_path, (str, Path)):
        # File path, load directly
        server.load_model(model_id, model_or_path)
    elif isinstance(model_or_path, Workflow):
        # Fitted Workflow, serialize the whole pipeline so the served model
        # applies its transformations too.
        if not model_or_path._is_fitted:
            raise RuntimeError("This Workflow is not fitted yet. Call fit() first.")
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp.close()
        save_model(model_or_path, tmp.name)
        server.load_model(model_id, tmp.name)
    else:
        # Assume it's a model object with predict()
        if not hasattr(model_or_path, 'predict'):
            raise ValueError("Object does not have a predict() method.")
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp.close()
        save_model(model_or_path, tmp.name)
        server.load_model(model_id, tmp.name)

    app = server.create_app()
    server_id = f"{host}:{port}"

    if not background:
        _SERVERS[server_id] = {
            "server_id": server_id,
            "host": host,
            "port": port,
            "model_id": model_id,
            "url": f"http://{host}:{port}",
            "server_obj": server,
        }
        try:
            uvicorn.run(app, host=host, port=port, reload=reload,
                        workers=workers, log_level="info")
        finally:
            _SERVERS.pop(server_id, None)
        return None

    # Background mode, run in daemon thread
    import asyncio
    import threading
    import time

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    uvicorn_server = uvicorn.Server(config)

    def run_server():
        """Run uvicorn server with proper event loop handling."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(uvicorn_server.serve())
        finally:
            loop.close()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait for uvicorn to actually bind before reporting success. Without this
    # a failure in the background thread, most often the port already being in
    # use, would go unnoticed and this call would hand back a URL for a server
    # that never started.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if getattr(uvicorn_server, "started", False):
            break
        if not thread.is_alive():
            raise RuntimeError(
                f"The server failed to start on {host}:{port}. The port is "
                f"most likely already in use, pick another port, or call "
                f"tuiml.stop_server() to shut down servers started earlier."
            )
        time.sleep(0.05)
    else:
        uvicorn_server.should_exit = True
        raise RuntimeError(
            f"The server did not become ready on {host}:{port} within 10s."
        )

    info = {
        "server_id": server_id,
        "host": host,
        "port": port,
        "model_id": model_id,
        "url": f"http://{host}:{port}",
        "endpoints": {
            "predict": f"http://{host}:{port}/models/{model_id}/predict",
            "health": f"http://{host}:{port}/health",
            "docs": f"http://{host}:{port}/docs",
        },
        "server_obj": server,
        "uvicorn_server": uvicorn_server,
        "thread": thread,
    }
    _SERVERS[server_id] = info

    # Return a clean dict without internal objects
    return {
        "server_id": server_id,
        "host": host,
        "port": port,
        "model_id": model_id,
        "url": f"http://{host}:{port}",
        "endpoints": info["endpoints"],
    }


def stop_server(server_id: Optional[str] = None) -> None:
    """Stop running model server(s).

    Parameters
    ----------
    server_id : str, optional
        The server ID (``"host:port"``) to stop. If None, stops all
        running servers.

    Examples
    --------
    Stop a specific server:

    >>> import tuiml
    >>> tuiml.stop_server("127.0.0.1:9999")

    Stop all servers:

    >>> tuiml.stop_server()
    """
    if server_id is not None:
        info = _SERVERS.pop(server_id, None)
        if info and "uvicorn_server" in info:
            info["uvicorn_server"].should_exit = True
    else:
        for sid in list(_SERVERS.keys()):
            info = _SERVERS.pop(sid, None)
            if info and "uvicorn_server" in info:
                info["uvicorn_server"].should_exit = True


def server_status() -> List[Dict]:
    """Get status of running model servers.

    Returns
    -------
    list of dict
        List of server info dicts, each containing ``server_id``,
        ``host``, ``port``, ``model_id``, and ``url``.

    Examples
    --------
    >>> import tuiml
    >>> tuiml.server_status()
    [{'server_id': '127.0.0.1:9999', 'url': 'http://127.0.0.1:9999', ...}]
    """
    return [
        {
            "server_id": info["server_id"],
            "host": info["host"],
            "port": info["port"],
            "model_id": info["model_id"],
            "url": info["url"],
        }
        for info in _SERVERS.values()
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TuiML Model Serving Server"
    )
    parser.add_argument("model", help="Path to the trained model file")
    parser.add_argument("--model-id", "-m", default="default")
    parser.add_argument("--host", "-H", default="127.0.0.1")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()
    serve(args.model, model_id=args.model_id, host=args.host,
          port=args.port, workers=args.workers, reload=args.reload,
          background=False)
