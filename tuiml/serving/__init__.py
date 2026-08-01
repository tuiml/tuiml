"""Serving trained models over HTTP.

A trained model is only useful once something else can call it. This puts one
behind a REST API — a FastAPI app with prediction and health endpoints — in a
single call, with no separate serving framework to stand up.

API
---
- **serve:** Load a model and start a server. Binds the port, waits until
  uvicorn actually reports ready, and returns the server's details, so a
  failure surfaces as an error rather than a URL for a server that never
  started.
- **stop_server / server_status:** Manage running servers by ``"host:port"``.
- **ModelServer:** The server itself, for hosting several models at once.
- **ModelManager:** Loading and holding those models.
- **create_app:** The FastAPI app, for mounting inside your own service.

Three ways in
-------------
All three share one registry, so a server started by any of them can be
inspected or stopped by the others::

    from tuiml.serving import serve       # Python
    serve("model.pkl", port=8000)

    tuiml serve model.pkl --port 8000     # CLI

    # or ask an agent, via the tuiml_serve_model MCP tool

Notes
-----
The port must be free: :func:`serve` raises rather than silently picking
another one, so nothing ends up listening where you did not expect.

Examples
--------
>>> from tuiml.serving import serve, stop_server
>>> info = serve("model.pkl", port=8000)          # doctest: +SKIP
>>> info["url"]                                   # doctest: +SKIP
'http://127.0.0.1:8000'
>>> stop_server("127.0.0.1:8000")                 # doctest: +SKIP
"""

from tuiml.serving.model_manager import ModelManager
from tuiml.serving.server import (
    ModelServer,
    create_app,
    serve,
    server_status,
    stop_server,
)

__all__ = [
    "ModelManager",
    "ModelServer",
    "create_app",
    "serve",
    "server_status",
    "stop_server",
]
