"""REST serving of trained models.

These are result-formatting wrappers over the root serving API
(``tuiml.serve`` / ``tuiml.stop_server`` / ``tuiml.server_status``). They own
no server state: the root ``tuiml.serving.server._SERVERS`` registry is the
only one, so a server started by an agent is visible to ``tuiml.server_status()``
and vice versa, and both use the same ``"host:port"`` server ids.
"""

from typing import Any, Dict

from .._spec import ToolSpec
from .._shared import _load_model_from_disk
from .._state import _MODEL_INDEX


def _endpoints(url: str, model_id: str) -> Dict[str, str]:
    """Build the endpoint map advertised by ``tuiml_serve_model``.

    Parameters
    ----------
    url : str
        Base URL of the running server.
    model_id : str
        Identifier the model is registered under on that server.

    Returns
    -------
    endpoints : dict
        Mapping of endpoint name to full URL.
    """
    return {
        'predict': f'{url}/predict',
        'predict_model': f'{url}/models/{model_id}/predict',
        'predict_proba': f'{url}/models/{model_id}/predict_proba',
        'health': f'{url}/health',
        'models': f'{url}/models',
        'docs': f'{url}/docs',
    }


def execute_serve_model(**kwargs) -> Dict[str, Any]:
    """Start a REST API server to serve a trained model.

    Backs the ``tuiml_serve_model`` tool. Resolves the model, then delegates
    the whole server lifecycle to ``tuiml.serve``, which binds the port, waits
    for uvicorn to actually report ready, and registers the server in the
    single process-wide registry.

    Parameters
    ----------
    model_id : str, default=None
        Identifier of a trained model from ``tuiml_train``. One of
        ``model_id`` / ``model_path`` is required (both arrive via
        ``**kwargs``, like all parameters below).
    model_path : str, default=None
        Explicit path to a serialized model file.
    port : int, default=8000
        TCP port to bind; must be free.
    host : str, default='127.0.0.1'
        Interface to bind the server to. Anything other than loopback puts the
        API on the network; it stays authenticated, but the traffic is
        unencrypted.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``server_id`` (``'host:port'``),
        ``model_id``, ``url``, ``auth_token``, ``auth``, ``endpoints``
        (name -> URL map) and ``example_curl``. On failure: ``status``
        (``'error'``), ``error``, ``error_type`` and optionally ``suggestion``.
    """
    import tuiml

    model_id = kwargs.get('model_id')
    model_path = kwargs.get('model_path')
    port = kwargs.get('port', 8000)
    host = kwargs.get('host', '127.0.0.1')

    # Resolve the model file the same way every other model-taking tool does.
    if model_id and model_id in _MODEL_INDEX:
        serve_target = _MODEL_INDEX[model_id]
    elif model_path:
        if _load_model_from_disk(model_path=model_path) is None:
            return {
                'status': 'error',
                'error': f'Could not load a model from model_path={model_path!r}.',
                'error_type': 'ValueError',
                'suggestion': 'Check the path, or pass a model_id from tuiml_train.'
            }
        serve_target = model_path
        model_id = model_id or _basename_id(model_path)
    else:
        return {
            'status': 'error',
            'error': 'Model not found. Provide model_id (from tuiml_train) or a valid model_path.',
            'error_type': 'ValueError',
            'suggestion': 'Train a model first with tuiml_train'
        }

    try:
        info = tuiml.serve(serve_target, host=host, port=port,
                           model_id=model_id, background=True)
    except ImportError as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': 'ImportError',
            'suggestion': 'Install with: pip install "tuiml[serving]" (requires fastapi and uvicorn)'
        }
    except RuntimeError as e:
        # tuiml.serve() raises this when the port is taken or uvicorn never
        # became ready; its message already names the port and the remedy.
        return {
            'status': 'error',
            'error': str(e),
            'error_type': 'RuntimeError',
            'suggestion': f'Use a different port, or stop the existing server with tuiml_stop_server'
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'error_type': type(e).__name__}

    url = info['url']
    token = info.get('auth_token')
    # The server generates its own token, so this response is the only place
    # the caller can learn it. Without it every endpoint answers 401.
    auth_header = f'-H "Authorization: Bearer {token}" ' if token else ''
    return {
        'status': 'success',
        'server_id': info['server_id'],
        'model_id': info['model_id'],
        'url': url,
        'auth_token': token,
        'auth': (
            'Send "Authorization: Bearer <auth_token>" on every request except '
            '/ and /health.'
        ) if token else 'This server has authentication disabled.',
        'endpoints': _endpoints(url, info['model_id']),
        'example_curl': (
            f'curl -X POST {url}/predict '
            f'-H "Content-Type: application/json" '
            f'{auth_header}'
            f'-d \'{{"features": [[5.1, 3.5, 1.4, 0.2]]}}\''
        ),
    }


def _basename_id(model_path: str) -> str:
    """Derive a model id from a file path when none was supplied.

    Parameters
    ----------
    model_path : str
        Path to a serialized model file.

    Returns
    -------
    model_id : str
        The file's basename without its extension.
    """
    import os
    return os.path.splitext(os.path.basename(model_path))[0]


def execute_stop_server(**kwargs) -> Dict[str, Any]:
    """Stop one or all running model serving servers.

    Backs the ``tuiml_stop_server`` tool; delegates to ``tuiml.stop_server``.

    Parameters
    ----------
    server_id : str, default=None
        Identifier of the server to stop (arrives via ``**kwargs``).
        When omitted, every running server is stopped.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``) and ``message``; stopping
        all servers also returns ``stopped`` (list of
        ``{server_id, model_id, port}``). On failure: ``status``
        (``'error'``), ``error`` and optionally ``suggestion`` /
        ``error_type``.
    """
    import tuiml

    try:
        server_id = kwargs.get('server_id')
        running = {s['server_id']: s for s in tuiml.server_status()}

        if server_id:
            if server_id not in running:
                return {
                    'status': 'error',
                    'error': f"Server '{server_id}' not found",
                    'suggestion': 'Use tuiml_server_status to see running servers'
                }
            info = running[server_id]
            tuiml.stop_server(server_id)
            return {
                'status': 'success',
                'message': (
                    f"Server {server_id} stopped "
                    f"(was serving {info['model_id']} on port {info['port']})"
                )
            }

        stopped = [
            {'server_id': s['server_id'], 'model_id': s['model_id'], 'port': s['port']}
            for s in running.values()
        ]
        tuiml.stop_server()
        return {
            'status': 'success',
            'message': f'Stopped {len(stopped)} server(s)',
            'stopped': stopped
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'error_type': type(e).__name__}


def execute_server_status(**kwargs) -> Dict[str, Any]:
    """Get status of running model serving servers.

    Backs the ``tuiml_server_status`` tool; delegates to
    ``tuiml.server_status``. Takes no arguments (any ``**kwargs`` are
    ignored). Reports every server in the process, whether it was started
    by an agent tool or by ``tuiml.serve()`` directly.

    Returns
    -------
    result : dict
        ``status`` (``'success'``), ``count``, and ``servers`` -- a list
        of ``{server_id, model_id, url, host, port}``.
    """
    import tuiml

    try:
        servers = tuiml.server_status()
        return {'status': 'success', 'count': len(servers), 'servers': servers}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'error_type': type(e).__name__}


SERVE_SPEC = ToolSpec(
    name='tuiml_serve_model',
    description="Start a REST API server to serve a trained model for predictions. "
        "Returns the URL with endpoints: POST /predict, POST /models/{id}/predict, "
        "GET /health, GET /models, GET /docs (Swagger UI).",
    input_schema={
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model ID returned by tuiml_train"
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to model file (alternative to model_id)"
                },
                "port": {
                    "type": "integer",
                    "default": 8000,
                    "minimum": 1024,
                    "maximum": 65535,
                    "description": "Port to serve on (default: 8000)"
                },
                "host": {
                    "type": "string",
                    "default": "127.0.0.1",
                    "description": "Host to bind to (default: 127.0.0.1)"
                }
            },
            "required": []
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "server_id": {"type": "string", "description": "Server ID ('host:port'); pass to tuiml_stop_server"},
                "model_id": {"type": "string"},
                "url": {"type": "string", "description": "Base URL of the serving API"},
                "auth_token": {
                    "type": "string",
                    "description": (
                        "Bearer token this server requires. Send as "
                        "'Authorization: Bearer <token>' on every request except / and "
                        "/health. Generated per server; this response is the only place "
                        "it appears."
                    ),
                },
                "endpoints": {"type": "object", "description": "Map of endpoint names to URLs"},
                "example_curl": {"type": "string"},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_serve_model,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=False, open_world=True,
)


STOP_SPEC = ToolSpec(
    name='tuiml_stop_server',
    description="Stop a running model serving API server.",
    input_schema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Server ID returned by tuiml_serve_model. If omitted, stops all servers."
                }
            },
            "required": []
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "message": {"type": "string"},
                "stopped": {"type": "array"},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_stop_server,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=True, open_world=False,
)


STATUS_SPEC = ToolSpec(
    name='tuiml_server_status',
    description="Get status of running model serving API servers.",
    input_schema={
            "type": "object",
            "properties": {},
            "required": []
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "count": {"type": "integer"},
                "servers": {"type": "array"},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_server_status,
    group='workflow',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
    reproducible=False,
)
