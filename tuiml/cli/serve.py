"""Serve Command - Start a REST API server for model predictions."""

import click


@click.command()
@click.option('--model-path', type=click.Path(exists=True), help='Path to a saved model file to serve (alternative to --model-id).')
@click.option('--model-id', '-m', help='ID of a model already trained by "tuiml train". Also names the model in the URL path.')
@click.option('--host', '-H', default='127.0.0.1',
              help='Interface to bind to (default: 127.0.0.1). Use 0.0.0.0 to accept connections from other machines.')
@click.option('--port', '-p', type=int, default=8000,
              help='TCP port to listen on (default: 8000).')
@click.option('--workers', '-w', type=int, default=1,
              help='Number of worker processes to run (default: 1). Raise this to serve concurrent requests.')
@click.option('--reload', is_flag=True,
              help='Restart the server whenever the source changes. For development only.')
def serve(model_path, model_id, host, port, workers, reload):
    """Serve a trained model over a REST API.

    Loads a model and runs it behind HTTP endpoints in the foreground, so
    other services can request predictions over the network. The server is
    backed by :mod:`tuiml.serving` and publishes interactive OpenAPI docs at
    ``/docs``. Press Ctrl-C to shut it down, or stop a detached server with
    ``tuiml stop-server``.

    Examples
    --------
    Serve a model on the default port:

    $ tuiml serve --model-path model.pkl

    Serve on a custom port with a specific model ID:

    $ tuiml serve --model-path classifier.pkl -m my_classifier -p 9000

    Serve with multiple workers, reachable from other machines:

    $ tuiml serve --model-path model.pkl -w 4 -H 0.0.0.0

    Serve a model already trained in this session, by ID:

    $ tuiml serve --model-id abc123

    Endpoints
    ---------
    - GET  /health - Health check
    - GET  /stats - Server statistics
    - GET  /models - List loaded models
    - POST /models - Load additional models
    - GET  /models/{id} - Get model info
    - POST /models/{id}/predict - Make predictions
    - POST /models/{id}/predict_proba - Get probabilities
    - POST /predict - Predict with default model

    API Documentation
    -----------------
    After starting the server, visit:
    - http://localhost:8000/docs - Swagger UI
    - http://localhost:8000/redoc - ReDoc
    """
    try:
        from tuiml.serving import serve as start_server
        from tuiml.agent.tools import _load_model_from_disk
        
        if not model_path and not model_id:
            raise click.UsageError("Must provide either --model-path or --model-id option.")
            
        if model_id and not model_path:
            model = _load_model_from_disk(model_id=model_id)
            if not model:
                raise click.ClickException(f"Model ID '{model_id}' not found.")
            
            start_server(
                model,
                model_id=model_id,
                host=host,
                port=port,
                workers=workers,
                reload=reload,
                background=False,
            )
        else:
            if not model_id:
                model_id = 'default'
            
            start_server(
                model_path,
                model_id=model_id,
                host=host,
                port=port,
                workers=workers,
                reload=reload,
                background=False,
            )
    except ImportError as e:
        raise click.ClickException(
            f"{e}\n\nInstall required packages with:\n  pip install fastapi uvicorn"
        )
    except Exception as e:
        raise click.ClickException(str(e))
