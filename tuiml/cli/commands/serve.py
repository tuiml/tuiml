"""Serve Command - Start a REST API server for model predictions."""

import click


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--model-id', '-m', default='default',
              help='Identifier for the model (default: "default")')
@click.option('--host', '-H', default='127.0.0.1',
              help='Host to bind to (default: 127.0.0.1)')
@click.option('--port', '-p', type=int, default=8000,
              help='Port to listen on (default: 8000)')
@click.option('--workers', '-w', type=int, default=1,
              help='Number of worker processes (default: 1)')
@click.option('--reload', is_flag=True,
              help='Enable auto-reload for development')
def serve(model_path, model_id, host, port, workers, reload):
    """Start a REST API server for model predictions.

    Load a trained model and serve predictions via HTTP endpoints.
    The server provides OpenAPI documentation at /docs.

    MODEL_PATH is the path to a saved model file (e.g., model.pkl).

    Examples
    --------
    Serve a model on the default port:

        $ tuiml serve model.pkl

    Serve on a custom port with a specific model ID:

        $ tuiml serve classifier.pkl -m my_classifier -p 9000

    Serve with multiple workers for production:

        $ tuiml serve model.pkl -w 4 -H 0.0.0.0

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
        start_server(
            model_path,
            model_id=model_id,
            host=host,
            port=port,
            workers=workers,
            reload=reload,
        )
    except ImportError as e:
        raise click.ClickException(
            f"{e}\n\nInstall required packages with:\n  pip install fastapi uvicorn"
        )
    except Exception as e:
        raise click.ClickException(str(e))
