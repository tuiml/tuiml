"""Serve Command - Start a REST API server for model predictions."""

import click


def _announce(token, host, port):
    """Print the bearer token before the server takes over the terminal.

    Parameters
    ----------
    token : str or False
        The token clients must send, or False when authentication is off.
    host, port : str, int
        Where the server is about to bind, for a copy-pasteable example.
    """
    if token is False:
        click.secho(
            "Authentication is OFF. Anyone who can reach this port can load and "
            "run models on this machine.",
            fg="red",
        )
        return
    click.secho(f"Auth token: {token}", fg="green", bold=True)
    click.echo(
        f'  curl -H "Authorization: Bearer {token}" http://{host}:{port}/models'
    )


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
@click.option('--auth-token', default=None,
              help='Bearer token clients must send. Generated and printed if omitted.')
@click.option('--no-auth', is_flag=True,
              help='Serve without authentication. Only safe behind a proxy that authenticates for you.')
@click.option('--models-dir', type=click.Path(exists=True, file_okay=False), default=None,
              help='Directory that POST /models may load from. Omitted, that endpoint is refused.')
def serve(model_path, model_id, host, port, workers, reload,
          auth_token, no_auth, models_dir):
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

    Authentication
    --------------
    Every endpoint except / and /health needs a bearer token, printed on
    startup. Loading a model unpickles a file and predicting runs it, so the
    server does not expose either without one.

    $ curl -H "Authorization: Bearer $TOKEN" localhost:8000/health
    """
    import secrets

    try:
        from tuiml.serving import serve as start_server
        from tuiml.agent.tools import _load_model_from_disk

        if no_auth and auth_token:
            raise click.UsageError("Pass either --auth-token or --no-auth, not both.")

        # Generated here rather than inside serve(): this command runs the
        # server in the foreground, so anything it only logged would be missed,
        # and a token nobody can read makes the server unusable.
        token = False if no_auth else (auth_token or secrets.token_urlsafe(32))
        
        if not model_path and not model_id:
            raise click.UsageError("Must provide either --model-path or --model-id option.")
            
        if model_id and not model_path:
            model = _load_model_from_disk(model_id=model_id)
            if not model:
                raise click.ClickException(f"Model ID '{model_id}' not found.")
            
            _announce(token, host, port)
            start_server(
                model,
                model_id=model_id,
                host=host,
                port=port,
                workers=workers,
                reload=reload,
                background=False,
                auth_token=token,
                models_dir=models_dir,
            )
        else:
            if not model_id:
                model_id = 'default'
            
            _announce(token, host, port)
            start_server(
                model_path,
                model_id=model_id,
                host=host,
                port=port,
                workers=workers,
                reload=reload,
                background=False,
                auth_token=token,
                models_dir=models_dir,
            )
    except ImportError as e:
        raise click.ClickException(
            f"{e}\n\nInstall required packages with:\n  pip install fastapi uvicorn"
        )
    except Exception as e:
        raise click.ClickException(str(e))
