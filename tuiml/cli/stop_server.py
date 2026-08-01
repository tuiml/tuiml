"""Stop Server Command - Shut down model serving API servers."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('stop-server')
@click.option('--server-id', type=str, help='ID of the server to stop. Omit to stop every running TuiML server.')
@click.option('--json-output', is_flag=True, help='Print the raw JSON result.')
def stop_server(server_id, json_output):
    """Shut down a background model serving API server.

    Stops a server started in the background by :func:`tuiml.serve`, and
    reports which servers were shut down. Without ``--server-id`` every
    running TuiML server is stopped. A foreground ``tuiml serve`` process is
    stopped with Ctrl-C instead.

    Examples
    --------
    Stop one specific server:

    $ tuiml stop-server --server-id srv_abc123

    Stop every running server:

    $ tuiml stop-server
    """
    kwargs = {
            'server_id': server_id,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_stop_server', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
