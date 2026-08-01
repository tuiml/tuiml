"""List Files Command - Discover built-in and user-authored algorithm files."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('list-files')
@click.option('--builtin/--no-builtin', default=True, help='Include built-in tuiml algorithm files.')
@click.option('--user/--no-user', default=True, help='Include user-authored algorithm files.')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def list_files(builtin, user, json_output):
    """List the algorithm source files available on this machine.

    Reports the path, category and metadata of every algorithm file, both
    the ones shipped with TuiML and the ones you have written yourself.
    Run this before tuiml read-algorithm or tuiml search-source to see
    what exists and to find the exact name to pass to them.

    Examples
    --------
    List every algorithm file:

    $ tuiml list-files

    List only the algorithms you have written:

    $ tuiml list-files --no-builtin

    Get the listing as raw JSON for scripting:

    $ tuiml list-files --json-output
    """
    kwargs = {
            'builtin': builtin,
            'user': user,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_list_files', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
