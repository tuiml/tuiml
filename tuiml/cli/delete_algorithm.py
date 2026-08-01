"""``tuiml delete-algorithm``: remove a user-authored algorithm from disk.

Thin CLI wrapper over the ``tuiml_delete_algorithm`` tool. It deletes one
version or the whole algorithm directory under ``~/.tuiml/user_algorithms/``.
"""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('delete-algorithm')
@click.option('--name', type=str, required=True, help='User algorithm name (directory name / class name).')
@click.option('--version', type=str, help='Version to delete, e.g. 1.0.0. If omitted, all versions are removed.')
@click.option('--json-output', is_flag=True, help='Print the raw JSON result.')
def delete_algorithm(name, version, json_output):
    """Delete a user-authored algorithm from disk.

    Pass only ``--name`` to remove every stored version of the algorithm, or add
    ``--version`` to remove a single one. Deletion is permanent: the files under
    ``~/.tuiml/user_algorithms/`` are gone, so keep a copy of anything you may
    want back.

    Classes that are already loaded stay in the component registry of a running
    process; a live MCP server keeps serving them until it restarts
    (``tuiml restart``).

    Requires ``TUIML_ALLOW_USER_ALGORITHMS=1`` in the environment.

    Examples
    --------
    Remove every version of an algorithm:

    $ tuiml delete-algorithm --name MyKNN

    Remove one version and keep the rest:

    $ tuiml delete-algorithm --name MyKNN --version 1.0.0

    Delete a version and inspect the raw result:

    $ tuiml delete-algorithm --name MyKNN --version 1.0.0 --json-output
    """
    kwargs = {
            'name': name,
            'version': version,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_delete_algorithm', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
