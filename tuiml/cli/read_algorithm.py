"""``tuiml read-algorithm``: print the source of any registered algorithm.

Thin CLI wrapper over the ``tuiml_read_algorithm`` tool. It reads both
user-authored algorithms stored under ``~/.tuiml/user_algorithms/`` and the
built-in algorithms shipped with the ``tuiml`` package.
"""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('read-algorithm')
@click.option('--name', type=str, required=True, help='Algorithm name (class name or directory name).')
@click.option('--version', type=str, help="Specific version to read (e.g. '1.0.2'). Defaults to latest.")
@click.option('--builtin/--no-builtin', default=False, help='Read a built-in tuiml algorithm instead of a user algorithm.')
@click.option('--json-output', is_flag=True, help='Print the raw JSON result.')
def read_algorithm(name, version, builtin, json_output):
    """Show the full source code of an algorithm, user-authored or built-in.

    For a user algorithm pass the directory name, which is also the class name;
    the latest version is read unless ``--version`` says otherwise. For a
    built-in algorithm add ``--builtin`` and pass either the class name
    (``RandomForestClassifier``) or the file stem (``random_forest``).

    The result carries the source twice, raw and with line numbers, which makes
    it easy to quote an exact snippet for ``tuiml edit-algorithm``. Built-in
    algorithms are read-only: fork one with ``tuiml create-algorithm`` to change
    its behaviour.

    Examples
    --------
    Read the latest version of a user algorithm:

    $ tuiml read-algorithm --name MyKNN

    Read one specific version:

    $ tuiml read-algorithm --name MyKNN --version 1.0.2

    Read a built-in algorithm by class name:

    $ tuiml read-algorithm --name RandomForestClassifier --builtin
    """
    kwargs = {
            'name': name,
            'version': version,
            'builtin': builtin,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_read_algorithm', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
