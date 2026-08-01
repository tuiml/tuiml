"""``tuiml edit-algorithm``: patch the source of a user-authored algorithm.

Thin CLI wrapper over the ``tuiml_edit_algorithm`` tool. It performs a single
exact string replacement in a stored algorithm, re-validates the result, and
re-registers the class in the component registry.
"""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('edit-algorithm')
@click.option('--name', type=str, required=True, help='User algorithm name (directory name / class name).')
@click.option('--old-string', type=str, required=True, help='Exact text to replace. Must appear exactly once in the file.')
@click.option('--new-string', type=str, required=True, help='Replacement text.')
@click.option('--version', type=str, help="Version to patch, e.g. '1.0.2'. Defaults to the latest version.")
@click.option('--bump-version/--no-bump-version', default=False, help='Save the edit as a new patch version instead of overwriting the current one.')
@click.option('--json-output', is_flag=True, help='Print the raw JSON result.')
def edit_algorithm(name, old_string, new_string, version, bump_version, json_output):
    """Apply a targeted string replacement to a user-authored algorithm.

    Replaces exactly one occurrence of ``--old-string`` with ``--new-string``.
    The edit fails loudly when the text is missing or matches more than once, so
    add surrounding context until the snippet is unique. The patched source is
    AST-validated and the class is re-registered in the component registry, so
    every command and MCP tool sees the change right away.

    The usual loop is ``tuiml read-algorithm`` to find the exact text, then this
    command to change it. Built-in algorithms are read-only; fork one with
    ``tuiml create-algorithm`` before editing it.

    Requires ``TUIML_ALLOW_USER_ALGORITHMS=1`` in the environment.

    Examples
    --------
    Change a default hyperparameter in the latest version:

    $ tuiml edit-algorithm --name MyKNN \\
        --old-string "n_neighbors: int = 5" --new-string "n_neighbors: int = 7"

    Keep the current version intact and save the edit as a new patch version:

    $ tuiml edit-algorithm --name MyKNN --bump-version \\
        --old-string "metric = 'euclidean'" --new-string "metric = 'manhattan'"

    Patch one specific version rather than the latest:

    $ tuiml edit-algorithm --name MyKNN --version 1.0.0 \\
        --old-string "return preds" --new-string "return preds.astype(int)"
    """
    kwargs = {
            'name': name,
            'old_string': old_string,
            'new_string': new_string,
            'version': version,
            'bump_version': bump_version,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_edit_algorithm', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
