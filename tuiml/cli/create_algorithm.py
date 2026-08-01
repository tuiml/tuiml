"""``tuiml create-algorithm``: register a user-authored algorithm.

Thin CLI wrapper over the ``tuiml_create_algorithm`` tool. It stores the
submitted source under ``~/.tuiml/user_algorithms/`` and adds the class to
the component registry, so the rest of the CLI and every MCP tool can use
it by class name.
"""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('create-algorithm')
@click.option('--name', type=str, required=True, help='Directory name, usually equal to the class name (Python identifier).')
@click.option('--kind', type=str, required=True, help="Task kind: 'classifier' or 'regressor'. Must match the decorator on the submitted class.")
@click.option('--code', type=str, required=True, help='Full Python source. Must define exactly one @classifier or @regressor class.')
@click.option('--version', type=str, help="Semver (MAJOR.MINOR.PATCH) for this submission, e.g. '1.0.1'. Defaults to '1.0.0'.")
@click.option('--description', type=str, help='Optional short description (falls back to the class docstring).')
@click.option('--force/--no-force', default=False, help='Overwrite an existing <name>/<version>/ on disk. Prefer bumping --version instead.')
@click.option('--json-output', is_flag=True, help='Print the raw JSON result.')
def create_algorithm(name, kind, code, version, description, force, json_output):
    """Validate, save, and register a new user-authored algorithm.

    The source must define exactly one ``@classifier`` or ``@regressor`` class.
    It is checked with an AST denylist before anything is written (no
    ``subprocess``, ``socket``, ``os``, ``urllib`` or ``requests`` imports; no
    ``eval``, ``exec``, ``open`` or ``__import__`` calls), then saved to
    ``~/.tuiml/user_algorithms/<name>/<version>/algorithm.py``.

    Once saved the class is added to the component registry under its own class
    name, so every other command and MCP tool (``tuiml train``,
    ``tuiml describe``, ``tuiml_benchmark``) can use it immediately. Each
    version is also registered under the pinned alias
    ``<ClassName>_v<major>_<minor>_<patch>``, so two versions of the same
    algorithm can compete in a single experiment.

    Requires ``TUIML_ALLOW_USER_ALGORITHMS=1`` in the environment.

    Examples
    --------
    Register a classifier whose source lives in a local file:

    $ tuiml create-algorithm --name MyKNN --kind classifier \\
        --code "$(cat my_knn.py)"

    Submit an explicit version and description:

    $ tuiml create-algorithm --name MyKNN --kind classifier --version 1.1.0 \\
        --description "kNN with distance weighting" --code "$(cat my_knn.py)"

    Overwrite an existing version instead of bumping it, printing raw JSON:

    $ tuiml create-algorithm --name MyKNN --kind classifier \\
        --code "$(cat my_knn.py)" --force --json-output
    """
    kwargs = {
            'name': name,
            'kind': kind,
            'code': code,
            'version': version,
            'description': description,
            'force': force,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_create_algorithm', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
