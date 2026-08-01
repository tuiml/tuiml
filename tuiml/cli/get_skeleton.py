"""``tuiml get-skeleton``: print a starter template for a new algorithm.

Thin CLI wrapper over the ``tuiml_get_skeleton`` tool. It emits a filled-in
``@classifier`` / ``@regressor`` class template that is ready to complete and
hand to ``tuiml create-algorithm``.
"""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('get-skeleton')
@click.option('--kind', type=str, required=True, help="Task kind the new algorithm targets: 'classifier' or 'regressor'.")
@click.option('--class-name', type=str, help="Python identifier for the new class, e.g. 'MyGradientBoosting'. Defaults to 'MyAlgorithm'.")
@click.option('--version', type=str, help="Initial semver baked into the decorator, e.g. '1.0.0'.")
@click.option('--description', type=str, help='One-line description used as the module and class docstring.')
@click.option('--json-output', is_flag=True, help='Print the raw JSON result.')
def get_skeleton(kind, class_name, version, description, json_output):
    """Print a ready-to-edit source template for a new algorithm.

    The template is a complete ``@classifier`` or ``@regressor`` class with the
    class name, version, and docstring already filled in, plus placeholder
    hyperparameters. Fill in ``fit()`` and ``predict()``, adjust ``__init__``
    and ``get_parameter_schema()``, then submit the finished source with
    ``tuiml create-algorithm``.

    The template comes back in the ``code`` field of the result, so pipe the
    output through a JSON reader when you want the bare source on disk.

    Requires ``TUIML_ALLOW_USER_ALGORITHMS=1`` in the environment.

    Examples
    --------
    Print a classifier template with a chosen class name:

    $ tuiml get-skeleton --kind classifier --class-name MyGradientBoosting

    Start a regressor at an explicit version, with a one-line description:

    $ tuiml get-skeleton --kind regressor --class-name MyRidge \\
        --version 1.0.0 --description "Ridge regression with a custom solver"

    Write just the source to a file, ready to edit and submit:

    $ tuiml get-skeleton --kind classifier --class-name MyKNN --json-output \\
        | jq -r .code > my_knn.py
    """
    kwargs = {
            'kind': kind,
            'class_name': class_name,
            'version': version,
            'description': description,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_get_skeleton', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
