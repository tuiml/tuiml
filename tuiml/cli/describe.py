"""Describe Command - Show the parameter schema for any TuiML component."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('describe')
@click.option('--name', type=str, required=True, help="Component name (e.g., 'RandomForestClassifier', 'SimpleImputer', 'iris')")
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def describe(name, json_output):
    """Show details and the parameter schema for a TuiML component.

    Looks a component up in the component registry by its exact class name
    and reports what it does along with the parameters it accepts, their
    types and their defaults. Works for algorithms, preprocessing steps,
    feature selectors and built-in datasets alike.

    Examples
    --------
    Inspect an algorithm and its hyperparameters:

    $ tuiml describe --name RandomForestClassifier

    Inspect a preprocessing step:

    $ tuiml describe --name SimpleImputer

    Inspect a built-in dataset:

    $ tuiml describe --name iris
    """
    kwargs = {
            'name': name,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_describe', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
