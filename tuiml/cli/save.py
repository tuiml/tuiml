"""Save Command - Copy a trained model to a path of your choosing."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('save')
@click.option('--model-id', type=str, required=True, help='Model ID returned by tuiml train')
@click.option('--destination', type=str, required=True, help="Destination file path (e.g., './my_model.joblib', '/home/user/models/rf.joblib')")
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def save(model_id, destination, json_output):
    """Copy a trained model to a path of your choosing.

    Takes a model produced by tuiml train and writes a copy to the given
    destination, so it can be archived, checked into version control or
    shipped to another machine. The original stays available under its
    model ID, so this is a copy rather than a move.

    Examples
    --------
    Save a model into the current directory:

    $ tuiml save --model-id m1 --destination ./my_model.joblib

    Save it into a models directory instead:

    $ tuiml save --model-id m1 --destination ~/models/rf.joblib
    """
    kwargs = {
            'model_id': model_id,
            'destination': destination,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_save_model', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
