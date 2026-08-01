"""Preprocess Command - Run preprocessing pipelines and atomic stages."""

import click
import json
from tuiml.agent.tools import execute_tool

def parse_extra_args(args):
    """Parse leftover command-line arguments into a keyword dictionary.

    Each flag is stripped of its leading dashes and paired with the token
    that follows it, coerced to a bool, int or float where possible. A flag
    with no value of its own becomes ``True``.

    Parameters
    ----------
    args : list of str
        Unparsed command-line tokens, e.g.
        ``['--kfold', '10', '--strategy', 'mean', '--cv']``.

    Returns
    -------
    kwargs : dict
        Parsed keyword arguments, e.g.
        ``{'kfold': 10, 'strategy': 'mean', 'cv': True}``.
    """
    kwargs = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            key = arg[2:]
        elif arg.startswith('-'):
            key = arg[1:]
        else:
            # Treating positional argument/malformed as flag key
            key = arg
            kwargs[key] = True
            i += 1
            continue

        # Check if next arg is the value
        if i + 1 < len(args) and not args[i+1].startswith('-'):
            val = args[i+1]
            if val.lower() == 'true':
                val = True
            elif val.lower() == 'false':
                val = False
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            kwargs[key] = val
            i += 2
        else:
            kwargs[key] = True
            i += 1
    return kwargs

@click.command('preprocess', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--data', type=str, required=True, help='Data file path or built-in dataset name')
@click.option('--target', type=str, help='Target column name (excluded from preprocessing, re-appended to output)')
@click.option('--steps', type=str, help='Preprocessing steps as a JSON array of class names, or of objects carrying their own params, e.g. \'["SimpleImputer", "StandardScaler"]\'')
@click.option('--stage', type=str, help="Run a single atomic stage instead of a pipeline: 'split', 'impute', 'balance', 'scale', 'encode' or 'discretize'")
@click.option('--output', type=str, help='Path to write the preprocessed file(s) to')
@click.option('--save-as', type=str, help='Alias for --output; ignored when --output is given')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
@click.pass_context
def preprocess(ctx, data, target, steps, stage, output, save_as, json_output):
    """Apply preprocessing steps or a single atomic stage to a dataset.

    Runs either a pipeline of named preprocessing steps or one atomic
    stage: split, impute, balance, scale, encode or discretize. The target
    column is held out while the features are transformed and re-appended
    to the result. Any option this command does not recognise is forwarded
    to the stage as a keyword argument, so stage-specific settings such as
    --kfold 10 can be given inline.

    Examples
    --------
    Run a preprocessing pipeline over a built-in dataset:

    $ tuiml preprocess --data iris --steps '["StandardScaler"]'

    Impute missing values and write the result to a file:

    $ tuiml preprocess --data data.csv --stage impute --output clean.csv

    Forward a stage-specific option to the split stage:

    $ tuiml preprocess --data data.csv --stage split --kfold 10
    """
    extra_kwargs = parse_extra_args(ctx.args)
    
    kwargs = {
            'data': data,
            'target': target,
            'steps': json.loads(steps) if steps else None,
            'stage': stage,
            'output': output or save_as,
            'stage_kwargs': extra_kwargs if extra_kwargs else None,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_preprocess', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            click.echo(json.dumps(result, indent=2, default=str))
