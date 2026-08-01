"""Predict Command - Make predictions using trained models via CLI."""

import click
import json
import numpy as np
from tuiml.agent.tools import execute_tool

def parse_extra_args(args):
    """Parse unrecognised command-line arguments into a keyword dictionary.

    Leading dashes are stripped from each flag name. A flag followed by a
    non-flag token takes that token as its value, coerced to ``bool``, ``int``,
    or ``float`` when possible; a flag with no value becomes ``True``.

    Parameters
    ----------
    args : list of str
        Raw leftover tokens, typically ``click.Context.args``.

    Returns
    -------
    kwargs : dict
        Mapping of flag name to parsed value.
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
            key = arg
            kwargs[key] = True
            i += 1
            continue

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

@click.command('predict', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--model-path', help='Path to a saved model file (alternative to --model-id).')
@click.option('--data', '-d', help='Path to a data file or a built-in dataset name to predict on.')
@click.option('--model-id', help='Model ID printed by "tuiml train" (alternative to --model-path).')
@click.option('--steps', type=int, help='Number of steps to forecast ahead. Timeseries models only.')
@click.option('--output', '-o', help='Write the predictions to this file as CSV.')
@click.option('--stage', help="Prediction stage to run: 'predict', 'predict_proba', or 'forecast'. Defaults to 'predict'.")
@click.option('--json-output', is_flag=True, help='Print the raw JSON result instead of the formatted summary.')
@click.option('--verbose', '-v', is_flag=True, help='Echo the resolved configuration and re-raise full tracebacks on error.')
@click.pass_context
def predict(ctx, model_path, data, model_id, steps, output, stage, json_output, verbose):
    """Make predictions with a trained model.

    Loads a model by ID or file path, applies it to a dataset, and prints a
    summary of the predictions, optionally writing them to CSV. Use ``--stage``
    to ask for class probabilities instead of labels, or to forecast future
    steps with a timeseries model. Anomaly detectors additionally report how
    many instances were flagged.

    Examples
    --------
    Predict with a saved model and write the results to CSV:

    $ tuiml predict --model-path model.pkl -d new_data.csv -o predictions.csv

    Predict using the model ID printed by "tuiml train":

    $ tuiml predict --model-id abc123 -d new_data.csv

    Get class probabilities instead of labels:

    $ tuiml predict --model-path model.pkl -d new_data.csv --stage predict_proba

    Forecast the next 12 steps of a timeseries model:

    $ tuiml predict --model-path arima.pkl --stage forecast --steps 12
    """
    try:
        extra_kwargs = parse_extra_args(ctx.args)

        if verbose:
            click.echo("Running prediction workflow...")
            if stage:
                click.echo(f"  Stage: {stage}")
            if model_id:
                click.echo(f"  Model ID: {model_id}")
            if model_path:
                click.echo(f"  Model Path: {model_path}")
            if data:
                click.echo(f"  Data: {data}")
            if steps:
                click.echo(f"  Steps: {steps}")
            if extra_kwargs:
                click.echo(f"  Stage arguments: {extra_kwargs}")

        kwargs = {
            'model_id': model_id,
            'model_path': model_path,
            'data': data,
            'steps': steps,
            'output_path': output,
            'stage': stage,
            'stage_kwargs': extra_kwargs if extra_kwargs else None
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        result = execute_tool('tuiml_predict', **kwargs)

        if result.get('status') == 'error':
            raise click.ClickException(result.get('error'))

        if json_output:
            click.echo(json.dumps(result, indent=2, default=str))
            return

        click.echo("\n" + "="*50)
        click.echo("Prediction Results")
        click.echo("="*50)

        if stage:
            click.echo(f"\nStage '{stage}' completed successfully.")

        if result.get('num_predictions') is not None:
            click.echo(f"\nNumber of predictions: {result.get('num_predictions')}")
        
        preview = result.get('predictions_preview')
        if preview is not None:
            click.echo(f"Predictions Preview: {preview}")

        if result.get('n_anomalies') is not None:
            click.echo(f"Normal instances: {result.get('n_normal')}")
            click.echo(f"Anomalies detected: {result.get('n_anomalies')}")
            click.echo(f"Anomaly ratio: {result.get('anomaly_ratio'):.4f}")

        if result.get('output_path'):
            click.echo(f"\nPredictions saved to: {result.get('output_path')}")

        click.echo("\n✓ Complete!")

    except Exception as e:
        if verbose:
            raise
        raise click.ClickException(str(e))
