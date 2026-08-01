"""Evaluate Command - Evaluate trained models on test data via CLI."""

import click
import json
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

@click.command('evaluate', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--model-path', help='Path to a saved model file (alternative to --model-id).')
@click.option('--data', '-d', help='Path to a test data file or a built-in dataset name.')
@click.option('--target', '-t', help='Name of the target column holding the true labels or values.')
@click.option('--model-id', help='Model ID printed by "tuiml train" (alternative to --model-path).')
@click.option('--metrics', '-m', multiple=True, help='Metric to compute, by function name. Repeatable; defaults to metrics chosen for the task.')
@click.option('--output', '-o', help='Write the full result record to this file as JSON.')
@click.option('--stage', help="Evaluation stage to run: 'metrics' for scores, or 'report' for a per-class text report.")
@click.option('--json-output', is_flag=True, help='Print the raw JSON result instead of the formatted summary.')
@click.option('--verbose', '-v', is_flag=True, help='Echo the resolved configuration and re-raise full tracebacks on error.')
@click.pass_context
def evaluate(ctx, model_path, data, target, model_id, metrics, output, stage, json_output, verbose):
    """Score a trained model against labelled test data.

    Loads a model by ID or file path, runs it over the given dataset, and
    prints the resulting metrics. With ``--stage report`` it prints a
    per-class classification report instead of a flat table of scores. Metrics
    are chosen automatically from the model's task unless ``--metrics`` is
    given.

    Examples
    --------
    Evaluate a saved model on a held-out CSV:

    $ tuiml evaluate --model-path model.pkl -d test.csv -t label

    Ask for specific metrics and save the scores:

    $ tuiml evaluate --model-id abc123 -d test.csv -t label -m accuracy -m f1_score -o scores.json

    Print a per-class classification report:

    $ tuiml evaluate --model-path model.pkl -d test.csv -t label --stage report
    """
    try:
        extra_kwargs = parse_extra_args(ctx.args)

        if verbose:
            click.echo("Running evaluation workflow...")
            if stage:
                click.echo(f"  Stage: {stage}")
            if model_id:
                click.echo(f"  Model ID: {model_id}")
            if model_path:
                click.echo(f"  Model Path: {model_path}")
            if data:
                click.echo(f"  Data: {data}")
            if target:
                click.echo(f"  Target: {target}")
            if metrics:
                click.echo(f"  Metrics: {metrics}")
            if extra_kwargs:
                click.echo(f"  Stage arguments: {extra_kwargs}")

        metrics_list = list(metrics) if metrics else None

        kwargs = {
            'model_id': model_id,
            'model_path': model_path,
            'data': data,
            'target': target,
            'metrics': metrics_list,
            'stage': stage,
            'stage_kwargs': extra_kwargs if extra_kwargs else None
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        result = execute_tool('tuiml_evaluate', **kwargs)

        if result.get('status') == 'error':
            raise click.ClickException(result.get('error'))

        if json_output:
            click.echo(json.dumps(result, indent=2, default=str))
            return

        # Display results
        click.echo("\n" + "="*50)
        click.echo("Evaluation Results")
        click.echo("="*50)

        # Print report if report is available
        report = result.get('report')
        if report:
            click.echo(f"\n{report}")
        else:
            metrics_data = result.get('metrics')
            if metrics_data:
                click.echo()
                for metric_name, value in metrics_data.items():
                    if isinstance(value, float):
                        click.echo(f"  {metric_name:25s}: {value:.4f}")
                    else:
                        click.echo(f"  {metric_name:25s}: {value}")

        # Save results to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"\nResults saved to: {output}")

        click.echo("\n✓ Complete!")

    except Exception as e:
        if verbose:
            raise
        raise click.ClickException(str(e))
