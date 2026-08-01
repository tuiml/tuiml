"""Train Command - Build and train machine learning models via CLI."""

import click
import json
import tuiml
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

    Examples
    --------
    >>> from tuiml.cli.train import parse_extra_args
    >>> parse_extra_args(['--kfold', '10', '--strategy', 'mean', '--cv'])
    {'kfold': 10, 'strategy': 'mean', 'cv': True}
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

@click.command('train', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--algorithm', '-a', help='Exact algorithm class name, e.g. RandomForestClassifier. Run "tuiml list" to browse.')
@click.option('--data', '-d', help='Path to a data file (CSV/ARFF/JSON) or a built-in dataset name, e.g. iris.')
@click.option('--target', '-t', help='Name of the target column in the data file. Built-in datasets define their own.')
@click.option('--preprocessing', '-p', multiple=True, help='Preprocessing step to apply, by exact class name. Repeatable; applied in the order given.')
@click.option('--feature-selection', '-f', help='Feature selection method, by exact class name.')
@click.option('--cv', type=int, default=None, help='Number of cross-validation folds. Omit to use a single train/test split.')
@click.option('--test-size', type=float, default=0.2, help='Fraction of the data held out for testing (default: 0.2).')
@click.option('--metrics', '-m', multiple=True, help='Metric to compute, by function name. Repeatable; defaults to metrics chosen for the task.')
@click.option('--preset', help='Preprocessing preset: minimal, fast, standard, full, or imbalanced.')
@click.option('--params', '-P', help='Algorithm hyperparameters as a JSON object, e.g. \'{"n_estimators": 200}\'.')
@click.option('--random-seed', type=int, help='Random seed for reproducible splits and model fitting.')
@click.option('--output', '-o', help='Write the full result record to this file as JSON.')
@click.option('--save-path', help='Path to save the trained model file. Defaults to a managed location.')
@click.option('--stage', help="Run a single atomic stage instead of the full workflow: 'init', 'fit', 'partial_fit', or 'cross_validate'.")
@click.option('--model-id', help='ID of a previously initialized or trained model to continue working with.')
@click.option('--model-path', help='File path of a previously initialized or trained model (alternative to --model-id).')
@click.option('--json-output', is_flag=True, help='Print the raw JSON result instead of the formatted summary.')
@click.option('--verbose', '-v', is_flag=True, help='Echo the resolved configuration and re-raise full tracebacks on error.')
@click.pass_context
def train(ctx, algorithm, data, target, preprocessing, feature_selection, cv, test_size,
          metrics, preset, params, random_seed, output, save_path, stage, model_id, model_path, json_output, verbose):
    """Train a machine learning model, end to end or one stage at a time.

    Loads a dataset, applies optional preprocessing and feature selection, fits
    the chosen algorithm, and reports metrics from a holdout split or
    cross-validation. The fitted model is saved and its ID and path are printed
    so later commands can reuse it. Pass ``--stage`` to run a single step of the
    workflow instead of the whole thing; unrecognised flags are forwarded to
    that stage as extra keyword arguments.

    Examples
    --------
    Train on a built-in dataset:

    $ tuiml train -a RandomForestClassifier -d iris -t class

    Train on a CSV, save the model to a chosen path, and store the report:

    $ tuiml train -a SVC -d data.csv -t label --save-path model.pkl -o report.json

    Add preprocessing and feature selection, then cross-validate:

    $ tuiml train -a LogisticRegression -d data.csv -t label -p StandardScaler -f SelectKBest --cv 10

    Pass hyperparameters as JSON and fix the seed:

    $ tuiml train -a RandomForestClassifier -d iris -t class -P '{"n_estimators": 200}' --random-seed 42

    Run one atomic stage against an existing model:

    $ tuiml train --stage partial_fit --model-id abc123 -d new_batch.csv -t label
    """
    try:
        # Parse algorithm parameters
        algo_params = {}
        if params:
            try:
                algo_params = json.loads(params)
            except json.JSONDecodeError:
                raise click.ClickException(f"Invalid JSON in --params: {params}")

        # Build preprocessing list
        preproc_list = list(preprocessing) if preprocessing else None

        # Build metrics list
        metrics_list = list(metrics) if metrics else None

        extra_kwargs = parse_extra_args(ctx.args)

        if verbose:
            click.echo("Running training workflow...")
            if stage:
                click.echo(f"  Stage: {stage}")
            if algorithm:
                click.echo(f"  Algorithm: {algorithm}")
            if data:
                click.echo(f"  Data: {data}")
            if target:
                click.echo(f"  Target: {target}")
            if model_id:
                click.echo(f"  Model ID: {model_id}")
            if model_path:
                click.echo(f"  Model Path: {model_path}")
            if extra_kwargs:
                click.echo(f"  Stage arguments: {extra_kwargs}")

        # Construct kwargs for tuiml_train
        kwargs = {
            'algorithm': algorithm,
            'data': data,
            'target': target,
            'preprocessing': preproc_list,
            'feature_selection': feature_selection,
            'cv': cv,
            'test_size': test_size,
            'metrics': metrics_list,
            'preset': preset,
            'algorithm_params': algo_params,
            'random_seed': random_seed,
            'save_path': save_path,
            'stage': stage,
            'model_id': model_id,
            'model_path': model_path,
            'stage_kwargs': extra_kwargs if extra_kwargs else None
        }
        # Filter None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Train model using agent tools backend
        result = execute_tool('tuiml_train', **kwargs)

        if result.get('status') == 'error':
            raise click.ClickException(result.get('error'))

        if json_output:
            click.echo(json.dumps(result, indent=2, default=str))
            return

        # Display results
        click.echo("\n" + "="*50)
        click.echo("Training Results")
        click.echo("="*50)

        if stage:
            click.echo(f"\nStage '{stage}' completed successfully.")
            if result.get('model_id'):
                click.echo(f"Model ID: {result.get('model_id')}")
            if result.get('model_path'):
                click.echo(f"Model Path: {result.get('model_path')}")
            if result.get('model_class'):
                click.echo(f"Model Class: {result.get('model_class')}")
        else:
            if result.get('model_id'):
                click.echo(f"Model ID: {result.get('model_id')}")
            if result.get('model_path'):
                click.echo(f"Model Path: {result.get('model_path')}")
            if result.get('model_class'):
                click.echo(f"Model Class: {result.get('model_class')}")

        metrics_data = result.get('metrics')
        if metrics_data:
            click.echo("\nMetrics:")
            for metric_name, value in metrics_data.items():
                if isinstance(value, float):
                    click.echo(f"  {metric_name}: {value:.4f}")
                else:
                    click.echo(f"  {metric_name}: {value}")

        cv_results = result.get('cv_results')
        if cv_results:
            click.echo("\nCross-Validation Results:")
            scores = cv_results.get('scores', {})
            for metric, val_list in scores.items():
                import numpy as np
                mean_val = np.mean(val_list) if val_list else 0.0
                std_val = np.std(val_list) if val_list else 0.0
                click.echo(f"  {metric}: {mean_val:.4f} (+/- {std_val:.4f})")

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
