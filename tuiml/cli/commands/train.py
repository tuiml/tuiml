"""Train Command - Build and train machine learning models via CLI."""

import click
import json
import tuiml
from tuiml.agent.tools import execute_tool

def parse_extra_args(args):
    """Parse extra command-line arguments into a dictionary.
    
    Examples:
        ['--kfold', '10', '--strategy', 'mean', '--cv'] -> {'kfold': 10, 'strategy': 'mean', 'cv': True}
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
@click.option('--algorithm', '-a', help='Algorithm class name (e.g., RandomForestClassifier)')
@click.option('--data', '-d', help='Path to data file or built-in dataset name (e.g., iris)')
@click.option('--target', '-t', help='Target column name')
@click.option('--preprocessing', '-p', multiple=True, help='Preprocessing steps (exact class names)')
@click.option('--feature-selection', '-f', help='Feature selection method (exact class name)')
@click.option('--cv', type=int, default=None, help='Number of cross-validation folds')
@click.option('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
@click.option('--metrics', '-m', multiple=True, help='Metrics to compute')
@click.option('--preset', help='Preprocessing preset (minimal, fast, standard, full, imbalanced)')
@click.option('--params', '-P', help='Algorithm parameters as JSON dict')
@click.option('--random-seed', type=int, help='Random seed for reproducibility')
@click.option('--output', '-o', help='Output file for results (JSON)')
@click.option('--save-path', help='Custom path to save the trained model file')
@click.option('--stage', help="Atomic training stage: 'init', 'fit', 'partial_fit', 'cross_validate'")
@click.option('--model-id', help='Unique identifier of a previously initialized/saved model')
@click.option('--model-path', help='File path of a previously initialized/saved model')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def train(ctx, algorithm, data, target, preprocessing, feature_selection, cv, test_size,
          metrics, preset, params, random_seed, output, save_path, stage, model_id, model_path, json_output, verbose):
    """Train a machine learning model with a complete workflow or an atomic stage.

    This command enables you to build and train models directly from the terminal,
    supporting preprocessing, feature selection, and multiple evaluation strategies.
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
