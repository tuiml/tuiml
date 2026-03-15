"""Train Command - Build and train machine learning models via CLI."""

import click
import json
import tuiml

@click.command()
@click.argument('algorithm')
@click.argument('data', type=click.Path(exists=True))
@click.argument('target')
@click.option('--preprocessing', '-p', multiple=True, help='Preprocessing steps (exact class names)')
@click.option('--feature-selection', '-f', help='Feature selection method (exact class name)')
@click.option('--cv', type=int, default=None, help='Number of cross-validation folds')
@click.option('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
@click.option('--metrics', '-m', multiple=True, help='Metrics to compute')
@click.option('--preset', help='Preprocessing preset (minimal, fast, standard, full, imbalanced)')
@click.option('--params', '-P', help='Algorithm parameters as JSON dict')
@click.option('--output', '-o', help='Output file for results (JSON)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def train(algorithm, data, target, preprocessing, feature_selection, cv, test_size,
          metrics, preset, params, output, verbose):
    """Train a machine learning model with a complete workflow.

    This command enables you to build and train models directly from the terminal,
    supporting preprocessing, feature selection, and multiple evaluation strategies.

    Parameters
    ----------
    algorithm : str
        Algorithm class name (e.g., ``"RandomForestClassifier"``, ``"SVM"``).
    data : str
        Path to the training data file (CSV, ARFF, etc.).
    target : str
        Name of the target column in the dataset.
    preprocessing : list of str, optional
        One or more preprocessing steps to apply (exact class names).
    feature_selection : str, optional
        Feature selection method to use (exact class name).
    cv : int, optional
        Number of cross-validation folds. If not set, holdout is used.
    test_size : float, default=0.2
        Proportion of data to use for testing in holdout evaluation.
    metrics : list of str, optional
        Metrics to compute during evaluation.
    preset : str, optional
        Preprocessing preset name (e.g., ``"standard"``, ``"full"``).
    params : str, optional
        Model hyperparameters as a JSON-encoded dictionary string.
    output : str, optional
        Path to save the training results as a JSON file.
    verbose : bool, default=False
        Whether to enable verbose output for progress tracking.

    Examples
    --------
    Simple training with default parameters:

    >>> tuiml train RandomForestClassifier iris.csv class

    Training with cross-validation and preprocessing:

    >>> tuiml train SVM data.csv target --cv 10 -p SimpleImputer -p MinMaxScaler

    Training with specific model parameters as JSON:

    >>> tuiml train RandomForestClassifier iris.csv class --params '{"n_trees": 100, "max_depth": 10}'
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

        if verbose:
            click.echo(f"Training {algorithm} on {data}...")
            click.echo(f"  Target: {target}")
            if cv:
                click.echo(f"  Cross-validation: {cv} folds")
            else:
                click.echo(f"  Test size: {test_size}")
            if preproc_list:
                click.echo(f"  Preprocessing: {preproc_list}")
            if preset:
                click.echo(f"  Preset: {preset}")
            if algo_params:
                click.echo(f"  Parameters: {algo_params}")

        # Train model
        result = tuiml.train(
            algorithm=algorithm,
            data=data,
            target=target,
            preprocessing=preproc_list,
            feature_selection=feature_selection,
            cv=cv,
            test_size=test_size,
            metrics=metrics_list,
            preset=preset,
            verbose=verbose,
            **algo_params
        )

        # Display results
        click.echo("\n" + "="*50)
        click.echo("Training Results")
        click.echo("="*50)

        if result.metrics:
            click.echo("\nMetrics:")
            for metric_name, value in result.metrics.items():
                click.echo(f"  {metric_name}: {value:.4f}")

        if result.cv_results:
            click.echo("\nCross-Validation Results:")
            click.echo(f"  Mean: {result.cv_results.get('mean', 0):.4f}")
            click.echo(f"  Std: {result.cv_results.get('std', 0):.4f}")

        # Save results to file if requested
        if output:
            output_data = {
                'algorithm': algorithm,
                'data': data,
                'target': target,
                'metrics': result.metrics,
                'cv_results': result.cv_results,
                'metadata': result.metadata
            }
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            click.echo(f"\nResults saved to: {output}")

        click.echo("\n✓ Training complete!")

    except Exception as e:
        if verbose:
            raise
        raise click.ClickException(str(e))
