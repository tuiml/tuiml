"""Experiment Command - Run comparative experiments via CLI."""

import click
import json
import tuiml

@click.command()
@click.option('--algorithms', '-a', multiple=True, required=True,
              help='Algorithm names (exact class names, e.g., RandomForestClassifier, SVM)')
@click.option('--data', '-d', multiple=True, required=True, help='Path to dataset file(s) or built-in dataset name(s)')
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--cv', type=int, default=10, help='Number of cross-validation folds (default: 10)')
@click.option('--metrics', '-m', multiple=True, help='Metrics to compute (default: accuracy)')
@click.option('--random-seed', type=int, help='Random seed for reproducibility')
@click.option('--output', '-o', help='Output file for results (JSON/Markdown/LaTeX)')
@click.option('--format', '-f', type=click.Choice(['json', 'markdown', 'latex', 'csv']),
              default='markdown', help='Output format (default: markdown)')
@click.option('--plot', is_flag=True, help='Generate critical difference plot')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def experiment(algorithms, data, target, cv, metrics, random_seed, output, format, plot, verbose):
    """Run cross-validation experiments to compare multiple algorithms.

    This command benchmarks multiple algorithms on one or more datasets using 
    cross-validation and generates summary statistics, comparison tables, 
    and optional visualization plots.

    Parameters
    ----------
    algorithms : list of str
        The names of the algorithms to compare (e.g., ``"RandomForestClassifier"``).
    data : list of str
        Path to the dataset file(s) (CSV, ARFF, etc.) or built-in dataset names.
    target : str
        Name of the target column in the dataset(s).
    cv : int, default=10
        Number of cross-validation folds.
    metrics : list of str, optional
        A list of metrics to compute for comparison.
    random_seed : int, optional
        Random seed for reproducibility.
    output : str, optional
        Path where the experiment results should be saved.
    format : {"json", "markdown", "latex", "csv"}, default="markdown"
        The desired format for the output results file.
    plot : bool, default=False
        Whether to generate a critical difference plot for statistical
        comparison.
    verbose : bool, default=False
        Whether to enable verbose output.

    Examples
    --------
    Benchmark three algorithms using 10-fold cross-validation:

    >>> tuiml experiment -a RandomForestClassifier -a SVM -a NaiveBayesClassifier -d iris.csv -t class --cv 10

    Compare algorithms across multiple datasets:

    >>> tuiml experiment -a SVM -a RandomForestClassifier -d iris -d wine -t target
    """
    try:
        from tuiml.datasets import load, load_dataset
        import os
        
        datasets_dict = {}
        for d in data:
            if verbose:
                click.echo(f"Loading data from: {d}")
            try:
                if os.path.exists(d):
                    dataset = load(d)
                else:
                    dataset = load_dataset(d)
                X = dataset.X
                y = dataset.get_target(target)
                datasets_dict[d] = (X, y)
            except Exception as ex:
                raise click.ClickException(f"Failed to load dataset '{d}': {ex}")

        # Build algorithm list
        algo_list = list(algorithms)

        # Build metrics list
        metrics_list = list(metrics) if metrics else None

        if verbose:
            click.echo(f"\nRunning experiment:")
            click.echo(f"  Algorithms: {', '.join(algo_list)}")
            click.echo(f"  Datasets: {', '.join(data)}")
            click.echo(f"  Cross-validation: {cv} folds")
            click.echo(f"  Metrics: {metrics_list or 'auto'}")
            if random_seed is not None:
                click.echo(f"  Random seed: {random_seed}")

        # Run experiment
        kwargs = dict(
            algorithms=algo_list,
            datasets=datasets_dict,
            cv=cv,
            metrics=metrics_list,
            verbose=1 if verbose else 0
        )
        if random_seed is not None:
            kwargs['random_seed'] = random_seed
            
        exp = tuiml.experiment(**kwargs)

        # Display results
        click.echo("\n" + "="*60)
        click.echo("Experiment Results")
        click.echo("="*60)
        click.echo()
        click.echo(exp.summary())

        # Save results to file
        if output:
            if format == 'json':
                # Save as JSON with structured metric values
                import json
                results_dict = {
                    'algorithms': algo_list,
                    'dataset': data,
                    'cv_folds': cv,
                    'results': {}
                }
                for ds_name, ds_result in exp.results.dataset_results.items():
                    results_dict['results'][ds_name] = {}
                    for model_name, model_result in ds_result.model_results.items():
                        stats = {}
                        for metric_name in exp._metric_funcs:
                            metric_stats = model_result.get_metric_stats(metric_name)
                            stats[metric_name] = {
                                'mean': metric_stats['mean'],
                                'std': metric_stats['std'],
                                'min': metric_stats['min'],
                                'max': metric_stats['max'],
                            }
                        results_dict['results'][ds_name][model_name] = stats
                with open(output, 'w') as f:
                    json.dump(results_dict, f, indent=2)

            elif format == 'markdown':
                with open(output, 'w') as f:
                    f.write(exp.to_markdown())

            elif format == 'latex':
                with open(output, 'w') as f:
                    f.write(exp.to_latex())

            elif format == 'csv':
                with open(output, 'w') as f:
                    f.write(exp.to_csv())

            click.echo(f"\nResults saved to: {output}")

        # Generate plot if requested
        if plot:
            try:
                click.echo("\nGenerating critical difference plot...")
                exp.plot_critical_difference()
                click.echo("✓ Plot displayed")
            except Exception as e:
                click.echo(f"Warning: Could not generate plot: {e}")

        click.echo("\n✓ Experiment complete!")

        # Show statistical comparison
        try:
            comparisons = exp.compare_models()
            if comparisons:
                click.echo("\nStatistical Comparisons:")
                click.echo("-" * 60)
                for key, comp in list(comparisons.items())[:5]:  # Show first 5
                    click.echo(f"  {comp['model']} vs {comp['baseline']}: "
                             f"p-value={comp['p_value']:.4f} "
                             f"{'✓ significant' if comp['significant'] else '✗ not significant'}")
        except Exception as e:
            click.echo(f"Warning: Statistical comparison failed: {e}", err=True)

    except Exception as e:
        if verbose:
            raise
        raise click.ClickException(str(e))
