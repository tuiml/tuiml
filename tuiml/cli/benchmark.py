"""Benchmark Command - Compare models across datasets via CLI."""

import click
import tuiml

@click.command()
@click.option('--models', '-a', 'models', multiple=True, required=True,
              help='Model to include, by exact class name, e.g. RandomForestClassifier. Repeatable.')
@click.option('--data', '-d', multiple=True, required=True,
              help='Dataset file path or built-in dataset name. Repeatable; every model runs on every dataset.')
@click.option('--target', '-t', help='Target column name. Applies to dataset files only; built-in datasets define their own.')
@click.option('--cv', type=int, default=10, help='Number of cross-validation folds (default: 10). Ignored when --test-size is given.')
@click.option('--test-size', type=float, help='Use a holdout split of this fraction instead of cross-validation.')
@click.option('--repeats', type=int, default=1, help='Repeat the whole split scheme this many times and pool the results (default: 1).')
@click.option('--metrics', '-m', multiple=True,
              help='Metric to report, by function name. Repeatable; defaults to metrics chosen for the task.')
@click.option('--random-seed', type=int, help='Random seed, so the splits and therefore the comparison are reproducible.')
@click.option('--output', '-o', help='Write the results table to this file, in the format given by --format.')
@click.option('--format', '-f', type=click.Choice(['markdown', 'latex', 'csv', 'html']),
              default='markdown', help='Format of the file written by --output (default: markdown).')
@click.option('--baseline', help='Model to use as the reference for pairwise significance tests.')
@click.option('--plot', is_flag=True, help='Display a critical difference diagram of the model rankings.')
@click.option('--verbose', '-v', is_flag=True, help='Report progress while running and re-raise full tracebacks on error.')
def benchmark(models, data, target, cv, test_size, repeats, metrics,
              random_seed, output, format, baseline, plot, verbose):
    """Benchmark models across datasets with cross-validation.

    Runs every model on every dataset, reports mean and spread per metric,
    marks winners, and optionally tests significance against a baseline.

    Examples
    --------
    Benchmark three models with 10-fold cross-validation:

    $ tuiml benchmark -a RandomForestClassifier -a SVC -a NaiveBayesClassifier -d iris.csv -t class --cv 10

    Compare across multiple datasets against a baseline:

    $ tuiml benchmark -a SVC -a RandomForestClassifier -d iris -d glass --baseline SVC

    Use a repeated holdout split instead of cross-validation:

    $ tuiml benchmark -a SVC -a RandomForestClassifier -d iris --test-size 0.3 --repeats 5

    Export a LaTeX table and show the critical difference diagram:

    $ tuiml benchmark -a SVC -a RandomForestClassifier -d iris -o results.tex -f latex --plot
    """
    import os

    try:
        dataset_specs = []
        for source in data:
            if os.path.exists(source) and target:
                dataset_specs.append({"source": source, "target": target})
            else:
                dataset_specs.append({"source": source})

        evaluation = {}
        if test_size:
            evaluation["test_size"] = test_size
        else:
            evaluation["cv"] = cv
        if repeats and repeats > 1:
            evaluation["repeats"] = repeats
        if metrics:
            evaluation["metrics"] = list(metrics)

        result = tuiml.Benchmark(
            models=[{"name": name} for name in models],
            datasets=dataset_specs,
            evaluation=evaluation,
            random_seed=random_seed,
        ).run(verbose=1 if verbose else 0)

        click.echo("\n" + "=" * 60)
        click.echo("Benchmark Results")
        click.echo("=" * 60)
        click.echo()
        click.echo(result.summary())

        if baseline:
            click.echo("\nSignificance vs baseline:")
            click.echo("-" * 60)
            click.echo(result.compare(baseline=baseline).to_string(index=False))

        if output:
            exporters = {
                'markdown': result.to_markdown,
                'latex': result.to_latex,
                'html': result.to_html,
            }
            if format == 'csv':
                result.to_csv(output)
            else:
                with open(output, 'w') as f:
                    f.write(exporters[format]())
            click.echo(f"\nResults saved to: {output}")

        if plot:
            try:
                click.echo("\nGenerating critical difference plot...")
                result.plot_critical_difference()
                click.echo("Plot displayed")
            except Exception as e:
                click.echo(f"Warning: Could not generate plot: {e}")

        click.echo("\nBenchmark complete!")
    except Exception as e:
        if verbose:
            raise
        raise click.ClickException(str(e))
