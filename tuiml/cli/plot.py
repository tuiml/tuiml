"""Plot Command - Visualize model behaviour and experiment comparisons."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('plot')
@click.option('--plot-type', type=str, required=True, help="Plot to generate. Scored against a dataset: 'confusion_matrix', 'roc_curve', 'pr_curve', 'learning_curve'. From the model alone: 'tree', 'feature_importance'. From experiment results: 'cd_diagram', 'boxplot_comparison', 'heatmap', 'ranking_table'")
@click.option('--model-id', type=str, help='Model ID returned by tuiml train (required for most plot types)')
@click.option('--model-path', type=str, help='Path to a saved model file (alternative to --model-id)')
@click.option('--data', type=str, help='Data file path or built-in dataset name (required for confusion_matrix, roc_curve, pr_curve, learning_curve)')
@click.option('--target', type=str, help='Target column name (required for confusion_matrix, roc_curve, pr_curve, learning_curve)')
@click.option('--algorithm', type=str, help='Algorithm class name (required for learning_curve)')
@click.option('--title', type=str, help='Custom plot title; a sensible one is chosen if omitted')
@click.option('--normalize/--no-normalize', default=False, help='Show the confusion matrix as percentages rather than counts (confusion_matrix only)')
@click.option('--benchmark-results', type=str, help='Cross-validation scores for the comparison plots, as a JSON object mapping each algorithm name to its list of fold scores')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def plot(plot_type, model_id, model_path, data, target, algorithm, title, normalize, benchmark_results, json_output):
    """Generate a plot for model or experiment analysis.

    Renders the requested chart and returns it as an inline image. What
    each plot needs differs: confusion_matrix, roc_curve and pr_curve want
    a trained model plus a labelled dataset; tree and feature_importance
    read the model alone; learning_curve retrains an --algorithm over
    growing subsets; and the comparison plots (cd_diagram,
    boxplot_comparison, heatmap, ranking_table) work purely from the
    cross-validation scores passed to --benchmark-results.

    Examples
    --------
    Chart which features a trained model relies on:

    $ tuiml plot --plot-type feature_importance --model-id m1

    Score a model on held-out data and plot its confusion matrix:

    $ tuiml plot --plot-type confusion_matrix --model-id m1 --data d.csv --target y

    Compare algorithms from cross-validation scores:

    $ tuiml plot --plot-type cd_diagram --benchmark-results "$SCORES"
    """
    kwargs = {
            'plot_type': plot_type,
            'model_id': model_id,
            'model_path': model_path,
            'data': data,
            'target': target,
            'algorithm': algorithm,
            'title': title,
            'normalize': normalize,
            'benchmark_results': json.loads(benchmark_results) if benchmark_results is not None else None,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_plot', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
