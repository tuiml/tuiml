"""Select Features Command - Rank and keep the most informative features."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('select-features')
@click.option('--data', type=str, required=True, help='Data file path or built-in dataset name')
@click.option('--target', type=str, required=True, help='Target column name')
@click.option('--method', type=str, required=True, help='Feature selection class name (e.g., SelectKBestSelector, CFSSelector)')
@click.option('--k', type=int, help='Number of top-scoring features to keep (SelectKBestSelector)')
@click.option('--threshold', type=float, help='Score cutoff for VarianceThresholdSelector or SelectThresholdSelector')
@click.option('--method-params', type=str, help='Additional method-specific parameters as a JSON object, e.g. \'{"percentile": 25}\'')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def select_features(data, target, method, k, threshold, method_params, json_output):
    """Select the most informative features in a dataset.

    Scores every feature against the target with the chosen method and
    reports the surviving feature names and their column indices. Filter
    methods (SelectKBestSelector, SelectPercentileSelector, SelectFprSelector,
    SelectThresholdSelector, VarianceThresholdSelector), correlation-based
    selection (CFSSelector) and wrapper search (WrapperSelector) are all
    supported. Use --k to fix how many features survive, --threshold for the
    score-cutoff methods, and --method-params for anything else.

    Examples
    --------
    Rank features by their correlation with the target:

    $ tuiml select-features --data iris --target class --method CFSSelector

    Keep the two highest-scoring features:

    $ tuiml select-features --data d.csv --target y --method SelectKBestSelector --k 2

    Search for a good subset with a wrapper method:

    $ tuiml select-features --data d.csv --target y --method WrapperSelector
    """
    kwargs = {
            'data': data,
            'target': target,
            'method': method,
            'k': k,
            'threshold': threshold,
            'method_params': json.loads(method_params) if method_params is not None else None,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_select_features', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
