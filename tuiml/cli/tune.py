"""Tune Command - Search for the best hyperparameters of an algorithm."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('tune')
@click.option('--algorithm', type=str, required=True, help="Exact algorithm class name to tune, e.g. 'RandomForestClassifier'.")
@click.option('--data', type=str, required=True, help='Path to a data file or a built-in dataset name.')
@click.option('--target', type=str, required=True, help='Name of the target column in the data.')
@click.option('--method', type=str, required=True, help="Search strategy: 'grid' (exhaustive), 'random' (sampled), or 'bayesian' (Gaussian-process guided).")
@click.option('--param-grid', type=str, required=True, help="Search space as a JSON object. For grid: {\"param\": [val1, val2]}. For random/bayesian a range may be given as {\"param\": [low, high, \"int\"]}.")
@click.option('--cv', type=int, help='Number of cross-validation folds used to score each candidate.')
@click.option('--scoring', type=str, help="Metric to maximise, e.g. 'accuracy', 'r2', or 'neg_mse'. Defaults to a metric chosen for the task.")
@click.option('--n-iter', type=int, help='Number of candidates to sample. Random search only.')
@click.option('--n-iterations', type=int, help='Number of optimisation rounds. Bayesian search only.')
@click.option('--random-seed', type=int, help='Random seed, so the sampled candidates and CV splits are reproducible.')
@click.option('--json-output', is_flag=True, help='Print the raw JSON result.')
def tune(algorithm, data, target, method, param_grid, cv, scoring, n_iter, n_iterations, random_seed, json_output):
    """Search for the hyperparameters that score best.

    Scores candidate configurations by cross-validation and reports the best
    parameters, the best score, and a model refitted with those settings.
    Grid search tries every combination, random search samples a fixed number
    of candidates, and Bayesian search uses a Gaussian process to decide where
    to look next.

    Examples
    --------
    Exhaustive grid search over two hyperparameters:

    $ tuiml tune --algorithm RandomForestClassifier --data iris --target class --method grid --param-grid '{"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10]}'

    Random search with a bounded integer range and a fixed budget:

    $ tuiml tune --algorithm RandomForestClassifier --data data.csv --target label --method random --param-grid '{"n_estimators": [10, 500, "int"]}' --n-iter 25 --random-seed 42

    Bayesian search optimising R-squared with 5-fold CV:

    $ tuiml tune --algorithm SVR --data housing.csv --target price --method bayesian --param-grid '{"C": [0.1, 100.0]}' --scoring r2 --cv 5
    """
    kwargs = {
            'algorithm': algorithm,
            'data': data,
            'target': target,
            'method': method,
            'param_grid': json.loads(param_grid) if param_grid else None,
            'cv': cv,
            'scoring': scoring,
            'n_iter': n_iter,
            'n_iterations': n_iterations,
            'random_seed': random_seed,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_tune', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
