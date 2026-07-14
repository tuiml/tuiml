"""Tune Command - Auto-generated from MCP schema."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('tune')
@click.option('--algorithm', type=str, required=True, help="Algorithm class name (e.g., 'RandomForestClassifier', 'SVM')")
@click.option('--data', type=str, required=True, help='Data file path or built-in dataset name')
@click.option('--target', type=str, required=True, help='Target column name')
@click.option('--method', type=str, required=True, help="Tuning method: 'grid' (exhaustive), 'random' (sampled), 'bayesian' (GP-based)")
@click.option('--param-grid', type=str, required=True, help="Parameter search space. For grid: {'param': [val1, val2]}. For random/bayesian: {'param': [low, high, 'int']} or {'param': [val1, val2]}. (pass as JSON string)")
@click.option('--cv', type=int, help='Number of cross-validation folds')
@click.option('--scoring', type=str, help="Scoring metric (e.g., 'accuracy', 'r2', 'neg_mse')")
@click.option('--n-iter', type=int, help='Number of iterations for random search')
@click.option('--n-iterations', type=int, help='Number of iterations for Bayesian search')
@click.option('--random-seed', type=int, help='Random seed for reproducibility')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def tune(algorithm, data, target, method, param_grid, cv, scoring, n_iter, n_iterations, random_seed, json_output):
    """Hyperparameter optimization for any algorithm. Supports grid search, random search, and Bayesian optimization. Returns best parameters, best score, and a trained model with optimal settings."""
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
