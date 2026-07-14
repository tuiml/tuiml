"""Generate Command - Auto-generated from MCP schema."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('generate')
@click.option('--generator', type=str, required=True, help='Generator class name')
@click.option('--n-samples', type=int, help='Number of samples to generate')
@click.option('--n-features', type=int, help='Number of features (not all generators support this)')
@click.option('--n-classes', type=int, help='Number of classes (classification generators only)')
@click.option('--n-clusters', type=int, help='Number of clusters (clustering generators only)')
@click.option('--noise', type=float, help='Noise level (regression generators only)')
@click.option('--random-seed', type=int, help='Random seed for reproducibility')
@click.option('--generator-params', type=str, help='Additional generator-specific parameters (pass as JSON string)')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def generate(generator, n_samples, n_features, n_classes, n_clusters, noise, random_seed, generator_params, json_output):
    """Generate synthetic datasets for testing and demos. Supports classification (RandomRBF, Agrawal, LED, Hyperplane), regression (Friedman, MexicanHat, Sine), and clustering (Blobs, Moons, Circles, SwissRoll) generators."""
    kwargs = {
            'generator': generator,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'n_clusters': n_clusters,
            'noise': noise,
            'random_seed': random_seed,
            'generator_params': json.loads(generator_params) if generator_params is not None else None,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_generate_data', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
