"""Generate Command - Create synthetic datasets for testing and demos."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('generate')
@click.option('--generator', type=str, required=True, help='Generator class name (e.g., RandomRBF, Friedman, Blobs)')
@click.option('--n-samples', type=int, help='Number of samples to generate')
@click.option('--n-features', type=int, help='Number of features (not all generators support this)')
@click.option('--n-classes', type=int, help='Number of classes (classification generators only)')
@click.option('--n-clusters', type=int, help='Number of clusters (clustering generators only)')
@click.option('--noise', type=float, help='Noise level (regression generators only)')
@click.option('--random-seed', type=int, help='Random seed for reproducibility')
@click.option('--generator-params', type=str, help='Additional generator-specific parameters as a JSON object, e.g. \'{"n_centroids": 20}\'')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def generate(generator, n_samples, n_features, n_classes, n_clusters, noise, random_seed, generator_params, json_output):
    """Generate a synthetic dataset for testing and demos.

    Writes a freshly generated dataset to a CSV file and reports its path,
    shape and a short preview, ready to feed straight into tuiml train.
    Classification generators (RandomRBF, Agrawal, LED, Hyperplane),
    regression generators (Friedman, MexicanHat, Sine) and clustering
    generators (Blobs, Moons, Circles, SwissRoll) are all supported. Pass
    --random-seed to make the output reproducible, and --generator-params
    for anything the dedicated options do not cover.

    Examples
    --------
    Generate a clustering dataset with three clusters:

    $ tuiml generate --generator Blobs --n-samples 500 --n-clusters 3

    Generate a reproducible classification dataset:

    $ tuiml generate --generator RandomRBF --n-classes 4 --random-seed 42

    Generate a noisy regression dataset:

    $ tuiml generate --generator Friedman --n-samples 200 --noise 0.1
    """
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
