"""List Command - Browse available algorithms in the registry via CLI."""

import click
import tuiml

DESC_MAX_LEN = 60

@click.command('list')
@click.option('--type', '-t', type=click.Choice(['classifier', 'regressor', 'clusterer', 'all']),
              default='all', help='Filter by algorithm type (default: all)')
@click.option('--search', '-s', help='Search query for filtering')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'names']),
              default='table', help='Output format (default: table)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def list_algorithms(type, search, format, verbose):
    """Browse and search for available algorithms in the registry.

    This command lists all registered algorithms, allowing you to filter by 
    type or search by keywords. It supports multiple output formats for 
    both human readability and scripting purposes.

    Parameters
    ----------
    type : {"classifier", "regressor", "clusterer", "all"}, default="all"
        Filter the list to a specific category of algorithms.
    search : str, optional
        A search query to filter algorithms by name or description.
    format : {"table", "json", "names"}, default="table"
        The desired output format for the results list.
    verbose : bool, default=False
        Whether to show detailed metadata for each algorithm.

    Examples
    --------
    List all registered algorithms in a table:

    >>> tuiml list

    Filter and show only classification algorithms:

    >>> tuiml list --type classifier

    Search for algorithms related to "forest":

    >>> tuiml list --search forest

    Output just the names for use in shell scripts:

    >>> tuiml list --format names
    """
    try:
        # Get algorithms
        if search:
            algorithms = tuiml.search_algorithms(search)
        elif type != 'all':
            algorithms = tuiml.list_algorithms(type=type)
        else:
            algorithms = tuiml.list_algorithms()

        if not algorithms:
            click.echo("No algorithms found.")
            return

        # Display based on format
        if format == 'names':
            # Just print names
            for algo in algorithms:
                click.echo(algo.get('name', 'Unknown'))

        elif format == 'json':
            # Print as JSON
            import json
            click.echo(json.dumps(algorithms, indent=2))

        else:  # table format
            click.echo("\n" + "="*80)
            click.echo("Available Algorithms")
            click.echo("="*80)
            click.echo()

            # Group by type if showing all
            if type == 'all':
                # Group algorithms by type
                grouped = {}
                for algo in algorithms:
                    algo_type = algo.get('type', 'Unknown')
                    if algo_type not in grouped:
                        grouped[algo_type] = []
                    grouped[algo_type].append(algo)

                for algo_type, algos in sorted(grouped.items()):
                    click.echo(f"\n{algo_type.upper()}S:")
                    click.echo("-" * 80)
                    for algo in sorted(algos, key=lambda x: x.get('name', '')):
                        name = algo.get('name', 'Unknown')
                        desc = algo.get('description', 'No description')
                        if verbose:
                            click.echo(f"\n  {name}")
                            click.echo(f"    {desc}")
                            if 'tags' in algo:
                                click.echo(f"    Tags: {', '.join(algo['tags'])}")
                        else:
                            # Truncate description
                            desc_short = desc[:DESC_MAX_LEN] + "..." if len(desc) > DESC_MAX_LEN else desc
                            click.echo(f"  {name:30s} - {desc_short}")
            else:
                click.echo(f"\n{type.upper()}S:")
                click.echo("-" * 80)
                for algo in sorted(algorithms, key=lambda x: x.get('name', '')):
                    name = algo.get('name', 'Unknown')
                    desc = algo.get('description', 'No description')
                    if verbose:
                        click.echo(f"\n  {name}")
                        click.echo(f"    {desc}")
                        if 'tags' in algo:
                            click.echo(f"    Tags: {', '.join(algo['tags'])}")
                    else:
                        desc_short = desc[:DESC_MAX_LEN] + "..." if len(desc) > DESC_MAX_LEN else desc
                        click.echo(f"  {name:30s} - {desc_short}")

            click.echo(f"\nTotal: {len(algorithms)} algorithm(s)")
            click.echo()

    except Exception as e:
        if verbose:
            raise
        raise click.ClickException(str(e))
