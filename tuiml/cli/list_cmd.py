"""List Command - Browse the TuiML component registry via CLI."""

import click
import tuiml

DESC_MAX_LEN = 60

@click.command('list')
@click.option('--category', '-c', type=click.Choice(['algorithm', 'preprocessing', 'dataset', 'feature', 'splitting', 'custom', 'all']),
              default='all', help='Only list components in this category (default: all).')
@click.option('--type', '-t', type=click.Choice(['classifier', 'regressor', 'clusterer', 'anomaly', 'timeseries', 'all']),
              default='all', help='Only list algorithms of this task type (default: all).')
@click.option('--search', '-s', help='Keep only components whose name or description matches this query.')
@click.option('--include-runs', is_flag=True, help='Include the recorded run history for custom algorithms.')
@click.option('--limit', type=int, default=50, help='Maximum number of results to return (default: 50).')
@click.option('--offset', type=int, default=0, help='Number of results to skip, for paging through a long list.')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'names']),
              default='table', help='Output format: grouped table, raw JSON, or bare names (default: table).')
@click.option('--verbose', '-v', is_flag=True, help='Show the full description, tags, and type for each component.')
def list_algorithms(category, type, search, include_runs, limit, offset, format, verbose):
    """Browse and search the component registry.

    Lists everything registered with TuiML, grouped by category: algorithms,
    preprocessing steps, datasets, feature selectors, splitting strategies, and
    any custom components you have added. The names printed here are the exact
    class names the other commands expect, so this is the place to look up what
    to pass to ``-a``, ``-p``, or ``-d``.

    Examples
    --------
    List everything the registry knows about:

    $ tuiml list

    Show only classifiers, with full descriptions and tags:

    $ tuiml list -c algorithm -t classifier -v

    Search for a component by keyword:

    $ tuiml list -s forest

    Emit bare names, ready to pipe into another command:

    $ tuiml list -c preprocessing -f names

    Page through a long listing:

    $ tuiml list --limit 20 --offset 40
    """
    try:
        from tuiml.agent.tools import execute_tool
        
        kwargs = {
            'category': category,
            'limit': limit,
            'offset': offset,
            'include_runs': include_runs
        }
        if search:
            kwargs['search'] = search
        if type != 'all':
            kwargs['type'] = type
            
        result = execute_tool('tuiml_list', **kwargs)
        
        if result.get('status') == 'error':
            raise click.ClickException(result.get('error', 'Unknown error'))
            
        components = result.get('components', result.get('algorithms', []))
        
        if not components:
            click.echo("No components found.")
            return

        # Display based on format
        if format == 'names':
            for comp in components:
                click.echo(comp.get('name', 'Unknown'))

        elif format == 'json':
            import json
            click.echo(json.dumps(result, indent=2))

        else:  # table format
            click.echo("\n" + "="*80)
            click.echo(f"Available Components (Total: {result.get('total', len(components))})")
            click.echo("="*80)
            click.echo()

            # Group by category
            grouped = {}
            for comp in components:
                comp_cat = comp.get('category', 'unknown')
                if comp_cat not in grouped:
                    grouped[comp_cat] = []
                grouped[comp_cat].append(comp)

            for comp_cat, comps in sorted(grouped.items()):
                click.echo(f"\n{comp_cat.upper()}S:")
                click.echo("-" * 80)
                for comp in sorted(comps, key=lambda x: x.get('name', '')):
                    name = comp.get('name', 'Unknown')
                    desc = comp.get('description', 'No description')
                    
                    if verbose:
                        click.echo(f"\n  {name}")
                        click.echo(f"    {desc}")
                        if 'tags' in comp:
                            click.echo(f"    Tags: {', '.join(comp['tags'])}")
                        if 'type' in comp and comp['type'] != comp_cat:
                            click.echo(f"    Type: {comp['type']}")
                    else:
                        desc_short = desc[:DESC_MAX_LEN] + "..." if len(desc) > DESC_MAX_LEN else desc
                        click.echo(f"  {name:30s} - {desc_short}")

            click.echo(f"\nShowing {len(components)} of {result.get('total', len(components))} items")
            if result.get('has_more'):
                click.echo(f"Use --offset {offset + limit} to see more")
            click.echo()

    except Exception as e:
        if verbose:
            raise
        raise click.ClickException(str(e))
