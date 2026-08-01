"""Search Source Command - Grep algorithm source files for a pattern."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('search-source')
@click.option('--query', type=str, required=True, help='Regular expression to search for.')
@click.option('--name', type=str, help='Scope the search to one user algorithm by name. Omit to search all.')
@click.option('--builtin/--no-builtin', default=True, help='Search built-in algorithm files.')
@click.option('--user/--no-user', default=True, help='Search user-authored algorithm files.')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def search_source(query, name, builtin, user, json_output):
    """Search algorithm source files for a regular expression.

    Greps the source of built-in and user-authored algorithms and returns
    every matching line with its file path and line number. Use it to
    pinpoint a function, attribute or piece of logic before reading or
    editing a file. Narrow the search with --name to target a single user
    algorithm, or with --no-builtin / --no-user to restrict the scope.

    Examples
    --------
    Find every implementation of a method:

    $ tuiml search-source --query "def partial_fit"

    Search only the algorithms you have written:

    $ tuiml search-source --query "gini" --no-builtin

    Search inside one algorithm:

    $ tuiml search-source --query "self._tree" --name MyTreeClassifier
    """
    kwargs = {
            'query': query,
            'name': name,
            'builtin': builtin,
            'user': user,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_search_source', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
