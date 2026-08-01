"""Read Data Command - Preview actual rows from a dataset."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('read-data')
@click.option('--data', type=str, required=True, help="Data file path or built-in dataset name (e.g., 'iris', '/tmp/tuiml_preprocessed/file.csv')")
@click.option('--n-rows', type=int, help='Number of rows to return (default: 10, max: 100)')
@click.option('--mode', type=str, help="Which rows to return: 'head' (first n-rows, the default), 'tail' (last n-rows), 'sample' (random n-rows), or 'indices' (the rows listed in --indices)")
@click.option('--indices', type=str, help="Row indices to return as a JSON array, e.g. '[0, 7, 42]'. Only used when --mode indices")
@click.option('--columns', type=str, help="Columns to return as a JSON array, e.g. '[\"age\", \"income\"]'. Returns all columns if omitted")
@click.option('--include-target/--no-include-target', default=True, help='Include the target column in the output (default: True)')
@click.option('--target', type=str, help='Target column name (used to label the target in the output)')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def read_data(data, n_rows, mode, indices, columns, include_target, target, json_output):
    """Preview actual rows from a dataset.

    Returns real rows as a list of records, so you can eyeball the values,
    column names and encodings without loading the file yourself. Rows can
    be taken from the head or the tail, drawn as a random sample, or picked
    out by index, and --columns narrows the output down to a few fields.
    The label column comes along unless you pass --no-include-target.

    Examples
    --------
    Preview the first rows of a built-in dataset:

    $ tuiml read-data --data iris

    Draw a random sample of five rows:

    $ tuiml read-data --data data.csv --mode sample --n-rows 5

    Pull out specific rows by index:

    $ tuiml read-data --data data.csv --mode indices --indices '[0,7]'
    """
    kwargs = {
            'data': data,
            'n_rows': n_rows,
            'mode': mode,
            'indices': json.loads(indices) if indices is not None else None,
            'columns': json.loads(columns) if columns is not None else None,
            'include_target': include_target,
            'target': target,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_read_data', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
