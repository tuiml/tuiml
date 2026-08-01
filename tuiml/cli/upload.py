"""Upload Command - Register a dataset for use by other TuiML commands."""

import click
import json
from tuiml.agent.tools import execute_tool

@click.command('upload')
@click.option('--file-path', type=str, help='Path to an existing dataset file on disk. Supported: .csv, .tsv, .arff, .parquet, .pq, .xlsx, .xls, .json, .jsonl, .ndjson, .npy, .npz')
@click.option('--content', type=str, help='Raw text content for small inline datasets (use together with --format)')
@click.option('--format', type=str, help='File format of --content; ignored for --file-path, where it is inferred from the extension')
@click.option('--name', type=str, help='Name to register the dataset under (without extension)')
@click.option('--json-output', is_flag=True, help='Output raw JSON')
def upload(file_path, content, format, name, json_output):
    """Register a dataset so other TuiML commands can use it.

    Validates a dataset and returns a canonical path you can hand to
    commands such as tuiml train, tuiml preprocess and tuiml profile.
    Supply either --file-path for a file already on disk (preferred, and
    the only practical option for large datasets) or --content with raw
    inline text plus --format. CSV, TSV, ARFF, Parquet, Excel, JSON,
    JSONL and NumPy npy/npz files are all supported.

    Examples
    --------
    Register a file that already exists on disk:

    $ tuiml upload --file-path ./data/customers.csv

    Register it under a friendlier name:

    $ tuiml upload --file-path ./raw.parquet --name customers

    Register a small dataset from inline text:

    $ tuiml upload --content "$CSV_TEXT" --format csv --name demo
    """
    kwargs = {
            'file_path': file_path,
            'content': content,
            'format': format,
            'name': name,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    result = execute_tool('tuiml_upload_data', **kwargs)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if result.get('status') == 'error':
            click.echo(f"Error: {result.get('error')}", err=True)
        else:
            # For now, print pretty JSON if not explicitly requested otherwise
            click.echo(json.dumps(result, indent=2, default=str))
