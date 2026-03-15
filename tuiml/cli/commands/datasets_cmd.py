"""Datasets Command - Browse, download, and upload datasets via CLI.

Examples:

    # Browse all datasets
    tuiml datasets list

    # Filter by task type
    tuiml datasets list --task classification

    # Show dataset details
    tuiml datasets info wine-quality

    # Download a dataset
    tuiml datasets download wine-quality

    # Upload a dataset
    tuiml datasets upload data.csv --name my-dataset --task classification
"""

import os
import json
import click
from pathlib import Path

# Default server URL
DEFAULT_SERVER = "http://localhost:8000"

DESC_MAX_LEN = 60


def get_server_url():
    """Get server URL from environment or default."""
    return os.environ.get("TUIML_HUB_URL", DEFAULT_SERVER)


def get_api_token():
    """Get API token from environment or config file."""
    token = os.environ.get("TUIML_API_TOKEN")
    if token:
        return token

    config_path = Path.home() / ".tuiml" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                return config.get("api_token")
        except (json.JSONDecodeError, OSError):
            pass

    return None


def make_request(endpoint, method="GET"):
    """Make HTTP request to the datasets API."""
    import urllib.request
    import urllib.error

    url = f"{get_server_url()}/api/datasets{endpoint}"

    try:
        req = urllib.request.Request(url, method=method)
        req.add_header("Accept", "application/json")

        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        try:
            error_json = json.loads(error_body)
            raise click.ClickException(f"API Error ({e.code}): {error_json.get('detail', error_body)}")
        except json.JSONDecodeError:
            raise click.ClickException(f"API Error ({e.code}): {error_body}")
    except urllib.error.URLError as e:
        raise click.ClickException(f"Cannot connect to hub at {get_server_url()}: {e}")


def format_size(size_bytes):
    """Format file size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


@click.group("datasets")
def datasets():
    """Browse, download, and upload datasets from TuiML Hub.

    Examples:

        tuiml datasets list
        tuiml datasets list --task classification
        tuiml datasets info wine-quality
        tuiml datasets download wine-quality
    """
    pass


@datasets.command("list")
@click.option("--task", "-t", type=click.Choice(["classification", "regression", "clustering", "other", "all"]),
              default="all", help="Filter by task type (default: all)")
@click.option("--format", "-f", "fmt", type=click.Choice(["table", "json", "names"]),
              default="table", help="Output format (default: table)")
@click.option("--limit", "-l", default=50, help="Max results (default: 50)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
def list_datasets(task, fmt, limit, verbose):
    """Browse datasets available on the hub.

    Examples
    -------
        tuiml datasets list
        tuiml datasets list --task classification
        tuiml datasets list --format json
    """
    try:
        endpoint = f"?limit={limit}"
        if task != "all":
            endpoint += f"&task_type={task}"

        data = make_request(endpoint)
        ds_list = data.get("datasets", [])

        if not ds_list:
            click.echo("No datasets found.")
            return

        if fmt == "names":
            for ds in ds_list:
                click.echo(ds["name"])

        elif fmt == "json":
            click.echo(json.dumps(ds_list, indent=2))

        else:  # table
            click.echo()
            click.echo("=" * 80)
            click.echo("Available Datasets")
            click.echo("=" * 80)

            # Group by task type
            if task == "all":
                grouped = {}
                for ds in ds_list:
                    tt = ds.get("task_type", "other")
                    if tt not in grouped:
                        grouped[tt] = []
                    grouped[tt].append(ds)

                for tt, items in sorted(grouped.items()):
                    click.echo(f"\n  {tt.upper()}:")
                    click.echo("  " + "-" * 78)
                    for ds in sorted(items, key=lambda x: x["name"]):
                        _print_dataset_row(ds, verbose)
            else:
                click.echo(f"\n  {task.upper()}:")
                click.echo("  " + "-" * 78)
                for ds in sorted(ds_list, key=lambda x: x["name"]):
                    _print_dataset_row(ds, verbose)

            click.echo(f"\n  Total: {data.get('total', len(ds_list))} dataset(s)")
            click.echo()

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


def _print_dataset_row(ds, verbose):
    """Print a single dataset row in the table."""
    name = ds["name"]
    desc = ds.get("description", "")
    rows = ds.get("rows", 0)
    cols = ds.get("columns", 0)
    fmt = ds.get("format", "csv").upper()
    size = format_size(ds.get("file_size", 0))

    if verbose:
        click.echo(f"\n    {name}")
        click.echo(f"      {desc}")
        click.echo(f"      {rows:,} rows x {cols} cols | {fmt} | {size} | {ds.get('downloads', 0)} downloads")
        if ds.get("tags"):
            click.echo(f"      Tags: {', '.join(ds['tags'])}")
    else:
        desc_short = desc[:DESC_MAX_LEN] + "..." if len(desc) > DESC_MAX_LEN else desc
        meta = f"[{fmt} {rows:,}x{cols}]"
        click.echo(f"    {name:25s} {meta:18s} {desc_short}")


@datasets.command("info")
@click.argument("name")
def info(name):
    """Show detailed information about a dataset.

    Examples
    -------
        tuiml datasets info wine-quality
    """
    try:
        # Find dataset by name
        data = make_request(f"?limit=100")
        ds_list = data.get("datasets", [])

        target = None
        for ds in ds_list:
            if ds["name"] == name:
                target = ds
                break

        if not target:
            raise click.ClickException(f"Dataset '{name}' not found")

        # Fetch full details
        detail = make_request(f"/{target['id']}")

        click.echo()
        click.echo("=" * 60)
        click.echo(f"  {detail['display_name']}")
        click.echo("=" * 60)
        click.echo()
        click.echo(f"  Name:        {detail['name']}")
        click.echo(f"  Task:        {detail['task_type']}")
        click.echo(f"  Format:      {detail['format'].upper()}")
        click.echo(f"  Rows:        {detail['rows']:,}")
        click.echo(f"  Columns:     {detail['columns']}")
        click.echo(f"  File size:   {detail['file_size_display']}")
        click.echo(f"  Downloads:   {detail['downloads']:,}")
        click.echo(f"  Views:       {detail['views']:,}")
        click.echo(f"  Source:      {detail['source']}")

        if detail.get("target_column"):
            click.echo(f"  Target:      {detail['target_column']}")

        if detail.get("author"):
            author = detail["author"]
            click.echo(f"  Author:      {author.get('full_name') or author.get('username')}")

        if detail.get("tags"):
            click.echo(f"  Tags:        {', '.join(detail['tags'])}")

        click.echo(f"  Created:     {detail['created_at'][:10]}")

        # Column info
        if detail.get("column_names"):
            click.echo()
            click.echo("  Columns:")
            click.echo("  " + "-" * 58)
            for i, (col_name, col_type) in enumerate(zip(detail["column_names"], detail.get("column_types", []))):
                marker = " *" if col_name == detail.get("target_column") else ""
                click.echo(f"    {i+1:3d}. {col_name:30s} {col_type:10s}{marker}")
            if detail.get("target_column"):
                click.echo(f"\n  * = target column")

        # Description
        if detail.get("description"):
            click.echo()
            click.echo(f"  Description:")
            click.echo(f"    {detail['description']}")

        # Usage snippet
        click.echo()
        click.echo("  Python usage:")
        click.echo(f"    from tuiml.hub import datasets")
        click.echo(f"    df = datasets.load(\"{detail['name']}\")")
        click.echo()

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


@datasets.command("download")
@click.argument("name")
@click.option("--output", "-o", type=click.Path(), help="Output directory (default: current dir)")
@click.option("--force", "-f", is_flag=True, help="Re-download even if cached")
def download(name, output, force):
    """Download a dataset from the hub.

    Downloads to ~/.tuiml/datasets/<name>/ by default,
    or to a specified output directory.

    Examples
    -------
        tuiml datasets download wine-quality
        tuiml datasets download wine-quality -o ./data/
    """
    try:
        from tuiml.hub.datasets_remote import datasets as ds_hub

        if output:
            # Download to custom location
            data = make_request(f"?limit=100")
            ds_list = data.get("datasets", [])

            target = None
            for ds in ds_list:
                if ds["name"] == name:
                    target = ds
                    break

            if not target:
                raise click.ClickException(f"Dataset '{name}' not found")

            # Get download info
            download_info = make_request(f"/{target['id']}/download")
            file_name = download_info.get("file_name", f"{name}.csv")

            # Download file
            import urllib.request
            file_url = f"{get_server_url()}/api/datasets/{target['id']}/file"
            out_dir = Path(output)
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / file_name

            click.echo(f"Downloading '{name}' to {dest}...")
            urllib.request.urlretrieve(file_url, dest)
            click.echo(f"  Saved: {dest} ({format_size(dest.stat().st_size)})")
        else:
            # Use the hub client (downloads to ~/.tuiml/datasets/)
            click.echo(f"Downloading '{name}'...")
            df = ds_hub.load(name, force=force)
            click.echo(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
            click.echo(f"  Cached at: ~/.tuiml/datasets/{name}/")

        click.echo("Done.")

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


@datasets.command("upload")
@click.argument("file", type=click.Path(exists=True))
@click.option("--name", "-n", required=True, help="Dataset name (slug, e.g. my-dataset)")
@click.option("--display-name", "-d", help="Display name (defaults to name)")
@click.option("--description", required=True, help="Short description")
@click.option("--task", "-t", type=click.Choice(["classification", "regression", "clustering", "other"]),
              default="other", help="Task type (default: other)")
@click.option("--target", help="Target column name")
@click.option("--tags", multiple=True, help="Tags (can be repeated)")
@click.option("--readme", type=click.Path(exists=True), help="Path to README.md file")
def upload(file, name, display_name, description, task, target, tags, readme):
    """Upload a dataset file to TuiML Hub.

    Supports CSV, JSON, Parquet, Excel, and ARFF files.

    Examples
    -------
        tuiml datasets upload data.csv --name wine-quality \\
            --description "Wine quality ratings" --task classification \\
            --target quality --tags wine --tags uci
    """
    token = get_api_token()
    if not token:
        raise click.ClickException(
            "No API token found. Set TUIML_API_TOKEN environment variable "
            "or run 'tuiml upload login <token>'"
        )

    try:
        import requests as req_lib
    except ImportError:
        raise click.ClickException(
            "The 'requests' library is required for uploads. "
            "Install with: pip install requests"
        )

    file_path = Path(file)
    display = display_name or name.replace("-", " ").replace("_", " ").title()

    click.echo(f"\n  Dataset: {display}")
    click.echo(f"  Name: {name}")
    click.echo(f"  File: {file_path.name} ({format_size(file_path.stat().st_size)})")
    click.echo(f"  Task: {task}")

    # Read README
    readme_content = None
    if readme:
        readme_content = Path(readme).read_text()

    # Build form data
    form_data = {
        "name": name,
        "display_name": display,
        "description": description,
        "task_type": task,
    }
    if target:
        form_data["target_column"] = target
    if tags:
        form_data["tags"] = ",".join(tags)
    if readme_content:
        form_data["readme"] = readme_content

    url = f"{get_server_url()}/api/datasets/"

    click.echo("\n  Uploading...")

    with open(file_path, "rb") as f:
        resp = req_lib.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (file_path.name, f)},
            data=form_data,
        )

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise click.ClickException(f"Upload failed ({resp.status_code}): {detail}")

    result = resp.json()

    click.echo(f"""
{'=' * 50}
  Dataset uploaded successfully!
{'=' * 50}
  ID:      {result['id']}
  Name:    {result['name']}
  Rows:    {result['rows']:,}
  Columns: {result['columns']}

  View at: {get_server_url()}/dataset/{result['id']}
""")
