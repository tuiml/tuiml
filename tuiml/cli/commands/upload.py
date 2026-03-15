"""Upload Command - Upload algorithms to TuiML Hub via CLI.

Simple folder-based upload workflow:

1. Create your algorithm folder with:
   - tuiml.json (configuration)
   - Your algorithm files (.py)
   - README.md (optional)

2. Upload with a single command:
   tuiml upload push ./my_algorithm/

Example tuiml.json:
{
    "name": "my_random_forest",
    "display_name": "My Random Forest",
    "description": "An optimized random forest implementation",
    "type": "classifier",
    "category": "ensemble",
    "version": "1.0.0",
    "tags": ["ml", "ensemble", "trees"],
    "main": "algorithm.py",
    "dependencies": ["numpy", "scikit-learn"]
}
"""

import os
import click
import json
from contextlib import ExitStack
from pathlib import Path

# Default server URL
DEFAULT_SERVER = "http://localhost:8000"

# Config file name
CONFIG_FILE = "tuiml.json"

# Default exclusions for upload
DEFAULT_EXCLUDES = {
    "__pycache__", ".git", ".gitignore", ".DS_Store", 
    "*.pyc", "*.pyo", ".env", "venv", ".venv", "node_modules"
}


def get_server_url():
    """Get server URL from environment or config."""
    return os.environ.get("TUIML_HUB_URL", DEFAULT_SERVER)


def get_api_token():
    """Get API token from environment or config file."""
    # Try environment variable first
    token = os.environ.get("TUIML_API_TOKEN")
    if token:
        return token
    
    # Try config file
    config_path = Path.home() / ".tuiml" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                return config.get("api_token")
        except (json.JSONDecodeError, OSError):
            pass

    return None


def make_request(method, endpoint, token, data=None, files=None, json_data=None):
    """Make HTTP request to the API."""
    import urllib.request
    import urllib.error
    
    url = f"{get_server_url()}/api/upload{endpoint}"
    headers = {"Authorization": f"Bearer {token}"}
    
    if json_data:
        headers["Content-Type"] = "application/json"
        req_data = json.dumps(json_data).encode('utf-8')
        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    elif files:
        # For multipart/form-data, we need requests library
        try:
            import requests
            if method == "POST":
                resp = requests.post(url, headers={"Authorization": f"Bearer {token}"}, 
                                     files=files, data=data)
            else:
                resp = requests.request(method, url, headers={"Authorization": f"Bearer {token}"}, 
                                        files=files, data=data)
            resp.raise_for_status()
            return resp.json()
        except ImportError:
            raise click.ClickException(
                "The 'requests' library is required for file uploads. "
                "Install it with: pip install requests"
            )
    else:
        req = urllib.request.Request(url, headers=headers, method=method)
    
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        try:
            error_json = json.loads(error_body)
            raise click.ClickException(f"API Error ({e.code}): {error_json.get('detail', error_body)}")
        except json.JSONDecodeError:
            raise click.ClickException(f"API Error ({e.code}): {error_body}")


@click.group()
def upload():
    """Upload algorithms to TuiML Hub.
    
    Simple folder-based workflow (recommended):
    
        # 1. Initialize config in your algorithm folder
        tuiml upload init ./my_algorithm/
        
        # 2. Edit tuiml.json with your algorithm details
        
        # 3. Push everything with one command
        tuiml upload push ./my_algorithm/
    
    Setup
    -----
    Set your API token via environment variable:
    
        export TUIML_API_TOKEN=wk_xxxxx
    
    Or run:
    
        tuiml upload login wk_xxxxx
    """
    pass


def load_config(folder_path: Path) -> dict:
    """Load and validate tuiml.json from a folder."""
    config_file = folder_path / CONFIG_FILE
    
    if not config_file.exists():
        raise click.ClickException(
            f"No {CONFIG_FILE} found in {folder_path}\n"
            f"Run 'tuiml upload init {folder_path}' to create one."
        )
    
    try:
        with open(config_file) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in {CONFIG_FILE}: {e}")
    
    # Validate required fields
    required = ["name", "display_name", "description", "type", "category"]
    missing = [f for f in required if f not in config]
    if missing:
        raise click.ClickException(
            f"Missing required fields in {CONFIG_FILE}: {', '.join(missing)}"
        )
    
    return config


def collect_files(folder_path: Path, exclude: set = None) -> list:
    """Collect all files from folder, respecting exclusions."""
    if exclude is None:
        exclude = DEFAULT_EXCLUDES
    
    files_to_upload = []
    
    for root, dirs, files in os.walk(folder_path):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in exclude and not any(
            d.endswith(e.lstrip("*")) for e in exclude if e.startswith("*")
        )]
        
        for file_name in files:
            # Skip config file (already processed)
            if file_name == CONFIG_FILE:
                continue
            
            # Check exclusions
            if file_name in exclude:
                continue
            if any(file_name.endswith(e.lstrip("*")) for e in exclude if e.startswith("*")):
                continue
            
            full_path = Path(root) / file_name
            rel_path = full_path.relative_to(folder_path)
            files_to_upload.append((full_path, str(rel_path)))
    
    return files_to_upload


@upload.command("init")
@click.argument("folder", type=click.Path(file_okay=False), default=".")
@click.option("--name", "-n", help="Algorithm name (lowercase, underscores)")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing config")
def init(folder, name, force):
    """Initialize a tuiml.json config file in a folder.
    
    Creates a template config file that you can edit with your algorithm details.
    
    Example
    -------
        # Initialize in current directory
        tuiml upload init
        
        # Initialize in a specific folder
        tuiml upload init ./my_algorithm/
        
        # With a specific name
        tuiml upload init ./my_algo/ --name my_random_forest
    """
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    config_file = folder_path / CONFIG_FILE
    
    if config_file.exists() and not force:
        raise click.ClickException(
            f"{CONFIG_FILE} already exists. Use --force to overwrite."
        )
    
    # Auto-detect algorithm name from folder
    algo_name = name or folder_path.name.lower().replace("-", "_").replace(" ", "_")
    
    # Create template config
    config = {
        "name": algo_name,
        "display_name": algo_name.replace("_", " ").title(),
        "description": "A brief description of your algorithm",
        "type": "classifier",  # classifier, regressor, clusterer, transformer, timeseries
        "category": "ensemble",  # ensemble, neural, bayesian, distance, etc.
        "version": "1.0.0",
        "tags": ["ml"],
        "main": "algorithm.py",  # Main algorithm file
        "dependencies": [],  # Python packages required
        "git_url": None,
        "paper_url": None,
        "documentation_url": None,
        "citation": None
    }
    
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    click.echo(f"""
✓ Created {config_file}

Next steps:
  1. Edit {CONFIG_FILE} with your algorithm details
  2. Add your algorithm files to the folder
  3. Run: tuiml upload push {folder}

Example folder structure:
  {folder_path}/
    ├── tuiml.json      (config - just created)
    ├── algorithm.py       (main algorithm file)
    ├── utils/             (optional helpers)
    │   └── helpers.py
    └── README.md          (optional documentation)
""")


@upload.command("push")
@click.argument("folder", type=click.Path(exists=True, file_okay=False), default=".")
@click.option("--update", "-u", is_flag=True, help="Update existing algorithm instead of creating new")
@click.option("--dry-run", is_flag=True, help="Show what would be uploaded without uploading")
def push(folder, update, dry_run):
    """Upload algorithm folder to TuiML Hub.
    
    Reads tuiml.json from the folder, creates the algorithm entry,
    and uploads all files in one step.
    
    Example
    -------
        # Upload from current directory
        tuiml upload push
        
        # Upload from specific folder
        tuiml upload push ./my_algorithm/
        
        # Preview what will be uploaded
        tuiml upload push ./my_algo/ --dry-run
    """
    token = get_api_token()
    if not token:
        raise click.ClickException(
            "No API token found. Set TUIML_API_TOKEN environment variable "
            "or run 'tuiml upload login <token>'"
        )
    
    folder_path = Path(folder).resolve()
    
    # Load config
    config = load_config(folder_path)
    
    # Collect files
    files_to_upload = collect_files(folder_path)
    
    # Read README if exists
    readme_content = None
    for readme_name in ["README.md", "README.txt", "readme.md", "README"]:
        readme_path = folder_path / readme_name
        if readme_path.exists():
            with open(readme_path) as f:
                readme_content = f.read()
            break
    
    click.echo(f"\n📦 Algorithm: {config['display_name']}")
    click.echo(f"   Name: {config['name']}")
    click.echo(f"   Type: {config['type']} / {config['category']}")
    click.echo(f"   Version: {config.get('version', '1.0.0')}")
    click.echo(f"   Files: {len(files_to_upload)}")
    
    if dry_run:
        click.echo("\n📁 Files to upload:")
        for full_path, rel_path in files_to_upload:
            size = full_path.stat().st_size
            click.echo(f"   • {rel_path} ({size:,} bytes)")
        click.echo("\n(Dry run - nothing uploaded)")
        return
    
    try:
        import requests
    except ImportError:
        raise click.ClickException(
            "The 'requests' library is required. Install with: pip install requests"
        )
    
    # Step 1: Create algorithm entry
    click.echo("\n⏳ Creating algorithm entry...")
    
    # Map config to API schema
    api_data = {
        "name": config["name"],
        "display_name": config["display_name"],
        "description": config["description"],
        "algorithm_type": config["type"],
        "category": config["category"],
        "version": config.get("version", "1.0.0"),
        "readme": readme_content,
        "git_url": config.get("git_url"),
        "paper_url": config.get("paper_url"),
        "documentation_url": config.get("documentation_url"),
        "citation": config.get("citation"),
        "dependencies": ",".join(config.get("dependencies", [])) if config.get("dependencies") else None,
        "tags": config.get("tags")
    }
    
    # Remove None values
    api_data = {k: v for k, v in api_data.items() if v is not None}
    
    result = make_request("POST", "/", token, json_data=api_data)
    algorithm_id = result["id"]
    
    click.echo(f"   ✓ Created algorithm (ID: {algorithm_id})")
    
    # Step 2: Upload files
    if files_to_upload:
        click.echo(f"\n⏳ Uploading {len(files_to_upload)} files...")
        
        url = f"{get_server_url()}/api/upload/{algorithm_id}/files"
        
        with ExitStack() as stack:
            file_objs = []
            paths = []

            for full_path, rel_path in files_to_upload:
                f = stack.enter_context(open(full_path, "rb"))
                file_objs.append(("files", (full_path.name, f)))
                paths.append(rel_path)

            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                files=file_objs,
                data={"paths": json.dumps(paths)}
            )
            resp.raise_for_status()
            upload_result = resp.json()
        
        click.echo(f"   ✓ Uploaded {upload_result['files_uploaded']} files ({upload_result['total_size']:,} bytes)")
    
    click.echo(f"""
{'='*50}
✅ Algorithm uploaded successfully!
{'='*50}
   ID: {algorithm_id}
   Name: {config['name']}
   Folder: {result['folder_path']}
   
   View at: {get_server_url()}/algorithms/{config['name']}
""")


@upload.command("login")
@click.argument("token")
def login(token):
    """Save API token for future use.
    
    The token is saved to ~/.tuiml/config.json
    
    Example
    -------
        tuiml upload login wk_abc123xyz
    """
    config_dir = Path.home() / ".tuiml"
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / "config.json"
    
    # Load existing config or create new
    config = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    
    config["api_token"] = token

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    os.chmod(config_path, 0o600)

    click.echo(f"✓ API token saved to {config_path}")


@upload.command("create")
@click.argument("name")
@click.argument("display_name")
@click.option("--description", "-d", required=True, help="Short description of the algorithm")
@click.option("--type", "algo_type", required=True, 
              type=click.Choice(["classifier", "regressor", "clusterer", "transformer", "evaluator", "timeseries"]),
              help="Algorithm type")
@click.option("--category", "-c", required=True,
              help="Category (e.g., ensemble, neural, bayesian, distance)")
@click.option("--version", "-v", default="1.0.0", help="Version string (default: 1.0.0)")
@click.option("--long-description", help="Detailed description")
@click.option("--readme", type=click.Path(exists=True), help="Path to README file")
@click.option("--git-url", help="Git repository URL")
@click.option("--paper-url", help="Research paper URL")
@click.option("--docs-url", help="Documentation URL")
@click.option("--citation", help="Citation text")
@click.option("--dependencies", help="Python dependencies (comma-separated)")
@click.option("--tags", "-t", multiple=True, help="Tags for the algorithm")
def create(name, display_name, description, algo_type, category, version, 
           long_description, readme, git_url, paper_url, docs_url, citation, 
           dependencies, tags):
    """Create a new algorithm entry on TuiML Hub.
    
    This creates the algorithm metadata. Files should be uploaded separately
    using the 'upload files' or 'upload folder' commands.
    
    Example
    -------
        tuiml upload create my_random_forest "My Random Forest" \\
            --description "An optimized random forest implementation" \\
            --type classifier --category ensemble \\
            --tags ml --tags ensemble --tags trees
    """
    token = get_api_token()
    if not token:
        raise click.ClickException(
            "No API token found. Set TUIML_API_TOKEN environment variable "
            "or run 'tuiml upload login <token>'"
        )
    
    # Read README content if provided
    readme_content = None
    if readme:
        with open(readme) as f:
            readme_content = f.read()
    
    data = {
        "name": name,
        "display_name": display_name,
        "description": description,
        "algorithm_type": algo_type,
        "category": category,
        "version": version,
        "long_description": long_description,
        "readme": readme_content,
        "git_url": git_url,
        "paper_url": paper_url,
        "documentation_url": docs_url,
        "citation": citation,
        "dependencies": dependencies,
        "tags": list(tags) if tags else None
    }
    
    # Remove None values
    data = {k: v for k, v in data.items() if v is not None}
    
    result = make_request("POST", "/", token, json_data=data)
    
    click.echo("\n" + "="*50)
    click.echo("Algorithm Created Successfully!")
    click.echo("="*50)
    click.echo(f"  ID: {result['id']}")
    click.echo(f"  Name: {result['name']}")
    click.echo(f"  Folder: {result['folder_path']}")
    click.echo(f"\n  Next step: Upload files using:")
    click.echo(f"    tuiml upload files {result['id']} <file1> <file2> ...")
    click.echo(f"    tuiml upload folder {result['id']} <path_to_folder>")


@upload.command("files")
@click.argument("algorithm_id", type=int)
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--path-prefix", help="Prefix to add to file paths in the algorithm folder")
def upload_files(algorithm_id, files, path_prefix):
    """Upload files to an existing algorithm.
    
    Upload one or more files to the algorithm folder. Files are uploaded
    with their original names unless --path-prefix is specified.
    
    Example
    -------
        # Upload to root folder
        tuiml upload files 1 my_algo.py README.md
        
        # Upload to a subdirectory
        tuiml upload files 1 helpers.py --path-prefix utils/
    """
    token = get_api_token()
    if not token:
        raise click.ClickException(
            "No API token found. Set TUIML_API_TOKEN environment variable "
            "or run 'tuiml upload login <token>'"
        )
    
    try:
        import requests
    except ImportError:
        raise click.ClickException(
            "The 'requests' library is required for file uploads. "
            "Install it with: pip install requests"
        )
    
    url = f"{get_server_url()}/api/upload/{algorithm_id}/files"
    
    # Prepare files and paths
    with ExitStack() as stack:
        file_objs = []
        paths = []

        for file_path in files:
            file_path = Path(file_path)
            rel_path = file_path.name
            if path_prefix:
                rel_path = f"{path_prefix.rstrip('/')}/{rel_path}"

            f = stack.enter_context(open(file_path, "rb"))
            file_objs.append(("files", (file_path.name, f)))
            paths.append(rel_path)

        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            files=file_objs,
            data={"paths": json.dumps(paths)}
        )
        resp.raise_for_status()
        result = resp.json()
    
    click.echo("\n" + "="*50)
    click.echo("Files Uploaded Successfully!")
    click.echo("="*50)
    click.echo(f"  Algorithm ID: {result['algorithm_id']}")
    click.echo(f"  Files uploaded: {result['files_uploaded']}")
    click.echo(f"  Total size: {result['total_size']:,} bytes")
    click.echo("\n  Uploaded files:")
    for f in result['files']:
        click.echo(f"    • {f['path']} ({f['size']:,} bytes)")


@upload.command("folder")
@click.argument("algorithm_id", type=int)
@click.argument("folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--exclude", "-e", multiple=True, help="Patterns to exclude (e.g., __pycache__, .git)")
def upload_folder(algorithm_id, folder, exclude):
    """Upload an entire folder to an algorithm.
    
    Recursively uploads all files in the folder, preserving the directory
    structure. Common patterns like __pycache__ and .git are excluded by default.
    
    Example
    -------
        tuiml upload folder 1 ./my_algorithm/
        
        # Exclude additional patterns
        tuiml upload folder 1 ./my_algo/ --exclude "*.pyc" --exclude "tests/"
    """
    token = get_api_token()
    if not token:
        raise click.ClickException(
            "No API token found. Set TUIML_API_TOKEN environment variable "
            "or run 'tuiml upload login <token>'"
        )
    
    try:
        import requests
    except ImportError:
        raise click.ClickException(
            "The 'requests' library is required for file uploads. "
            "Install it with: pip install requests"
        )
    
    folder_path = Path(folder)
    all_exclude = DEFAULT_EXCLUDES | set(exclude)

    # Reuse collect_files with merged exclusions
    files_to_upload = collect_files(folder_path, exclude=all_exclude)

    if not files_to_upload:
        raise click.ClickException("No files found to upload")

    click.echo(f"Found {len(files_to_upload)} files to upload...")

    url = f"{get_server_url()}/api/upload/{algorithm_id}/files"

    with ExitStack() as stack:
        file_objs = []
        paths = []

        for full_path, rel_path in files_to_upload:
            f = stack.enter_context(open(full_path, "rb"))
            file_objs.append(("files", (full_path.name, f)))
            paths.append(rel_path)

        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            files=file_objs,
            data={"paths": json.dumps(paths)}
        )
        resp.raise_for_status()
        result = resp.json()
    
    click.echo("\n" + "="*50)
    click.echo("Folder Uploaded Successfully!")
    click.echo("="*50)
    click.echo(f"  Algorithm ID: {result['algorithm_id']}")
    click.echo(f"  Files uploaded: {result['files_uploaded']}")
    click.echo(f"  Total size: {result['total_size']:,} bytes")


@upload.command("delete")
@click.argument("algorithm_id", type=int)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def delete(algorithm_id, yes):
    """Delete an algorithm and all its files.
    
    Example
    -------
        tuiml upload delete 1
        
        # Skip confirmation
        tuiml upload delete 1 --yes
    """
    token = get_api_token()
    if not token:
        raise click.ClickException(
            "No API token found. Set TUIML_API_TOKEN environment variable "
            "or run 'tuiml upload login <token>'"
        )
    
    if not yes:
        if not click.confirm(f"Are you sure you want to delete algorithm {algorithm_id}?"):
            click.echo("Cancelled.")
            return
    
    result = make_request("DELETE", f"/{algorithm_id}", token)
    click.echo(f"✓ {result['message']}")


@upload.command("update")
@click.argument("algorithm_id", type=int)
@click.option("--display-name", help="New display name")
@click.option("--description", "-d", help="New description")
@click.option("--version", "-v", help="New version")
@click.option("--git-url", help="Git repository URL")
@click.option("--paper-url", help="Research paper URL")
@click.option("--tags", "-t", multiple=True, help="New tags (replaces existing)")
def update(algorithm_id, display_name, description, version, git_url, paper_url, tags):
    """Update algorithm metadata.
    
    Only the specified fields will be updated.
    
    Example
    -------
        tuiml upload update 1 --version "1.1.0" --description "Updated description"
    """
    token = get_api_token()
    if not token:
        raise click.ClickException(
            "No API token found. Set TUIML_API_TOKEN environment variable "
            "or run 'tuiml upload login <token>'"
        )
    
    # Fetch existing algorithm data so we only override provided fields
    existing = make_request("GET", f"/{algorithm_id}", token)

    data = {
        "name": existing.get("name", ""),
        "display_name": display_name or existing.get("display_name", ""),
        "description": description or existing.get("description", ""),
        "algorithm_type": existing.get("algorithm_type", "classifier"),
        "category": existing.get("category", "misc"),
        "version": version or existing.get("version", "1.0.0"),
        "git_url": git_url if git_url is not None else existing.get("git_url"),
        "paper_url": paper_url if paper_url is not None else existing.get("paper_url"),
        "tags": list(tags) if tags else existing.get("tags"),
    }

    result = make_request("PUT", f"/{algorithm_id}", token, json_data=data)
    click.echo(f"✓ {result['message']}")
