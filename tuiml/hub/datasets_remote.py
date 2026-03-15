"""TuiML Hub Dataset Client for browsing and loading datasets."""

import os
import json
from pathlib import Path
from typing import List, Optional, Any
from dataclasses import dataclass

# Default server URL
DEFAULT_SERVER = "http://localhost:8000"

# Local cache directory
CACHE_DIR = Path.home() / ".tuiml" / "datasets"


@dataclass
class DatasetInfo:
    """Information about a remote dataset.

    Attributes
    ----------
    id : str
        The unique identifier for the dataset.
    name : str
        The unique slug/name of the dataset.
    display_name : str
        The human-readable name of the dataset.
    description : str
        A brief description of the dataset.
    format : str
        The file format (csv, arff, json, parquet, excel).
    task_type : str
        The task type (classification, regression, clustering, other).
    rows : int
        Number of rows in the dataset.
    columns : int
        Number of columns in the dataset.
    file_size : int
        File size in bytes.
    downloads : int
        Number of times the dataset has been downloaded.
    target_column : str
        Name of the target/label column.
    """
    id: str
    name: str
    display_name: str
    description: str
    format: str = "csv"
    task_type: str = "other"
    rows: int = 0
    columns: int = 0
    file_size: int = 0
    downloads: int = 0
    target_column: str = None

    def __repr__(self):
        return f"DatasetInfo(name='{self.name}', task='{self.task_type}', {self.rows}x{self.columns})"


class DatasetHub:
    """Client for browsing and loading datasets from the TuiML Hub.

    Examples
    --------
    >>> from tuiml.hub import dataset_hub
    >>> # Browse datasets
    >>> for ds in dataset_hub.browse():
    ...     print(ds.name, ds.rows, ds.columns)
    >>> # Load a dataset
    >>> df = dataset_hub.load("wine-quality")
    """

    def __init__(self, server_url: str = None):
        """Initialize the dataset hub client.

        Parameters
        ----------
        server_url : str, optional, default=None
            URL of the TuiML Hub server. Defaults to ``TUIML_HUB_URL``
            environment variable or localhost.
        """
        self.server_url = server_url or os.environ.get("TUIML_HUB_URL", DEFAULT_SERVER)
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_request(self, endpoint: str) -> Any:
        """Make HTTP request to the hub API."""
        import urllib.request
        import urllib.error

        url = f"{self.server_url}/api{endpoint}"

        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            try:
                error_json = json.loads(error_body)
                raise RuntimeError(f"Hub API error ({e.code}): {error_json.get('detail', error_body)}")
            except json.JSONDecodeError:
                raise RuntimeError(f"Hub API error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot connect to hub at {self.server_url}: {e}")

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file from URL to destination."""
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)

    def browse(
        self,
        task_type: str = None,
        format: str = None,
        limit: int = 50,
    ) -> List[DatasetInfo]:
        """Browse datasets available on the remote hub.

        Parameters
        ----------
        task_type : str, optional, default=None
            Filter by task type (e.g., 'classification').
        format : str, optional, default=None
            Filter by file format (e.g., 'csv').
        limit : int, default=50
            Maximum number of results to return.

        Returns
        -------
        datasets : List[DatasetInfo]
            List of dataset information objects.

        Examples
        --------
        >>> classification_datasets = dataset_hub.browse(task_type="classification")
        """
        endpoint = f"/datasets?limit={limit}"
        if task_type:
            endpoint += f"&task_type={task_type}"
        if format:
            endpoint += f"&format={format}"

        data = self._make_request(endpoint)

        return [
            DatasetInfo(
                id=ds.get("id", ""),
                name=ds.get("name", ""),
                display_name=ds.get("display_name", ""),
                description=ds.get("description", ""),
                format=ds.get("format", "csv"),
                task_type=ds.get("task_type", "other"),
                rows=ds.get("rows", 0),
                columns=ds.get("columns", 0),
                file_size=ds.get("file_size", 0),
                downloads=ds.get("downloads", 0),
            )
            for ds in data.get("datasets", [])
        ]

    def search(self, query: str) -> List[DatasetInfo]:
        """Search for datasets by keyword.

        Parameters
        ----------
        query : str
            Search query string.

        Returns
        -------
        results : List[DatasetInfo]
            List of matching dataset information objects.

        Examples
        --------
        >>> results = dataset_hub.search("wine")
        """
        import urllib.parse

        encoded = urllib.parse.quote(query)
        # Use the list endpoint with no specific search API yet
        data = self._make_request(f"/datasets?limit=50")

        query_lower = query.lower()
        return [
            DatasetInfo(
                id=ds.get("id", ""),
                name=ds.get("name", ""),
                display_name=ds.get("display_name", ""),
                description=ds.get("description", ""),
                format=ds.get("format", "csv"),
                task_type=ds.get("task_type", "other"),
                rows=ds.get("rows", 0),
                columns=ds.get("columns", 0),
                file_size=ds.get("file_size", 0),
                downloads=ds.get("downloads", 0),
            )
            for ds in data.get("datasets", [])
            if query_lower in ds.get("name", "").lower()
            or query_lower in ds.get("display_name", "").lower()
            or query_lower in ds.get("description", "").lower()
        ]

    def info(self, name: str) -> DatasetInfo:
        """Get detailed information about a dataset.

        Parameters
        ----------
        name : str
            The name (slug) of the dataset.

        Returns
        -------
        info : DatasetInfo
            Detailed information about the dataset.
        """
        # Find dataset by name via list endpoint
        data = self._make_request(f"/datasets?limit=100")
        for ds in data.get("datasets", []):
            if ds.get("name") == name:
                return DatasetInfo(
                    id=ds.get("id", ""),
                    name=ds.get("name", ""),
                    display_name=ds.get("display_name", ""),
                    description=ds.get("description", ""),
                    format=ds.get("format", "csv"),
                    task_type=ds.get("task_type", "other"),
                    rows=ds.get("rows", 0),
                    columns=ds.get("columns", 0),
                    file_size=ds.get("file_size", 0),
                    downloads=ds.get("downloads", 0),
                    target_column=ds.get("target_column"),
                )
        raise ValueError(f"Dataset '{name}' not found on hub")

    def load(self, name: str, force: bool = False):
        """Download a dataset and return it as a pandas DataFrame.

        Downloads the dataset file to ``~/.tuiml/datasets/{name}/``
        and reads it into a DataFrame.

        Parameters
        ----------
        name : str
            The name (slug) of the dataset to load.
        force : bool, default=False
            If True, re-download even if cached locally.

        Returns
        -------
        df : pandas.DataFrame
            The dataset as a DataFrame.

        Examples
        --------
        >>> df = dataset_hub.load("wine-quality")
        >>> print(df.shape)
        (6497, 13)
        """
        import pandas as pd

        # Check cache
        ds_dir = self.cache_dir / name
        meta_file = ds_dir / "meta.json"

        if meta_file.exists() and not force:
            meta = json.loads(meta_file.read_text())
            file_path = ds_dir / meta["file_name"]
            if file_path.exists():
                return self._read_file(file_path, meta["format"])

        # Look up dataset info to get the ID
        ds_info = self.info(name)

        # Increment download count
        try:
            self._make_request(f"/datasets/{ds_info.id}/download")
        except Exception:
            pass

        # Download the file
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Get file info from download endpoint
        download_info = self._make_request(f"/datasets/{ds_info.id}/download")
        file_name = download_info.get("file_name", f"{name}.csv")
        file_format = download_info.get("format", "csv")
        file_dest = ds_dir / file_name

        file_url = f"{self.server_url}/api/datasets/{ds_info.id}/file"
        print(f"Downloading '{name}' from hub...")
        self._download_file(file_url, file_dest)

        # Save metadata
        meta = {"name": name, "file_name": file_name, "format": file_format}
        meta_file.write_text(json.dumps(meta, indent=2))

        print(f"Cached to {ds_dir}")
        return self._read_file(file_dest, file_format)

    def _read_file(self, path: Path, format: str):
        """Read a data file into a pandas DataFrame.

        Parameters
        ----------
        path : Path
            Path to the data file.
        format : str
            File format (csv, arff, json, parquet, excel).

        Returns
        -------
        df : pandas.DataFrame
            The loaded DataFrame.
        """
        import pandas as pd

        if format == "csv":
            return pd.read_csv(path)
        elif format == "json":
            return pd.read_json(path)
        elif format == "parquet":
            return pd.read_parquet(path)
        elif format in ("excel", "xls", "xlsx"):
            return pd.read_excel(path)
        elif format == "arff":
            content = path.read_text(errors="replace")
            lines = content.splitlines()
            import io
            data_start = 0
            attrs = []
            for i, line in enumerate(lines):
                stripped = line.strip().lower()
                if stripped.startswith("@attribute"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        attrs.append(parts[1])
                elif stripped == "@data":
                    data_start = i + 1
                    break
            data_lines = "\n".join(lines[data_start:])
            df = pd.read_csv(io.StringIO(data_lines), header=None)
            if attrs:
                df.columns = attrs[:len(df.columns)]
            return df
        else:
            raise ValueError(f"Unsupported format: {format}")

    def upload(
        self,
        file_path: str,
        name: str,
        display_name: str = None,
        description: str = "",
        task_type: str = "other",
        target_column: str = None,
        tags: List[str] = None,
        readme: str = None,
        token: str = None,
    ) -> DatasetInfo:
        """Upload a dataset file to the TuiML Hub.

        Parameters
        ----------
        file_path : str
            Path to the dataset file (csv, json, parquet, excel, arff).
        name : str
            Unique slug/name for the dataset (lowercase, hyphens).
        display_name : str, optional
            Human-readable name. Defaults to name with title case.
        description : str, default=""
            Brief description of the dataset.
        task_type : str, default="other"
            Task type: "classification", "regression", "clustering", or "other".
        target_column : str, optional
            Name of the target/label column.
        tags : List[str], optional
            List of tags for search and filtering.
        readme : str, optional
            Long description or documentation in markdown.
        token : str, optional
            API token. Defaults to ``TUIML_API_TOKEN`` environment variable.

        Returns
        -------
        info : DatasetInfo
            Information about the uploaded dataset.

        Examples
        --------
        >>> from tuiml.hub import datasets
        >>> info = datasets.upload(
        ...     "my_data.csv",
        ...     name="customer-churn",
        ...     description="Customer churn prediction dataset",
        ...     task_type="classification",
        ...     target_column="churn",
        ...     tags=["customer", "marketing"]
        ... )
        >>> print(info.id)
        """
        import urllib.request
        import urllib.error
        import mimetypes
        import uuid

        # Resolve file path
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get API token
        api_token = token or os.environ.get("TUIML_API_TOKEN")
        if not api_token:
            raise ValueError(
                "API token required. Set TUIML_API_TOKEN environment variable "
                "or pass token parameter."
            )

        # Prepare form data
        if display_name is None:
            display_name = name.replace("-", " ").replace("_", " ").title()

        tags_str = ",".join(tags) if tags else ""

        # Read file content
        file_content = path.read_bytes()
        file_name = path.name

        # Build multipart form data
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex[:16]}"

        def encode_field(name: str, value: str) -> bytes:
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")

        def encode_file(field_name: str, filename: str, content: bytes) -> bytes:
            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            header = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8")
            return header + content + b"\r\n"

        # Build body
        body = b""
        body += encode_field("name", name)
        body += encode_field("display_name", display_name)
        body += encode_field("description", description)
        body += encode_field("task_type", task_type)
        if target_column:
            body += encode_field("target_column", target_column)
        if tags_str:
            body += encode_field("tags", tags_str)
        if readme:
            body += encode_field("readme", readme)
        body += encode_file("file", file_name, file_content)
        body += f"--{boundary}--\r\n".encode("utf-8")

        # Make request
        url = f"{self.server_url}/api/datasets/"
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        req.add_header("Authorization", f"Bearer {api_token}")

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode())
                print(f"Dataset '{name}' uploaded successfully!")
                print(f"  ID: {result.get('id')}")
                print(f"  URL: {self.server_url}/dataset/{result.get('id')}")
                return DatasetInfo(
                    id=result.get("id", ""),
                    name=result.get("name", name),
                    display_name=result.get("display_name", display_name),
                    description=result.get("description", description),
                    format=result.get("format", path.suffix.lstrip(".")),
                    task_type=result.get("task_type", task_type),
                    rows=result.get("rows", 0),
                    columns=result.get("columns", 0),
                    file_size=result.get("file_size", len(file_content)),
                    downloads=0,
                )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            try:
                error_json = json.loads(error_body)
                raise RuntimeError(f"Upload failed ({e.code}): {error_json.get('detail', error_body)}")
            except json.JSONDecodeError:
                raise RuntimeError(f"Upload failed ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot connect to hub at {self.server_url}: {e}")


# Singleton instance
datasets = DatasetHub()
