"""TuiML Hub Remote Client for downloading and using algorithms."""

import os
import sys
import json
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Type

from tuiml.hub.types import AlgorithmInfo

# Default server URL
DEFAULT_SERVER = "http://localhost:8000"

# Local cache directory
CACHE_DIR = Path.home() / ".tuiml" / "algorithms"


class RemoteHub:
    """Client for interacting with the remote TuiML Hub.
    
    Provides methods to browse, download, and use algorithms from the hub.

    Examples
    --------
    >>> from tuiml.hub.remote import remote
    >>> # Browse algorithms
    >>> for algo in remote.browse():
    ...     print(algo.name, algo.description)
    >>> # Install and use
    >>> remote.install("my_algo")
    >>> MyAlgo = remote.load("my_algo")
    """
    
    def __init__(self, server_url: str = None):
        """Initialize the remote hub client.

        Parameters
        ----------
        server_url : str, optional, default=None
            URL of the TuiML Hub server. Defaults to ``TUIML_HUB_URL`` 
            environment variable or localhost.
        """
        self.server_url = server_url or os.environ.get("TUIML_HUB_URL", DEFAULT_SERVER)
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache of loaded modules
        self._loaded_modules: Dict[str, Any] = {}
    
    def _make_request(self, endpoint: str, method: str = "GET") -> Any:
        """Make HTTP request to the hub API."""
        import urllib.request
        import urllib.error
        
        url = f"{self.server_url}/api{endpoint}"
        
        try:
            req = urllib.request.Request(url, method=method)
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
    
    def _resolve_id(self, name: str) -> str:
        """Resolve an algorithm name to its hub ID.

        Parameters
        ----------
        name : str
            Algorithm name.

        Returns
        -------
        algorithm_id : str
            The hub ID for the algorithm.
        """
        data = self._make_request("/algorithms?limit=100")

        for algo in data.get("algorithms", data):
            if algo.get("name") == name:
                return str(algo["id"])

        raise ValueError(f"Algorithm '{name}' not found on hub")

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file from URL to destination."""
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    
    def browse(
        self, 
        algorithm_type: str = None, 
        category: str = None,
        limit: int = 50
    ) -> List[AlgorithmInfo]:
        """Browse algorithms available on the remote hub.

        Parameters
        ----------
        algorithm_type : str, optional, default=None
            Filter by algorithm type (e.g., 'classifier').
        category : str, optional, default=None
            Filter by category (e.g., 'ensemble').
        limit : int, default=50
            Maximum number of results to return.

        Returns
        -------
        algorithms : List[AlgorithmInfo]
            List of algorithm information objects.

        Examples
        --------
        >>> classifiers = remote.browse(algorithm_type="classifier")
        """
        endpoint = f"/algorithms?limit={limit}"
        
        if algorithm_type:
            endpoint += f"&type={algorithm_type}"
        if category:
            endpoint += f"&category={category}"
        
        data = self._make_request(endpoint)
        
        return [
            AlgorithmInfo(
                id=algo.get("id"),
                name=algo.get("name"),
                display_name=algo.get("display_name"),
                description=algo.get("description", ""),
                algorithm_type=algo.get("algorithm_type"),
                category=algo.get("category"),
                version=algo.get("version", "1.0.0"),
                author=algo.get("author"),
                tags=algo.get("tags", []),
                downloads=algo.get("downloads", 0),
            )
            for algo in data.get("algorithms", data)  # Handle both list and paginated response
        ]
    
    def search(self, query: str) -> List[AlgorithmInfo]:
        """Search for algorithms by keyword.

        Parameters
        ----------
        query : str
            Search query string (matches name, description, tags).

        Returns
        -------
        results : List[AlgorithmInfo]
            List of matching algorithm information objects.

        Examples
        --------
        >>> results = remote.search("random forest")
        """
        data = self._make_request("/algorithms?limit=100")

        query_lower = query.lower()
        return [
            AlgorithmInfo(
                id=algo.get("id"),
                name=algo.get("name"),
                display_name=algo.get("display_name"),
                description=algo.get("description", ""),
                algorithm_type=algo.get("algorithm_type"),
                category=algo.get("category"),
                version=algo.get("version", "1.0.0"),
                author=algo.get("author"),
                tags=algo.get("tags", []),
                downloads=algo.get("downloads", 0),
            )
            for algo in data.get("algorithms", data)
            if query_lower in algo.get("name", "").lower()
            or query_lower in algo.get("display_name", "").lower()
            or query_lower in algo.get("description", "").lower()
        ]
    
    def info(self, name: str) -> AlgorithmInfo:
        """Get detailed information about an algorithm.

        Parameters
        ----------
        name : str
            The name of the algorithm.

        Returns
        -------
        info : AlgorithmInfo
            Detailed information about the algorithm.
        """
        algo_id = self._resolve_id(name)
        data = self._make_request(f"/algorithms/{algo_id}")
        
        return AlgorithmInfo(
            id=data.get("id"),
            name=data.get("name"),
            display_name=data.get("display_name"),
            description=data.get("description", ""),
            algorithm_type=data.get("algorithm_type"),
            category=data.get("category"),
            version=data.get("version", "1.0.0"),
            author=data.get("author"),
            tags=data.get("tags", []),
            downloads=data.get("downloads", 0),
        )
    
    def install(
        self, 
        name: str, 
        force: bool = False,
        version: str = None
    ) -> Path:
        """Download and install an algorithm from the hub.

        Downloads algorithm files to ``~/.tuiml/algorithms/{name}/``.

        Parameters
        ----------
        name : str
            The name of the algorithm to install.
        force : bool, default=False
            If True, reinstall even if the algorithm is already installed.
        version : str, optional, default=None
            Specific version to install. If None, installs the latest version.

        Returns
        -------
        path : Path
            The path to the installed algorithm directory.

        Examples
        --------
        >>> remote.install("my_random_forest")
        """
        algo_dir = self.cache_dir / name
        config_file = algo_dir / "tuiml.json"
        
        # Check if already installed
        if config_file.exists() and not force:
            print(f"Algorithm '{name}' is already installed. Use force=True to reinstall.")
            return algo_dir
        
        print(f"Installing '{name}' from hub...")

        # Resolve name to hub ID
        try:
            algo_id = self._resolve_id(name)
        except (ValueError, RuntimeError) as e:
            raise ValueError(f"Algorithm '{name}' not found on hub: {e}")

        # Get algorithm info
        try:
            algo_info = self._make_request(f"/algorithms/{algo_id}")
        except RuntimeError as e:
            raise ValueError(f"Algorithm '{name}' not found on hub: {e}")

        # Get file list
        try:
            files_response = self._make_request(f"/algorithms/{algo_id}/files")
            files = files_response.get("files", [])
        except RuntimeError:
            files = []

        # Create algorithm directory
        algo_dir.mkdir(parents=True, exist_ok=True)

        # Download files (skip directories)
        downloaded = 0
        for file_info in files:
            if file_info.get("type") == "directory":
                continue
            file_path = file_info.get("path", file_info.get("name"))

            # The file endpoint returns JSON with a "content" field
            file_data = self._make_request(
                f"/algorithms/{algo_id}/file/{file_path}"
            )
            content = file_data.get("content", "")

            dest = algo_dir / file_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            print(f"  Downloaded {file_path} ({len(content)} bytes)")
            downloaded += 1

        if downloaded == 0:
            raise ValueError(
                f"Algorithm '{name}' has no source files uploaded on the hub. "
                f"The author needs to upload files first."
            )

        # Save config
        config = {
            "name": algo_info.get("name"),
            "display_name": algo_info.get("display_name"),
            "description": algo_info.get("description"),
            "type": algo_info.get("algorithm_type"),
            "category": algo_info.get("category"),
            "version": algo_info.get("version", "1.0.0"),
            "main": algo_info.get("main", "algorithm.py"),
            "installed_from": self.server_url,
        }
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Installed '{name}' to {algo_dir}")
        return algo_dir
    
    def is_installed(self, name: str) -> bool:
        """Check if an algorithm is installed locally.

        Parameters
        ----------
        name : str
            The name of the algorithm.

        Returns
        -------
        installed : bool
            True if the algorithm is installed, False otherwise.
        """
        config_file = self.cache_dir / name / "tuiml.json"
        return config_file.exists()
    
    def load(self, name: str, auto_install: bool = True) -> Type:
        """Load an algorithm class from the cache.

        If the algorithm is not installed and ``auto_install`` is True, 
        it will be installed first.

        Parameters
        ----------
        name : str
            The name of the algorithm to load.
        auto_install : bool, default=True
            If True, automatically install the algorithm if it is not 
            found locally.

        Returns
        -------
        cls : Type
            The loaded algorithm class, ready to be instantiated.

        Examples
        --------
        >>> MyAlgo = remote.load("my_random_forest")
        >>> model = MyAlgo(n_estimators=100)
        """
        # Check if already loaded
        if name in self._loaded_modules:
            return self._loaded_modules[name]
        
        algo_dir = self.cache_dir / name
        config_file = algo_dir / "tuiml.json"
        
        # Install if needed
        if not config_file.exists():
            if auto_install:
                self.install(name)
            else:
                raise ValueError(
                    f"Algorithm '{name}' not installed. "
                    f"Run remote.install('{name}') first."
                )
        
        # Load config
        with open(config_file) as f:
            config = json.load(f)
        
        # Find main file
        main_file = config.get("main", "algorithm.py")
        main_path = algo_dir / main_file
        
        if not main_path.exists():
            # Try to find any .py file
            py_files = list(algo_dir.glob("*.py"))
            if py_files:
                main_path = py_files[0]
            else:
                raise ValueError(f"No Python files found in {algo_dir}")
        
        # Load the module dynamically
        module_name = f"tuiml_hub_{name}"
        spec = importlib.util.spec_from_file_location(module_name, main_path)
        module = importlib.util.module_from_spec(spec)
        
        # Add algorithm dir to path for relative imports
        if str(algo_dir) not in sys.path:
            sys.path.insert(0, str(algo_dir))
        
        try:
            spec.loader.exec_module(module)
        finally:
            # Remove from path
            if str(algo_dir) in sys.path:
                sys.path.remove(str(algo_dir))
        
        # Find the main class (look for class with same name as algorithm)
        class_name = "".join(word.title() for word in name.split("_"))
        
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
        else:
            # Find first class that looks like an algorithm
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and hasattr(attr, "fit"):
                    cls = attr
                    break
            else:
                raise ValueError(
                    f"Could not find algorithm class in {main_path}. "
                    f"Expected class named '{class_name}' or a class with 'fit' method."
                )
        
        # Cache the loaded class
        self._loaded_modules[name] = cls
        return cls
    
    def use(self, name: str, **kwargs) -> Any:
        """Load and instantiate an algorithm in one step.

        This is a convenience method that combines ``load()`` and 
        instantiation.

        Parameters
        ----------
        name : str
            The name of the algorithm to use.
        **kwargs : Any
            Arguments passed to the algorithm's constructor.

        Returns
        -------
        instance : Any
            An instance of the requested algorithm.

        Examples
        --------
        >>> model = remote.use("my_random_forest", n_estimators=100)
        """
        cls = self.load(name)
        return cls(**kwargs)
    
    def uninstall(self, name: str) -> bool:
        """Remove an installed algorithm.

        Parameters
        ----------
        name : str
            The name of the algorithm to uninstall.

        Returns
        -------
        success : bool
            True if uninstallation was successful, False otherwise.
        """
        import shutil
        
        algo_dir = self.cache_dir / name
        
        if not algo_dir.exists():
            return False
        
        shutil.rmtree(algo_dir)
        
        # Remove from loaded cache
        if name in self._loaded_modules:
            del self._loaded_modules[name]
        
        print(f"✓ Uninstalled '{name}'")
        return True
    
    def installed(self) -> List[str]:
        """List all locally installed algorithms.

        Returns
        -------
        installed : List[str]
            A list of names of all installed algorithms.
        """
        algorithms = []
        
        for algo_dir in self.cache_dir.iterdir():
            if algo_dir.is_dir():
                config_file = algo_dir / "tuiml.json"
                if config_file.exists():
                    algorithms.append(algo_dir.name)
        
        return algorithms
    
    def update(self, name: str = None) -> None:
        """Update installed algorithm(s) to the latest version.

        Parameters
        ----------
        name : str, optional, default=None
            The name of the specific algorithm to update. If None, updates 
            all installed algorithms.
        """
        if name:
            if self.is_installed(name):
                self.install(name, force=True)
            else:
                print(f"Algorithm '{name}' is not installed.")
        else:
            # Update all
            for algo_name in self.installed():
                print(f"Updating {algo_name}...")
                self.install(algo_name, force=True)


# Singleton instance
remote = RemoteHub()


# Convenience function for quick imports
def use(name: str, **kwargs) -> Any:
    """Quick way to import and use a hub algorithm.

    Parameters
    ----------
    name : str
        The name of the algorithm to use.
    **kwargs : Any
        Arguments passed to the algorithm's constructor.

    Returns
    -------
    instance : Any
        An instance of the requested algorithm.

    Examples
    --------
    >>> from tuiml.hub.remote import use
    >>> model = use("my_random_forest", n_estimators=100)
    """
    return remote.use(name, **kwargs)
