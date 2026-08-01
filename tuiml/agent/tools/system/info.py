"""Installation introspection."""

from pathlib import Path
from typing import Any, Dict

from .._spec import ToolSpec


def _detect_install_method() -> Dict[str, Any]:
    """Inspect sys.prefix / sys.executable to guess how tuiml was installed."""
    import sys
    # Don't resolve(): that follows the python symlink out of the tool venv.
    prefix = sys.prefix.replace("\\", "/")
    exe = sys.executable.replace("\\", "/")

    # Editable install wins over path-matching: check first so a dev checkout
    # imported into any venv is still reported as editable-dev.
    try:
        import tuiml as _pkg
        pkg_dir = Path(_pkg.__file__).resolve().parent
        if (pkg_dir.parent / "pyproject.toml").exists():
            return {"method": "editable-dev", "writable": False,
                    "upgrade_hint": "cd <checkout> && git pull"}
    except Exception:
        pass

    # uv tool install puts the venv under .../uv/tools/tuiml/
    if "/uv/tools/tuiml" in prefix or "/uv/tools/tuiml" in exe:
        return {"method": "uv-tool", "writable": True,
                "upgrade_hint": "uv tool install --reinstall --force tuiml"}

    # Default: assume pip / uv pip
    return {"method": "pip", "writable": True,
            "upgrade_hint": f"{sys.executable} -m pip install --upgrade tuiml"}


def _query_latest_pypi_version(package: str = "tuiml", timeout: float = 5.0) -> Dict[str, Any]:
    """Look up the latest released version of a package on PyPI.

    Parameters
    ----------
    package : str, default='tuiml'
        PyPI package name to query.
    timeout : float, default=5.0
        HTTP timeout in seconds.

    Returns
    -------
    result : dict
        On success: ``ok`` (True), ``version`` and ``released`` (upload
        timestamp). On failure: ``ok`` (False) and ``error``.
    """
    import json as _json
    import urllib.request
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "version": data["info"]["version"],
                "released": data["releases"].get(data["info"]["version"], [{}])[0].get("upload_time")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def execute_system_info(**kwargs) -> Dict[str, Any]:
    """Report installation details for the running TuiML install.

    Backs the ``tuiml_system_info`` tool.

    Parameters
    ----------
    check_latest : bool, default=True
        Also query PyPI for the latest released version (arrives via
        ``**kwargs``).

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``version``,
        ``install_method``, ``upgrade_hint``, ``package_path``,
        ``site_packages``, ``python_executable``, ``python_version``,
        ``platform``, and -- when ``check_latest`` -- ``latest_version``
        and ``update_available`` (or ``latest_version_error``). On
        failure: ``status`` (``'error'``), ``error`` and ``error_type``.
    """
    import sys
    import platform as _plat
    try:
        import tuiml as _pkg
        pkg_dir = Path(_pkg.__file__).resolve().parent
        version = getattr(_pkg, "__version__", "unknown")
    except Exception as e:
        return {"status": "error", "error": f"cannot import tuiml: {e}",
                "error_type": type(e).__name__}

    install = _detect_install_method()
    result: Dict[str, Any] = {
        "status": "success",
        "version": version,
        "install_method": install["method"],
        "upgrade_hint": install["upgrade_hint"],
        "package_path": str(pkg_dir),
        "site_packages": str(pkg_dir.parent),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": _plat.platform(),
    }

    if kwargs.get("check_latest", True):
        pypi = _query_latest_pypi_version()
        if pypi["ok"]:
            latest = pypi["version"]
            result["latest_version"] = latest
            result["update_available"] = (latest != version)
            if pypi.get("released"):
                result["latest_released"] = pypi["released"]
        else:
            result["latest_version_error"] = pypi["error"]

    return result


SPEC = ToolSpec(
    name='tuiml_system_info',
    description="Report details about the TuiML installation on this machine: "
        "installed version, install method (uv tool / pip / editable), "
        "package location, Python executable, platform, and the latest "
        "version available on PyPI. Agents can use this to decide whether "
        "an update is worth running via tuiml_self_update.",
    input_schema={
            "type": "object",
            "properties": {
                "check_latest": {
                    "type": "boolean",
                    "description": "Query PyPI for the latest released version. Defaults to true.",
                    "default": True,
                }
            },
        },
    # No dedicated output schema; falls back to COMPONENT_OUTPUT_SCHEMA.
    output_schema=None,
    execute=execute_system_info,
    group='workflow',
    read_only=True, destructive=False,
    idempotent=True, open_world=True,
    reproducible=False,
)
