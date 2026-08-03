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


def _detect_install_source() -> Dict[str, Any]:
    """Report which channel tuiml was installed from, via PEP 610 metadata.

    ``direct_url.json`` is written into the ``.dist-info`` only for installs
    that did not come from an index, which makes its absence the signal for a
    PyPI install. When present it records exactly what was installed: for a
    VCS install that includes the resolved commit, which is what lets an
    update check compare against a branch head rather than a version string.

    Returns
    -------
    source : dict
        ``kind`` is one of ``'pypi'``, ``'git'``, ``'editable'`` or
        ``'local'``. Git installs also carry ``url``, ``commit`` and
        ``requested_revision`` (the branch or tag asked for, if any).
    """
    import json as _json
    import importlib.metadata as _md

    try:
        raw = _md.distribution("tuiml").read_text("direct_url.json")
    except Exception:
        raw = None

    if not raw:
        # No direct_url.json => resolved from an index, i.e. PyPI.
        return {"kind": "pypi"}

    try:
        data = _json.loads(raw)
    except Exception as e:
        return {"kind": "unknown", "error": f"unreadable direct_url.json: {e}"}

    if "vcs_info" in data:
        vcs = data["vcs_info"]
        return {
            "kind": "git" if vcs.get("vcs") == "git" else vcs.get("vcs", "vcs"),
            "url": data.get("url"),
            "commit": vcs.get("commit_id"),
            "requested_revision": vcs.get("requested_revision"),
        }

    if data.get("dir_info", {}).get("editable"):
        return {"kind": "editable", "url": data.get("url")}

    return {"kind": "local", "url": data.get("url")}


def _query_latest_git_commit(url: str, ref: str = "main",
                             timeout: float = 5.0) -> Dict[str, Any]:
    """Resolve a remote ref to its current commit with ``git ls-remote``.

    Preferred over the GitHub API for the routine check: it needs no
    authentication and has no rate limit, where unauthenticated API calls are
    capped per IP and a long-lived MCP server can exhaust that budget.

    Parameters
    ----------
    url : str
        Repository URL. A ``git+`` prefix, as stored in PEP 610 metadata, is
        stripped before use.
    ref : str, default='main'
        Branch or tag name to resolve.
    timeout : float, default=5.0
        Seconds to wait for git before giving up.

    Returns
    -------
    result : dict
        On success: ``ok`` (True) and ``commit``. On failure: ``ok`` (False)
        and ``error``.
    """
    import subprocess

    clean = (url or "").removeprefix("git+")
    clean = clean.split("@", 1)[0] if clean.startswith("git+") else clean
    if not clean:
        return {"ok": False, "error": "no repository URL recorded"}

    try:
        proc = subprocess.run(
            ["git", "ls-remote", clean, f"refs/heads/{ref}"],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "git not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"git ls-remote timed out after {timeout}s"}

    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or "git ls-remote failed").strip()}

    out = proc.stdout.strip()
    if not out:
        return {"ok": False, "error": f"remote has no ref refs/heads/{ref}"}

    return {"ok": True, "commit": out.split()[0]}


def execute_system_info(**kwargs) -> Dict[str, Any]:
    """Report installation details for the running TuiML install.

    Backs the ``tuiml_system_info`` tool.

    Parameters
    ----------
    check_latest : bool, default=True
        Also check whether an update is available, against whichever channel
        this install came from (arrives via ``**kwargs``).

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``version``,
        ``install_method``, ``install_source``, ``upgrade_hint``,
        ``package_path``, ``site_packages``, ``python_executable``,
        ``python_version``, ``platform``, ``session_seed``, plus
        ``installed_commit`` for a VCS install.

        When ``check_latest``, the comparison follows ``install_source``:

        - ``pypi`` — ``latest_version``, ``update_available`` and
          ``latest_released`` (or ``latest_version_error``).
        - ``git`` — ``tracking_ref``, ``latest_commit`` and
          ``update_available`` (or ``latest_commit_error``), since a branch
          install moves without the version string changing.
        - ``editable`` — ``update_available`` is False; a dev checkout is
          updated with ``git pull``.

        On failure: ``status`` (``'error'``), ``error`` and ``error_type``.
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

    # The seed every unseeded call in this session runs under. Reporting it is
    # what makes a session reproducible from the outside: quote it back as
    # `random_seed` (or export TUIML_SEED before starting the server) and the
    # numbers repeat. Note this is the *default*, not necessarily the seed of
    # the last call, which may have passed its own.
    from .._state import get_session_seed

    install = _detect_install_method()
    result: Dict[str, Any] = {
        "status": "success",
        "version": version,
        "session_seed": get_session_seed(),
        "install_method": install["method"],
        "upgrade_hint": install["upgrade_hint"],
        "package_path": str(pkg_dir),
        "site_packages": str(pkg_dir.parent),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": _plat.platform(),
    }

    source = _detect_install_source()
    result["install_source"] = source["kind"]
    if source.get("commit"):
        result["installed_commit"] = source["commit"]

    # A git install has to be refreshed from its own remote. The default hint
    # names the PyPI package, so following it would quietly move the install
    # onto the released channel instead of updating it in place.
    src_url = source.get("url")
    if source["kind"] == "git" and src_url:
        if install["method"] == "uv-tool":
            result["upgrade_hint"] = (
                f'uv tool install --reinstall --force "tuiml @ {src_url}"'
            )
        elif install["method"] == "pip":
            result["upgrade_hint"] = (
                f'{sys.executable} -m pip install --upgrade --force-reinstall "{src_url}"'
            )

    if kwargs.get("check_latest", True):
        # A git install tracks a branch, not a version: main's pyproject and
        # the last PyPI release usually carry the same version string, so
        # comparing versions would report "up to date" however far behind the
        # checkout is. Compare commits instead.
        if source["kind"] == "git":
            ref = source.get("requested_revision") or "main"
            remote = _query_latest_git_commit(source.get("url", ""), ref)
            result["tracking_ref"] = ref
            if remote["ok"]:
                result["latest_commit"] = remote["commit"]
                installed = source.get("commit")
                result["update_available"] = bool(
                    installed and remote["commit"] != installed
                )
            else:
                result["latest_commit_error"] = remote["error"]
        elif source["kind"] == "editable":
            # A dev checkout is updated with git pull, not by us; claiming an
            # update is available against PyPI would be noise.
            result["update_available"] = False
        else:
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
