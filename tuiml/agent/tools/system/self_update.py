"""In-place package upgrade."""

from typing import Any, Dict

from .._spec import ToolSpec
from .info import (
    _detect_install_method,
    _detect_install_source,
    execute_system_info,
)


def execute_self_update(**kwargs) -> Dict[str, Any]:
    """Upgrade tuiml to the latest PyPI version using the detected installer.

    Backs the ``tuiml_self_update`` tool. Editable/dev checkouts are
    refused; use ``git pull`` there instead.

    Parameters
    ----------
    target_version : str, default=None
        Specific version to install; defaults to the latest release
        (arrives via ``**kwargs``, like all parameters below).
    dry_run : bool, default=False
        Only report the command that would run; make no changes.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``install_method``,
        ``command``, ``returncode``, ``stdout``, ``stderr``,
        ``version_before``, ``version_after``, ``restart_required`` and
        ``note`` (dry runs return ``dry_run`` and ``command`` only). On
        failure: ``status`` (``'error'``), ``error`` and ``error_type``.

        A dry run always succeeds: in an editable checkout, where a real
        upgrade is refused, it reports that refusal as its prediction with
        ``command`` set to None, rather than failing.
    """
    import subprocess
    import sys

    install = _detect_install_method()
    method = install["method"]

    if method == "editable-dev":
        # A dry run promises to report what would happen and change nothing, so
        # it answers rather than failing: what *would* happen here is a refusal.
        if kwargs.get("dry_run"):
            return {
                "status": "success",
                "dry_run": True,
                "install_method": method,
                "command": None,
                "note": "would refuse: this is an editable / dev checkout, "
                        "run `git pull` in the source tree instead",
            }
        return {
            "status": "error",
            "error": "refusing to upgrade an editable / dev checkout, run `git pull` in the source tree instead",
            "error_type": "EditableInstallError",
            "install_method": method,
        }

    # Update along whichever channel this install came from. Always naming the
    # PyPI package would quietly move a git install onto the released channel:
    # the upgrade "succeeds", reports the same version on both sides because
    # main and the last release share a version string, and the user is left
    # on a different channel than the one they chose.
    source = _detect_install_source()
    target = kwargs.get("target_version")

    if source["kind"] == "git" and source.get("url") and not target:
        spec = f"tuiml @ {source['url']}"
    else:
        spec = f"tuiml=={target}" if target else "tuiml"

    if source["kind"] == "git" and target:
        return {
            "status": "error",
            "error": (
                "this is a git install tracking "
                f"{source.get('requested_revision') or 'main'}; it has no PyPI "
                f"version to pin to. Re-install from PyPI to use target_version, "
                f"or drop it to pull the latest commit."
            ),
            "error_type": "ChannelMismatchError",
            "install_method": method,
            "install_source": source["kind"],
        }

    if method == "uv-tool":
        cmd = ["uv", "tool", "install", "--reinstall", "--force", spec]
    else:  # pip
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", spec]

    if kwargs.get("dry_run"):
        return {
            "status": "success",
            "dry_run": True,
            "install_method": method,
            "command": cmd,
            "note": "no changes made; set dry_run=false to actually run",
        }

    try:
        before = execute_system_info(check_latest=False).get("version")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "upgrade timed out after 300s",
                "error_type": "TimeoutExpired", "command": cmd}
    except FileNotFoundError as e:
        return {"status": "error",
                "error": f"installer not found on PATH: {e.filename or str(e)}",
                "error_type": "FileNotFoundError", "command": cmd}

    # After upgrade the on-disk package has changed, but the running process is
    # still holding the old module objects. Re-reading what landed requires a
    # subprocess. The commit is read alongside the version because on the git
    # channel the version alone cannot show that anything moved.
    probe_src = (
        "import importlib, json, importlib.metadata as m; "
        "importlib.invalidate_caches(); "
        "d = m.distribution('tuiml'); "
        "raw = d.read_text('direct_url.json') or ''; "
        "vcs = (json.loads(raw).get('vcs_info') or {}) if raw else {}; "
        "print(m.version('tuiml')); print(vcs.get('commit_id') or '')"
    )
    after = after_commit = None
    try:
        probe = subprocess.run(
            [sys.executable, "-c", probe_src],
            capture_output=True, text=True, timeout=15,
        )
        lines = probe.stdout.strip().splitlines()
        after = lines[0].strip() if lines else None
        after_commit = lines[1].strip() if len(lines) > 1 and lines[1].strip() else None
    except Exception:
        pass

    ok = (proc.returncode == 0)
    result: Dict[str, Any] = {
        "status": "success" if ok else "error",
        "install_method": method,
        "install_source": source["kind"],
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "version_before": before,
        "version_after": after,
        "restart_required": ok,
        "note": "Call tuiml_restart to reload the MCP servers (or restart the client manually).",
    }

    # Only meaningful on the git channel, where two different commits routinely
    # carry the same version string.
    if source.get("commit") or after_commit:
        result["commit_before"] = source.get("commit")
        result["commit_after"] = after_commit

    return result


SPEC = ToolSpec(
    name='tuiml_self_update',
    description="Upgrade the TuiML installation to the latest PyPI release. "
        "Auto-detects the installer (uv tool install vs pip) and runs the "
        "appropriate upgrade command. Returns the command, its stdout / "
        "stderr, and the resulting version. Editable / dev installs are "
        "refused, those need a git pull instead.\n\n"
        "IMPORTANT: the running MCP process is still using the old package "
        "in memory. Call tuiml_restart immediately after, or restart the "
        "client manually, for the new version to take effect.",
    input_schema={
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "description": "Print the command that would be run without executing it.",
                    "default": False,
                },
                "target_version": {
                    "type": "string",
                    "description": "Upgrade to a specific version (e.g. '0.1.3'). Defaults to latest.",
                },
            },
        },
    # No dedicated output schema; falls back to COMPONENT_OUTPUT_SCHEMA.
    output_schema=None,
    execute=execute_self_update,
    group='workflow',
    read_only=False, destructive=True,
    idempotent=False, open_world=True,
    reproducible=False,
)
