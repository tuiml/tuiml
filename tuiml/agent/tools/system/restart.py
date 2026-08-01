"""Restarting tuiml-mcp processes.

stdio MCP servers are spawned as child processes by their parent AI
clients (Claude Desktop, Cursor, ...). When the child exits, every
mainstream client respawns it on the next request. So "restart" really
means "kill the running children and let the clients respawn them with
the freshly installed code".

Used by both `tuiml restart` (CLI) and the `tuiml_restart` MCP tool.
"""

import os
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional

from .._spec import ToolSpec


def find_mcp_processes(exclude_self: bool = True) -> List[Dict]:
    """Return a list of running tuiml-mcp processes.

    Parameters
    ----------
    exclude_self : bool, default=True
        When True, omit the current process so a tuiml_restart MCP call
        running inside one of the targets doesn't kill itself before
        returning a response.

    Returns
    -------
    procs : List[Dict]
        One dict per process with keys ``pid`` (int), ``ppid`` (int), and
        ``command`` (str). Empty list on Windows or if ``ps`` fails.
    """
    self_pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,command="],
            text=True, timeout=5,
        )
    except Exception:
        return []

    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_s, ppid_s, cmd = parts
        if "tuiml-mcp" not in cmd:
            continue
        try:
            pid = int(pid_s)
            ppid = int(ppid_s)
        except ValueError:
            continue
        if exclude_self and pid == self_pid:
            continue
        rows.append({"pid": pid, "ppid": ppid, "command": cmd})
    return rows


def kill_mcp_processes(
    procs: Optional[List[Dict]] = None,
    grace_seconds: float = 2.0,
    include_self: bool = False,
    self_delay_seconds: float = 0.5,
) -> Dict:
    """Send SIGTERM to each tuiml-mcp process; SIGKILL after grace.

    Parameters
    ----------
    procs : list of dict, optional
        Explicit list of process dicts to kill. If None, the function
        re-discovers running tuiml-mcp processes itself (excluding the
        current one).
    grace_seconds : float, default=2.0
        How long to wait between SIGTERM and SIGKILL for each PID.
    include_self : bool, default=False
        If True, schedule a delayed self-exit AFTER killing other
        processes. Used by the MCP tool so the agent receives the
        response before the server dies.
    self_delay_seconds : float, default=0.5
        Delay before the deferred self-exit (allows the caller to
        flush a response).

    Returns
    -------
    result : Dict
        Dict with keys ``killed`` (list of PIDs successfully terminated),
        ``failed`` (list of dicts with ``pid`` and ``error``), and
        ``self_exit_scheduled`` (bool).
    """
    if procs is None:
        procs = find_mcp_processes(exclude_self=True)

    killed: List[int] = []
    failed: List[Dict] = []

    for p in procs:
        pid = p["pid"]
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            # Already gone, fine
            continue
        except PermissionError as e:
            failed.append({"pid": pid, "error": f"permission denied: {e}"})
            continue
        except Exception as e:
            failed.append({"pid": pid, "error": str(e)})
            continue

        # Wait for graceful exit, then SIGKILL if still alive
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)  # signal 0 = exists check
            except ProcessLookupError:
                killed.append(pid)
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                killed.append(pid)
            except Exception as e:
                failed.append({"pid": pid, "error": f"SIGKILL failed: {e}"})

    self_exit = False
    if include_self:
        # Defer the self-exit so the calling MCP response can be flushed.
        import threading

        def _delayed_exit():
            time.sleep(self_delay_seconds)
            os._exit(0)

        threading.Thread(target=_delayed_exit, daemon=True).start()
        self_exit = True

    return {
        "killed": killed,
        "failed": failed,
        "self_exit_scheduled": self_exit,
    }


def execute_restart(**kwargs) -> Dict[str, Any]:
    """Restart every running tuiml-mcp child process.

    When called from an MCP context the current server is one of those
    children. We schedule a deferred self-exit (after a short delay)
    so this response can be flushed back to the agent before the
    process dies; the parent client (Claude Desktop, Cursor, …) will
    auto-respawn the child with the newly installed code on its next
    request.

    Parameters
    ----------
    include_self : bool, default=True
        Also schedule a deferred exit of the current server process
        (arrives via ``**kwargs``).

    Returns
    -------
    result : dict
        ``status`` (``'success'``), ``killed_other`` (count),
        ``failed``, ``self_exit_scheduled`` and ``note``.
    """
    include_self = kwargs.get("include_self", True)
    others = find_mcp_processes(exclude_self=True)
    result = kill_mcp_processes(
        procs=others,
        grace_seconds=2.0,
        include_self=include_self,
        self_delay_seconds=0.5,
    )

    return {
        "status": "success",
        "killed_other": result["killed"],
        "failed": result["failed"],
        "self_exit_scheduled": result["self_exit_scheduled"],
        "note": (
            "Clients automatically respawn their tuiml-mcp child on the next "
            "request. If you called this right after tuiml_self_update, the "
            "new version will be loaded then."
        ),
    }


SPEC = ToolSpec(
    name='tuiml_restart',
    description="Restart every running tuiml-mcp process so AI clients pick up "
        "freshly installed code (e.g. right after tuiml_self_update). "
        "Sends SIGTERM (then SIGKILL after a grace period) to every "
        "tuiml-mcp child; each parent client (Claude Desktop, Cursor, "
        "Codex, ...) automatically respawns its child on the next "
        "request. The current MCP process schedules a self-exit AFTER "
        "this response is sent, so the agent should expect a brief "
        "reconnect.",
    input_schema={
            "type": "object",
            "properties": {
                "include_self": {
                    "type": "boolean",
                    "description": (
                        "Also exit the current MCP process after responding. "
                        "True is the usual case, it forces the calling "
                        "client to respawn with the new code too. Set False "
                        "to restart only the other clients' instances."
                    ),
                    "default": True,
                },
            },
        },
    # No dedicated output schema; falls back to COMPONENT_OUTPUT_SCHEMA.
    output_schema=None,
    execute=execute_restart,
    group='workflow',
    read_only=False, destructive=True,
    idempotent=False, open_world=False,
    reproducible=False,
)
