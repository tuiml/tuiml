"""Helpers for restarting tuiml-mcp processes.

stdio MCP servers are spawned as child processes by their parent AI
clients (Claude Desktop, Cursor, …). When the child exits, every
mainstream client respawns it on the next request. So "restart" really
means "kill the running children and let the clients respawn them with
the freshly installed code".

Used by both `tuiml restart` (CLI) and `tuiml_restart` (MCP tool).
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Dict, List, Optional


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
