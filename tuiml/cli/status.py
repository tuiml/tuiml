"""``tuiml status``: show every running tuiml-mcp instance and which
AI client launched it.

stdio MCP servers are spawned as child processes by their clients, so
"is tuiml running?" really means "are any clients keeping a tuiml-mcp
child alive?". This command answers that without any guessing: it walks
the process table, keeps the ``tuiml-mcp`` processes, and maps each one
back to the application that started it.
"""
from __future__ import annotations

import json as _json
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import click

# Best-effort map from a parent-process command line to a friendly client
# name. The lookup is by substring against ps -o command output.
_CLIENT_HINTS = [
    ("Claude.app",          "Claude Desktop"),
    ("Cursor.app",          "Cursor"),
    ("Codex.app",           "OpenAI Codex"),
    ("Code Helper",         "VS Code"),
    ("Code.app",            "VS Code"),
    ("Windsurf.app",        "Windsurf"),
    ("Zed.app",             "Zed"),
    ("Goose.app",           "Goose"),
    ("Perplexity.app",      "Perplexity"),
    ("ChatGPT.app",         "ChatGPT Desktop"),
    ("Antigravity",         "Antigravity"),
    ("claude-dev",          "Cline (VS Code)"),
    ("roo-cline",           "Roo Code (VS Code)"),
    ("kilo-code",           "Kilo Code (VS Code)"),
    ("opencode",            "OpenCode"),
    ("gemini",              "Gemini CLI"),
    ("openclaw",            "OpenClaw"),
    ("/claude",             "Claude Code"),
]


def _identify(parent_cmd: str) -> str:
    """Map a parent command line to a friendly AI client name.

    Parameters
    ----------
    parent_cmd : str
        Command line of the process that spawned a tuiml-mcp child.

    Returns
    -------
    client : str
        Matching label from ``_CLIENT_HINTS``, or ``"unknown"`` when no hint
        matches.
    """
    for needle, label in _CLIENT_HINTS:
        if needle in parent_cmd:
            return label
    return "unknown"


def _ps_rows() -> List[dict]:
    """Collect one row per running tuiml-mcp process.

    Shells out to ``ps``, keeps the processes whose command line mentions
    ``tuiml-mcp``, and resolves each one's parent (skipping wrapper processes
    such as ``python`` or ``node``) to name the AI client behind it.

    Returns
    -------
    rows : List[dict]
        One dict per process with keys ``pid``, ``ppid``, ``command``,
        ``parent_cmd``, and ``client``. Empty on Windows or when ``ps`` fails.
    """
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,command="],
            text=True, timeout=5,
        )
    except Exception:
        return []

    procs = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, cmd = parts[0], parts[1], parts[2]
        procs[pid] = {"pid": pid, "ppid": ppid, "command": cmd}

    rows = []
    for p in procs.values():
        if "tuiml-mcp" not in p["command"]:
            continue
        parent = procs.get(p["ppid"], {})
        # Walk up one more hop if the immediate parent is a wrapper like
        # "disclaimer" or python, gives a more useful client name.
        gparent = procs.get(parent.get("ppid", ""), {}) if parent else {}
        candidate = parent.get("command", "")
        if any(w in candidate for w in ("disclaimer", "/python", "/python3", "node ")):
            candidate = gparent.get("command", candidate)
        rows.append({
            **p,
            "parent_cmd": candidate or parent.get("command", "?"),
            "client":     _identify(candidate or parent.get("command", "")),
        })
    return rows


@click.command()
@click.option("--json", "as_json", is_flag=True,
              help="Emit raw JSON instead of the formatted table.")
def status(as_json: bool) -> None:
    """Show every running tuiml-mcp process and its parent AI client.

    Prints a table of process ids, the client that spawned each server (Claude
    Desktop, Cursor, Claude Code, ...), and the command line behind it, followed
    by the state of the tool-call trace log: whether tracing is on, where the
    file lives, and how large it has grown. An empty table means no AI client
    has loaded TuiML yet, which is the usual first thing to check after
    ``tuiml setup``.

    Examples
    --------
    List the running servers and the trace-log summary:

    $ tuiml status

    Feed the same information to another program:

    $ tuiml status --json

    Restart the servers this command lists, after an upgrade:

    $ tuiml update && tuiml restart
    """
    rows = _ps_rows()

    trace_path = Path(os.environ.get(
        "TUIML_MCP_TRACE_FILE",
        str(Path.home() / ".tuiml" / "logs" / "mcp.jsonl"),
    ))
    trace_info = {
        "enabled": os.environ.get("TUIML_MCP_TRACE", "1") != "0",
        "path":    str(trace_path),
        "exists":  trace_path.exists(),
        "size_kb": round(trace_path.stat().st_size / 1024, 1) if trace_path.exists() else 0,
    }

    if as_json:
        click.echo(_json.dumps({"instances": rows, "trace": trace_info}, indent=2))
        return

    if not rows:
        click.echo("No tuiml-mcp processes running.")
        click.echo("This usually means no MCP client has loaded tuiml yet.")
        click.echo("Open Claude Desktop / Cursor / etc. and re-run `tuiml status`.")
    else:
        click.echo(f"{'PID':>7}  {'CLIENT':<22}  COMMAND")
        click.echo("-" * 70)
        for r in rows:
            click.echo(f"{r['pid']:>7}  {r['client']:<22}  {r['command'][:120]}")
        click.echo()
        click.echo(f"{len(rows)} tuiml-mcp instance(s) running.")

    click.echo()
    click.echo("Trace log:")
    click.echo(f"  enabled: {trace_info['enabled']}  (TUIML_MCP_TRACE=0 to disable)")
    click.echo(f"  path:    {trace_info['path']}")
    if trace_info["exists"]:
        click.echo(f"  size:    {trace_info['size_kb']} KB")
        click.echo("  follow:  tuiml trace -f")
    else:
        click.echo("  (no calls logged yet)")
