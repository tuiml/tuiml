"""``tuiml trace``: view or live-tail MCP tool-call traces.

Every MCP call from an AI client is appended to ``~/.tuiml/logs/mcp.jsonl``
by the running tuiml-mcp server, one JSON record per line: a ``call`` record
when a tool starts and a ``return`` record when it finishes. This command
pretty-prints those records and can follow new entries in real time.
"""
from __future__ import annotations

import json as _json
import os
import time as _time
from pathlib import Path

import click


def _default_path() -> Path:
    """Return the trace file path used when ``--path`` is not given.

    Returns
    -------
    path : Path
        ``$TUIML_MCP_TRACE_FILE`` when set, otherwise
        ``~/.tuiml/logs/mcp.jsonl``.
    """
    return Path(os.environ.get(
        "TUIML_MCP_TRACE_FILE",
        str(Path.home() / ".tuiml" / "logs" / "mcp.jsonl"),
    ))


def _format_record(rec: dict) -> str:
    """Render one trace record as a colourised terminal line.

    ``call`` records show the tool name and the first few arguments; ``return``
    records show a status marker, the duration, and either the error or the keys
    of the returned payload.

    Parameters
    ----------
    rec : dict
        One decoded JSONL record, with keys such as ``ts``, ``pid``, ``tool``,
        ``phase``, ``args``, ``duration_ms``, ``error``, and ``summary``.

    Returns
    -------
    line : str
        Single line of text including ANSI colour escapes.
    """
    ts = rec.get("ts", "")
    pid = rec.get("pid", "?")
    tool = rec.get("tool", "?")
    phase = rec.get("phase", "?")

    if phase == "call":
        args = rec.get("args", {})
        arg_summary = ", ".join(f"{k}={v}" for k, v in list(args.items())[:5])
        if len(args) > 5:
            arg_summary += f", … +{len(args) - 5} more"
        return f"\033[36m{ts}\033[0m  pid={pid}  \033[1m→ {tool}\033[0m({arg_summary})"

    # phase == "return"
    dur = rec.get("duration_ms", "?")
    err = rec.get("error")
    summary = rec.get("summary", {})
    status = summary.get("status", "?")
    if err:
        color, marker = "\033[31m", "✗"
        body = f"error: {err}"
    elif status == "success":
        color, marker = "\033[32m", "✓"
        body = f"keys=[{', '.join(summary.get('keys', []))}]"
    else:
        color, marker = "\033[33m", "!"
        body = f"status={status}"
    return f"\033[36m{ts}\033[0m  pid={pid}  {color}{marker}\033[0m {tool}  {dur}ms  {body}"


@click.command()
@click.option("-f/-F", "--follow/--no-follow", default=True,
              help="Follow new entries in real time (default: on). --no-follow to print and exit.")
@click.option("-n", "--lines", type=int, default=50,
              help="Number of recent records to print before following (default: 50).")
@click.option("--tool", "tool_filter", metavar="NAME",
              help="Only show records for this tool, e.g. tuiml_train.")
@click.option("--json", "as_json", is_flag=True,
              help="Emit the raw JSONL records instead of formatted lines.")
@click.option("--clear", is_flag=True,
              help="Delete the trace log file and exit without printing.")
@click.option("--path", "path_override", type=click.Path(),
              help="Path to the trace file. Defaults to "
                   "$TUIML_MCP_TRACE_FILE or ~/.tuiml/logs/mcp.jsonl.")
def trace(follow: bool, lines: int, tool_filter: str | None,
          as_json: bool, clear: bool, path_override: str | None) -> None:
    """View or follow MCP tool-call traces (live by default, Ctrl-C to stop).

    Prints the most recent records from the trace log, then keeps following new
    ones as connected AI clients call TuiML tools; ``--no-follow`` prints the
    tail and exits instead. Each entry shows the timestamp, the server process
    id, the tool name, and either its arguments or its result and duration. If
    the log does not exist yet, the command waits for it rather than failing, so
    it can be started before the first tool call.

    Tracing is on by default and is disabled by setting ``TUIML_MCP_TRACE=0``
    for the process that launches the server.

    Examples
    --------
    Follow tool calls as they happen:

    $ tuiml trace -f

    Print the last 100 records and exit:

    $ tuiml trace --no-follow -n 100

    Watch a single tool, as raw JSONL for further processing:

    $ tuiml trace --tool tuiml_train --json

    Start a clean log before reproducing a problem:

    $ tuiml trace --clear
    """
    path = Path(path_override) if path_override else _default_path()

    if clear:
        if path.exists():
            path.unlink()
            click.echo(f"Removed {path}")
        else:
            click.echo("Nothing to remove.")
        return

    if not path.exists():
        if not follow:
            click.echo(f"No trace log at {path}", err=True)
            click.echo("Trigger any MCP tool call from a connected AI client, "
                       "then re-run.", err=True)
            raise click.exceptions.Exit(1)
        click.echo(f"Waiting for trace log at {path} (Ctrl-C to stop)…", err=True)
        try:
            while not path.exists():
                _time.sleep(0.5)
        except KeyboardInterrupt:
            click.echo("", err=True)
            return

    def _matches(rec: dict) -> bool:
        return not tool_filter or rec.get("tool") == tool_filter

    # ── Print historical tail ─────────────────────────────────────────
    with path.open() as f:
        records = []
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = _json.loads(raw)
            except _json.JSONDecodeError:
                continue
            if _matches(rec):
                records.append((raw, rec))

    for raw, rec in records[-lines:]:
        click.echo(raw if as_json else _format_record(rec))

    if not follow:
        return

    # ── Live tail (poll-based, no extra deps) ─────────────────────────
    click.echo("\n--- live (Ctrl-C to stop  |  --no-follow to print and exit) ---", err=True)
    with path.open() as f:
        f.seek(0, 2)  # to EOF
        try:
            while True:
                line = f.readline()
                if not line:
                    _time.sleep(0.3)
                    continue
                raw = line.strip()
                if not raw:
                    continue
                try:
                    rec = _json.loads(raw)
                except _json.JSONDecodeError:
                    continue
                if _matches(rec):
                    click.echo(raw if as_json else _format_record(rec))
        except KeyboardInterrupt:
            click.echo("", err=True)
