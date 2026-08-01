"""``tuiml restart``: kill every running tuiml-mcp; clients respawn it.

Most useful right after ``tuiml update`` to pick up the new code without
manually quitting each AI client. The process discovery and signalling live in
``tuiml.agent.tools.system.restart``; this module is only the CLI front end.
"""
from __future__ import annotations

import json as _json

import click


@click.command()
@click.option("--json", "as_json", is_flag=True,
              help="Emit raw JSON instead of human-readable output.")
@click.option("--grace", type=float, default=2.0, show_default=True,
              help="Seconds to wait for SIGTERM before sending SIGKILL.")
def restart(as_json: bool, grace: float) -> None:
    """Restart every running tuiml-mcp process.

    Each AI client (Claude Desktop, Cursor, Codex, ...) automatically respawns
    its tuiml-mcp child when the previous one exits, so stopping the servers is
    all it takes to pick up a freshly installed version. Every instance gets a
    SIGTERM, then a SIGKILL if it is still alive after the grace period, and the
    stopped process ids are reported. Nothing is restarted eagerly: the clients
    spawn a new server on their next request.

    Use ``tuiml status`` first to see what is running.

    Examples
    --------
    Pick up a new version straight after upgrading:

    $ tuiml update && tuiml restart

    Give slow servers longer to shut down cleanly:

    $ tuiml restart --grace 5

    Restart from a script and inspect what was stopped:

    $ tuiml restart --json
    """
    from tuiml.agent.tools.system.restart import find_mcp_processes, kill_mcp_processes

    procs = find_mcp_processes(exclude_self=True)

    if as_json:
        result = kill_mcp_processes(procs=procs, grace_seconds=grace) if procs else {
            "killed": [], "failed": [], "self_exit_scheduled": False,
        }
        result["candidates"] = procs
        click.echo(_json.dumps(result, indent=2))
        return

    if not procs:
        click.echo("No tuiml-mcp processes running, nothing to restart.")
        click.echo("Open an AI client (Claude Desktop, Cursor, ...) to spawn one.")
        return

    click.echo(f"Restarting {len(procs)} tuiml-mcp instance(s):")
    for p in procs:
        click.echo(f"  pid {p['pid']:>6}  (parent pid {p['ppid']})")
    click.echo()

    result = kill_mcp_processes(procs=procs, grace_seconds=grace)

    if result["killed"]:
        click.echo(f"✓ Stopped: {', '.join(str(p) for p in result['killed'])}")
    if result["failed"]:
        for f in result["failed"]:
            click.echo(f"✗ pid {f['pid']}: {f['error']}", err=True)

    click.echo()
    click.echo("Your AI clients will respawn tuiml-mcp on the next request.")
