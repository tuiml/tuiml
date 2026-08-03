"""`tuiml update`: upgrade tuiml to the latest PyPI version.

Thin CLI wrapper around the existing `execute_self_update` MCP tool.
Refuses to upgrade editable / dev checkouts (use `git pull` instead).
"""
from __future__ import annotations

import json as _json

import click


@click.command()
@click.option(
    "--target", "target_version",
    metavar="VERSION",
    help="Install this exact version (e.g. 0.1.4) instead of the latest release. Can be used to downgrade.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the upgrade command that would run, and change nothing.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit raw JSON instead of the human-readable summary.",
)
def update(target_version: str, dry_run: bool, as_json: bool) -> None:
    """Upgrade TuiML to the latest release.

    Detects how TuiML was installed and re-runs that same installer, so a
    ``uv tool`` install is upgraded with ``uv`` and a pip install with pip,
    then reports the version before and after. Editable and development
    checkouts are refused, since those should be updated with ``git pull``.
    Use ``tuiml info`` first to see whether an upgrade is available.

    Examples
    --------
    Upgrade to the newest published release:

    $ tuiml update

    See what would happen, without changing anything:

    $ tuiml update --dry-run

    Pin to a specific version:

    $ tuiml update --target 0.1.4
    """
    from tuiml.agent.tools import execute_self_update

    result = execute_self_update(target_version=target_version, dry_run=dry_run)

    if as_json:
        click.echo(_json.dumps(result, indent=2, default=str))
        raise click.exceptions.Exit(0 if result.get("status") == "success" else 1)

    if result.get("status") != "success":
        click.echo(f"✗ {result.get('error', 'upgrade failed')}", err=True)
        if "command" in result:
            click.echo(f"  command: {' '.join(result['command'])}", err=True)
        raise click.exceptions.Exit(1)

    if result.get("dry_run"):
        click.echo(f"would run: {' '.join(result['command'])}")
        click.echo(f"install method: {result['install_method']}")
        return

    # execute_self_update reports these as version_before/version_after, and
    # either can be None when the probe that reads them fails — so fall back
    # on a falsy value, not just a missing key.
    before = result.get("version_before") or "?"
    after  = result.get("version_after") or "?"
    click.echo(f"✓ TuiML upgraded: {before} → {after}")
    if result.get("install_method"):
        click.echo(f"  via: {result['install_method']}")
