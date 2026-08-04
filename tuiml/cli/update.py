"""`tuiml update`: upgrade tuiml to the latest PyPI version.

Thin CLI wrapper around the existing `execute_self_update` MCP tool.
Refuses to upgrade editable / dev checkouts (use `git pull` instead).
"""
from __future__ import annotations

import itertools
import json as _json
import sys
import threading
import time

import click

from tuiml.cli.setup import C


class _Status:
    """A single-line progress indicator on stderr.

    The upgrade spends nearly all its wall time inside a ``pip`` / ``uv``
    subprocess whose output ``execute_self_update`` captures, so without
    this the terminal stays blank for up to five minutes and the command
    reads as hung.

    Writes to stderr so that ``--json`` stdout stays machine-parseable.
    Animation is used only on a TTY; when piped, each phase is printed once
    as a plain line, which keeps CI logs readable instead of filling them
    with escape codes.

    Parameters
    ----------
    enabled : bool, default=True
        False silences the indicator entirely.
    """

    FRAMES = "|/-\\"
    INTERVAL = 0.12

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._animate = enabled and sys.stderr.isatty()
        self._text = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def _paint(self, frame: str) -> None:
        """Redraw the status line in place."""
        elapsed = int(time.monotonic() - self._started)
        sys.stderr.write(
            f"\r{C.CYAN}{frame}{C.RESET} {self._text}{C.DIM} ({elapsed}s){C.RESET}\033[K"
        )
        sys.stderr.flush()

    def _run(self) -> None:
        """Animate the status line until stopped."""
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                return
            self._paint(frame)
            self._stop.wait(self.INTERVAL)

    def start(self, text: str) -> None:
        """Show `text` and begin animating.

        Parameters
        ----------
        text : str
            Initial status message.
        """
        if not self._enabled:
            return
        self._text = text
        self._started = time.monotonic()
        if not self._animate:
            click.echo(f"{text}...", err=True)
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, text: str) -> None:
        """Replace the status message.

        Parameters
        ----------
        text : str
            New status message.
        """
        if not self._enabled:
            return
        self._text = text
        if not self._animate:
            click.echo(f"{text}...", err=True)

    def stop(self) -> None:
        """Stop animating and clear the status line.

        Safe to call more than once, and from a ``finally`` block on
        Ctrl-C, so an interrupted upgrade never leaves a stray line behind.
        """
        if not self._enabled or not self._animate:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


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
    # Started before the tuiml.agent.tools import below, which alone costs
    # about half a second: the point is that *something* appears the instant
    # the user hits enter.
    status = _Status(enabled=not as_json)
    status.start("Starting TuiML update")

    try:
        from tuiml.agent.tools import execute_self_update

        result = execute_self_update(
            target_version=target_version,
            dry_run=dry_run,
            _progress_callback=lambda info: status.update(info.get("message", "Working")),
        )
    finally:
        status.stop()

    if as_json:
        click.echo(_json.dumps(result, indent=2, default=str))
        raise click.exceptions.Exit(0 if result.get("status") == "success" else 1)

    if result.get("status") != "success":
        click.echo(f"✗ {result.get('error', 'upgrade failed')}", err=True)
        if result.get("command"):
            click.echo(f"  command: {' '.join(result['command'])}", err=True)
        raise click.exceptions.Exit(1)

    if result.get("dry_run"):
        # An editable checkout has no upgrade command to show: the tool
        # reports the refusal it would raise in `note`, with command=None.
        if result.get("command"):
            click.echo(f"would run: {' '.join(result['command'])}")
        elif result.get("note"):
            click.echo(f"would not run: {result['note']}")
        click.echo(f"install method: {result['install_method']}")
        return

    # execute_self_update reports these as version_before/version_after, and
    # either can be None when the probe that reads them fails — so fall back
    # on a falsy value, not just a missing key.
    before = result.get("version_before") or "?"
    after  = result.get("version_after") or "?"
    click.echo(f"✓ TuiML upgraded: {before} → {after}")

    # A git install tracks a branch, so successive commits share a version
    # string and the line above would read "0.1.9 → 0.1.9" on a real upgrade.
    # The commits are what actually moved.
    c_before = result.get("commit_before")
    c_after = result.get("commit_after")
    if c_before or c_after:
        short = lambda c: (c or "?")[:12]
        if c_before and c_after and c_before == c_after:
            click.echo(f"  commit: {short(c_after)} (already current)")
        else:
            click.echo(f"  commit: {short(c_before)} → {short(c_after)}")

    if result.get("install_method"):
        via = result["install_method"]
        if result.get("install_source") == "git":
            via += " (git channel)"
        click.echo(f"  via: {via}")
