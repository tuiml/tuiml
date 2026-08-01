"""Uninstall Command, remove TuiML's MCP wiring from every detected AI client.

This command is the inverse of ``tuiml setup``. It does NOT remove the
``tuiml`` package itself (use ``uv tool uninstall tuiml`` or ``pip uninstall
tuiml`` for that). It only unwires the MCP server entries and the Claude Code
skill file that ``tuiml setup`` wrote.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import click

from tuiml.cli.setup import (
    ALL_CLIENT_IDS,
    C,
    backup_file,
    client_specs,
    confirm,
    describe_target,
    error,
    info,
    resolve_client_ids,
    section,
    success,
    warn,
    _get_nested,
    _set_nested,
)


SERVER_NAME = "tuiml"


def _banner() -> None:
    """Print the uninstall wizard's title banner."""
    click.echo()
    click.echo(f"  {C.BOLD}{C.BLUE}TuiML Uninstall Wizard{C.RESET}")
    click.echo(f"  {C.DIM}Remove TuiML from your AI agents{C.RESET}")
    click.echo()


# ---------------------------------------------------------------------------
# Per-kind removers
# ---------------------------------------------------------------------------

def remove_json_entry(config_path: Path, key: str) -> tuple[bool, str]:
    """Remove the ``tuiml`` entry from a JSON config file.

    The file is backed up before being rewritten. A missing file, unparseable
    JSON, or an absent entry is reported rather than raised.

    Parameters
    ----------
    config_path : Path
        JSON config file to edit.
    key : str
        Key holding the server map, dotted for nesting, e.g. ``'mcp.servers'``.

    Returns
    -------
    changed : bool
        True if the file was modified.
    reason : str
        Short human-readable description of what happened.
    """
    if not config_path.exists():
        return False, "config file not present"
    try:
        data = json.loads(config_path.read_text() or "{}")
    except json.JSONDecodeError as exc:
        return False, f"existing config is not valid JSON: {exc}"

    block = _get_nested(data, key)
    if not isinstance(block, dict) or SERVER_NAME not in block:
        return False, "no tuiml entry found"

    backup_file(config_path)
    del block[SERVER_NAME]
    _set_nested(data, key, block)
    config_path.write_text(json.dumps(data, indent=2))
    return True, "removed tuiml entry"


def remove_toml_entry(config_path: Path) -> tuple[bool, str]:
    """Strip the ``[mcp_servers.tuiml]`` block from a TOML config.

    Follows the OpenAI Codex CLI layout. The file is backed up before being
    rewritten.

    Parameters
    ----------
    config_path : Path
        TOML config file to edit.

    Returns
    -------
    changed : bool
        True if the file was modified.
    reason : str
        Short human-readable description of what happened.
    """
    if not config_path.exists():
        return False, "config file not present"

    text = config_path.read_text()
    header = f"[mcp_servers.{SERVER_NAME}]"
    if header not in text:
        return False, "no tuiml entry found"

    backup_file(config_path)
    pattern = re.compile(
        rf"\[mcp_servers\.{re.escape(SERVER_NAME)}\][^\[]*",
        re.MULTILINE,
    )
    cleaned = pattern.sub("", text).rstrip() + "\n"
    config_path.write_text(cleaned)
    return True, "removed [mcp_servers.tuiml] block"


def remove_opencode_entry(config_path: Path) -> tuple[bool, str]:
    """Remove the ``tuiml`` entry from OpenCode's ``opencode.json``.

    OpenCode nests servers under a top-level ``"mcp"`` key rather than
    ``"mcpServers"``, so it needs its own remover.

    Parameters
    ----------
    config_path : Path
        Path to ``opencode.json``.

    Returns
    -------
    changed : bool
        True if the file was modified.
    reason : str
        Short human-readable description of what happened.
    """
    return remove_json_entry(config_path, "mcp")


def remove_openclaw_entry(spec: dict) -> tuple[bool, str]:
    """Remove TuiML from OpenClaw, preferring its own CLI.

    Uses ``openclaw mcp remove`` when the executable is on ``PATH``, and falls
    back to editing the nested JSON config directly if that command is missing
    or fails.

    Parameters
    ----------
    spec : dict
        Client spec, providing ``config`` and ``key``.

    Returns
    -------
    changed : bool
        True if the client configuration was modified.
    reason : str
        Short human-readable description of what happened.
    """
    openclaw = shutil.which("openclaw")
    if openclaw:
        result = subprocess.run(
            [openclaw, "mcp", "remove", SERVER_NAME],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return True, "removed via openclaw mcp remove"
        detail = (result.stderr or result.stdout).strip()
        if detail:
            warn(f"  OpenClaw CLI removal failed: {detail}")
        warn("  Falling back to direct config edit.")

    return remove_json_entry(spec["config"], spec["key"])


def remove_skill_dir(skills_dir: Path) -> tuple[bool, str]:
    """Delete the installed ``tuiml`` skill directory.

    Removes ``<skills_dir>/tuiml`` and everything under it.

    Parameters
    ----------
    skills_dir : Path
        The agent's skills directory.

    Returns
    -------
    changed : bool
        True if the directory was deleted.
    reason : str
        Short human-readable description of what happened.
    """
    target_dir = skills_dir / "tuiml"
    if not target_dir.exists():
        return False, "skill directory not present"
    shutil.rmtree(target_dir)
    return True, f"deleted {target_dir}"


def print_yaml_instructions(spec: dict) -> tuple[bool, str]:
    """Print the manual removal step for YAML-config clients (Goose and similar).

    YAML configs are not edited automatically, to avoid destroying comments
    and formatting.

    Parameters
    ----------
    spec : dict
        Client spec, providing ``config``.

    Returns
    -------
    changed : bool
        Always False; nothing is written.
    reason : str
        Note explaining that this is a manual step.
    """
    info(f"  Edit {spec['config']} and remove the `extensions.tuiml` block.")
    return False, "manual step (YAML config not auto-edited)"


def unconfigure(spec: dict) -> tuple[bool, str]:
    """Remove TuiML's wiring from one client, dispatching on the spec's ``kind``.

    Parameters
    ----------
    spec : dict
        Client spec from ``tuiml.cli.setup.client_specs``.

    Returns
    -------
    changed : bool
        True if the client configuration was modified.
    reason : str
        Short human-readable description of what happened.
    """
    kind = spec["kind"]
    if kind == "openclaw":
        return remove_openclaw_entry(spec)
    if kind == "json-mcp":
        return remove_json_entry(spec["config"], "mcpServers")
    if kind == "json-key":
        return remove_json_entry(spec["config"], spec["key"])
    if kind == "toml-mcp":
        return remove_toml_entry(spec["config"])
    if kind == "opencode":
        return remove_opencode_entry(spec["config"])
    if kind == "skill":
        return remove_skill_dir(spec["skills_dir"])
    if kind == "yaml-instructions":
        return print_yaml_instructions(spec)
    if kind == "instructions":
        info(f"  {spec['name']} wiring lives outside the host (see `tuiml setup` instructions). Skipped.")
        return False, "manual step (config lives outside the host)"
    return False, f"unknown client kind: {kind}"


def prompt_mode(default: str = "auto") -> str:
    """Ask whether to unwire every client at once or one by one.

    Parameters
    ----------
    default : str, default="auto"
        Choice used when the user just presses Enter.

    Returns
    -------
    mode : {"auto", "manual", "quit"}
        The selected mode.
    """
    click.echo()
    click.echo(f"  {C.BOLD}Mode:{C.RESET}")
    click.echo(f"    [{C.GREEN}a{C.RESET}] Auto  , remove TuiML from every detected client")
    click.echo(f"    [{C.YELLOW}m{C.RESET}] Manual, ask for each client individually")
    click.echo(f"    [{C.DIM}q{C.RESET}] Quit")
    choice = click.prompt("  Choose", default=default, show_default=True).strip().lower()
    if choice.startswith("q"):
        return "quit"
    if choice.startswith("m"):
        return "manual"
    return "auto"


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

@click.command("uninstall")
@click.option("--yes", "-y", "assume_yes", is_flag=True,
              help="Auto mode: unwire every client without prompting.")
@click.option("--manual", "force_manual", is_flag=True,
              help="Manual mode: ask before unwiring each client, skipping the Auto/Manual menu.")
@click.option("--client", "clients", multiple=True,
              help="Unwire only this client, by ID. Repeatable. "
                   "Run 'tuiml setup --list' to see valid IDs.")
def uninstall(assume_yes: bool, force_manual: bool, clients: tuple[str, ...]) -> None:
    """Remove TuiML from your AI agents, undoing ``tuiml setup``.

    Scans every supported client, and for each one that currently has a
    ``tuiml`` MCP entry or skill file, removes it. Every known client is
    checked rather than only the detected ones, so stale entries left behind by
    an uninstalled client are cleaned up too. Each config file is backed up
    before being modified.

    This command does not uninstall the ``tuiml`` Python package itself; it
    prints the right command to finish that job.

    Examples
    --------
    Run the interactive wizard:

    $ tuiml uninstall

    Unwire every client without prompting:

    $ tuiml uninstall --yes

    Unwire only specific clients:

    $ tuiml uninstall --client claude-code --client cursor

    Then remove the package itself, using whichever installer you used:

    $ uv tool uninstall tuiml

    $ pip uninstall tuiml
    """
    _banner()

    info("Scanning AI clients for TuiML wiring ...")

    # Consider every known client (not just detected ones): a config file may
    # still have a stale tuiml entry even if the detect path is gone.
    all_specs = client_specs()

    if clients:
        wanted, unknown = resolve_client_ids(clients)
        if unknown:
            error(f"Unknown client(s): {', '.join(unknown)}")
            info(f"Valid IDs: {', '.join(ALL_CLIENT_IDS)}")
            sys.exit(1)
        all_specs = [s for s in all_specs if s["id"] in wanted]

    if assume_yes:
        mode = "auto"
    elif force_manual:
        mode = "manual"
    else:
        mode = prompt_mode(default="auto")

    if mode == "quit":
        info("Cancelled, no changes made.")
        click.echo()
        return

    auto = (mode == "auto")

    section("Removing wiring:")
    changes_made = 0
    for spec in all_specs:
        if not auto and not confirm(f"Unwire {spec['name']}?", default=True):
            info(f"  Skipped {spec['name']}")
            continue

        try:
            changed, reason = unconfigure(spec)
        except Exception as exc:
            error(f"  {spec['name']}: {exc}")
            continue

        if changed:
            success(f"  {spec['name']}: {reason}")
            changes_made += 1
        else:
            info(f"  {spec['name']:22} {C.DIM}({reason}){C.RESET}")

    section("Done.")
    if changes_made:
        info(f"Removed TuiML from {changes_made} client(s). Restart any running clients.")
    else:
        info("No client wiring was found to remove.")
    click.echo()

    click.echo(f"  {C.DIM}To remove the Python package as well, run one of:{C.RESET}")
    click.echo(f"    {C.BOLD}uv tool uninstall tuiml{C.RESET}     {C.DIM}# if installed via `uv tool install`{C.RESET}")
    click.echo(f"    {C.BOLD}pip uninstall tuiml{C.RESET}         {C.DIM}# if installed via pip{C.RESET}")
    click.echo()
