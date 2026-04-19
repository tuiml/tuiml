"""Setup Command - Interactive wizard to connect TuiML to AI agents."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def banner() -> None:
    click.echo()
    click.echo(f"{C.BOLD}{C.BLUE}  TuiML Setup Wizard{C.RESET}")
    click.echo(f"{C.DIM}  Connect TuiML to your AI agents{C.RESET}")
    click.echo()


def info(msg: str) -> None:
    click.echo(f"{C.DIM}·{C.RESET} {msg}")


def success(msg: str) -> None:
    click.echo(f"{C.GREEN}✓{C.RESET} {msg}")


def warn(msg: str) -> None:
    click.echo(f"{C.YELLOW}!{C.RESET} {msg}")


def error(msg: str) -> None:
    click.echo(f"{C.RED}✗{C.RESET} {msg}")


def section(title: str) -> None:
    click.echo()
    click.echo(f"{C.BOLD}{title}{C.RESET}")


def confirm(prompt: str, default: bool = True, assume_yes: bool = False) -> bool:
    if assume_yes:
        return True
    return click.confirm(f"  {prompt}", default=default)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def claude_desktop_config_path() -> Path:
    """Return Claude Desktop's MCP config file path for the current OS."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if sys.platform.startswith("linux"):
        return home / ".config" / "Claude" / "claude_desktop_config.json"
    if sys.platform == "win32":
        return home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    return home / ".claude_desktop_config.json"


def cursor_mcp_config_path() -> Path:
    """Return Cursor's MCP config file path."""
    return Path.home() / ".cursor" / "mcp.json"


def claude_code_skills_dir() -> Path:
    """Return Claude Code skills directory."""
    return Path.home() / ".claude" / "skills"


def detect_clients() -> dict:
    """Detect which AI clients are installed by checking standard config paths."""
    detected = {}

    # Claude Desktop: detected if config dir exists
    claude_desktop = claude_desktop_config_path()
    if claude_desktop.parent.exists():
        detected["claude-desktop"] = {
            "name": "Claude Desktop",
            "type": "mcp",
            "config_path": claude_desktop,
        }

    # Cursor: detected if .cursor dir exists
    if (Path.home() / ".cursor").exists():
        detected["cursor"] = {
            "name": "Cursor",
            "type": "mcp",
            "config_path": cursor_mcp_config_path(),
        }

    # Claude Code: detected if .claude dir exists
    claude_code = Path.home() / ".claude"
    if claude_code.exists():
        detected["claude-code"] = {
            "name": "Claude Code",
            "type": "skill",
            "skills_dir": claude_code_skills_dir(),
        }

    return detected


# ---------------------------------------------------------------------------
# MCP config writers
# ---------------------------------------------------------------------------

def backup_file(path: Path) -> Optional[Path]:
    """Back up a file with a timestamp suffix. Returns the backup path, or None if no file existed."""
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, backup)
    return backup


def add_mcp_server(config_path: Path, server_name: str, command: str) -> tuple[bool, str]:
    """Add a TuiML MCP server entry to a JSON config without clobbering existing servers.

    Returns (changed, reason).
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        try:
            data = json.loads(config_path.read_text() or "{}")
        except json.JSONDecodeError as exc:
            return False, f"existing config is not valid JSON: {exc}"
    else:
        data = {}

    servers = data.setdefault("mcpServers", {})
    if server_name in servers:
        existing = servers[server_name].get("command")
        if existing == command:
            return False, "already configured"
        # Update existing to point to current command
        backup_file(config_path)
        servers[server_name] = {"command": command}
        config_path.write_text(json.dumps(data, indent=2))
        return True, f"updated existing entry (was: {existing})"

    backup_file(config_path)
    servers[server_name] = {"command": command}
    config_path.write_text(json.dumps(data, indent=2))
    return True, "added new entry"


# ---------------------------------------------------------------------------
# Skill file installer
# ---------------------------------------------------------------------------

def install_skill_file(skills_dir: Path) -> tuple[bool, str]:
    """Copy the bundled SKILL.md into a coding agent's skills directory."""
    # Locate the skill file inside the installed tuiml package
    try:
        from importlib.resources import files  # py3.9+
        skill_src = files("tuiml.llm").joinpath("Skill.md")
        if not skill_src.is_file():
            return False, f"skill file not found in package: {skill_src}"
        src_text = skill_src.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"could not read package skill file: {exc}"

    target_dir = skills_dir / "tuiml"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "SKILL.md"

    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if existing == src_text:
            return False, "already up to date"
        backup_file(target)

    target.write_text(src_text, encoding="utf-8")
    return True, f"installed to {target}"


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

@click.command("setup")
@click.option("--yes", "-y", "assume_yes", is_flag=True,
              help="Skip prompts and configure all detected clients.")
@click.option("--list", "list_only", is_flag=True,
              help="List detected clients without making any changes.")
@click.option("--client", multiple=True,
              type=click.Choice(["claude-desktop", "cursor", "claude-code"]),
              help="Configure only the specified client(s). Can be passed multiple times.")
def setup(assume_yes: bool, list_only: bool, client: tuple[str, ...]) -> None:
    """Connect TuiML to your AI agents.

    Detects installed clients (Claude Desktop, Cursor, Claude Code, ...) and
    interactively wires them up: appends an MCP server entry for MCP clients,
    or copies the SKILL.md file for coding agents that use skills.
    """
    banner()

    info("Detecting installed AI clients ...")
    detected = detect_clients()

    if not detected:
        warn("No AI clients detected.")
        click.echo()
        info("Looked in:")
        info(f"  Claude Desktop: {claude_desktop_config_path().parent}")
        info(f"  Cursor: {Path.home() / '.cursor'}")
        info(f"  Claude Code: {Path.home() / '.claude'}")
        click.echo()
        info("Install one of these clients, then re-run: tuiml setup")
        return

    # Print detection summary
    section("Detected:")
    for key, c in detected.items():
        if c["type"] == "mcp":
            info(f"  {C.GREEN}✓{C.RESET} {c['name']:20} ({c['config_path']})")
        else:
            info(f"  {C.GREEN}✓{C.RESET} {c['name']:20} (skills dir: {c['skills_dir']})")

    if list_only:
        click.echo()
        return

    # Filter by --client if provided
    if client:
        detected = {k: v for k, v in detected.items() if k in client}
        if not detected:
            error("None of the specified clients were detected.")
            sys.exit(1)

    section("Configuration:")
    changes_made = 0
    for key, c in detected.items():
        if not confirm(f"Configure {c['name']}?", default=True, assume_yes=assume_yes):
            info(f"  Skipped {c['name']}")
            continue

        if c["type"] == "mcp":
            changed, reason = add_mcp_server(c["config_path"], "tuiml", "tuiml-mcp")
            if changed:
                success(f"  {c['name']}: {reason}")
                changes_made += 1
            else:
                info(f"  {c['name']}: {reason}")
        elif c["type"] == "skill":
            changed, reason = install_skill_file(c["skills_dir"])
            if changed:
                success(f"  {c['name']}: {reason}")
                changes_made += 1
            else:
                info(f"  {c['name']}: {reason}")

    # Final message
    section("Done.")
    if changes_made:
        info("Restart your AI client to activate TuiML.")
        click.echo()
        info(f"Try asking your agent: {C.CYAN}\"Train a random forest on iris and report accuracy\"{C.RESET}")
    else:
        info("No changes were made.")
    click.echo()
