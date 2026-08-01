"""Setup Command - Interactive wizard to connect TuiML to AI agents."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
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
    """Print the setup wizard's title banner."""
    click.echo()
    click.echo(f"{C.BOLD}{C.BLUE}  TuiML Setup Wizard{C.RESET}")
    click.echo(f"{C.DIM}  Connect TuiML to your AI agents{C.RESET}")
    click.echo()


def info(msg: str) -> None:
    """Print a dimmed informational line.

    Parameters
    ----------
    msg : str
        Message to display.
    """
    click.echo(f"{C.DIM}·{C.RESET} {msg}")


def success(msg: str) -> None:
    """Print a green check-marked success line.

    Parameters
    ----------
    msg : str
        Message to display.
    """
    click.echo(f"{C.GREEN}✓{C.RESET} {msg}")


def warn(msg: str) -> None:
    """Print a yellow warning line.

    Parameters
    ----------
    msg : str
        Message to display.
    """
    click.echo(f"{C.YELLOW}!{C.RESET} {msg}")


def error(msg: str) -> None:
    """Print a red error line.

    Parameters
    ----------
    msg : str
        Message to display.
    """
    click.echo(f"{C.RED}✗{C.RESET} {msg}")


def section(title: str) -> None:
    """Print a bold section heading preceded by a blank line.

    Parameters
    ----------
    title : str
        Heading text.
    """
    click.echo()
    click.echo(f"{C.BOLD}{title}{C.RESET}")


def confirm(prompt: str, default: bool = True, assume_yes: bool = False) -> bool:
    """Ask the user a yes/no question, unless answering has been pre-empted.

    Parameters
    ----------
    prompt : str
        Question to put to the user.
    default : bool, default=True
        Answer used when the user just presses Enter.
    assume_yes : bool, default=False
        If True, return True immediately without prompting.

    Returns
    -------
    answer : bool
        The user's answer, or True when ``assume_yes`` is set.
    """
    if assume_yes:
        return True
    return click.confirm(f"  {prompt}", default=default)


# ---------------------------------------------------------------------------
# Client registry
# ---------------------------------------------------------------------------
#
# Each client spec has:
#   id        : short slug used by --client flag
#   name      : human-readable label
#   detect    : Path that, if it exists, indicates the client is installed
#   kind      : "openclaw"  (OpenClaw CLI registry, fallback to JSON)
#                "json-mcp"  (JSON config with mcpServers key)
#                "json-key" (JSON config with custom key, eg Zed)
#                "toml-mcp" (TOML config, eg OpenAI Codex CLI)
#                "skill"    (drop SKILL.md in a skills directory)
#   config    : Path to the config file (for json/toml)
#   key       : Top-level config key (default "mcpServers", overridable)
#   skills_dir: Path to skills directory (for skill kind)

HOME = Path.home()


def _xdg_config(name: str) -> Path:
    """Return the XDG config directory for an application.

    Parameters
    ----------
    name : str
        Application directory name.

    Returns
    -------
    path : Path
        ``$XDG_CONFIG_HOME/<name>``, falling back to ``~/.config/<name>``.
    """
    import os
    base = Path(os.environ.get("XDG_CONFIG_HOME", HOME / ".config"))
    return base / name


def _platform_app_dir(macos: str, linux: str, windows: str) -> Path:
    """Return the per-platform application support directory.

    Parameters
    ----------
    macos : str
        Directory name under ``~/Library/Application Support``.
    linux : str
        Directory name under the XDG config home.
    windows : str
        Directory name under ``~/AppData/Roaming``.

    Returns
    -------
    path : Path
        The directory appropriate to the current platform.
    """
    if sys.platform == "darwin":
        return HOME / "Library" / "Application Support" / macos
    if sys.platform == "win32":
        return HOME / "AppData" / "Roaming" / windows
    # linux / wsl
    return _xdg_config(linux)


def _vscode_global_storage(ext_id: str) -> Path:
    """Return the VS Code globalStorage settings directory for an extension.

    Cline, Roo Code, and Kilo Code all store their MCP settings under VS Code's
    per-extension globalStorage path,
    ``<vscode-user>/globalStorage/<publisher.extension>/settings/``.

    Parameters
    ----------
    ext_id : str
        Extension identifier, in ``publisher.extension`` form.

    Returns
    -------
    path : Path
        The extension's ``settings`` directory.
    """
    user = _platform_app_dir("Code/User", "Code/User", "Code/User")
    return user / "globalStorage" / ext_id / "settings"


def _fork_global_storage(app: str, ext_id: str) -> Path:
    """Return the globalStorage settings directory for a VS Code fork.

    Forks such as Antigravity keep the same ``User/globalStorage`` layout as
    VS Code, but under their own application-support directory, e.g.
    ``~/Library/Application Support/Antigravity/User/globalStorage/<ext>/settings``.

    Parameters
    ----------
    app : str
        Application-support directory name of the fork, e.g. ``"Antigravity"``.
    ext_id : str
        Extension identifier, in ``publisher.extension`` form.

    Returns
    -------
    path : Path
        The extension's ``settings`` directory inside the fork.
    """
    user = _platform_app_dir(f"{app}/User", f"{app}/User", f"{app}/User")
    return user / "globalStorage" / ext_id / "settings"


def client_specs() -> list[dict]:
    """Return the spec for every AI client TuiML knows how to wire up.

    Returns
    -------
    specs : list of dict
        One spec per client, each with at least ``id``, ``name``, ``detect``,
        and ``kind`` keys, plus ``config``, ``key``, or ``skills_dir``
        depending on the kind.
    """
    return [
        # ----- OpenClaw (featured partner) ---------------------------------
        # Config lives at ~/.openclaw/openclaw.json with nested "mcp.servers".
        {
            "id": "openclaw",
            "name": "OpenClaw",
            "kind": "openclaw",
            "key": "mcp.servers",
            "detect_command": "openclaw",
            "detect": HOME / ".openclaw",
            "config": HOME / ".openclaw" / "openclaw.json",
        },
        # ----- NVIDIA NemoClaw ---------------------------------------------
        # NemoClaw runs OpenClaw inside a sandboxed Docker container via
        # OpenShell. The host `~/.nemoclaw/` directory only holds sandbox
        # metadata (sandboxes.json, onboard-session.json): the real OpenClaw
        # config with `mcp.servers` lives INSIDE the sandbox. We can detect
        # NemoClaw, but wiring must happen from inside the sandbox shell.
        {
            "id": "nemoclaw",
            "name": "NemoClaw (NVIDIA)",
            "kind": "instructions",
            "detect": HOME / ".nemoclaw",
            "config": HOME / ".nemoclaw" / "sandboxes.json",
            "instructions": (
                "NemoClaw runs OpenClaw inside a sandboxed Docker container,\n"
                "so its OpenClaw MCP config is not reachable from the host.\n"
                "To wire TuiML into a NemoClaw sandbox:\n"
                "  1. Allow PyPI package installs from the host:\n"
                "       nemoclaw <sandbox-name> policy-add pypi --yes\n"
                "  2. Connect to the sandbox:\n"
                "       nemoclaw <sandbox-name> connect\n"
                "  3. Install TuiML inside the sandbox:\n"
                "       python -m venv /sandbox/.openclaw/workspace/tuiml_venv\n"
                "       . /sandbox/.openclaw/workspace/tuiml_venv/bin/activate\n"
                "       pip install tuiml\n"
                "  4. Configure OpenClaw from inside the sandbox:\n"
                "       tuiml setup --client openclaw -y"
            ),
        },
        # ----- Claude Desktop ----------------------------------------------
        {
            "id": "claude-desktop",
            "name": "Claude Desktop",
            "kind": "json-mcp",
            "detect": _platform_app_dir("Claude", "Claude", "Claude"),
            "config": _platform_app_dir("Claude", "Claude", "Claude") / "claude_desktop_config.json",
        },
        # ----- Claude Code (skill file) ------------------------------------
        {
            "id": "claude-code",
            "name": "Claude Code",
            "kind": "skill",
            "detect": HOME / ".claude",
            "skills_dir": HOME / ".claude" / "skills",
        },
        # ----- ChatGPT Desktop ---------------------------------------------
        # OpenAI added MCP support in 2026; config layout mirrors Claude Desktop.
        {
            "id": "chatgpt-desktop",
            "name": "ChatGPT Desktop",
            "kind": "json-mcp",
            "detect": _platform_app_dir("ChatGPT", "ChatGPT", "ChatGPT"),
            "config": _platform_app_dir("ChatGPT", "ChatGPT", "ChatGPT") / "mcp_config.json",
        },
        # ----- Perplexity Desktop (Comet) ----------------------------------
        {
            "id": "perplexity",
            "name": "Perplexity Desktop",
            "kind": "json-mcp",
            "detect": _platform_app_dir("Perplexity", "Perplexity", "Perplexity"),
            "config": _platform_app_dir("Perplexity", "Perplexity", "Perplexity") / "mcp_config.json",
        },
        # ----- OpenAI Codex CLI --------------------------------------------
        {
            "id": "codex",
            "name": "OpenAI Codex CLI",
            "kind": "toml-mcp",
            "detect": HOME / ".codex",
            "config": HOME / ".codex" / "config.toml",
        },
        # ----- Cursor ------------------------------------------------------
        {
            "id": "cursor",
            "name": "Cursor",
            "kind": "json-mcp",
            "detect": HOME / ".cursor",
            "config": HOME / ".cursor" / "mcp.json",
        },
        # ----- Windsurf (Codeium) ------------------------------------------
        {
            "id": "windsurf",
            "name": "Windsurf",
            "kind": "json-mcp",
            "detect": HOME / ".codeium" / "windsurf",
            "config": HOME / ".codeium" / "windsurf" / "mcp_config.json",
        },
        # ----- Zed (uses "context_servers" key) ----------------------------
        {
            "id": "zed",
            "name": "Zed",
            "kind": "json-key",
            "key": "context_servers",
            "detect": _xdg_config("zed"),
            "config": _xdg_config("zed") / "settings.json",
        },
        # ----- Continue (VS Code) ------------------------------------------
        {
            "id": "continue",
            "name": "Continue (VS Code)",
            "kind": "json-mcp",
            "detect": HOME / ".continue",
            "config": HOME / ".continue" / "config.json",
        },
        # ----- VS Code MCP (1.99+ native) ----------------------------------
        # VS Code stores MCP config in user settings under "mcp.servers".
        {
            "id": "vscode",
            "name": "VS Code (Copilot)",
            "kind": "json-key",
            "key": "mcp.servers",
            "detect": _platform_app_dir("Code/User", "Code/User", "Code/User"),
            "config": _platform_app_dir("Code/User", "Code/User", "Code/User") / "settings.json",
        },
        # ----- Goose (Block) -----------------------------------------------
        # Goose uses YAML; we don't write it automatically, print instructions.
        {
            "id": "goose",
            "name": "Goose",
            "kind": "yaml-instructions",
            "detect": _xdg_config("goose"),
            "config": _xdg_config("goose") / "config.yaml",
        },
        # ----- Gemini CLI (Google) -----------------------------------------
        # ~/.gemini/settings.json with top-level "mcpServers" key.
        {
            "id": "gemini",
            "name": "Gemini CLI",
            "kind": "json-mcp",
            "detect_command": "gemini",
            "detect": HOME / ".gemini",
            "config": HOME / ".gemini" / "settings.json",
        },
        # ----- Cline (VS Code) ---------------------------------------------
        # Stores MCP settings inside VS Code globalStorage; uses "mcpServers".
        {
            "id": "cline",
            "name": "Cline (VS Code)",
            "kind": "json-mcp",
            "detect": _vscode_global_storage("saoudrizwan.claude-dev"),
            "config": _vscode_global_storage("saoudrizwan.claude-dev") / "cline_mcp_settings.json",
        },
        # ----- Roo Code (VS Code) ------------------------------------------
        {
            "id": "roo",
            "name": "Roo Code (VS Code)",
            "kind": "json-mcp",
            "detect": _vscode_global_storage("rooveterinaryinc.roo-cline"),
            "config": _vscode_global_storage("rooveterinaryinc.roo-cline") / "mcp_settings.json",
        },
        # ----- Kilo Code (VS Code) -----------------------------------------
        {
            "id": "kilo",
            "name": "Kilo Code (VS Code)",
            "kind": "json-mcp",
            "detect": _vscode_global_storage("kilocode.kilo-code"),
            "config": _vscode_global_storage("kilocode.kilo-code") / "mcp_settings.json",
        },
        # ----- OpenCode (terminal) -----------------------------------------
        # Uses a unique schema where `mcp.<name>` is {"type":"local","command":[...]}.
        # Handled by a dedicated writer.
        {
            "id": "opencode",
            "name": "OpenCode",
            "kind": "opencode",
            "detect_command": "opencode",
            "detect": _xdg_config("opencode"),
            "config": _xdg_config("opencode") / "opencode.json",
        },
        # ----- Antigravity (Google): VS Code fork -------------------------
        # Antigravity has NO native global MCP config; MCP comes through
        # extensions, each keeping config under
        #   <Antigravity>/User/globalStorage/<ext-id>/settings/
        # mirroring VS Code. We add concrete entries for the MCP-capable
        # extensions, detected by their install dir under ~/.antigravity/
        # extensions/<ext-id>-<version>/ (glob, since the dir is versioned).
        {
            "id": "antigravity-kilo",
            "name": "Antigravity · Kilo Code",
            "kind": "json-mcp",
            "detect_glob": (HOME / ".antigravity" / "extensions", "kilocode.kilo-code-*"),
            "detect": _fork_global_storage("Antigravity", "kilocode.kilo-code"),
            "config": _fork_global_storage("Antigravity", "kilocode.kilo-code") / "mcp_settings.json",
        },
        {
            "id": "antigravity-cline",
            "name": "Antigravity · Cline",
            "kind": "json-mcp",
            "detect_glob": (HOME / ".antigravity" / "extensions", "saoudrizwan.claude-dev-*"),
            "detect": _fork_global_storage("Antigravity", "saoudrizwan.claude-dev"),
            "config": _fork_global_storage("Antigravity", "saoudrizwan.claude-dev") / "cline_mcp_settings.json",
        },
        {
            "id": "antigravity-roo",
            "name": "Antigravity · Roo Code",
            "kind": "json-mcp",
            "detect_glob": (HOME / ".antigravity" / "extensions", "rooveterinaryinc.roo-cline-*"),
            "detect": _fork_global_storage("Antigravity", "rooveterinaryinc.roo-cline"),
            "config": _fork_global_storage("Antigravity", "rooveterinaryinc.roo-cline") / "mcp_settings.json",
        },
    ]


def detect_clients() -> list[dict]:
    """Return the specs of clients that appear to be installed.

    A client counts as installed if its ``detect`` path exists, if its
    ``detect_glob`` pattern matches a versioned install directory, or if its
    ``detect_command`` is on ``PATH``.

    Returns
    -------
    detected : list of dict
        The subset of :func:`client_specs` found on this machine.
    """
    detected = []
    for spec in client_specs():
        # 1) config/install dir exists
        if spec["detect"].exists():
            detected.append(spec)
            continue
        # 2) a versioned install dir matches a glob (e.g. an extension folder
        #    like ~/.antigravity/extensions/kilocode.kilo-code-5.11.0)
        detect_glob = spec.get("detect_glob")
        if detect_glob:
            base, pattern = detect_glob
            try:
                if base.exists() and any(base.glob(pattern)):
                    detected.append(spec)
                    continue
            except OSError:
                pass
        # 3) a CLI command is on PATH
        command = spec.get("detect_command")
        if command and shutil.which(command):
            detected.append(spec)
    return detected


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def backup_file(path: Path) -> Optional[Path]:
    """Copy a file alongside itself with a timestamped ``.bak`` suffix.

    Parameters
    ----------
    path : Path
        File to back up.

    Returns
    -------
    backup : Path or None
        Path of the backup copy, or None if ``path`` did not exist.
    """
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, backup)
    return backup


def _set_nested(obj: dict, dotted_key: str, value) -> None:
    """Assign a value at a dotted path, creating intermediate dicts.

    For ``dotted_key='a.b.c'`` this sets ``obj['a']['b']['c'] = value``.

    Parameters
    ----------
    obj : dict
        Dictionary to modify in place.
    dotted_key : str
        Dot-separated path to the target key.
    value : object
        Value to store at the target key.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an intermediate key already holds a non-dict value.
    """
    parts = dotted_key.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
        if not isinstance(cur, dict):
            raise ValueError(f"Conflicting non-dict value at '{p}' in config")
    cur[parts[-1]] = value


def _get_nested(obj: dict, dotted_key: str):
    """Read the value at a dotted path, if the whole path exists.

    Parameters
    ----------
    obj : dict
        Dictionary to read from.
    dotted_key : str
        Dot-separated path to the target key.

    Returns
    -------
    value : object or None
        The value found, or None if any part of the path is missing or holds a
        non-dict value.
    """
    parts = dotted_key.split(".")
    cur = obj
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def write_json_mcp(config_path: Path, key: str, server_name: str, command: str) -> tuple[bool, str]:
    """Write an MCP server entry into a JSON config file.

    The config file and its parent directories are created if missing, and an
    existing file is backed up before being rewritten. Idempotent: an entry
    that already names the same command is left untouched.

    Parameters
    ----------
    config_path : Path
        JSON config file to write.
    key : str
        Key holding the server map, dotted for nesting, e.g. ``'mcp.servers'``.
    server_name : str
        Name to register the server under.
    command : str
        Executable the client should launch.

    Returns
    -------
    changed : bool
        True if the file was modified.
    reason : str
        Short human-readable description of what happened.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text() or "{}")
        except json.JSONDecodeError as exc:
            return False, f"existing config is not valid JSON: {exc}"
    else:
        data = {}

    existing_block = _get_nested(data, key)
    if not isinstance(existing_block, dict):
        existing_block = {}

    if server_name in existing_block:
        if existing_block[server_name].get("command") == command:
            return False, "already configured"
        backup_file(config_path)
        existing_block[server_name] = {"command": command}
        _set_nested(data, key, existing_block)
        config_path.write_text(json.dumps(data, indent=2))
        return True, "updated existing entry"

    backup_file(config_path)
    existing_block[server_name] = {"command": command}
    _set_nested(data, key, existing_block)
    config_path.write_text(json.dumps(data, indent=2))
    return True, "added new entry"


def _tuiml_mcp_command() -> str:
    """Return the command clients should run to start the TuiML MCP server.

    Returns
    -------
    command : str
        Absolute path to ``tuiml-mcp`` when it is on ``PATH``, otherwise the
        bare name ``"tuiml-mcp"``.
    """
    return shutil.which("tuiml-mcp") or "tuiml-mcp"


def configure_openclaw(spec: dict) -> tuple[bool, str]:
    """Register TuiML with OpenClaw, preferring its own CLI.

    Uses ``openclaw mcp set`` when the executable is on ``PATH``, and falls
    back to editing the JSON config directly if that command is missing or
    fails.

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
    mcp_command = _tuiml_mcp_command()
    if openclaw:
        payload = json.dumps({"command": mcp_command}, separators=(",", ":"))
        result = subprocess.run(
            [openclaw, "mcp", "set", "tuiml", payload],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return True, f"registered via openclaw mcp set ({mcp_command})"

        detail = (result.stderr or result.stdout).strip()
        if detail:
            warn(f"  OpenClaw CLI setup failed: {detail}")
        warn("  Falling back to direct config edit.")

    return write_json_mcp(spec["config"], spec["key"], "tuiml", mcp_command)


def write_opencode_mcp(config_path: Path, server_name: str, command: str) -> tuple[bool, str]:
    """Write an MCP entry into OpenCode's ``opencode.json``.

    OpenCode's schema places servers under a top-level ``"mcp"`` key with the
    shape ``{"mcp": {"<name>": {"type": "local", "command": ["<exec>"],
    "enabled": true}}}``. The file is backed up before being rewritten.

    Parameters
    ----------
    config_path : Path
        Path to ``opencode.json``.
    server_name : str
        Name to register the server under.
    command : str
        Executable OpenCode should launch.

    Returns
    -------
    changed : bool
        True if the file was modified.
    reason : str
        Short human-readable description of what happened.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text() or "{}")
        except json.JSONDecodeError as exc:
            return False, f"existing config is not valid JSON: {exc}"
    else:
        data = {"$schema": "https://opencode.ai/config.json"}

    mcp_block = data.get("mcp")
    if not isinstance(mcp_block, dict):
        mcp_block = {}

    new_entry = {"type": "local", "command": [command], "enabled": True}
    if mcp_block.get(server_name) == new_entry:
        return False, "already configured"

    backup_file(config_path)
    mcp_block[server_name] = new_entry
    data["mcp"] = mcp_block
    config_path.write_text(json.dumps(data, indent=2))
    return True, "added entry" if server_name not in mcp_block else "updated entry"


def write_toml_mcp(config_path: Path, server_name: str, command: str) -> tuple[bool, str]:
    """Append an ``[mcp_servers.<name>]`` section to a TOML config.

    Follows the OpenAI Codex CLI layout. Idempotent: if the section already
    exists with the same command, nothing changes; otherwise any prior block
    for that server is replaced and the file is backed up first.

    Parameters
    ----------
    config_path : Path
        TOML config file to write.
    server_name : str
        Name to register the server under.
    command : str
        Executable the client should launch.

    Returns
    -------
    changed : bool
        True if the file was modified.
    reason : str
        Short human-readable description of what happened.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    section_header = f"[mcp_servers.{server_name}]"
    new_block = f'{section_header}\ncommand = "{command}"\n'

    if not config_path.exists():
        config_path.write_text(new_block)
        return True, "created new config"

    text = config_path.read_text()

    # If the exact line is already present, do nothing.
    if section_header in text and f'command = "{command}"' in text:
        return False, "already configured"

    backup_file(config_path)

    # Remove any existing [mcp_servers.<name>] block (and following lines until next [section] or EOF)
    pattern = re.compile(
        rf"\[mcp_servers\.{re.escape(server_name)}\][^\[]*",
        re.MULTILINE,
    )
    cleaned = pattern.sub("", text).rstrip() + "\n"

    config_path.write_text(cleaned + "\n" + new_block)
    return True, "added entry"


def install_skill_file(skills_dir: Path) -> tuple[bool, str]:
    """Copy the bundled ``SKILL.md`` into a coding agent's skills directory.

    The file is written to ``<skills_dir>/tuiml/SKILL.md``. An existing file
    with different contents is backed up before being overwritten.

    Parameters
    ----------
    skills_dir : Path
        The agent's skills directory.

    Returns
    -------
    changed : bool
        True if the skill file was written.
    reason : str
        Short human-readable description of what happened.
    """
    try:
        from importlib.resources import files
        skill_src = files("tuiml.agent").joinpath("SKILL.md")
        if not skill_src.is_file():
            return False, f"skill file not found in package: {skill_src}"
        src_text = skill_src.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"could not read package skill file: {exc}"

    target_dir = skills_dir / "tuiml"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "SKILL.md"

    if target.exists():
        if target.read_text(encoding="utf-8") == src_text:
            return False, "already up to date"
        backup_file(target)

    target.write_text(src_text, encoding="utf-8")
    return True, f"installed to {target}"


def print_yaml_instructions(spec: dict) -> tuple[bool, str]:
    """Print the YAML snippet the user must add by hand (Goose and similar).

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
    snippet = (
        "extensions:\n"
        "  tuiml:\n"
        "    type: stdio\n"
        "    command: tuiml-mcp\n"
        "    enabled: true\n"
    )
    info(f"  Add to {spec['config']}:")
    for line in snippet.splitlines():
        click.echo(f"      {line}")
    return False, "manual step (YAML config not auto-edited)"


def print_instructions(spec: dict) -> tuple[bool, str]:
    """Print client-specific setup instructions without editing any config.

    Used for clients that cannot be safely auto-wired from the host, e.g.
    NemoClaw, whose OpenClaw config lives inside a Docker sandbox.

    Parameters
    ----------
    spec : dict
        Client spec, providing ``instructions``.

    Returns
    -------
    changed : bool
        Always False; nothing is written.
    reason : str
        Note explaining that this is a manual step.
    """
    text = spec.get("instructions", "")
    for line in text.splitlines():
        click.echo(f"    {line}")
    return False, "manual step (config lives outside the host, see instructions)"


# ---------------------------------------------------------------------------
# Per-client dispatcher
# ---------------------------------------------------------------------------

def configure(spec: dict) -> tuple[bool, str]:
    """Wire TuiML into one client, dispatching on the spec's ``kind``.

    Parameters
    ----------
    spec : dict
        Client spec from :func:`client_specs`.

    Returns
    -------
    changed : bool
        True if the client configuration was modified.
    reason : str
        Short human-readable description of what happened.
    """
    kind = spec["kind"]
    if kind == "openclaw":
        return configure_openclaw(spec)
    if kind == "json-mcp":
        return write_json_mcp(spec["config"], "mcpServers", "tuiml", "tuiml-mcp")
    if kind == "json-key":
        return write_json_mcp(spec["config"], spec["key"], "tuiml", "tuiml-mcp")
    if kind == "toml-mcp":
        return write_toml_mcp(spec["config"], "tuiml", "tuiml-mcp")
    if kind == "opencode":
        return write_opencode_mcp(spec["config"], "tuiml", _tuiml_mcp_command())
    if kind == "skill":
        return install_skill_file(spec["skills_dir"])
    if kind == "yaml-instructions":
        return print_yaml_instructions(spec)
    if kind == "instructions":
        return print_instructions(spec)
    return False, f"unknown client kind: {kind}"


def describe_target(spec: dict) -> str:
    """Return the path this client's wiring will be written to, for display.

    Parameters
    ----------
    spec : dict
        Client spec from :func:`client_specs`.

    Returns
    -------
    target : str
        The skills directory for skill-based clients, otherwise the config
        file path.
    """
    if spec["kind"] == "skill":
        return f"skills dir: {spec['skills_dir']}"
    return str(spec["config"])


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

ALL_CLIENT_IDS = [c["id"] for c in client_specs()]


def prompt_mode(default: str = "auto") -> str:
    """Ask whether to configure all detected clients at once or one by one.

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
    click.echo(f"    [{C.GREEN}a{C.RESET}] Auto  , configure every detected client")
    click.echo(f"    [{C.YELLOW}m{C.RESET}] Manual, ask for each client individually")
    click.echo(f"    [{C.DIM}q{C.RESET}] Quit")
    choice = click.prompt("  Choose", default=default, show_default=True).strip().lower()
    if choice.startswith("q"):
        return "quit"
    if choice.startswith("m"):
        return "manual"
    return "auto"


@click.command("setup")
@click.option("--yes", "-y", "assume_yes", is_flag=True,
              help="Auto mode: configure every detected client without prompting.")
@click.option("--manual", "force_manual", is_flag=True,
              help="Manual mode: ask before configuring each client, skipping the Auto/Manual menu.")
@click.option("--list", "list_only", is_flag=True,
              help="List the detected clients and their config paths, then exit without changing anything.")
@click.option("--client", "clients", multiple=True,
              help="Configure only this client, by ID. Repeatable. "
                   "Run 'tuiml setup --list' to see valid IDs.")
def setup(assume_yes: bool, force_manual: bool, list_only: bool, clients: tuple[str, ...]) -> None:
    """Connect TuiML to your AI agents.

    Detects installed clients and wires them up: appends an MCP server entry
    for MCP clients, copies the SKILL.md file for skill-based agents, or
    prints manual instructions for YAML and unsupported configs. Every config
    file is backed up before it is modified, and re-running is safe: a client
    that is already wired correctly is left alone.

    By default the wizard asks whether to configure every detected client at
    once (Auto) or one by one (Manual). Pass ``-y`` / ``--yes`` to skip the
    menu and go straight to Auto, or ``--manual`` to go straight to Manual.
    Undo everything with ``tuiml uninstall``.

    Examples
    --------
    Run the interactive wizard:

    $ tuiml setup

    See which clients are detected, without changing anything:

    $ tuiml setup --list

    Wire up every detected client without prompting:

    $ tuiml setup --yes

    Configure only specific clients:

    $ tuiml setup --client claude-code --client cursor
    """
    banner()

    info("Detecting installed AI clients ...")
    detected = detect_clients()

    if not detected:
        warn("No AI clients detected.")
        click.echo()
        info("Looked in:")
        for spec in client_specs():
            info(f"  {spec['name']:22} {spec['detect']}")
        click.echo()
        info("Install one of these clients, then re-run: tuiml setup")
        return

    section("Detected:")
    for spec in detected:
        info(f"  {C.GREEN}✓{C.RESET} {spec['name']:22} ({describe_target(spec)})")

    if list_only:
        click.echo()
        return

    # Filter by --client if provided
    if clients:
        unknown = [c for c in clients if c not in ALL_CLIENT_IDS]
        if unknown:
            error(f"Unknown client(s): {', '.join(unknown)}")
            info(f"Valid IDs: {', '.join(ALL_CLIENT_IDS)}")
            sys.exit(1)
        detected = [s for s in detected if s["id"] in clients]
        if not detected:
            error("None of the specified clients were detected on this machine.")
            sys.exit(1)

    # Decide mode: auto (configure all) vs manual (prompt per client)
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

    section("Configuration:")
    changes_made = 0
    for spec in detected:
        if not auto and not confirm(f"Configure {spec['name']}?", default=True):
            info(f"  Skipped {spec['name']}")
            continue

        try:
            changed, reason = configure(spec)
        except Exception as exc:
            error(f"  {spec['name']}: {exc}")
            continue

        if changed:
            success(f"  {spec['name']}: {reason}")
            changes_made += 1
        else:
            info(f"  {spec['name']}: {reason}")

    section("Done.")
    if changes_made:
        info("Restart your AI client(s) to activate TuiML.")
        click.echo()
        info(f"Try asking your agent: {C.CYAN}\"Train a random forest on iris and report accuracy using TuiML\"{C.RESET}")
    else:
        info("No changes were made.")
    click.echo()
