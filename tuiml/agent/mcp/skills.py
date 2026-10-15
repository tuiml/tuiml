"""Skills over MCP (SEP-2640) support for the TuiML MCP server.

The bundled ``SKILL.md`` (the canonical TuiML agent guide, also installed by
``tuiml setup`` as a Claude Code skill) is served over MCP so any client that
speaks the ``io.modelcontextprotocol/skills`` extension can discover and read
it without a separate install step. The server:

- advertises ``capabilities.extensions["io.modelcontextprotocol/skills"]``;
- serves ``skills/list`` and ``skills/get`` (the extension's two methods);
- exposes every skill file as an ordinary MCP resource under the
  ``skill://`` scheme (``skill://tuiml/SKILL.md``), so clients *without* the
  extension can still read the guide via ``resources/read``.

The wire shapes follow the SEP-2640 working draft maintained by the Skills
Over MCP working group (https://github.com/modelcontextprotocol/experimental-ext-skills).

Examples
--------
>>> from tuiml.agent.mcp.skills import get_skill_entry, SKILL_MD_URI
>>> entry = get_skill_entry()
>>> entry["uri"] == SKILL_MD_URI
True
>>> entry["frontmatter"]["name"]
'tuiml'
>>> [r["uri"] for r in entry["resources"]][:2]
['skill://tuiml/SKILL.md', 'skill://tuiml/reference/cli.md']
>>> entry["resources"][0]["digest"][:7]
'sha256:'
"""

from __future__ import annotations

import hashlib
import pathlib
from functools import lru_cache
from importlib import resources
from typing import Any, Dict, List, Optional, Tuple

#: SEP-2133 extension identifier advertised under ``ServerCapabilities.extensions``.
SKILLS_EXTENSION_ID = "io.modelcontextprotocol/skills"

#: Extension settings. ``directoryRead`` is the optional
#: ``resources/directory/read`` method, which a single-file skill does not need.
SKILLS_EXTENSION_SETTINGS: Dict[str, Any] = {"directoryRead": False}

#: Skill path (last segment is the skill name, per the ``skill://`` scheme).
SKILL_NAME = "tuiml"
SKILL_ROOT_URI = f"skill://{SKILL_NAME}"
SKILL_MD_URI = f"{SKILL_ROOT_URI}/SKILL.md"

#: MIME type per file extension for files shipped inside the skill.
_MIME_BY_SUFFIX: Dict[str, str] = {".md": "text/markdown", ".py": "text/x-python",
                                   ".json": "application/json", ".txt": "text/plain"}


@lru_cache(maxsize=1)
def _skill_files() -> Dict[str, str]:
    """Discover every file of the bundled skill and its MIME type.

    Walks ``tuiml.agent.prompts`` (``SKILL.md`` plus the ``reference/`` and
    ``workflows/`` trees) so a new reference file is served the moment it is
    added. Python package files and caches are skipped.

    Returns
    -------
    dict of str -> str
        Relative POSIX path -> MIME type, ``SKILL.md`` first.
    """
    root = resources.files("tuiml.agent.prompts")
    found: Dict[str, str] = {}

    def walk(node, prefix: str) -> None:
        for child in sorted(node.iterdir(), key=lambda c: c.name):
            name = child.name
            if name.startswith((".", "__")):
                continue
            rel = f"{prefix}{name}"
            if child.is_dir():
                walk(child, rel + "/")
                continue
            mime = _MIME_BY_SUFFIX.get(pathlib.PurePosixPath(name).suffix)
            if mime is not None and name != "__init__.py":
                found[rel] = mime

    walk(root, "")
    ordered = {"SKILL.md": found.pop("SKILL.md")}
    ordered.update(found)
    return ordered


def parse_frontmatter(text: str) -> Dict[str, str]:
    """Parse the flat ``key: value`` YAML frontmatter at the top of a SKILL.md.

    Only the flat subset used by Agent Skills frontmatter is supported
    (``name``, ``description``, ``version``, ...). Nested or multi-line values
    are not needed by the bundled skill, so no YAML dependency is pulled in.

    Parameters
    ----------
    text : str
        Full Markdown text, optionally starting with a ``---`` block.

    Returns
    -------
    dict of str -> str
        Frontmatter fields, or an empty dict if there is no frontmatter.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: Dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line or line.startswith((" ", "\t", "#")):
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        fields[key.strip()] = value
    return fields


@lru_cache(maxsize=1)
def _load_skill_files() -> Dict[str, bytes]:
    """Read every bundled skill file from ``tuiml.agent.prompts``.

    Returns
    -------
    dict of str -> bytes
        Relative file path -> raw contents. Cached; the files are static
        package data.
    """
    root = resources.files("tuiml.agent.prompts")
    return {rel: root.joinpath(rel).read_bytes() for rel in _skill_files()}


def _file_uri(rel_path: str) -> str:
    """Build the ``skill://`` URI for a file inside the TuiML skill.

    Parameters
    ----------
    rel_path : str
        File path relative to the skill root, e.g. ``"SKILL.md"``.

    Returns
    -------
    str
        ``skill://tuiml/<rel_path>``.
    """
    return f"{SKILL_ROOT_URI}/{rel_path}"


def get_skill_frontmatter() -> Dict[str, str]:
    """Return the SKILL.md frontmatter as a dict.

    Returns
    -------
    dict of str -> str
        At least ``name`` and ``description``; the bundled file also carries
        ``version`` and ``mcp_server``.
    """
    return parse_frontmatter(_load_skill_files()["SKILL.md"].decode("utf-8"))


def get_skill_entry() -> Dict[str, Any]:
    """Build the SEP-2640 skill entry returned by ``skills/list`` and ``skills/get``.

    Returns
    -------
    dict
        ``{"uri", "frontmatter", "resources"}`` where ``resources`` lists
        every file with its ``uri``, ``sha256:`` ``digest`` and ``size`` so a
        client can install and verify the skill.
    """
    files = _load_skill_files()
    manifest: List[Dict[str, Any]] = []
    for rel, data in files.items():
        manifest.append({
            "uri": _file_uri(rel),
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size": len(data),
        })
    return {
        "uri": SKILL_MD_URI,
        "frontmatter": get_skill_frontmatter(),
        "resources": manifest,
    }


def list_skills() -> Dict[str, Any]:
    """Build the ``skills/list`` result.

    Returns
    -------
    dict
        ``{"skills": [entry]}`` with the single bundled TuiML skill. The list
        is complete, so no ``nextCursor`` is set.
    """
    return {"skills": [get_skill_entry()]}


def get_skill(uri: str) -> Optional[Dict[str, Any]]:
    """Resolve a ``skills/get`` request.

    Parameters
    ----------
    uri : str
        Either the skill's ``SKILL.md`` URI or its root URI.

    Returns
    -------
    dict or None
        The skill entry, or None if ``uri`` names no bundled skill.
    """
    if uri.rstrip("/") in (SKILL_ROOT_URI, SKILL_MD_URI):
        return get_skill_entry()
    return None


def list_skill_resources() -> List[Dict[str, Any]]:
    """Describe every skill file for ``resources/list``.

    Returns
    -------
    list of dict
        One entry per file with ``uri``, ``name``, ``description``,
        ``mime_type`` and ``size``, ready to unpack into ``mcp.types.Resource``.
        ``SKILL.md`` takes its name and description from the frontmatter, as
        SEP-2640 recommends.
    """
    fm = get_skill_frontmatter()
    files = _load_skill_files()
    out: List[Dict[str, Any]] = []
    for rel, mime in _skill_files().items():
        is_skill_md = rel == "SKILL.md"
        out.append({
            "uri": _file_uri(rel),
            "name": fm.get("name", SKILL_NAME) if is_skill_md else rel,
            "description": (
                fm.get("description", "TuiML agent skill")
                if is_skill_md else f"{rel} (part of the {SKILL_NAME} skill)"
            ),
            "mime_type": mime,
            "size": len(files[rel]),
        })
    return out


def read_skill_resource(uri: str) -> Optional[Tuple[str, str]]:
    """Resolve a ``resources/read`` for a ``skill://`` URI.

    Parameters
    ----------
    uri : str
        A ``skill://tuiml/<file>`` URI.

    Returns
    -------
    tuple of (str, str) or None
        ``(mime_type, text)`` for a known file, else None.
    """
    prefix = SKILL_ROOT_URI + "/"
    if not uri.startswith(prefix):
        return None
    rel = uri[len(prefix):]
    mime = _skill_files().get(rel)
    if mime is None:
        return None
    return mime, _load_skill_files()[rel].decode("utf-8")
