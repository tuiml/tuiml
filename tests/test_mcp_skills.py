"""Skills over MCP (SEP-2640): the server serves its bundled SKILL.md.

Runs the real server behind the SDK's in-memory client, so the assertions
cover the wire shapes a client sees — capability advertisement, the
``skill://`` resource, and the ``skills/list`` / ``skills/get`` methods.
"""

import asyncio
import hashlib
from typing import Any

import pytest

pytest.importorskip("mcp")

import mcp.types as t  # noqa: E402
from mcp.client import Client  # noqa: E402
from mcp.shared.exceptions import MCPError  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

from tuiml.agent.mcp.server import create_server  # noqa: E402
from tuiml.agent.mcp.skills import (  # noqa: E402
    SKILL_MD_URI,
    SKILLS_EXTENSION_ID,
    get_skill_frontmatter,
    parse_frontmatter,
)


class _Raw(t.Request[dict, str]):
    """A request for a method the client SDK has no typed wrapper for."""


_DICT = TypeAdapter(dict[str, Any])


def _run(coro):
    """Run a coroutine against a fresh in-memory server.

    Parameters
    ----------
    coro : callable
        ``async def coro(client)``; receives the connected client.

    Returns
    -------
    Any
        Whatever ``coro`` returns.
    """
    async def main():
        async with Client(create_server()) as client:
            return await coro(client)
    return asyncio.run(main())


def test_parse_frontmatter_flat_block():
    text = '---\nname: "Quoted"\ndescription: plain: with colon\n---\n# Body\nkey: not frontmatter\n'
    assert parse_frontmatter(text) == {"name": "Quoted", "description": "plain: with colon"}
    assert parse_frontmatter("# no frontmatter\n") == {}


def test_bundled_frontmatter_has_required_fields():
    fm = get_skill_frontmatter()
    assert fm["name"] and fm["description"], "Agent Skills frontmatter needs name + description"


def test_server_advertises_skills_extension():
    async def check(client):
        return client.server_capabilities.extensions
    exts = _run(check)
    assert SKILLS_EXTENSION_ID in exts
    assert exts[SKILLS_EXTENSION_ID] == {"directoryRead": False}


def test_skill_md_is_a_readable_resource():
    async def check(client):
        listed = await client.list_resources()
        read = await client.read_resource(SKILL_MD_URI)
        return listed.resources, read.contents[0]
    resources, content = _run(check)
    skill = next(r for r in resources if str(r.uri) == SKILL_MD_URI)
    assert skill.mime_type == "text/markdown"
    assert skill.name == get_skill_frontmatter()["name"]
    assert content.mime_type == "text/markdown"
    assert content.text.startswith("---\nname:")


def test_skills_list_and_get_agree_with_resources_read():
    async def check(client):
        listed = await client.session.send_request(_Raw(method="skills/list", params={}), _DICT)
        got = await client.session.send_request(
            _Raw(method="skills/get", params={"uri": SKILL_MD_URI}), _DICT)
        read = await client.read_resource(SKILL_MD_URI)
        return listed, got, read.contents[0].text
    listed, got, text = _run(check)

    assert [s["uri"] for s in listed["skills"]] == [SKILL_MD_URI]
    entry = listed["skills"][0]
    assert entry["frontmatter"]["name"] == get_skill_frontmatter()["name"]
    # The runner stamps ``resultType`` / ``_meta`` onto every result; compare
    # the skill entry fields only.
    assert {k: got[k] for k in ("uri", "frontmatter", "resources")} == entry

    # The digest manifest must match the bytes resources/read serves, or a
    # client installing the skill (fast-agent) rejects it.
    manifest = {r["uri"]: r for r in entry["resources"]}
    raw = text.encode("utf-8")
    assert manifest[SKILL_MD_URI]["digest"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert manifest[SKILL_MD_URI]["size"] == len(raw)


def test_skills_get_unknown_uri_is_invalid_params():
    async def check(client):
        with pytest.raises(MCPError) as exc:
            await client.session.send_request(
                _Raw(method="skills/get", params={"uri": "skill://nope/SKILL.md"}), _DICT)
        return exc.value.error.code
    assert _run(check) == t.INVALID_PARAMS


def test_reference_and_workflow_files_are_served():
    """Every file of the skill tree is listed, digested and readable."""
    async def check(client):
        listed = await client.session.send_request(_Raw(method="skills/list", params={}), _DICT)
        uris = [r["uri"] for r in listed["skills"][0]["resources"]]
        read = await client.read_resource("skill://tuiml/reference/mistakes.md")
        return uris, read.contents[0]
    uris, content = _run(check)
    for rel in ("reference/python-api.md", "reference/mcp-tools.md", "reference/cli.md",
                "reference/mistakes.md", "workflows/research.md", "workflows/industry.md"):
        assert f"skill://tuiml/{rel}" in uris, rel
    assert content.mime_type == "text/markdown"
    assert content.text.startswith("# Mistakes")


def test_router_links_every_bundled_file():
    """SKILL.md must link each reference/workflow file one level deep."""
    from tuiml.agent.mcp.skills import _load_skill_files
    files = _load_skill_files()
    router = files["SKILL.md"].decode("utf-8")
    for rel in files:
        if rel != "SKILL.md":
            assert f"]({rel})" in router, f"SKILL.md does not link {rel}"
    assert router.count("\n") < 500, "SKILL.md body must stay under 500 lines"
