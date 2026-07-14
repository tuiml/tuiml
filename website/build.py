#!/usr/bin/env python3
"""Freeze the TuiML site to static HTML for GitHub Pages.

GitHub Pages serves static files only — it cannot run the FastAPI app. This
script renders every route through the *real* app (so templates, the changelog
parser, notebook conversion, and generate_docs output all run exactly as they do
in production) and writes the results into ``_site/`` as plain files.

Run from the website directory:

    uv run python build.py

Then ``_site/`` is a self-contained static copy ready to publish.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import httpx

import app as webapp

HERE = Path(__file__).resolve().parent
OUT = HERE / "_site"
STATIC_SRC = HERE / "static"
DOCS_API = HERE / "templates" / "docs_api"
TUTORIALS_DIR = webapp.TUTORIALS_DIR
DOMAIN = "https://tuiml.ai"

# Routes with no parameters — rendered verbatim. Redirect routes are handled
# separately below (Pages can't issue a 301, so we emit a meta-refresh stub).
STATIC_ROUTES = [
    "/",
    "/projects",
    "/docs/getting_started.html",
    "/docs/tutorials.html",
    "/docs/api-reference.html",
    "/docs/benchmarks.html",
    "/docs/changelog.html",
    "/docs/contributing.html",
    "/docs/remote-mcp.html",
    "/docs/about.html",
    "/docs/privacy.html",
    "/docs/terms.html",
    "/robots.txt",
    "/sitemap.xml",
    "/install.sh",
    "/install",
]

# /about -> /docs/about.html etc. Real app returns 301; on Pages we emit a
# static HTML redirect that preserves the same destination.
REDIRECTS = {
    "/about.html": "/docs/about.html",
    "/privacy.html": "/docs/privacy.html",
    "/terms.html": "/docs/terms.html",
}

REDIRECT_STUB = """<!doctype html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={dest}">
<link rel="canonical" href="{dest}">
<title>Redirecting…</title>
<a href="{dest}">Redirecting to {dest}</a>
"""


def out_path(url_path: str) -> Path:
    """Map a URL path to a file inside ``_site/``.

    ``/`` and dir URLs become ``index.html``; extensionless "pretty" URLs
    (``/projects``, ``/tutorials/x``, ``/install``) get a ``.html`` suffix, which
    GitHub Pages serves back at the clean path. Paths that already carry an
    extension (``.html``, ``.txt``, ``.xml``, ``.sh``) are written verbatim.
    """
    clean = url_path.lstrip("/")
    if clean == "" or clean.endswith("/"):
        clean += "index.html"
    elif "." not in clean.rsplit("/", 1)[-1]:
        clean += ".html"
    return OUT / clean


def write(url_path: str, content: bytes) -> None:
    p = out_path(url_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def _collect_urls() -> list[str]:
    """Every parameterised URL the app should render, in output order."""
    urls = list(STATIC_ROUTES)

    # Generated API reference — every docs_api/*.html, rendered through the
    # catch-all so it gets the header/footer wrapper.
    for html in sorted(DOCS_API.rglob("*.html")):
        urls.append(f"/docs/{html.relative_to(DOCS_API).as_posix()}")

    # Tutorials — notebooks converted at build time (nbconvert runs now instead
    # of per request).
    if TUTORIALS_DIR.exists():
        for nb in sorted(TUTORIALS_DIR.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in nb.parts:
                continue
            urls.append(f"/tutorials/{nb.relative_to(TUTORIALS_DIR).with_suffix('').as_posix()}")

    return urls


async def _render_all(urls: list[str]) -> int:
    # base_url set to the real domain so any absolute URLs the app builds
    # (sitemap entries, canonical links from request.base_url) are correct.
    transport = httpx.ASGITransport(app=webapp.app)
    async with httpx.AsyncClient(transport=transport, base_url=DOMAIN) as client:
        for url in urls:
            r = await client.get(url)
            if r.status_code != 200:
                raise SystemExit(f"  ✗ {url} -> HTTP {r.status_code}")
            write(url, r.content)
    return len(urls)


def freeze() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    count = asyncio.run(_render_all(_collect_urls()))

    # Redirect stubs
    for src, dest in REDIRECTS.items():
        write(src, REDIRECT_STUB.format(dest=dest).encode())
        count += 1

    # 5. Static assets
    shutil.copytree(STATIC_SRC, OUT / "static")

    # 6. Pages hygiene:
    #    .nojekyll  — REQUIRED. Jekyll strips files/dirs beginning with "_"
    #                 (docs_api has __init__.html and _cpp/), which would
    #                 delete a chunk of the API docs otherwise.
    #    CNAME      — keeps the tuiml.ai custom domain bound to the Pages site.
    (OUT / ".nojekyll").write_text("")
    (OUT / "CNAME").write_text("tuiml.ai\n")

    # 404 fallback so unknown paths render the site's own not-found styling.
    err = OUT / "docs" / "getting_started.html"
    if err.exists():
        shutil.copyfile(err, OUT / "404.html")

    print(f"  ✓ froze {count} pages + static assets -> {OUT.relative_to(HERE)}/")


if __name__ == "__main__":
    freeze()
