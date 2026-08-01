#!/usr/bin/env python3
"""Build the TuiML static site for GitHub Pages.

This is the entire website toolchain in one file: it renders the Jinja
templates in ``templates/`` (pages, docs, generated API reference), converts
the tutorial notebooks in ``../tutorials/`` to HTML, parses the root
``../CHANGELOG.md``, and writes everything into ``_site/`` as plain static
files ready to publish. Nothing here runs in production — GitHub Pages serves
the output directly.

Run from the website directory:

    uv run python build.py

Then ``_site/`` is a self-contained static copy of the site. Preview it with

    uv run python build.py serve [port]

which serves ``_site/`` with GitHub Pages' pretty-URL rules (``/tutorials/x``
-> ``tutorials/x.html``) — a plain ``http.server`` would 404 on those links.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup, escape

HERE = Path(__file__).resolve().parent
OUT = HERE / "_site"
STATIC_SRC = HERE / "static"
TEMPLATES = HERE / "templates"
DOCS_API = TEMPLATES / "_generated"
TUTORIALS_DIR = HERE.parent / "tutorials"     # single source of truth (repo root)
CHANGELOG = HERE.parent / "CHANGELOG.md"      # single source of truth (repo root)
DOMAIN = "https://tuiml.ai"

# ---------------------------------------------------------------------------
# Site settings (formerly core/config.py). The version constant below is
# rewritten by scripts/bump_version.py — don't change its assignment form.
# ---------------------------------------------------------------------------
APP_NAME = "TuiML"
APP_VERSION = "0.1.6"
APP_STATUS = "Alpha"
PROJECT_NAME = "TuiML"
GITHUB_URL = "https://github.com/tuiml/tuiml"

# The Tutorials nav opens the first notebook directly — there is no index page
# to step through, since every tutorial already carries the full sidebar.
# Kept in sync with TUTORIALS by an assertion below.
FIRST_TUTORIAL = "quickstart/01_hello_tuiml"
TUTORIALS_URL = f"/tutorials/{FIRST_TUTORIAL}"

CONFIG = {
    "project_name": PROJECT_NAME,
    "app_name": APP_NAME,
    "version": APP_VERSION,
    "status": APP_STATUS,
    "version_label": f"v{APP_VERSION} {APP_STATUS}",
    "github_url": GITHUB_URL,
    "tutorials_url": TUTORIALS_URL,
    "copyright_year": datetime.now().year,
    "default_meta_description": (
        "TuiML Hub is an open-source machine learning platform for discovering algorithms, "
        "datasets, tutorials, benchmarks, and agent-ready MCP workflows."
    ),
}

# Optional URL prefix for hosting under a subpath, e.g. a GitHub project page at
# tuiml.github.io/tuiml/ needs BASE_PATH=/tuiml so the site's root-absolute links
# (/static, /docs, …) resolve. Leave empty for root/custom-domain hosting, where
# a CNAME is written instead. Set via the BASE_PATH env var.
BASE = "/" + os.environ.get("BASE_PATH", "").strip("/") if os.environ.get("BASE_PATH", "").strip("/") else ""

env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
env.globals["config"] = CONFIG


class _Request:
    """Minimal stand-in for the FastAPI request object templates reference.

    Templates only use ``request.url`` (canonical/og:url tags) and
    ``request.url_for('static', path=...)`` (og:image), both piped through
    ``|string`` — so plain strings are enough.
    """

    def __init__(self, url_path: str):
        self.url = f"{DOMAIN}{url_path}"

    def url_for(self, name: str, path: str = "") -> str:
        if name != "static":
            raise ValueError(f"url_for: unknown route {name!r}")
        return f"{DOMAIN}/static/{path}"


# ---------------------------------------------------------------------------
# Page definitions — template + per-page context for every Jinja-rendered URL.
# ---------------------------------------------------------------------------
PAGES: dict[str, tuple[str, dict]] = {
    "/": ("pages/index.html", {
        "title": "TuiML — Machine Learning for AI Agents",
        "meta_description": (
            "Ask your agent to train a model, tune it, compare it to the last run, "
            "or find an algorithm that fits your data. Open-source MCP-native ML "
            "runtime with 200+ typed tools. Local-first."
        ),
        "meta_keywords": (
            "machine learning, MCP, Model Context Protocol, AI agents, agentic ML, "
            "Claude Desktop, ChatGPT, Cursor, Windsurf, Codex, Perplexity, "
            "agent-native ML, open source ML, Python ML toolkit, LLM tools, "
            "classification, regression, clustering, scikit-learn alternative"
        ),
        "canonical_url": "https://tuiml.ai/",
        "og_image_url": "https://tuiml.ai/static/images/tuiml_logo.png",
    }),
    "/projects": ("pages/projects.html", {
        "title": "Build Board — TuiML",
        "meta_description": (
            "A living list of algorithms, integrations, and tools the community can "
            "build with TuiML — GPU backends, streaming, AutoML, deep-learning wrappers, "
            "and good-first-issue algorithms. Perfect for students and contributors."
        ),
        "meta_keywords": (
            "TuiML contribute, open source ML projects, good first issue, "
            "student projects, RAPIDS, JAX, Keras, PyTorch, CapyMOA, AutoML, "
            "missing algorithms, machine learning tasks"
        ),
        "canonical_url": "https://tuiml.ai/projects",
        "og_image_url": "https://tuiml.ai/static/images/tuiml_logo.png",
    }),
    "/docs/getting_started.html": ("pages/getting_started.html", {
        "active_nav": "docs", "page_title": "Getting Started",
    }),
    "/docs/api-reference.html": ("pages/api-reference.html", {
        "active_nav": "api", "page_title": "API Reference", "title": "API Reference",
        "meta_description": (
            "Complete API reference for TuiML: algorithms, datasets, preprocessing, "
            "evaluation, feature selection, and MCP server modules."
        ),
    }),
    "/docs/benchmarks.html": ("pages/benchmarks.html", {
        "active_nav": "benchmarks", "page_title": "Benchmarks",
    }),
    "/docs/contributing.html": ("pages/contributing.html", {
        "active_nav": "contributing", "page_title": "Contributing",
    }),
    "/docs/privacy.html": ("pages/privacy.html", {
        "title": "Privacy Policy — TuiML",
        "meta_description": (
            "Privacy policy for TuiML, an open-source agent-native ML runtime. "
            "TuiML runs locally and does not collect personal data."
        ),
        "active_nav": "", "page_title": "Privacy Policy",
    }),
    "/docs/terms.html": ("pages/terms.html", {
        "title": "Terms of Service — TuiML",
        "meta_description": (
            "Terms of service for the TuiML open-source project, distributed "
            "under the BSD-3-Clause license."
        ),
        "active_nav": "", "page_title": "Terms of Service",
    }),
    "/docs/about.html": ("pages/about.html", {
        "title": "About TuiML",
        "meta_description": (
            "About TuiML: agent-native ML runtime developed at the AI Institute, "
            "University of Waikato. Meet the team."
        ),
        "active_nav": "about", "page_title": "About",
    }),
    # /docs/changelog.html is rendered separately (needs the parsed CHANGELOG).
}

# /about -> /docs/about.html etc. The old server answered 301; on Pages we emit
# a static HTML redirect that preserves the same destination.
REDIRECTS = {
    "/docs/tutorials.html": TUTORIALS_URL,
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

# Core public URLs listed in sitemap.xml.
SITEMAP_PATHS = [
    "/",
    "/docs/getting_started.html",
    TUTORIALS_URL,
    "/docs/api-reference.html",
    "/docs/benchmarks.html",
    "/docs/changelog.html",
    "/docs/contributing.html",
    "/docs/about.html",
    "/docs/privacy.html",
    "/docs/terms.html",
]

# Machine-readable install + setup instructions for LLM agents, served at
# /install. Paired with the "Agent" tab on the landing page — a user pastes
# "Help me install https://tuiml.ai/install" into any agent with web access;
# the agent fetches this plain-text guide and follows the steps.
INSTALL_GUIDE = (
    "# TuiML install + setup guide (for agents)\n"
    "\n"
    "You are being asked to install TuiML and wire it into the current AI client.\n"
    "TuiML is an open-source Python ML library with an MCP server that exposes\n"
    "200+ algorithms, datasets, and workflow tools to any MCP-capable agent.\n"
    "\n"
    "## Prerequisites (check before running)\n"
    "- A POSIX shell (bash/zsh). macOS or Linux; Windows users should run in WSL.\n"
    "- A C++ compiler (macOS: `xcode-select --install`; Debian/Ubuntu: `build-essential`).\n"
    "- Network access to github.com and pypi.org.\n"
    "\n"
    "## NemoClaw sandbox note\n"
    "NemoClaw sandboxes are deny-by-default. Do not use the tuiml.ai curl\n"
    "installer from inside the sandbox unless that domain has been explicitly\n"
    "allowed. The simpler path is PyPI:\n"
    "\n"
    "    # On the host\n"
    "    nemoclaw <sandbox-name> policy-add pypi --yes\n"
    "    nemoclaw <sandbox-name> connect\n"
    "\n"
    "    # Inside the sandbox\n"
    "    python -m venv /sandbox/.openclaw/workspace/tuiml_venv\n"
    "    . /sandbox/.openclaw/workspace/tuiml_venv/bin/activate\n"
    "    pip install tuiml\n"
    "    openclaw mcp set tuiml '{\"command\":\"/sandbox/.openclaw/workspace/tuiml_venv/bin/tuiml-mcp\"}'\n"
    "    openclaw gateway restart\n"
    "\n"
    "## Step 1 — Install the package\n"
    "Run this single command in a shell tool. It installs `uv` if missing,\n"
    "then builds and installs tuiml as an isolated uv tool (no global Python\n"
    "pollution). The script is idempotent and safe to re-run.\n"
    "\n"
    "    curl -fsSL https://tuiml.ai/install.sh | bash\n"
    "\n"
    "After it finishes the commands `tuiml` and `tuiml-mcp` are available on\n"
    "the PATH (usually `~/.local/bin`).\n"
    "\n"
    "## Step 2 — Wire TuiML into the current AI client\n"
    "Run the setup CLI. It auto-detects installed MCP clients (Claude Desktop,\n"
    "Claude Code, Cursor, OpenClaw, ChatGPT Desktop, Codex CLI, Zed,\n"
    "Continue, Windsurf, VS Code Copilot, Perplexity Desktop, Goose, NemoClaw)\n"
    "and writes a `tuiml` MCP server entry into each client's config. For\n"
    "NemoClaw, host-side setup prints the sandbox commands because the\n"
    "OpenClaw config lives inside the sandbox.\n"
    "\n"
    "    tuiml setup -y\n"
    "\n"
    "`-y` skips the interactive Auto/Manual prompt and configures every\n"
    "detected client. Use `tuiml setup --manual` for per-client prompting,\n"
    "or `tuiml setup --list` to see what was detected without writing.\n"
    "\n"
    "## Step 3 — Restart the AI client\n"
    "The client loads MCP server definitions on startup, so the tuiml server\n"
    "will only appear after a restart. Tell the user to restart their client\n"
    "(Claude Desktop, Cursor, etc.) once step 2 succeeds.\n"
    "\n"
    "## Step 4 (optional) — Enable agent-authored algorithms\n"
    "Set this environment variable in whichever shell spawns the MCP server,\n"
    "then restart the client:\n"
    "\n"
    "    export TUIML_ALLOW_USER_ALGORITHMS=1\n"
    "\n"
    "That unlocks `tuiml_algorithm_skeleton`, `tuiml_create_algorithm`,\n"
    "`tuiml_list_user_algorithms`, `tuiml_delete_user_algorithm`, and\n"
    "`tuiml_research_log` — the Auto-Research loop. See\n"
    "https://tuiml.ai/docs/getting_started.html for details.\n"
    "\n"
    "## Verify the install\n"
    "Once restarted, the agent should call `tuiml_system_info` and confirm\n"
    "the expected version, install method (`uv-tool`), and that\n"
    "`update_available` is false. Then try:\n"
    "\n"
    "    tuiml_train(algorithm=\"RandomForestClassifier\",\n"
    "                data=\"iris\", target=\"target\", cv=5)\n"
    "\n"
    "Success: `status: \"success\"` plus a `metrics` block. Report the\n"
    "accuracy back to the user and TuiML is ready.\n"
    "\n"
    "## Uninstall\n"
    "    tuiml uninstall          # removes tuiml wiring from every client\n"
    "    uv tool uninstall tuiml  # removes the package itself\n"
    "\n"
    "## Documentation\n"
    "- Getting started: https://tuiml.ai/docs/getting_started.html\n"
    "- API reference:   https://tuiml.ai/docs/api-reference.html\n"
    "- Changelog:       https://tuiml.ai/docs/changelog.html\n"
    "- Source:          https://github.com/tuiml/tuiml\n"
)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

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


def write(url_path: str, content: str | bytes) -> None:
    p = out_path(url_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode("utf-8")
    p.write_bytes(content)


def render(url_path: str, template_name: str, **context) -> None:
    """Render one Jinja template to its static output file."""
    context.setdefault("request", _Request(url_path))
    write(url_path, env.get_template(template_name).render(**context))


# ---------------------------------------------------------------------------
# Changelog (parsed from the root CHANGELOG.md — the single canonical copy)
# ---------------------------------------------------------------------------

def _inline_md(text: str) -> Markup:
    """Render the two inline-markdown forms the changelog uses: **bold**, `code`.

    The raw text is escaped first, so changelog content can never inject HTML.
    """
    s = str(escape(text))
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`", r'<code class="font-mono text-[0.85em] bg-gray-100 px-1 py-0.5 rounded">\1</code>', s)
    return Markup(s)


def parse_changelog() -> list[dict]:
    """Parse ``## [version] - date`` release blocks into template-ready dicts.

    The ``[Unreleased]`` section is skipped — the public changelog page lists
    shipped versions only.
    """
    releases: list[dict] = []
    content = CHANGELOG.read_text()
    for block in re.split(r"\n## ", content)[1:]:  # skip header
        lines = block.strip().split("\n")
        header = lines[0]
        match = re.match(r"\[(.+?)\]\s*-\s*(.+)", header)
        version = match.group(1) if match else header.strip("[] ")
        date = match.group(2).strip() if match else ""
        if version.lower() == "unreleased":
            continue
        sections: list[dict] = []
        current_section = None
        for line in lines[1:]:
            sec_match = re.match(r"### (.+)", line)
            if sec_match:
                current_section = {"type": sec_match.group(1), "items": []}
                sections.append(current_section)
            elif line.startswith("- ") and current_section is not None:
                current_section["items"].append(line[2:])
            elif line.startswith("  ") and current_section is not None and current_section["items"]:
                # Wrapped bullet — join the continuation onto the last item.
                current_section["items"][-1] += " " + line.strip()
        for section in sections:
            section["items"] = [_inline_md(i) for i in section["items"]]
        releases.append({"version": version, "date": date, "sections": sections})
    return releases


# ---------------------------------------------------------------------------
# Tutorials (Jupyter notebooks -> HTML with the site's header/sidebar/footer)
# ---------------------------------------------------------------------------

# Tutorial list for the sidebar: one flat, start-to-finish sequence. The order
# is the reading order — connect an agent, learn the Python API, then ship it —
# but it is deliberately NOT grouped: the tracks split a short list into stubs
# and forced readers to pick a lane before they knew which one they wanted.
TUTORIALS = [
    ("quickstart/01_hello_tuiml", "Hello TuiML", "fa-solid fa-play"),
    ("llm_friendly/02_mcp_server", "Connect Your Agent", "fa-solid fa-bolt"),
    ("llm_friendly/01_llm_tools", "Tools an Agent Can Call", "fa-solid fa-robot"),
    ("llm_friendly/04_agent_doing_ml", "Watch an Agent Do ML", "fa-solid fa-comments"),
    ("ml_simplified/01_high_level_api", "High-Level API", "fa-solid fa-rocket"),
    ("ml_simplified/02_workflow_builder", "Workflow Builder", "fa-solid fa-code"),
    ("ml_simplified/08_preprocessing", "Preprocessing", "fa-solid fa-wand-magic-sparkles"),
    ("ml_simplified/09_feature_engineering", "Feature Engineering", "fa-solid fa-filter"),
    ("ml_simplified/10_benchmarking", "Benchmarking", "fa-solid fa-flask"),
    ("deploy/02_model_serving", "Model Serving", "fa-solid fa-server"),
    ("case_studies/01_diabetes_prediction", "Case Study: Diabetes", "fa-solid fa-heart-pulse"),
]

# Head additions for tutorial pages: favicons, fonts, nav/footer chrome deps,
# and a style block that maps nbconvert's exported JupyterLab CSS onto the oc
# design system (see website/DESIGN.md). Appended at the END of nbconvert's
# <head> so these rules win source-order ties against the exported lab CSS.
TUTORIAL_HEAD_EXTRAS = '''
<link rel="icon" type="image/png" sizes="32x32" href="/static/images/favicon-32.png?v=10">
<link rel="icon" type="image/png" sizes="512x512" href="/static/images/favicon.png?v=10">
<link rel="apple-touch-icon" sizes="180x180" href="/static/images/apple-touch-icon.png?v=10">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<script src="https://cdn.tailwindcss.com"></script>
<style>
/* ---- JupyterLab theme variables -> oc palette (oc.css :root) ---- */
body[data-jp-theme-light="true"] {
    --jp-content-font-family: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
    --jp-ui-font-family: var(--jp-content-font-family);
    --jp-code-font-family: var(--jp-content-font-family);
    --jp-code-font-family-default: var(--jp-content-font-family);
    --jp-cell-prompt-font-family: var(--jp-content-font-family);
    --jp-content-font-size1: 16px;
    --jp-content-line-height: 1.5;
    --jp-code-font-size: 12.5px;
    --jp-code-line-height: 1.7;
    --jp-code-padding: 16px;
    --jp-layout-color0: var(--canvas);
    --jp-layout-color1: var(--canvas);
    --jp-layout-color2: var(--soft);
    --jp-content-font-color0: var(--ink);
    --jp-content-font-color1: var(--body);
    --jp-content-font-color2: var(--mute);
    --jp-content-font-color3: var(--mute);
    --jp-content-link-color: var(--ink);
    --jp-content-heading-font-weight: 700;
    --jp-border-color0: var(--hairline);
    --jp-border-color1: var(--hairline);
    --jp-border-color2: var(--hairline);
    --jp-border-color3: var(--hairline);
    --jp-cell-editor-background: var(--card);
    --jp-cell-editor-active-background: var(--card);
    --jp-cell-editor-border-color: transparent;
    --jp-cell-editor-active-border-color: transparent;
    --jp-cell-inprompt-font-color: var(--mute);
    --jp-cell-outprompt-font-color: var(--mute);
    --jp-mirror-editor-keyword-color: var(--accent-deep);
    --jp-mirror-editor-variable-color: var(--ink);
    --jp-mirror-editor-string-color: var(--green);
    --jp-mirror-editor-comment-color: var(--mute);
    --jp-mirror-editor-number-color: var(--orange);
    --jp-mirror-editor-operator-color: var(--body);
    --jp-mirror-editor-punctuation-color: var(--body);
    --jp-mirror-editor-error-color: var(--orange);
    --jp-rendermime-table-row-background: var(--soft);
    --jp-rendermime-table-row-hover-background: var(--card);
    --jp-rendermime-error-background: rgba(194, 98, 12, 0.08);
}
/* Tutorial rail: wrap long titles onto the next line instead of ellipsizing
   (oc.css default). The [+] marker is lifted out of the flow so wrapped
   lines stay aligned with the label, not the bracket. */
.oc-toc.nb-rail a {
    position: relative;
    padding-left: 34px;
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
}
.oc-toc.nb-rail a::before { position: absolute; left: 0; }

/* One face, no italics — comments included (DESIGN.md: no italic style). */
.nb-doc .highlight .c, .nb-doc .highlight .c1, .nb-doc .highlight .cm,
.nb-doc .highlight .ch, .nb-doc .highlight .cs, .nb-doc .highlight .cp,
.nb-doc .highlight .cpf { font-style: normal; }

/* ---- Cell chrome: flat, no prompts, code on the card surface ---- */
.nb-doc .jp-Collapser,
.nb-doc .jp-InputPrompt,
.nb-doc .jp-OutputPrompt { display: none; }
.nb-doc .jp-Cell { padding: 0; }
.nb-doc .jp-CodeCell .jp-Cell-inputWrapper { margin: 24px 0 0; }
.nb-doc .jp-InputArea-editor {
    position: relative;
    border: none;
    border-radius: 4px;
    background: var(--card);
}
.nb-doc .highlight { background: transparent; }
.nb-doc .highlight pre { margin: 0; padding: 16px 64px 16px 16px; overflow-x: auto; }
.nb-doc .jp-Cell-outputWrapper { margin: 8px 0 0; }
.nb-doc .jp-OutputArea-child { margin: 8px 0 0; }
.nb-doc .jp-OutputArea-output {
    background: var(--soft);
    border-radius: 4px;
    color: var(--body);
}
.nb-doc .jp-OutputArea-output pre {
    margin: 0;
    padding: 12px 16px;
    font-size: 12.5px;
    line-height: 1.7;
    overflow-x: auto;
}
.nb-doc .jp-RenderedImage { background: #fff; border: 1px solid var(--hairline); border-radius: 4px; padding: 8px; }
.nb-doc .jp-RenderedImage img { max-width: 100%; height: auto; }

/* ---- Markdown cells in the oc type roles ---- */
.nb-doc .jp-RenderedHTMLCommon { color: var(--body); font-size: 16px; line-height: 1.5; padding: 0; }
/* The lead h1 duplicates the page header above — drop it. */
.nb-doc main > .jp-Cell:first-child .jp-RenderedMarkdown > h1:first-child { display: none; }
.nb-doc .jp-RenderedHTMLCommon h1 { font-size: 28px; font-weight: 700; color: var(--ink); line-height: 1.5; margin: 48px 0 8px; }
.nb-doc .jp-RenderedHTMLCommon h2 {
    font-size: 16px;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.5;
    margin: 48px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--hairline);
}
.nb-doc .jp-RenderedHTMLCommon h2::before { content: "## "; color: var(--mute); font-weight: 400; }
.nb-doc .jp-RenderedHTMLCommon h3 { font-size: 14px; font-weight: 700; color: var(--ink); line-height: 2; margin: 32px 0 4px; }
.nb-doc .jp-RenderedHTMLCommon h4,
.nb-doc .jp-RenderedHTMLCommon h5,
.nb-doc .jp-RenderedHTMLCommon h6 { font-size: 14px; font-weight: 700; color: var(--ink); margin: 24px 0 4px; }
.nb-doc .jp-RenderedHTMLCommon p { margin: 0 0 12px; max-width: 720px; }
.nb-doc .jp-RenderedHTMLCommon strong { color: var(--ink); font-weight: 500; }
.nb-doc .jp-RenderedHTMLCommon em { font-style: normal; color: var(--ink); }
.nb-doc .jp-RenderedHTMLCommon a { color: var(--ink); text-decoration: underline; }
.nb-doc .jp-RenderedHTMLCommon a:hover { color: var(--accent-deep); }
.nb-doc .jp-RenderedHTMLCommon blockquote {
    margin: 16px 0;
    padding: 8px 16px;
    border-left: 2px solid var(--hairline-strong);
    border-radius: 0 4px 4px 0;
    background: var(--soft);
    color: var(--mute);
}
.nb-doc .jp-RenderedHTMLCommon blockquote p { margin: 0; }
.nb-doc .jp-RenderedHTMLCommon hr { border: 0; height: 1px; background: var(--hairline); margin: 32px 0; }
/* ASCII bracket markers instead of bullet discs — the brackets ARE the icons. */
.nb-doc .jp-RenderedHTMLCommon ul { list-style: none; padding-left: 0; margin: 8px 0 16px; }
.nb-doc .jp-RenderedHTMLCommon ul > li { position: relative; padding: 2px 0 2px 28px; }
.nb-doc .jp-RenderedHTMLCommon ul > li::before {
    content: "[+]";
    position: absolute;
    left: 0;
    color: var(--accent-deep);
    font-weight: 700;
}
.nb-doc .jp-RenderedHTMLCommon ul ul > li::before { content: "[-]"; font-weight: 400; }
.nb-doc .jp-RenderedHTMLCommon ol { padding-left: 24px; margin: 8px 0 16px; }
.nb-doc .jp-RenderedHTMLCommon li > p { margin: 0; }
/* Fenced code inside markdown */
.nb-doc .jp-RenderedHTMLCommon pre {
    background: var(--card);
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 12.5px;
    line-height: 1.7;
    overflow-x: auto;
}
/* Heading anchors: hidden until the heading is hovered */
.nb-doc .anchor-link { color: var(--mute); text-decoration: none; margin-left: 6px; opacity: 0; }
.nb-doc h1:hover .anchor-link,
.nb-doc h2:hover .anchor-link,
.nb-doc h3:hover .anchor-link,
.nb-doc h4:hover .anchor-link { opacity: 0.7; }

/* ---- Tables (markdown + dataframes): the oc-table look ---- */
.nb-doc table { border-collapse: collapse; font-size: 13px; line-height: 1.5; margin: 16px 0; }
.nb-doc table th {
    text-align: left;
    font-size: 12px;
    font-weight: 700;
    color: var(--mute);
    padding: 8px 12px;
    border-bottom: 1px solid var(--hairline-strong);
    background: transparent;
    vertical-align: bottom;
}
.nb-doc table td { padding: 6px 12px; border-bottom: 1px solid var(--hairline); color: var(--body); background: transparent; }
.nb-doc table tbody tr:hover td { background: var(--soft); }
.nb-doc table tbody tr:last-child td { border-bottom: none; }
.nb-doc .jp-OutputArea-output table { margin: 0; }
.nb-doc .jp-OutputArea-output:has(table) { padding: 8px 16px; overflow-x: auto; }
</style>
'''

# Adds a [copy] button to every code cell, reusing oc.css's .code-copy styling
# (oc.js only wires .code-block elements, which notebook cells are not).
TUTORIAL_COPY_JS = '''
<script>
document.querySelectorAll('.nb-doc .jp-InputArea-editor').forEach(function (el) {
    var pre = el.querySelector('pre');
    if (!pre) return;
    var text = pre.innerText.replace(/\\s+$/, '');
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'code-copy';
    btn.textContent = '[copy]';
    btn.setAttribute('aria-label', 'Copy code');
    btn.addEventListener('click', function () {
        navigator.clipboard.writeText(text);
        btn.textContent = '[ok]';
        setTimeout(function () { btn.textContent = '[copy]'; }, 1500);
    });
    el.appendChild(btn);
});
</script>
'''

# The Tutorials nav and the /docs/tutorials.html redirect both point at
# FIRST_TUTORIAL, so a reordering of TUTORIALS must not leave them pointing at
# a notebook that is no longer first.
assert TUTORIALS[0][0] == FIRST_TUTORIAL, (
    f"FIRST_TUTORIAL is {FIRST_TUTORIAL!r} but TUTORIALS now starts with "
    f"{TUTORIALS[0][0]!r} — update FIRST_TUTORIAL."
)

# Every listed tutorial must exist on disk, and every notebook on disk must be
# listed — otherwise a deleted notebook leaves a dead rail link, and a new one
# renders to a page nothing links to.
_listed = {nb_id for nb_id, _t, _i in TUTORIALS}
_on_disk = {
    nb.relative_to(TUTORIALS_DIR).with_suffix("").as_posix()
    for nb in TUTORIALS_DIR.rglob("*.ipynb")
    if ".ipynb_checkpoints" not in nb.parts
}
assert _listed == _on_disk, (
    f"TUTORIALS is out of sync with {TUTORIALS_DIR}/ — "
    f"listed but missing: {sorted(_listed - _on_disk)}; "
    f"on disk but unlisted: {sorted(_on_disk - _listed)}"
)


# Tutorial pages use the shared site navbar (rendered once, reused per page).
def _tutorial_header_html() -> str:
    """Render the shared dark site nav with Tutorials active."""
    return env.get_template("components/_site_nav.html").render(active_nav="tutorials")


def render_notebook(nb_file: Path) -> str:
    """Convert one tutorial notebook to a full oc-styled HTML page.

    nbconvert's ``basic`` template supplies the cell markup plus the exported
    JupyterLab/pygments CSS (keyed to ``--jp-*`` variables, remapped to the oc
    palette in TUTORIAL_HEAD_EXTRAS); the chrome around it — navbar, header
    section, tutorial rail, footer — matches getting_started/benchmarks.
    """
    import nbformat
    from nbconvert import HTMLExporter

    notebook = nbformat.read(nb_file, as_version=4)
    html_exporter = HTMLExporter()
    html_exporter.template_name = "basic"
    full_html, _resources = html_exporter.from_notebook_node(notebook)

    current_notebook = nb_file.relative_to(TUTORIALS_DIR).with_suffix("").as_posix()

    # Title for the header section. The caption above it is the same on every
    # page now that the list is flat — there is no owning group to name.
    tutorial_title, tutorial_group = current_notebook, "Tutorials"
    for nb_id, nb_title, _nb_icon in TUTORIALS:
        if nb_id == current_notebook:
            tutorial_title = nb_title

    # Tutorial rail: same oc-toc component as the benchmarks algorithm rail —
    # fixed beside the column on wide screens, in-flow block on narrow ones.
    # TUTORIALS order reads start-to-finish. Extensionless links: /tutorials/<id>
    # serves <id>.html.
    rail_items = ""
    for nb_id, nb_title, _nb_icon in TUTORIALS:
        active = ' class="active"' if current_notebook == nb_id else ""
        rail_items += f'                <a href="/tutorials/{nb_id}"{active}>{nb_title}</a>\n'
    rail_items += (
        f'                <a href="{GITHUB_URL}/tree/main/tutorials"'
        ' target="_blank" rel="noopener">Notebook on GitHub</a>\n'
    )

    header_section = f'''
    <div class="oc-wrap">

        <!-- ===================== HEADER ===================== -->
        <section class="oc-section" style="padding-top: 64px; padding-bottom: 40px;">
            <p class="oc-caption" style="margin: 0;">{tutorial_group}</p>
            <h1 class="oc-display">{tutorial_title}</h1>

            <nav class="oc-toc oc-toc-flow nb-rail" aria-label="Tutorials">
                <div class="oc-toc-label">Tutorials</div>
{rail_items}            </nav>
        </section>

        <!-- ===================== NOTEBOOK ===================== -->
        <section class="oc-section" style="padding-top: 48px;">
            <div class="nb-doc jp-Notebook">
'''

    # Shared footer component, rendered through the site env (config global)
    footer_html = env.get_template("components/_footer.html").render()

    # Splice the oc chrome into nbconvert's document
    head_end = full_html.find("</head>")
    body_start = full_html.find("<body")
    body_tag_end = full_html.find(">", body_start) + 1
    body_end = full_html.find("</body>")

    head = full_html[:head_end].replace(
        "<title>Notebook</title>",
        f"<title>{tutorial_title}, TuiML Tutorials</title>"
        f'\n<link rel="canonical" href="{DOMAIN}/tutorials/{current_notebook}">',
        1,
    )

    return (
        head
        + TUTORIAL_HEAD_EXTRAS
        + "</head>\n"
        # Keep the data-jp-theme attributes: the exported lab CSS scopes its
        # theme variables to them, and the oc remapping overrides on top.
        + '<body class="landing-page antialiased overflow-x-hidden" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">\n'
        + _tutorial_header_html()
        + '\n<link rel="stylesheet" href="/static/css/oc.css">\n'
        + '<script src="/static/js/oc.js"></script>\n'
        + header_section
        + full_html[body_tag_end:body_end]
        + "\n            </div>\n        </section>\n    </div>\n"
        + footer_html
        + TUTORIAL_COPY_JS
        + full_html[body_end:]
    )


# ---------------------------------------------------------------------------
# Post-processing passes over the rendered output
# ---------------------------------------------------------------------------

def _normalize_tutorial_links() -> None:
    """Strip the ``.ipynb`` suffix from in-page ``/tutorials/...`` links.

    Notebook markdown cells (and any template) may link tutorials with the
    ``.ipynb`` extension, but the static freeze only writes ``/tutorials/x.html``,
    so those links would 404. Rewrite them to the extensionless form, which
    GitHub Pages serves back as the ``.html`` page. Must run BEFORE
    _apply_base_path (matches the un-prefixed path).
    """
    pat = re.compile(r"(/tutorials/[A-Za-z0-9_/\-]+)\.ipynb")
    for html in OUT.rglob("*.html"):
        text = html.read_text(encoding="utf-8")
        new = pat.sub(r"\1", text)
        if new != text:
            html.write_text(new, encoding="utf-8")


def _apply_base_path() -> None:
    """Rewrite root-absolute links (href/src/action="/...") to sit under BASE.

    Only touches rendered HTML. Protocol-relative (``//``) and absolute URLs are
    left alone. No-op when BASE is empty (root/custom-domain hosting).
    """
    if not BASE:
        return
    pat = re.compile(r'(href|src|action)="/(?!/)')
    repl = r'\1="' + BASE + "/"
    for html in OUT.rglob("*.html"):
        text = html.read_text(encoding="utf-8")
        new = pat.sub(repl, text)
        if new != text:
            html.write_text(new, encoding="utf-8")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def freeze() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    count = 0

    # Jinja-rendered pages
    for url, (template_name, ctx) in PAGES.items():
        render(url, template_name, **ctx)
        count += 1

    # Changelog page (from the root CHANGELOG.md)
    render(
        "/docs/changelog.html", "pages/changelog.html",
        releases=parse_changelog(), title="Changelog",
        meta_description=(
            "Release history and changelog for TuiML. Track new algorithms, bug fixes, "
            "API changes, and MCP server updates across versions."
        ),
        active_nav="changelog", page_title="Changelog",
    )
    count += 1

    # Generated API reference — every _generated/*.html rendered through Jinja so
    # it picks up the shared navbar/footer components.
    for html in sorted(DOCS_API.rglob("*.html")):
        rel = html.relative_to(DOCS_API).as_posix()
        render(f"/docs/{rel}", f"_generated/{rel}", active_nav="api", page_title="API Reference")
        count += 1

    # Tutorials — notebooks converted to HTML at build time
    for nb in sorted(TUTORIALS_DIR.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in nb.parts:
            continue
        url = f"/tutorials/{nb.relative_to(TUTORIALS_DIR).with_suffix('').as_posix()}"
        write(url, render_notebook(nb))
        count += 1

    # Plain-text/XML endpoints
    write("/robots.txt", f"User-agent: *\nAllow: /\nSitemap: {DOMAIN}/sitemap.xml\n")
    urlset = "\n".join(f"  <url><loc>{DOMAIN}{p}</loc></url>" for p in SITEMAP_PATHS)
    write(
        "/sitemap.xml",
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urlset}\n"
        "</urlset>\n",
    )
    write("/install.sh", (STATIC_SRC / "install.sh").read_text())
    write("/install", INSTALL_GUIDE)
    count += 4

    # Redirect stubs (dest prefixed with BASE so subpath hosting still lands right)
    for src, dest in REDIRECTS.items():
        write(src, REDIRECT_STUB.format(dest=BASE + dest))
        count += 1

    # Static assets
    shutil.copytree(STATIC_SRC, OUT / "static")

    # Static-hosting link fixes
    _normalize_tutorial_links()   # /tutorials/x.ipynb -> /tutorials/x (served as .html)
    _apply_base_path()            # prefix root-absolute links for subpath hosting (no-op at root)

    # Pages hygiene:
    #   .nojekyll  — REQUIRED. Jekyll strips files/dirs beginning with "_"
    #                (_generated has __init__.html and _cpp/), which would delete a
    #                chunk of the API docs otherwise.
    #   CNAME      — binds the tuiml.ai custom domain. Only written for ROOT
    #                hosting; a subpath preview (BASE set) must NOT claim the
    #                apex domain, so it's skipped there.
    (OUT / ".nojekyll").write_text("")
    if not BASE:
        (OUT / "CNAME").write_text("tuiml.ai\n")

    # 404 page. This was a byte-for-byte copy of getting_started.html, which
    # meant a broken link looked like a successful navigation to the wrong
    # page: no error, no clue, and nothing to report. Serve a real not-found
    # page instead, built from the same components so it still looks like the
    # site.
    render("/404.html", "pages/404.html")

    mode = f"subpath {BASE}" if BASE else "root (+ CNAME tuiml.ai)"
    print(f"  ✓ froze {count} pages + static assets -> {OUT.relative_to(HERE)}/  [{mode}]")


def serve(port: int = 8000) -> None:
    """Preview ``_site/`` with GitHub Pages' URL semantics.

    Pages serves the pretty URL ``/tutorials/x`` from ``tutorials/x.html`` and
    falls back to ``404.html``; the stock ``http.server`` does neither, so
    extensionless links 404 in a plain preview. This handler adds both rules.
    """
    import http.server

    class PagesHandler(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path: str) -> str:
            resolved = super().translate_path(path)
            p = Path(resolved)
            if not p.exists() and not p.suffix and p.with_suffix(".html").exists():
                return str(p.with_suffix(".html"))
            if not p.exists() and (OUT / "404.html").exists():
                return str(OUT / "404.html")
            return resolved

        def log_message(self, fmt, *args):  # quieter default output
            print(f"  {self.address_string()} - {fmt % args}")

    handler = lambda *a, **kw: PagesHandler(*a, directory=str(OUT), **kw)
    with http.server.ThreadingHTTPServer(("", port), handler) as httpd:
        print(f"  serving {OUT.relative_to(HERE)}/ at http://localhost:{port} (Pages-style URLs)")
        httpd.serve_forever()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        if not OUT.exists():
            freeze()
        serve(int(sys.argv[2]) if len(sys.argv) > 2 else 8000)
    else:
        freeze()
