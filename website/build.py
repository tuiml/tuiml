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

Then ``_site/`` is a self-contained static copy of the site.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
from markupsafe import Markup, escape

HERE = Path(__file__).resolve().parent
OUT = HERE / "_site"
STATIC_SRC = HERE / "static"
TEMPLATES = HERE / "templates"
DOCS_API = TEMPLATES / "docs_api"
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

CONFIG = {
    "project_name": PROJECT_NAME,
    "app_name": APP_NAME,
    "version": APP_VERSION,
    "status": APP_STATUS,
    "version_label": f"v{APP_VERSION} {APP_STATUS}",
    "github_url": GITHUB_URL,
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
    "/docs/getting_started.html": ("docs/getting_started.html", {
        "active_nav": "docs", "page_title": "Getting Started",
    }),
    "/docs/tutorials.html": ("docs/tutorials.html", {
        "active_nav": "tutorials", "page_title": "Tutorials",
    }),
    "/docs/api-reference.html": ("docs/api-reference.html", {
        "active_nav": "api", "page_title": "API Reference", "title": "API Reference",
        "meta_description": (
            "Complete API reference for TuiML: algorithms, datasets, preprocessing, "
            "evaluation, feature selection, and MCP server modules."
        ),
    }),
    "/docs/benchmarks.html": ("docs/benchmarks.html", {
        "active_nav": "benchmarks", "page_title": "Benchmarks",
    }),
    "/docs/contributing.html": ("docs/contributing.html", {
        "active_nav": "contributing", "page_title": "Contributing",
    }),
    "/docs/remote-mcp.html": ("docs/remote-mcp.html", {
        "active_nav": "docs", "page_title": "Remote MCP Setup",
    }),
    "/docs/privacy.html": ("docs/privacy.html", {
        "title": "Privacy Policy — TuiML",
        "meta_description": (
            "Privacy policy for TuiML, an open-source agent-native ML runtime. "
            "TuiML runs locally and does not collect personal data."
        ),
        "active_nav": "docs", "page_title": "Privacy Policy",
    }),
    "/docs/terms.html": ("docs/terms.html", {
        "title": "Terms of Service — TuiML",
        "meta_description": (
            "Terms of service for the TuiML open-source project, distributed "
            "under the BSD-3-Clause license."
        ),
        "active_nav": "docs", "page_title": "Terms of Service",
    }),
    "/docs/about.html": ("docs/about.html", {
        "title": "About TuiML",
        "meta_description": (
            "About TuiML: agent-native ML runtime developed at the AI Institute, "
            "University of Waikato. Meet the team."
        ),
        "active_nav": "docs",
    }),
    # /docs/changelog.html is rendered separately (needs the parsed CHANGELOG).
}

# /about -> /docs/about.html etc. The old server answered 301; on Pages we emit
# a static HTML redirect that preserves the same destination.
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

# Core public URLs listed in sitemap.xml.
SITEMAP_PATHS = [
    "/",
    "/docs/getting_started.html",
    "/docs/tutorials.html",
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

# Tutorial list for the sidebar, organised into three tracks:
# A — agent-first (the homepage promise), B — Python APIs, C — ship it.
TUTORIAL_GROUPS = [
    ("Start Here", [
        ("quickstart/01_hello_tuiml", "Hello TuiML", "fa-solid fa-play"),
    ]),
    ("Track A · I have an agent", [
        ("llm_friendly/02_mcp_server", "1. Connect Your Agent", "fa-solid fa-bolt"),
        ("llm_friendly/01_llm_tools", "2. Tools an Agent Can Call", "fa-solid fa-robot"),
        ("llm_friendly/04_agent_doing_ml", "3. Watch an Agent Do ML", "fa-solid fa-comments"),
        ("llm_friendly/03_agentic_workflows", "4. Build an Agentic Workflow", "fa-solid fa-wand-sparkles"),
    ]),
    ("Track B · I want the Python API", [
        ("ml_simplified/01_high_level_api", "High-Level API", "fa-solid fa-rocket"),
        ("ml_simplified/02_workflow_builder", "Workflow Builder", "fa-solid fa-code"),
        ("ml_simplified/08_preprocessing", "Preprocessing", "fa-solid fa-wand-magic-sparkles"),
        ("ml_simplified/09_feature_engineering", "Feature Engineering", "fa-solid fa-filter"),
        ("ml_simplified/03_classification", "Classification", "fa-solid fa-tags"),
        ("ml_simplified/04_regression", "Regression", "fa-solid fa-chart-line"),
        ("ml_simplified/05_clustering", "Clustering", "fa-solid fa-circle-nodes"),
        ("ml_simplified/06_anomaly_detection", "Anomaly Detection", "fa-solid fa-triangle-exclamation"),
        ("ml_simplified/07_timeseries", "Time Series", "fa-solid fa-chart-area"),
        ("ml_simplified/10_experiments", "Experiments", "fa-solid fa-flask"),
    ]),
    ("Track C · I want to ship it", [
        ("deploy/01_cli", "CLI", "fa-solid fa-terminal"),
        ("deploy/02_model_serving", "Model Serving", "fa-solid fa-server"),
        ("case_studies/01_diabetes_prediction", "Case Study: Diabetes", "fa-solid fa-heart-pulse"),
        ("case_studies/02_credit_scoring", "Case Study: Credit Scoring", "fa-solid fa-credit-card"),
    ]),
]

# CSS matching getting_started.html design
TUTORIAL_LAYOUT_CSS = '''
<link rel="icon" type="image/svg+xml" href="/static/images/tuiml_logo.png">
<link rel="shortcut icon" href="/static/images/tuiml_logo.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {
    theme: {
        extend: {
            fontFamily: {
                sans: ['"IBM Plex Sans"', 'sans-serif'],
                mono: ['"IBM Plex Mono"', 'monospace'],
            }
        }
    }
}
</script>
<style>
.sidebar-link {
    display: flex;
    align-items: center;
    gap: 0.625rem;
    padding: 0.5rem 0.75rem;
    font-size: 0.875rem;
    font-weight: 500;
    color: #4b5563;
    border-radius: 0.5rem;
    transition: all 0.15s ease;
    text-decoration: none;
}
.sidebar-link:hover {
    background-color: #f3f4f6;
    color: #111827;
}
.sidebar-link.active {
    background-color: #fff7ed;
    color: #ea580c;
}
.sidebar-link i {
    width: 1rem;
    text-align: center;
    font-size: 0.875rem;
}
.sidebar-group-label {
    display: block;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9ca3af;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    padding-left: 0.5rem;
}
.sidebar-group-label:first-child {
    margin-top: 0;
}
/* Fix cell number prompts to show fully */
.jp-InputPrompt, .jp-OutputPrompt {
    min-width: 80px !important;
    flex-shrink: 0 !important;
    white-space: nowrap !important;
    overflow: visible !important;
}
</style>
'''

# Header HTML matching the docs navbar component
TUTORIAL_HEADER_HTML = f'''
<nav class="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200/50 shadow-sm">
    <div class="container mx-auto px-4 h-16 flex items-center justify-between gap-4">
        <a href="/docs/getting_started.html" class="flex items-center gap-2.5 flex-shrink-0 group">
            <div class="text-3xl text-blue-700 transition-transform group-hover:scale-110 duration-300">
                <img src="/static/images/tuiml_logo.png" alt="TuiML" class="w-8 h-8 rounded-full">
            </div>
            <div class="flex flex-col">
                <span class="font-bold text-xl tracking-tight text-gray-900 leading-none">{PROJECT_NAME}</span>
                <span class="text-gray-400 font-medium text-[10px] uppercase tracking-[0.2em] mt-0.5">Documentation</span>
            </div>
        </a>
        <div class="hidden md:flex items-center gap-6 text-sm font-medium text-gray-600">
            <a href="/docs/getting_started.html" class="hover:text-gray-900 px-3 py-1.5 rounded-full hover:bg-gray-100 transition-colors">Documentation</a>
            <a href="/docs/api-reference.html" class="hover:text-gray-900 px-3 py-1.5 rounded-full hover:bg-gray-100 transition-colors">API Reference</a>
            <a href="/docs/tutorials.html" class="text-blue-700 px-3 py-1.5 rounded-full bg-blue-50 transition-colors">Tutorials</a>
            <a href="/docs/benchmarks.html" class="hover:text-gray-900 px-3 py-1.5 rounded-full hover:bg-gray-100 transition-colors">Benchmarks</a>
            <a href="/docs/changelog.html" class="hover:text-gray-900 px-3 py-1.5 rounded-full hover:bg-gray-100 transition-colors">Changelog</a>
            <a href="/browse" class="hover:text-gray-900 px-3 py-1.5 rounded-full hover:bg-gray-100 transition-colors">Platform</a>
            <a href="{GITHUB_URL}" class="hover:text-gray-900 transition-all hover:scale-110">
                <i class="fa-brands fa-github text-xl"></i>
            </a>
        </div>
    </div>
</nav>
'''


def render_notebook(nb_file: Path) -> str:
    """Convert one tutorial notebook to a full HTML page (header/sidebar/footer)."""
    import nbformat
    from nbconvert import HTMLExporter

    notebook = nbformat.read(nb_file, as_version=4)
    html_exporter = HTMLExporter()
    html_exporter.template_name = "lab"
    full_html, _resources = html_exporter.from_notebook_node(notebook)

    current_notebook = nb_file.relative_to(TUTORIALS_DIR).with_suffix("").as_posix()

    # Flatten groups for the breadcrumb title lookup
    tutorials_flat = {
        nb_id: (nb_title, nb_icon)
        for _group, items in TUTORIAL_GROUPS
        for nb_id, nb_title, nb_icon in items
    }
    tutorial_title = tutorials_flat.get(current_notebook, (current_notebook, "fa-solid fa-file"))[0]

    subheader_html = f'''
<div class="bg-gray-50 border-b border-gray-200 sticky top-16 z-40">
    <div class="container mx-auto px-4 py-3 text-sm flex items-center justify-between">
        <div class="flex items-center gap-2 text-gray-500 overflow-x-auto whitespace-nowrap">
            <a href="/docs/tutorials.html" class="hover:text-gray-900">Tutorials</a>
            <i class="fa-solid fa-chevron-right text-[10px] text-gray-300"></i>
            <span class="text-gray-900 font-semibold">{tutorial_title}</span>
        </div>
        <div class="flex items-center gap-3">
            <span class="hidden sm:inline text-xs text-gray-400 uppercase font-bold tracking-widest">v{APP_VERSION} {APP_STATUS}</span>
            <a href="{GITHUB_URL}/stargazers"
                class="flex items-center gap-1.5 bg-white border border-gray-200 px-2.5 py-1 rounded text-xs font-medium hover:bg-gray-50 transition-colors">
                <i class="fa-solid fa-star text-yellow-400"></i> Star
            </a>
        </div>
    </div>
</div>
'''

    # Build sidebar HTML from TUTORIAL_GROUPS (extensionless links — the static
    # site serves /tutorials/<id> as <id>.html).
    sidebar_nav = ""
    for group_name, items in TUTORIAL_GROUPS:
        sidebar_nav += f'<div class="sidebar-group-label">{group_name}</div>\n'
        for nb_id, nb_title, nb_icon in items:
            active = "active" if current_notebook == nb_id else ""
            sidebar_nav += f'            <a href="/tutorials/{nb_id}" class="sidebar-link {active}"><i class="{nb_icon}"></i> {nb_title}</a>\n'

    sidebar_html = f'''
<aside class="hidden lg:block w-72 flex-shrink-0">
    <div class="sticky top-32">
        <nav class="space-y-1">
            {sidebar_nav}
        </nav>
    </div>
</aside>
'''

    # Shared footer component, rendered standalone
    footer_template = (TEMPLATES / "components" / "_footer.html").read_text()
    footer_html = Template(footer_template).render(compact=False, config=CONFIG)

    # Splice the layout into nbconvert's document
    head_end = full_html.find("</head>")
    body_start = full_html.find("<body")
    body_tag_end = full_html.find(">", body_start) + 1
    body_end = full_html.find("</body>")

    return (
        full_html[:head_end]
        + TUTORIAL_LAYOUT_CSS
        + full_html[head_end:body_tag_end].replace(
            "<body", '<body class="bg-white text-gray-900 font-sans antialiased flex flex-col min-h-screen"'
        )
        + TUTORIAL_HEADER_HTML
        + subheader_html
        + '<div class="container mx-auto px-4 py-10 flex flex-col lg:flex-row gap-16 flex-grow">'
        + sidebar_html
        + '<main class="flex-1 min-w-0">'
        + full_html[body_tag_end:body_end]
        + "</main></div>"
        + footer_html
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
        "/docs/changelog.html", "docs/changelog.html",
        releases=parse_changelog(), title="Changelog",
        meta_description=(
            "Release history and changelog for TuiML. Track new algorithms, bug fixes, "
            "API changes, and MCP server updates across versions."
        ),
        active_nav="changelog", page_title="Changelog",
    )
    count += 1

    # Generated API reference — every docs_api/*.html rendered through Jinja so
    # it picks up the shared navbar/footer components.
    for html in sorted(DOCS_API.rglob("*.html")):
        rel = html.relative_to(DOCS_API).as_posix()
        render(f"/docs/{rel}", f"docs_api/{rel}", active_nav="api", page_title="API Reference")
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
    #                (docs_api has __init__.html and _cpp/), which would delete a
    #                chunk of the API docs otherwise.
    #   CNAME      — binds the tuiml.ai custom domain. Only written for ROOT
    #                hosting; a subpath preview (BASE set) must NOT claim the
    #                apex domain, so it's skipped there.
    (OUT / ".nojekyll").write_text("")
    if not BASE:
        (OUT / "CNAME").write_text("tuiml.ai\n")

    # 404 fallback so unknown paths render the site's own not-found styling.
    err = OUT / "docs" / "getting_started.html"
    if err.exists():
        shutil.copyfile(err, OUT / "404.html")

    mode = f"subpath {BASE}" if BASE else "root (+ CNAME tuiml.ai)"
    print(f"  ✓ froze {count} pages + static assets -> {OUT.relative_to(HERE)}/  [{mode}]")


if __name__ == "__main__":
    freeze()
