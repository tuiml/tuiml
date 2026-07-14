"""TuiML landing site + documentation server."""

from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, Response, RedirectResponse
import uvicorn
from pathlib import Path

from core.config import settings


app = FastAPI(
    title=settings.APP_NAME,
    description="TuiML landing and documentation site",
    version=settings.APP_VERSION,
    # Disable auto-generated Swagger/ReDoc/OpenAPI endpoints. This is a
    # public-facing docs site, not an API; exposing /docs, /redoc, and
    # /openapi.json leaks every internal route and route signature.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Tutorials directory path - check local tutorials/ first, then fall back to sibling tuiml/tutorials
_app_dir = Path(__file__).resolve().parent
TUTORIALS_DIR = _app_dir / "tutorials" if (_app_dir / "tutorials").exists() else _app_dir.parent / "tuiml" / "tutorials"

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Inject global config into all Jinja2 templates
templates.env.globals["config"] = {
    "project_name": settings.PROJECT_NAME,
    "app_name": settings.APP_NAME,
    "version": settings.APP_VERSION,
    "status": settings.APP_STATUS,
    "version_label": f"v{settings.APP_VERSION} {settings.APP_STATUS}",
    "github_url": settings.GITHUB_URL,
    "copyright_year": settings.COPYRIGHT_YEAR,
    "default_meta_description": (
        "TuiML Hub is an open-source machine learning platform for discovering algorithms, "
        "datasets, tutorials, benchmarks, and agent-ready MCP workflows."
    ),
}


def _build_sitemap_urls(request: Request) -> list[str]:
    """Return the core public URLs that should appear in the XML sitemap."""
    base = str(request.base_url).rstrip("/")
    public_paths = [
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
    return [f"{base}{path}" for path in public_paths]

# ============================================================================
# Documentation Pages (Jinja2-rendered, override StaticFiles mount)
# ============================================================================

@app.get("/docs/getting_started.html", response_class=HTMLResponse)
async def docs_readme(request: Request):
    """Documentation landing page."""
    return templates.TemplateResponse("docs/getting_started.html", {"request": request, "active_nav": "docs", "page_title": "Getting Started"})


@app.get("/docs/tutorials.html", response_class=HTMLResponse)
async def docs_tutorials(request: Request):
    """Tutorials landing page."""
    return templates.TemplateResponse("docs/tutorials.html", {"request": request, "active_nav": "tutorials", "page_title": "Tutorials"})


@app.get("/docs/api-reference.html", response_class=HTMLResponse)
async def docs_api_reference(request: Request):
    """API Reference page."""
    return templates.TemplateResponse(
        "docs/api-reference.html",
        {
            "request": request,
            "active_nav": "api",
            "page_title": "API Reference",
            "title": "API Reference",
            "meta_description": (
                "Complete API reference for TuiML: algorithms, datasets, preprocessing, "
                "evaluation, feature selection, and MCP server modules."
            ),
        }
    )


@app.get("/docs/benchmarks.html", response_class=HTMLResponse)
async def docs_benchmarks(request: Request):
    """Benchmarks page."""
    return templates.TemplateResponse("docs/benchmarks.html", {"request": request, "active_nav": "benchmarks", "page_title": "Benchmarks"})


@app.get("/docs/changelog.html", response_class=HTMLResponse)
async def docs_changelog(request: Request):
    """Changelog page - parsed from CHANGELOG.md."""
    import re
    changelog_path = Path(__file__).resolve().parent / "CHANGELOG.md"
    releases = []
    try:
        content = changelog_path.read_text()
        # Parse releases: ## [version] - date
        release_blocks = re.split(r'\n## ', content)[1:]  # skip header
        for block in release_blocks:
            lines = block.strip().split('\n')
            header = lines[0]
            match = re.match(r'\[(.+?)\]\s*-\s*(.+)', header)
            version = match.group(1) if match else header.strip('[] ')
            date = match.group(2).strip() if match else ""
            sections = []
            current_section = None
            for line in lines[1:]:
                sec_match = re.match(r'### (.+)', line)
                if sec_match:
                    current_section = {"type": sec_match.group(1), "items": []}
                    sections.append(current_section)
                elif line.startswith('- ') and current_section is not None:
                    current_section["items"].append(line[2:])
            releases.append({"version": version, "date": date, "sections": sections})
    except FileNotFoundError:
        pass
    return templates.TemplateResponse(
        "docs/changelog.html",
        {"request": request, "releases": releases, "title": "Changelog",
         "meta_description": (
             "Release history and changelog for TuiML. Track new algorithms, bug fixes, "
             "API changes, and MCP server updates across versions."
         ),
         "active_nav": "changelog", "page_title": "Changelog"},
    )


@app.get("/docs/contributing.html", response_class=HTMLResponse)
async def docs_contributing(request: Request):
    """Contributing guide page."""
    return templates.TemplateResponse(
        "docs/contributing.html",
        {"request": request, "active_nav": "contributing", "page_title": "Contributing"},
    )


@app.get("/docs/remote-mcp.html", response_class=HTMLResponse)
async def docs_remote_mcp(request: Request):
    """Self-hosted remote MCP setup guide."""
    return templates.TemplateResponse(
        "docs/remote-mcp.html",
        {"request": request, "active_nav": "docs", "page_title": "Remote MCP Setup"},
    )


# ============================================================================
# Web UI Routes (Jinja2 Templates)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Landing page."""
    return templates.TemplateResponse(
        "pages/index.html",
        {
            "request": request,
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
        }
    )


@app.get("/projects", response_class=HTMLResponse)
async def projects(request: Request):
    """Community build board — open projects, missing algorithms, and integrations."""
    return templates.TemplateResponse(
        "pages/projects.html",
        {
            "request": request,
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
        }
    )


@app.get("/about", response_class=HTMLResponse)
async def about_redirect(request: Request):
    """Redirect /about to /docs/about.html."""
    return RedirectResponse(url="/docs/about.html", status_code=301)


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_redirect(request: Request):
    return RedirectResponse(url="/docs/privacy.html", status_code=301)


@app.get("/terms", response_class=HTMLResponse)
async def terms_redirect(request: Request):
    return RedirectResponse(url="/docs/terms.html", status_code=301)


@app.get("/docs/privacy.html", response_class=HTMLResponse)
async def docs_privacy(request: Request):
    """Privacy policy page."""
    return templates.TemplateResponse(
        "docs/privacy.html",
        {
            "request": request,
            "title": "Privacy Policy — TuiML",
            "meta_description": (
                "Privacy policy for TuiML, an open-source agent-native ML runtime. "
                "TuiML runs locally and does not collect personal data."
            ),
            "active_nav": "docs",
            "page_title": "Privacy Policy",
        }
    )


@app.get("/docs/terms.html", response_class=HTMLResponse)
async def docs_terms(request: Request):
    """Terms of service page."""
    return templates.TemplateResponse(
        "docs/terms.html",
        {
            "request": request,
            "title": "Terms of Service — TuiML",
            "meta_description": (
                "Terms of service for the TuiML open-source project, distributed "
                "under the BSD-3-Clause license."
            ),
            "active_nav": "docs",
            "page_title": "Terms of Service",
        }
    )


@app.get("/docs/about.html", response_class=HTMLResponse)
async def docs_about(request: Request):
    """About page (now under docs)."""
    return templates.TemplateResponse(
        "docs/about.html",
        {
            "request": request,
            "title": "About TuiML",
            "meta_description": (
                "About TuiML: agent-native ML runtime developed at the AI Institute, "
                "University of Waikato. Meet the team."
            ),
            "active_nav": "docs",
        }
    )


# ============================================================================
# Tutorial Routes (Jupyter Notebook Rendering)
# ============================================================================

@app.get("/tutorials/{notebook_path:path}", response_class=HTMLResponse)
async def render_tutorial(request: Request, notebook_path: str):
    """Render a Jupyter notebook as HTML with header, sidebar, and footer."""
    # Ensure .ipynb extension
    if not notebook_path.endswith('.ipynb'):
        notebook_path += '.ipynb'

    notebook_name = notebook_path  # Keep full path for lookups
    full_path = TUTORIALS_DIR / notebook_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Tutorial not found: {notebook_path}")

    try:
        import nbformat
        import re
        from nbconvert import HTMLExporter

        # Read the notebook
        with open(full_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)

        # Convert to HTML using lab template
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'lab'

        (full_html, resources) = html_exporter.from_notebook_node(notebook)

        # Tutorial list for sidebar, organised into three tracks:
        # A — agent-first (the homepage promise), B — Python APIs, C — ship it.
        tutorial_groups = [
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

        current_notebook = notebook_name.replace('.ipynb', '')

        # Flatten for lookups
        tutorials_flat = {}
        for group_name, items in tutorial_groups:
            for nb_id, nb_title, nb_icon in items:
                tutorials_flat[nb_id] = (nb_title, nb_icon)

        # Find where to inject the layout
        head_end = full_html.find('</head>')
        body_start = full_html.find('<body')
        body_tag_end = full_html.find('>', body_start) + 1
        body_end = full_html.find('</body>')

        # CSS matching getting_started.html design
        layout_css = '''
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

        # Header HTML matching docs navbar component
        header_html = f'''
<nav class="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200/50 shadow-sm">
    <div class="container mx-auto px-4 h-16 flex items-center justify-between gap-4">
        <a href="/docs/getting_started.html" class="flex items-center gap-2.5 flex-shrink-0 group">
            <div class="text-3xl text-blue-700 transition-transform group-hover:scale-110 duration-300">
                <img src="/static/images/tuiml_logo.png" alt="TuiML" class="w-8 h-8 rounded-full">
            </div>
            <div class="flex flex-col">
                <span class="font-bold text-xl tracking-tight text-gray-900 leading-none">{settings.PROJECT_NAME}</span>
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
            <a href="{settings.GITHUB_URL}" class="hover:text-gray-900 transition-all hover:scale-110">
                <i class="fa-brands fa-github text-xl"></i>
            </a>
        </div>
    </div>
</nav>
'''

        # Subheader with breadcrumb
        tutorial_info = tutorials_flat.get(current_notebook, (current_notebook, "fa-solid fa-file"))
        tutorial_title = tutorial_info[0]
        subheader_html = f'''
<div class="bg-gray-50 border-b border-gray-200 sticky top-16 z-40">
    <div class="container mx-auto px-4 py-3 text-sm flex items-center justify-between">
        <div class="flex items-center gap-2 text-gray-500 overflow-x-auto whitespace-nowrap">
            <a href="/docs/tutorials.html" class="hover:text-gray-900">Tutorials</a>
            <i class="fa-solid fa-chevron-right text-[10px] text-gray-300"></i>
            <span class="text-gray-900 font-semibold">{tutorial_title}</span>
        </div>
        <div class="flex items-center gap-3">
            <span class="hidden sm:inline text-xs text-gray-400 uppercase font-bold tracking-widest">v{settings.APP_VERSION} {settings.APP_STATUS}</span>
            <a href="{settings.GITHUB_URL}/stargazers"
                class="flex items-center gap-1.5 bg-white border border-gray-200 px-2.5 py-1 rounded text-xs font-medium hover:bg-gray-50 transition-colors">
                <i class="fa-solid fa-star text-yellow-400"></i> Star
            </a>
        </div>
    </div>
</div>
'''

        # Build sidebar HTML dynamically from tutorial_groups
        sidebar_nav = ""
        for group_name, items in tutorial_groups:
            sidebar_nav += f'<div class="sidebar-group-label">{group_name}</div>\n'
            for nb_id, nb_title, nb_icon in items:
                active = "active" if current_notebook == nb_id else ""
                sidebar_nav += f'            <a href="/tutorials/{nb_id}.ipynb" class="sidebar-link {active}"><i class="{nb_icon}"></i> {nb_title}</a>\n'

        sidebar_html = f'''
<aside class="hidden lg:block w-72 flex-shrink-0">
    <div class="sticky top-32">
        <nav class="space-y-1">
            {sidebar_nav}
        </nav>
    </div>
</aside>
'''

        # Read footer from shared component (compact version for notebook pages)
        footer_path = Path(__file__).resolve().parent / "templates" / "components" / "_footer.html"
        footer_template = footer_path.read_text()
        # Render the Jinja2 template with compact=true
        from jinja2 import Template
        footer_html = Template(footer_template).render(compact=False, config=templates.env.globals["config"])

        # Inject into the full HTML
        html_content = (
            full_html[:head_end] +
            layout_css +
            full_html[head_end:body_tag_end].replace('<body', '<body class="bg-white text-gray-900 font-sans antialiased flex flex-col min-h-screen"') +
            header_html +
            subheader_html +
            '<div class="container mx-auto px-4 py-10 flex flex-col lg:flex-row gap-16 flex-grow">' +
            sidebar_html +
            '<main class="flex-1 min-w-0">' +
            full_html[body_tag_end:body_end] +
            '</main></div>' +
            footer_html +
            full_html[body_end:]
        )
        return HTMLResponse(content=html_content)

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="nbconvert not installed. Run: pip install nbconvert nbformat"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rendering notebook: {str(e)}")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.APP_VERSION}


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots_txt(request: Request):
    """Serve robots.txt for public crawl directives."""
    sitemap_url = str(request.url_for("sitemap_xml"))
    return PlainTextResponse(
        f"User-agent: *\nAllow: /\nSitemap: {sitemap_url}\n"
    )


@app.get("/install.sh", response_class=PlainTextResponse)
async def install_sh():
    """Serve the install script at the root URL.

    Allows: curl -fsSL https://tuiml.ai/install.sh | bash
    """
    install_path = Path(__file__).parent / "static" / "install.sh"
    return PlainTextResponse(
        install_path.read_text(),
        media_type="text/x-shellscript",
    )


@app.get("/install", response_class=PlainTextResponse)
async def install_instructions_for_agents():
    """Machine-readable install + setup instructions for LLM agents.

    Paired with the "Agent" tab on the landing page — a user pastes
    "Help me install https://tuiml.ai/install" into any agent with web access;
    the agent fetches this plain-text guide and follows the steps.
    """
    text = (
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
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")


@app.get("/sitemap.xml", name="sitemap_xml")
async def sitemap_xml(request: Request):
    """Serve a minimal XML sitemap for core public pages."""
    urls = _build_sitemap_urls(request)
    urlset = "\n".join(
        f"  <url><loc>{url}</loc></url>" for url in urls
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urlset}\n"
        "</urlset>\n"
    )
    return Response(content=xml, media_type="application/xml")


# Catch-all route for generated documentation pages (Jinja2-rendered)
# This MUST come after all explicit /docs/ routes so they take precedence
@app.get("/docs/{path:path}", response_class=HTMLResponse)
async def docs_catchall(request: Request, path: str):
    """Render generated doc pages through Jinja2 for navbar/footer components."""
    template_path = f"docs_api/{path}"
    if not path.endswith(".html"):
        template_path = f"docs_api/{path}/index.html"
    try:
        return templates.TemplateResponse(
            template_path,
            {"request": request, "active_nav": "api", "page_title": "API Reference"},
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Page not found")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
