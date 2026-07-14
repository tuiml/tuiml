#!/usr/bin/env bash
# TuiML Installer
# Usage: curl -fsSL https://tuiml.ai/install.sh | bash
#
# What this does:
#   1. Detects your OS (macOS or Linux)
#   2. Verifies a C/C++ compiler is available (TuiML has C++ extensions)
#   3. Installs uv if missing (Python package manager)
#   4. Installs tuiml from the latest source on GitHub
#      (always the freshest fixes, not just the last PyPI release)
#   5. Verifies the install
#   6. Prompts you to run `tuiml setup` to wire up your AI agent
#
# This script is idempotent and safe to re-run.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable: where the latest source lives
# ---------------------------------------------------------------------------
TUIML_GIT_URL="${TUIML_GIT_URL:-git+https://github.com/tuiml/tuiml.git}"

# ---------------------------------------------------------------------------
# Colors and UI helpers
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    BLUE=$'\033[34m'
    CYAN=$'\033[36m'
    GREEN=$'\033[32m'
    YELLOW=$'\033[33m'
    RED=$'\033[31m'
    NC=$'\033[0m'
else
    BOLD="" DIM="" BLUE="" CYAN="" GREEN="" YELLOW="" RED="" NC=""
fi

info()    { echo "${DIM}·${NC} $*"; }
success() { echo "${GREEN}✓${NC} $*"; }
warn()    { echo "${YELLOW}!${NC} $*"; }
err()     { echo "${RED}✗${NC} $*" >&2; }

banner() {
    echo
    echo "${BOLD}${BLUE}  TuiML Installer${NC}"
    echo "${DIM}  Machine Learning that agents can actually call.${NC}"
    echo
}

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
detect_os() {
    case "$OSTYPE" in
        darwin*)        OS="macos" ;;
        linux-gnu*)     OS="linux" ;;
        linux*)         OS="linux" ;;
        msys*|cygwin*|win32*)
            err "Windows is not supported by this installer."
            echo "  Use WSL (Windows Subsystem for Linux), or install via pip:"
            echo "    pip install git+https://github.com/tuiml/tuiml.git"
            exit 1
            ;;
        *)
            err "Unsupported OS: $OSTYPE"
            echo "  Install manually: pip install git+https://github.com/tuiml/tuiml.git"
            exit 1
            ;;
    esac
    success "Detected: $OS"
}

# ---------------------------------------------------------------------------
# Prerequisite: C/C++ compiler (TuiML builds C++ extensions from source)
# ---------------------------------------------------------------------------
ensure_compiler() {
    if command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1; then
        success "C++ compiler found"
        return 0
    fi

    err "No C++ compiler found. TuiML builds C++ extensions from source."
    if [[ "$OS" == "macos" ]]; then
        echo "  Install Apple Command Line Tools, then re-run:"
        echo "    xcode-select --install"
    else
        echo "  Install build tools, e.g. on Ubuntu/Debian:"
        echo "    sudo apt-get update && sudo apt-get install -y build-essential"
        echo "  On Fedora/RHEL:"
        echo "    sudo dnf install -y gcc-c++ make"
    fi
    exit 1
}

# ---------------------------------------------------------------------------
# Prerequisite: git (uv needs it for source installs)
# ---------------------------------------------------------------------------
ensure_git() {
    if command -v git >/dev/null 2>&1; then
        success "git is available"
        return 0
    fi
    err "git is required to install from source."
    if [[ "$OS" == "macos" ]]; then
        echo "  Run: xcode-select --install"
    else
        echo "  Run: sudo apt-get install -y git    (Debian/Ubuntu)"
        echo "    or sudo dnf install -y git        (Fedora/RHEL)"
    fi
    exit 1
}

# ---------------------------------------------------------------------------
# Prerequisite: uv (Python package manager)
# ---------------------------------------------------------------------------
ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV_VERSION=$(uv --version 2>/dev/null | awk '{print $2}')
        success "uv is already installed (v${UV_VERSION})"
        return 0
    fi

    info "uv not found. Installing it now ..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        err "Neither curl nor wget found. Install one and re-run."
        exit 1
    fi

    # uv installs to ~/.local/bin or ~/.cargo/bin; make sure it's on PATH for this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        err "uv install reported success but the binary is not on PATH."
        echo "  Add this to your shell profile and re-run:"
        echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
        exit 1
    fi
    success "uv installed"
}

# ---------------------------------------------------------------------------
# Install tuiml from GitHub source (latest main branch)
# ---------------------------------------------------------------------------
install_tuiml() {
    info "Installing TuiML from latest source: ${DIM}${TUIML_GIT_URL}${NC}"
    info "This builds C++ extensions and may take a minute the first time."

    if command -v tuiml >/dev/null 2>&1; then
        # Reinstall to pull the latest commits (uv tool upgrade only checks PyPI)
        uv tool install --reinstall --force "$TUIML_GIT_URL"
    else
        uv tool install "$TUIML_GIT_URL"
    fi

    if ! command -v tuiml >/dev/null 2>&1; then
        warn "tuiml binary not found on PATH after install."
        echo "  Run: uv tool update-shell"
        echo "  Then restart your shell and re-run this installer."
        exit 1
    fi
    success "tuiml installed: $(tuiml --version)"
}

# ---------------------------------------------------------------------------
# Final guidance
# ---------------------------------------------------------------------------
print_next_steps() {
    echo
    echo "${BOLD}${GREEN}  ✓ TuiML is installed.${NC}"
    echo
    echo "  ${BOLD}Next:${NC} connect TuiML to your AI agent."
    echo
    echo "    ${CYAN}tuiml setup${NC}"
    echo
    echo "  This wizard auto-detects OpenClaw, Claude Desktop, Claude Code,"
    echo "  ChatGPT Desktop, Perplexity Desktop, Cursor, Windsurf, OpenAI Codex CLI,"
    echo "  Zed, VS Code, Continue, and Goose. For NemoClaw, it prints the"
    echo "  sandbox commands to run from inside the OpenClaw environment."
    echo
    echo "  ${DIM}Docs:   https://tuiml.ai/docs/getting_started.html${NC}"
    echo "  ${DIM}Source: https://github.com/tuiml/tuiml${NC}"
    echo
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
banner
detect_os
ensure_compiler
ensure_git
ensure_uv
install_tuiml
print_next_steps
