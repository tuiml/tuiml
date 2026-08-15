#!/usr/bin/env bash
# TuiML Installer
# Usage: curl -fsSL https://tuiml.ai/install.sh | bash
#
# What this does:
#   1. Detects your OS (macOS or Linux)
#   2. Verifies a C/C++ compiler is available (TuiML has C++ extensions)
#   3. Installs uv if missing (Python package manager)
#   4. Asks whether to include the optional integrations
#      (gradient-boosting backends, scikit-learn / CapyMOA / Weka wrappers),
#      and warns if a JVM-backed extra (CapyMOA, Weka) was picked without a
#      Java runtime on PATH
#   4b. Detects your GPU (CUDA / ROCm / Apple Metal) and, if one is present,
#      offers the PyTorch-backed neural models and the TabICL foundation
#      model. Both work on CPU too, so they are still offered — just not
#      defaulted to yes
#   5. Installs tuiml — the latest PyPI release by default
#   6. Verifies the install
#   7. Prompts you to run `tuiml setup` to wire up your AI agent
#
# This script is idempotent and safe to re-run.
#
# Which version you get:
#   The installer asks, offering the stable PyPI release (prebuilt, no
#   compiler needed) or the newest code from GitHub main (unreleased, built
#   from source). Stable is the default and what a non-interactive run takes.
#
#   Skip the question with TUIML_CHANNEL:
#     curl -fsSL https://tuiml.ai/install.sh | TUIML_CHANNEL=stable bash
#     curl -fsSL https://tuiml.ai/install.sh | TUIML_CHANNEL=git bash
#
# Non-interactive / automation:
#   Set TUIML_EXTRAS to skip the prompts, e.g.
#     curl -fsSL https://tuiml.ai/install.sh | TUIML_EXTRAS="boosting,sklearn,capymoa,weka" bash
#     curl -fsSL https://tuiml.ai/install.sh | TUIML_EXTRAS="none" bash   # core only
#     curl -fsSL https://tuiml.ai/install.sh | TUIML_EXTRAS="torch,foundation" bash
#
#   Skip GPU probing with TUIML_GPU=cuda|rocm|mps|cpu (default: auto-detect).

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable: which channel to install from, and where the source lives
# ---------------------------------------------------------------------------
TUIML_CHANNEL="${TUIML_CHANNEL:-}"   # "stable" (PyPI) or "git" (GitHub main);
                                     # empty means ask, or stable if no terminal
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
            err "This is the macOS/Linux installer; you are on Windows."
            echo "  Use the PowerShell installer instead — open PowerShell and run:"
            echo "    irm https://tuiml.ai/install.ps1 | iex"
            echo "  ${DIM}Or run this script from inside WSL (Windows Subsystem for Linux).${NC}"
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
# Accelerator detection.
#
# Sets ACCEL to one of cuda|rocm|mps|cpu, and ACCEL_DESC to something a human
# can read. Used to decide whether it is worth offering the neural extras:
# FT-Transformer, SAINT, NODE, N-BEATS, NHITS, PatchTST and the TabICL
# foundation model all run on CPU, but a GPU is worth roughly an order of
# magnitude on every one of them.
#
# Detection is deliberately conservative — a missing tool means "no", never an
# error. This only ever gates a question, so a wrong guess costs nothing.
# ---------------------------------------------------------------------------
detect_accelerator() {
    ACCEL="cpu"
    ACCEL_DESC="CPU only"
    ACCEL_VRAM_MB=0

    # Honour an explicit override before probing anything.
    if [[ -n "${TUIML_GPU:-}" && "${TUIML_GPU}" != "auto" ]]; then
        ACCEL="${TUIML_GPU}"
        ACCEL_DESC="forced by TUIML_GPU=${TUIML_GPU}"
        return 0
    fi

    # NVIDIA. Query name and VRAM in one go; if nvidia-smi exists but fails
    # (driver mismatch is common), fall through to CPU rather than trusting it.
    if command -v nvidia-smi >/dev/null 2>&1; then
        local line
        if line=$(nvidia-smi --query-gpu=name,memory.total \
                             --format=csv,noheader,nounits 2>/dev/null | head -1); then
            if [[ -n "$line" ]]; then
                ACCEL="cuda"
                # nvidia-smi already reports the vendor in the product name
                # ("NVIDIA GeForce RTX 4090"), so do not prepend it again.
                ACCEL_DESC="${line%%,*}"
                ACCEL_VRAM_MB="$(echo "$line" | awk -F', *' '{print $2}' | tr -dc '0-9')"
                [[ -n "$ACCEL_VRAM_MB" ]] || ACCEL_VRAM_MB=0
                return 0
            fi
        fi
    fi

    # AMD ROCm.
    if command -v rocminfo >/dev/null 2>&1 || [[ -d /opt/rocm ]]; then
        ACCEL="rocm"
        ACCEL_DESC="AMD GPU (ROCm)"
        return 0
    fi

    # Apple Silicon — torch reaches the GPU through Metal (MPS). Intel Macs
    # have no such path, so check the architecture rather than just the OS.
    if [[ "$OS" == "macos" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        ACCEL="mps"
        ACCEL_DESC="Apple Silicon GPU (Metal)"
        return 0
    fi

    return 0
}

# ---------------------------------------------------------------------------
# Offer the PyTorch-backed extras, informed by what we just detected.
#
# Two separate questions, because they are two separate decisions:
#   tuiml[torch]      — six neural models TuiML implements itself
#   tuiml[foundation] — TabICL, a *pretrained* model whose weights are
#                       downloaded on first use (~150 MB)
#
# Neither is offered by default on a CPU-only machine: they work, but slowly
# enough that a user who did not ask for them would not thank us.
# ---------------------------------------------------------------------------
select_neural_extras() {
    # TUIML_EXTRAS is the non-interactive contract and already covers these.
    [[ -n "${TUIML_EXTRAS:-}" ]] && return 0
    if [[ ! -t 1 ]] || [[ ! -r /dev/tty ]]; then
        return 0
    fi

    local ans default_hint gpu_found="no"
    [[ "$ACCEL" == "cuda" || "$ACCEL" == "rocm" || "$ACCEL" == "mps" ]] && gpu_found="yes"

    echo
    echo "  ${BOLD}Neural models${NC} ${DIM}(PyTorch)${NC}"
    if [[ "$gpu_found" == "yes" ]]; then
        success "Accelerator detected: ${BOLD}${ACCEL_DESC}${NC}"
        default_hint="[Y/n] "
    else
        info "No GPU detected — ${ACCEL_DESC}. These still run, just slowly."
        default_hint="[y/N] "
    fi

    # Warn when the card is too small to be comfortable. 8 GB is where the
    # transformer models stop needing their batch size lowered.
    if [[ "$ACCEL" == "cuda" && "$ACCEL_VRAM_MB" -gt 0 && "$ACCEL_VRAM_MB" -lt 8000 ]]; then
        warn "Only ${ACCEL_VRAM_MB} MB of VRAM — you may need a smaller batch_size."
    fi

    echo "    ${DIM}FT-Transformer, SAINT, NODE, N-BEATS, NHITS, PatchTST${NC}"
    printf "  Install neural models? ${DIM}tuiml[torch]${NC} %s" "$default_hint"
    read -r ans < /dev/tty || ans=""
    if [[ "$gpu_found" == "yes" ]]; then
        [[ ! "$ans" =~ ^[Nn] ]] && EXTRAS="${EXTRAS:+$EXTRAS,}torch"
    else
        [[ "$ans" =~ ^[Yy] ]] && EXTRAS="${EXTRAS:+$EXTRAS,}torch"
    fi

    # The foundation model only makes sense alongside torch, which it pulls in
    # anyway — so only ask once torch is on the list.
    if [[ "${EXTRAS:-}" == *torch* ]]; then
        echo
        echo "    ${DIM}TabICL — a pretrained model that predicts without training.${NC}"
        echo "    ${DIM}Downloads a ~150 MB checkpoint on first use, into${NC}"
        echo "    ${DIM}~/.cache/huggingface. Code and weights are BSD-3-Clause,${NC}"
        echo "    ${DIM}the same license as TuiML — nothing to accept.${NC}"
        printf "  Install the TabICL foundation model? ${DIM}tuiml[foundation]${NC} [y/N] "
        read -r ans < /dev/tty || ans=""
        [[ "$ans" =~ ^[Yy] ]] && EXTRAS="${EXTRAS:+$EXTRAS,}foundation"
    fi
}

# ---------------------------------------------------------------------------
# Prerequisite: C/C++ compiler.
#
# Only a hard requirement on the git channel, which always builds the C++
# extensions from source. The PyPI channel ships prebuilt wheels, so a missing
# compiler matters there only if no wheel matches this platform and Python and
# uv has to fall back to the sdist — worth a warning, not a refusal.
# ---------------------------------------------------------------------------
ensure_compiler() {
    local requirement="${1:-required}"

    if command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1; then
        success "C++ compiler found"
        return 0
    fi

    if [[ "$requirement" == "optional" ]]; then
        warn "No C++ compiler found. Installing a prebuilt wheel, so this is"
        echo "  ${DIM}usually fine — it only matters if no wheel matches your platform.${NC}"
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
# Optional integrations — ask the user which extras to include.
#
# TuiML core is always installed. sklearn, capymoa and weka are optional
# extras (see pyproject.toml). We prompt for each, reading from /dev/tty so this
# works even under `curl | bash` (where stdin is the script, not the keyboard).
# Sets the global EXTRAS to a comma-separated list, e.g. "sklearn,capymoa,weka".
# ---------------------------------------------------------------------------
select_extras() {
    EXTRAS=""

    # Explicit env var wins — non-interactive override for automation/CI.
    if [[ -n "${TUIML_EXTRAS:-}" ]]; then
        if [[ "$TUIML_EXTRAS" == "none" ]]; then
            info "TUIML_EXTRAS=none — installing core TuiML only."
        else
            EXTRAS="$TUIML_EXTRAS"
            info "Optional extras from TUIML_EXTRAS: ${BOLD}${EXTRAS}${NC}"
        fi
        return 0
    fi

    # Need a real terminal to prompt. Under `curl | bash`, stdin is the script,
    # so read from /dev/tty. If there's no terminal (CI, Docker), default core.
    if [[ ! -t 1 ]] || [[ ! -r /dev/tty ]]; then
        info "Non-interactive install — core TuiML only."
        info "Add extras later: ${DIM}TUIML_EXTRAS=boosting,sklearn,capymoa,weka${NC} and re-run."
        return 0
    fi

    local ans
    echo
    echo "  ${BOLD}Optional integrations${NC} ${DIM}(you can always add these later)${NC}"
    echo

    printf "  Install scikit-learn wrappers? ${DIM}tuiml[sklearn]${NC} [y/N] "
    read -r ans < /dev/tty || ans=""
    [[ "$ans" =~ ^[Yy] ]] && EXTRAS="sklearn"

    # XGBoost / LightGBM / CatBoost were required dependencies until they were
    # made optional. Defaulted to yes because they are the usual accuracy
    # ceiling on tabular data and users upgrading from an older TuiML expect
    # them present.
    echo "    ${DIM}XGBoost, LightGBM, CatBoost — usually the strongest on tabular data${NC}"
    printf "  Install gradient-boosting backends? ${DIM}tuiml[boosting]${NC} [Y/n] "
    read -r ans < /dev/tty || ans=""
    [[ ! "$ans" =~ ^[Nn] ]] && EXTRAS="${EXTRAS:+$EXTRAS,}boosting"

    printf "  Install CapyMOA streaming wrappers? ${DIM}tuiml[capymoa], needs Java${NC} [y/N] "
    read -r ans < /dev/tty || ans=""
    [[ "$ans" =~ ^[Yy] ]] && EXTRAS="${EXTRAS:+$EXTRAS,}capymoa"

    printf "  Install Weka wrappers? ${DIM}tuiml[weka], needs Java${NC} [y/N] "
    read -r ans < /dev/tty || ans=""
    [[ "$ans" =~ ^[Yy] ]] && EXTRAS="${EXTRAS:+$EXTRAS,}weka"

    if [[ -n "$EXTRAS" ]]; then
        success "Will include extras: ${BOLD}${EXTRAS}${NC}"
    else
        info "No extras selected — installing core TuiML."
    fi
}

# ---------------------------------------------------------------------------
# CapyMOA (MOA) and Weka both run on the JVM. Installing their wheels without a
# Java runtime works, but every learner then fails at fit time, so warn while it
# is still cheap to act on. Not fatal: the rest of TuiML is unaffected.
# ---------------------------------------------------------------------------
check_jvm_extras_java() {
    local needs=""
    [[ "${EXTRAS:-}" == *capymoa* ]] && needs="CapyMOA"
    [[ "${EXTRAS:-}" == *weka* ]] && needs="${needs:+$needs and }Weka"
    [[ -n "$needs" ]] || return 0
    if command -v java >/dev/null 2>&1; then
        success "Java found (required by $needs)"
        return 0
    fi
    warn "$needs selected but no 'java' on PATH. It needs a JVM (Java 11+)."
    if [[ "$OS" == "macos" ]]; then
        echo "  Install one, then re-run:  brew install openjdk"
    else
        echo "  Install one, then re-run:  sudo apt-get install -y default-jre"
        echo "                         or: sudo dnf install -y java-latest-openjdk"
    fi
    echo "  ${DIM}Continuing — only the $needs wrappers need it.${NC}"
}

# ---------------------------------------------------------------------------
# Decide which channel to install from, setting the global CHANNEL.
#
# Asks when there is a terminal to ask on. TUIML_CHANNEL skips the question,
# and a non-interactive run (CI, Docker, `| bash` with no tty) takes stable,
# so an unattended install is never left waiting on input.
# ---------------------------------------------------------------------------
select_channel() {
    # Explicit env var wins — non-interactive override for automation/CI.
    if [[ -n "$TUIML_CHANNEL" ]]; then
        case "$TUIML_CHANNEL" in
            stable|pypi|release) CHANNEL="stable" ;;
            git|main|source|dev) CHANNEL="git" ;;
            *)
                err "Unknown TUIML_CHANNEL: '${TUIML_CHANNEL}' (use 'stable' or 'git')."
                exit 1
                ;;
        esac
        _announce_channel
        return 0
    fi

    # Under `curl | bash` stdin is the script itself, so prompt via /dev/tty.
    if [[ ! -t 1 ]] || [[ ! -r /dev/tty ]]; then
        CHANNEL="stable"
        _announce_channel
        info "Non-interactive — set ${DIM}TUIML_CHANNEL=git${NC} for the development build."
        return 0
    fi

    local ans
    echo
    echo "  ${BOLD}Which version?${NC}"
    echo
    echo "    ${BOLD}1${NC}) Stable    ${DIM}latest release from PyPI, prebuilt — recommended${NC}"
    echo "    ${BOLD}2${NC}) Developer ${DIM}newest code from GitHub main, unreleased,${NC}"
    echo "                 ${DIM}built from source (needs git + a C++ compiler)${NC}"
    echo
    printf "  Choice [1]: "
    read -r ans < /dev/tty || ans=""

    case "$ans" in
        2|g|git|dev|d) CHANNEL="git" ;;
        *)             CHANNEL="stable" ;;
    esac
    _announce_channel
}

_announce_channel() {
    if [[ "$CHANNEL" == "stable" ]]; then
        info "Channel: ${BOLD}stable${NC} ${DIM}(latest PyPI release)${NC}"
    else
        info "Channel: ${BOLD}git${NC} ${DIM}(GitHub main — unreleased, built from source)${NC}"
    fi
}

# ---------------------------------------------------------------------------
# Install tuiml from the selected channel
# ---------------------------------------------------------------------------
install_tuiml() {
    # Fold any selected extras into the spec. PyPI takes "tuiml[a,b]"; a git
    # install needs PEP 508 form, "tuiml[a,b] @ git+https://...".
    local suffix="" spec
    [[ -n "${EXTRAS:-}" ]] && suffix="[${EXTRAS}]"

    if [[ "$CHANNEL" == "stable" ]]; then
        spec="tuiml${suffix}"
        info "Installing TuiML: ${DIM}${spec}${NC}"
    else
        if [[ -n "$suffix" ]]; then
            spec="tuiml${suffix} @ ${TUIML_GIT_URL}"
        else
            spec="$TUIML_GIT_URL"
        fi
        info "Installing TuiML from source: ${DIM}${spec}${NC}"
        info "This builds C++ extensions and may take a minute the first time."
    fi

    # --compile-bytecode writes the .pyc files during install, where uv shows
    # progress. Without it the first `tuiml` command pays that cost instead,
    # compiling bytecode for the whole dependency tree — numpy, pandas,
    # xgboost, matplotlib and the rest — while printing nothing, which reads
    # as a hang right after "Installed 2 executables".
    if command -v tuiml >/dev/null 2>&1; then
        # Reinstall rather than upgrade: `uv tool upgrade` only ever checks
        # PyPI, so it would not move a git install onto newer commits.
        uv tool install --compile-bytecode --reinstall --force "$spec"
    else
        uv tool install --compile-bytecode "$spec"
    fi

    if ! command -v tuiml >/dev/null 2>&1; then
        warn "tuiml binary not found on PATH after install."
        echo "  Run: uv tool update-shell"
        echo "  Then restart your shell and re-run this installer."
        exit 1
    fi
    # Say so before running it: this first invocation loads the package and
    # can take a few seconds, and a silent pause here looks like a stall.
    info "Verifying install..."
    success "tuiml installed: $(tuiml --version)"
}

# ---------------------------------------------------------------------------
# Confirm the extras that were asked for actually landed.
#
# uv only *warns* when a release does not carry a requested extra and still
# exits 0, so "tuiml[weka]" against a release predating that extra installs
# core and reports success while `import tuiml.weka` fails later. Each backend
# registers namespaced hub keys (sklearn.SVC, capymoa.HoeffdingTree,
# weka.J48), so asking the registry is a direct check that the wrappers are
# usable rather than merely requested.
# ---------------------------------------------------------------------------
verify_extras() {
    [[ -n "${EXTRAS:-}" ]] || return 0
    local e missing="" sel
    IFS=',' read -ra sel <<< "$EXTRAS"
    for e in "${sel[@]}"; do
        e="${e//[[:space:]]/}"
        [[ -n "$e" ]] || continue

        # `torch` and `boosting` are not wrapper namespaces. The algorithms they
        # unlock are registered under bare names and are listed whether or not
        # the library is installed — that is the whole point of the lazy-import
        # contract. So the registry cannot tell us anything about them; check
        # that the libraries themselves import instead.
        local probe="" label=""
        case "$e" in
            torch)    probe="import torch"; label="PyTorch available — neural models ready to fit" ;;
            boosting) probe="import xgboost, lightgbm, catboost"; label="XGBoost, LightGBM and CatBoost available" ;;
        esac
        if [[ -n "$probe" ]]; then
            if uv run --no-project python -c "$probe" >/dev/null 2>&1 \
               || python3 -c "$probe" >/dev/null 2>&1; then
                success "$label"
            else
                missing="${missing:+$missing, }$e"
            fi
            continue
        fi

        if tuiml list -s "${e}." -f names 2>/dev/null | grep -q "${e}\."; then
            success "${e} wrappers registered"
        else
            missing="${missing:+$missing, }$e"
        fi
    done
    [[ -n "$missing" ]] || return 0
    warn "Selected but not usable: ${missing}"
    echo "  The backing library did not install, or this release does not ship"
    echo "  that extra yet. The newest code always has it:"
    echo "  ${DIM}curl -fsSL https://tuiml.ai/install.sh | TUIML_CHANNEL=git TUIML_EXTRAS=\"${EXTRAS}\" bash${NC}"
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
    echo "  OpenAI Codex (which covers the ChatGPT Desktop app, the Codex CLI,"
    echo "  and the IDE extension — they share one config), Antigravity and its"
    echo "  agy CLI, Cursor, Windsurf, Zed, VS Code, Continue, Goose, OpenCode,"
    echo "  Perplexity Desktop, Cline, Roo Code, Kilo Code, and Gemini CLI."
    echo "  For NemoClaw, it prints the sandbox commands to run from inside"
    echo "  the OpenClaw environment."
    echo
    echo "  ${DIM}See exactly what was found, without changing anything:${NC}"
    echo "    ${CYAN}tuiml setup --list${NC}"
    echo
    echo "  ${DIM}Docs:   https://tuiml.ai/getting_started.html${NC}"
    echo "  ${DIM}Source: https://github.com/tuiml/tuiml${NC}"
    echo
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
banner
detect_os
select_channel
# git builds from source and so needs both toolchains; the PyPI channel
# installs a wheel and needs neither.
if [[ "$CHANNEL" == "git" ]]; then
    ensure_compiler required
    ensure_git
else
    ensure_compiler optional
fi
ensure_uv
detect_accelerator
select_extras
select_neural_extras
check_jvm_extras_java
install_tuiml
verify_extras
print_next_steps
