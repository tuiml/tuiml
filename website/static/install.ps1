# TuiML Installer (Windows / PowerShell)
# Usage: irm https://tuiml.ai/install.ps1 | iex
#
# What this does:
#   1. Verifies it is running on Windows PowerShell 5.1+ / PowerShell 7+
#   2. Checks for an MSVC toolchain (only needed when building from source)
#   3. Installs uv if missing (Python package manager)
#   4. Asks whether to include the optional integrations
#      (scikit-learn wrappers, CapyMOA streaming wrappers), and warns if
#      a JVM-backed extra (CapyMOA, Weka) was picked without a Java runtime
#      on PATH
#   5. Installs tuiml - the latest PyPI release by default
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
#     $env:TUIML_CHANNEL = "stable"; irm https://tuiml.ai/install.ps1 | iex
#     $env:TUIML_CHANNEL = "git";    irm https://tuiml.ai/install.ps1 | iex
#
# Non-interactive / automation:
#   Set TUIML_EXTRAS to skip the prompts, e.g.
#     $env:TUIML_EXTRAS = "sklearn,capymoa,weka"; irm https://tuiml.ai/install.ps1 | iex
#     $env:TUIML_EXTRAS = "none";            irm https://tuiml.ai/install.ps1 | iex   # core only
#
# Notes for `irm | iex`:
#   This script deliberately has NO param() block. `iex` evaluates the text in
#   the caller's scope, where a param() block either fails outright or collides
#   with the automatic $args variable, so every knob is an environment variable.
#   Nothing here needs an elevated prompt; everything installs per-user.

# ---------------------------------------------------------------------------
# Configurable: which channel to install from, and where the source lives
# ---------------------------------------------------------------------------
$TuimlChannelEnv = $env:TUIML_CHANNEL   # "stable" (PyPI) or "git" (GitHub main);
                                        # empty means ask, or stable if no console
$TuimlGitUrl = $env:TUIML_GIT_URL
if (-not $TuimlGitUrl) { $TuimlGitUrl = 'git+https://github.com/tuiml/tuiml.git' }

# ---------------------------------------------------------------------------
# UI helpers
#
# Write-Host with -ForegroundColor rather than ANSI escapes: the legacy
# conhost console that still ships with Windows PowerShell 5.1 does not
# enable virtual-terminal processing by default and would print the raw
# escape bytes. Markers are ASCII for the same reason - a code page 437
# console renders "OK" fine and a check mark as a box.
# ---------------------------------------------------------------------------
function Write-Info {
    param([string]$Message)
    Write-Host "  - $Message" -ForegroundColor DarkGray
}

function Write-Ok {
    param([string]$Message)
    Write-Host '[ok] ' -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Note {
    param([string]$Message)
    Write-Host '[!]  ' -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Fail {
    param([string]$Message)
    Write-Host '[x]  ' -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Detail {
    param([string]$Message)
    Write-Host "     $Message" -ForegroundColor DarkGray
}

function Show-Banner {
    Write-Host ''
    Write-Host '  TuiML Installer' -ForegroundColor Blue
    Write-Host '  Machine Learning that agents can actually call.' -ForegroundColor DarkGray
    Write-Host ''
}

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Read-Answer {
    param([string]$Prompt)
    if (-not $script:Interactive) { return '' }
    Write-Host $Prompt -NoNewline
    try { return (Read-Host) } catch { return '' }
}

# Is there a human at the keyboard? Under `irm | iex` stdin is NOT consumed by
# the pipeline (unlike `curl | bash`), so Read-Host talks to the real console
# and no /dev/tty trick is needed. It is still absent in CI and in
# `powershell -Command` runs with redirected input, so check before prompting.
function Test-Interactive {
    try {
        return ([Environment]::UserInteractive -and (-not [Console]::IsInputRedirected))
    } catch {
        return $false
    }
}

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
function Confirm-Platform {
    # $IsWindows only exists on PowerShell 6+; on 5.1 the host is Windows by
    # definition, so treat a missing variable as Windows.
    $onWindows = $true
    if ($PSVersionTable.PSVersion.Major -ge 6) { $onWindows = $IsWindows }

    if (-not $onWindows) {
        Write-Fail 'This installer is for Windows.'
        Write-Detail 'On macOS or Linux use the shell installer:'
        Write-Detail '  curl -fsSL https://tuiml.ai/install.sh | bash'
        throw 'Unsupported platform'
    }

    if ($PSVersionTable.PSVersion.Major -lt 5) {
        Write-Fail "PowerShell $($PSVersionTable.PSVersion) is too old (need 5.1 or newer)."
        Write-Detail 'Install PowerShell 7:  winget install --id Microsoft.PowerShell -e'
        throw 'Unsupported PowerShell version'
    }

    $script:Arch = $env:PROCESSOR_ARCHITECTURE
    Write-Ok "Detected: Windows ($script:Arch), PowerShell $($PSVersionTable.PSVersion)"
}

# ---------------------------------------------------------------------------
# Prerequisite: MSVC C++ toolchain.
#
# Only a hard requirement on the git channel, which always builds the C++
# extensions from source. The PyPI channel ships prebuilt wheels for 64-bit
# Windows on CPython 3.10-3.13, so a missing compiler matters there only if no
# wheel matches this platform and Python and uv has to fall back to the sdist -
# worth a warning, not a refusal. Note there is no prebuilt wheel for Windows
# on ARM64 yet, so those machines always build from source.
# ---------------------------------------------------------------------------
function Test-MsvcToolchain {
    if (Test-CommandExists 'cl.exe') { return $true }

    # Not in a Developer Prompt: ask the VS installer where the C++ tools are.
    # vswhere.exe always lives under the 32-bit Program Files, on every arch.
    $programFilesX86 = ${env:ProgramFiles(x86)}
    if (-not $programFilesX86) { return $false }

    $vswhere = Join-Path $programFilesX86 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        try {
            $found = & $vswhere -latest -products '*' `
                -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                -property installationPath 2>$null
            if ($found) { return $true }
        } catch {
            # vswhere present but unhappy - fall through and report "not found".
        }
    }
    return $false
}

function Confirm-Compiler {
    param([ValidateSet('required', 'optional')][string]$Requirement = 'required')

    if (Test-MsvcToolchain) {
        Write-Ok 'MSVC C++ build tools found'
        return
    }

    if ($Requirement -eq 'optional' -and $script:Arch -ne 'ARM64') {
        Write-Note 'No MSVC C++ build tools found. Installing a prebuilt wheel, so this is'
        Write-Detail 'usually fine - it only matters if no wheel matches your Python version.'
        return
    }

    if ($Requirement -eq 'optional') {
        Write-Note 'No MSVC C++ build tools found, and Windows on ARM64 has no prebuilt'
        Write-Detail 'wheel yet, so TuiML will be built from source. Install the tools below,'
        Write-Detail 'or install into an x64 Python via emulation.'
    } else {
        Write-Fail 'No MSVC C++ build tools found. TuiML builds C++ extensions from source.'
    }

    Write-Detail 'Install the Visual Studio Build Tools, then re-run:'
    Write-Detail '  winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
    Write-Detail 'Or download them: https://visualstudio.microsoft.com/visual-cpp-build-tools/'
    Write-Detail 'Pick the "Desktop development with C++" workload.'

    if ($Requirement -eq 'required') { throw 'C++ toolchain missing' }
}

# ---------------------------------------------------------------------------
# Prerequisite: git (uv needs it for source installs)
# ---------------------------------------------------------------------------
function Confirm-Git {
    if (Test-CommandExists 'git') {
        Write-Ok 'git is available'
        return
    }
    Write-Fail 'git is required to install from source.'
    Write-Detail 'Install it, then re-run:  winget install --id Git.Git -e'
    throw 'git missing'
}

# ---------------------------------------------------------------------------
# PATH refresh.
#
# The uv installer edits the *persisted* user PATH, which the already-running
# process does not see. Rebuild the session PATH from the registry (plus the
# entries this session already had) so `uv` resolves without a restart.
# ---------------------------------------------------------------------------
function Update-SessionPath {
    $candidates = @(
        $env:Path,
        [Environment]::GetEnvironmentVariable('Path', 'Machine'),
        [Environment]::GetEnvironmentVariable('Path', 'User'),
        (Join-Path $env:USERPROFILE '.local\bin')
    )
    $seen = New-Object System.Collections.Generic.HashSet[string]
    $parts = New-Object System.Collections.Generic.List[string]
    foreach ($candidate in $candidates) {
        if (-not $candidate) { continue }
        foreach ($entry in $candidate.Split(';')) {
            $trimmed = $entry.Trim()
            if ($trimmed -and $seen.Add($trimmed.ToLowerInvariant())) { $parts.Add($trimmed) }
        }
    }
    $env:Path = $parts -join ';'
}

# ---------------------------------------------------------------------------
# Prerequisite: uv (Python package manager)
# ---------------------------------------------------------------------------
function Confirm-Uv {
    if (Test-CommandExists 'uv') {
        $version = (& uv --version) -replace '^uv\s+', ''
        Write-Ok "uv is already installed (v$version)"
        return
    }

    Write-Info 'uv not found. Installing it now ...'

    # Run astral's installer in a child process from a temp file rather than
    # `iex`-ing it here: it declares its own parameters and would collide with
    # this script's scope, and a child process keeps its $ErrorActionPreference
    # and StrictMode settings out of ours.
    $temp = Join-Path $env:TEMP "uv-install-$PID.ps1"
    try {
        Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1' -OutFile $temp
        $shell = if (Test-CommandExists 'powershell.exe') { 'powershell.exe' } else { 'pwsh' }
        & $shell -NoProfile -ExecutionPolicy Bypass -File $temp
        if ($LASTEXITCODE -ne 0) { throw "uv installer exited with code $LASTEXITCODE" }
    } catch {
        Write-Fail "Could not install uv: $($_.Exception.Message)"
        Write-Detail 'Install it manually, then re-run:  winget install --id astral-sh.uv -e'
        throw 'uv install failed'
    } finally {
        Remove-Item $temp -Force -ErrorAction SilentlyContinue
    }

    Update-SessionPath

    if (-not (Test-CommandExists 'uv')) {
        Write-Fail 'uv install reported success but the binary is not on PATH.'
        Write-Detail 'Open a new terminal and re-run this installer.'
        throw 'uv not on PATH'
    }
    Write-Ok 'uv installed'
}

# ---------------------------------------------------------------------------
# Optional integrations - ask the user which extras to include.
#
# TuiML core is always installed. sklearn, capymoa and weka are optional
# extras (see pyproject.toml). Sets $script:Extras to a comma-separated list,
# e.g. "sklearn,capymoa,weka".
# ---------------------------------------------------------------------------
function Select-Extras {
    $script:Extras = ''

    # Explicit env var wins - non-interactive override for automation/CI.
    if ($env:TUIML_EXTRAS) {
        if ($env:TUIML_EXTRAS -eq 'none') {
            Write-Info 'TUIML_EXTRAS=none - installing core TuiML only.'
        } else {
            $script:Extras = $env:TUIML_EXTRAS
            Write-Info "Optional extras from TUIML_EXTRAS: $script:Extras"
        }
        return
    }

    if (-not $script:Interactive) {
        Write-Info 'Non-interactive install - core TuiML only.'
        Write-Info 'Add wrappers later: $env:TUIML_EXTRAS = "sklearn,capymoa,weka" and re-run.'
        return
    }

    Write-Host ''
    Write-Host '  Optional integrations ' -NoNewline
    Write-Host '(you can always add these later)' -ForegroundColor DarkGray
    Write-Host ''

    $selected = @()
    if ((Read-Answer '  Install scikit-learn wrappers? tuiml[sklearn] [y/N] ') -match '^[Yy]') {
        $selected += 'sklearn'
    }
    if ((Read-Answer '  Install CapyMOA streaming wrappers? tuiml[capymoa], needs Java [y/N] ') -match '^[Yy]') {
        $selected += 'capymoa'
    }
    if ((Read-Answer '  Install Weka wrappers? tuiml[weka], needs Java [y/N] ') -match '^[Yy]') {
        $selected += 'weka'
    }
    $script:Extras = $selected -join ','

    if ($script:Extras) {
        Write-Ok "Will include extras: $script:Extras"
    } else {
        Write-Info 'No extras selected - installing core TuiML.'
    }
}

# ---------------------------------------------------------------------------
# CapyMOA (MOA) and Weka both run on the JVM. Installing their wheels without a
# Java runtime works, but every learner then fails at fit time, so warn while it
# is still cheap to act on. Not fatal: the rest of TuiML is unaffected.
# ---------------------------------------------------------------------------
function Confirm-JvmExtrasJava {
    $needs = @()
    if ($script:Extras -like '*capymoa*') { $needs += 'CapyMOA' }
    if ($script:Extras -like '*weka*') { $needs += 'Weka' }
    if (-not $needs) { return }
    $label = $needs -join ' and '
    if (Test-CommandExists 'java') {
        Write-Ok "Java found (required by $label)"
        return
    }
    Write-Note "$label selected but no 'java' on PATH. It needs a JVM (Java 11+)."
    Write-Detail 'Install one, then re-run:  winget install --id Microsoft.OpenJDK.21 -e'
    Write-Detail "Continuing - only the $label wrappers need it."
}

# ---------------------------------------------------------------------------
# Decide which channel to install from, setting $script:Channel.
#
# Asks when there is a console to ask on. TUIML_CHANNEL skips the question,
# and a non-interactive run (CI, containers, redirected input) takes stable,
# so an unattended install is never left waiting on input.
# ---------------------------------------------------------------------------
function Select-Channel {
    if ($TuimlChannelEnv) {
        switch -Regex ($TuimlChannelEnv) {
            '^(stable|pypi|release)$' { $script:Channel = 'stable' }
            '^(git|main|source|dev)$' { $script:Channel = 'git' }
            default {
                Write-Fail "Unknown TUIML_CHANNEL: '$TuimlChannelEnv' (use 'stable' or 'git')."
                throw 'Bad TUIML_CHANNEL'
            }
        }
        Show-Channel
        return
    }

    if (-not $script:Interactive) {
        $script:Channel = 'stable'
        Show-Channel
        Write-Info 'Non-interactive - set $env:TUIML_CHANNEL = "git" for the development build.'
        return
    }

    Write-Host ''
    Write-Host '  Which version?'
    Write-Host ''
    Write-Host '    1) Stable    ' -NoNewline
    Write-Host 'latest release from PyPI, prebuilt - recommended' -ForegroundColor DarkGray
    Write-Host '    2) Developer ' -NoNewline
    Write-Host 'newest code from GitHub main, unreleased,' -ForegroundColor DarkGray
    Write-Host '                 built from source (needs git + MSVC build tools)' -ForegroundColor DarkGray
    Write-Host ''

    $answer = Read-Answer '  Choice [1]: '
    if ($answer -match '^(2|g|git|dev|d)$') { $script:Channel = 'git' } else { $script:Channel = 'stable' }
    Show-Channel
}

function Show-Channel {
    if ($script:Channel -eq 'stable') {
        Write-Info 'Channel: stable (latest PyPI release)'
    } else {
        Write-Info 'Channel: git (GitHub main - unreleased, built from source)'
    }
}

# ---------------------------------------------------------------------------
# Install tuiml from the selected channel
# ---------------------------------------------------------------------------
function Install-Tuiml {
    # Fold any selected extras into the spec. PyPI takes "tuiml[a,b]"; a git
    # install needs PEP 508 form, "tuiml[a,b] @ git+https://...".
    $suffix = ''
    if ($script:Extras) { $suffix = "[$($script:Extras)]" }

    if ($script:Channel -eq 'stable') {
        $spec = "tuiml$suffix"
        Write-Info "Installing TuiML: $spec"
    } else {
        if ($suffix) { $spec = "tuiml$suffix @ $TuimlGitUrl" } else { $spec = $TuimlGitUrl }
        Write-Info "Installing TuiML from source: $spec"
        Write-Info 'This builds C++ extensions and may take a minute the first time.'
    }

    # --compile-bytecode writes the .pyc files during install, where uv shows
    # progress. Without it the first `tuiml` command pays that cost instead,
    # compiling bytecode for the whole dependency tree - numpy, pandas,
    # xgboost, matplotlib and the rest - while printing nothing, which reads
    # as a hang right after "Installed 2 executables".
    if (Test-CommandExists 'tuiml') {
        # Reinstall rather than upgrade: `uv tool upgrade` only ever checks
        # PyPI, so it would not move a git install onto newer commits.
        & uv tool install --compile-bytecode --reinstall --force $spec
    } else {
        & uv tool install --compile-bytecode $spec
    }
    if ($LASTEXITCODE -ne 0) { throw "uv tool install failed (exit code $LASTEXITCODE)" }

    Update-SessionPath

    if (-not (Test-CommandExists 'tuiml')) {
        Write-Note 'tuiml binary not found on PATH after install.'
        Write-Detail 'Run: uv tool update-shell'
        Write-Detail 'Then open a new terminal and re-run this installer.'
        throw 'tuiml not on PATH'
    }
    # Say so before running it: this first invocation loads the package and
    # can take a few seconds, and a silent pause here looks like a stall.
    Write-Info 'Verifying install...'
    Write-Ok "tuiml installed: $(& tuiml --version)"
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
function Confirm-Extras {
    if (-not $script:Extras) { return }
    $missing = @()
    foreach ($e in ($script:Extras -split ',')) {
        $e = $e.Trim()
        if (-not $e) { continue }
        $names = & tuiml list -s "$e." -f names 2>$null
        if ($names -match [regex]::Escape("$e.")) {
            Write-Ok "$e wrappers registered"
        } else {
            $missing += $e
        }
    }
    if (-not $missing) { return }
    Write-Note "Selected but not usable: $($missing -join ', ')"
    Write-Detail 'The backing library did not install, or this release does not ship that extra yet.'
    Write-Detail 'The newest code always has it:'
    # Backtick-escaped $env: so the command prints literally instead of being
    # interpolated to the (empty) value of the variable in this session.
    Write-Detail "  `$env:TUIML_CHANNEL = 'git'; `$env:TUIML_EXTRAS = '$($script:Extras)'; irm https://tuiml.ai/install.ps1 | iex"
}

# ---------------------------------------------------------------------------
# Final guidance
# ---------------------------------------------------------------------------
function Show-NextSteps {
    Write-Host ''
    Write-Host '  TuiML is installed.' -ForegroundColor Green
    Write-Host ''
    Write-Host '  Next: connect TuiML to your AI agent.'
    Write-Host ''
    Write-Host '    tuiml setup' -ForegroundColor Cyan
    Write-Host ''
    Write-Host '  This wizard auto-detects OpenClaw, Claude Desktop, Claude Code,'
    Write-Host '  OpenAI Codex (which covers the ChatGPT Desktop app, the Codex CLI,'
    Write-Host '  and the IDE extension - they share one config), Antigravity and its'
    Write-Host '  agy CLI, Cursor, Windsurf, Zed, VS Code, Continue, Goose, OpenCode,'
    Write-Host '  Perplexity Desktop, Cline, Roo Code, Kilo Code, and Gemini CLI.'
    Write-Host '  For NemoClaw, it prints the sandbox commands to run from inside'
    Write-Host '  the OpenClaw environment.'
    Write-Host ''
    Write-Host '  See exactly what was found, without changing anything:' -ForegroundColor DarkGray
    Write-Host '    tuiml setup --list' -ForegroundColor Cyan
    Write-Host ''
    Write-Host '  Docs:   https://tuiml.ai/getting_started.html' -ForegroundColor DarkGray
    Write-Host '  Source: https://github.com/tuiml/tuiml' -ForegroundColor DarkGray
    Write-Host ''
}

# ---------------------------------------------------------------------------
# Main
#
# Wrapped in a function so failures can `throw` instead of `exit`: this script
# is normally evaluated by `iex` in the user's own session, where a bare `exit`
# would close their console window along with the error message they need.
# ---------------------------------------------------------------------------
function Invoke-TuimlInstall {
    # Set inside the function, not at the top of the file: under `iex` the file
    # runs in the caller's own session, and a StrictMode left switched on there
    # would change how every later command they type behaves. Scoped here, it
    # covers this function and everything it calls, then reverts on return.
    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    # Windows PowerShell 5.1 still defaults to SSL3/TLS1.0 on older boxes, which
    # fails against astral.sh and PyPI. Opt into TLS 1.2 before any download.
    try {
        [Net.ServicePointManager]::SecurityProtocol =
            [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch {
        # PowerShell 7 on modern Windows manages this itself; nothing to do.
    }

    $script:Interactive = Test-Interactive

    Show-Banner
    Confirm-Platform
    Select-Channel
    # git builds from source and so needs both toolchains; the PyPI channel
    # installs a wheel and needs neither.
    if ($script:Channel -eq 'git') {
        Confirm-Compiler -Requirement required
        Confirm-Git
    } else {
        Confirm-Compiler -Requirement optional
    }
    Confirm-Uv
    Select-Extras
    Confirm-JvmExtrasJava
    Install-Tuiml
    Confirm-Extras
    Show-NextSteps
}

try {
    Invoke-TuimlInstall
} catch {
    Write-Host ''
    Write-Fail "Install aborted: $($_.Exception.Message)"
    Write-Detail 'Need a hand? https://github.com/tuiml/tuiml/issues'
    Write-Host ''
    $global:LASTEXITCODE = 1
    # Only hard-exit when nobody is watching: in CI the caller needs a non-zero
    # status, but in an interactive session `exit` would kill the console.
    if (-not $script:Interactive) { exit 1 }
}
