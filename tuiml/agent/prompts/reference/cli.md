# TuiML CLI, Install and Client Setup

Read this for shell commands: installing TuiML, wiring it into an AI client,
and the `tuiml` command line. MCP tool arguments are in `mcp-tools.md`.

## Contents
- Install, update, uninstall
- Wire TuiML into an AI client (`tuiml setup`)
- CLI commands (train, list, benchmark, tune, plot, serve, ...)
- Manual MCP client configuration
- Keeping the skill file fresh

### Install

One-liner (recommended, installs `uv` if missing, pulls the latest TuiML from GitHub, builds C++ extensions):

```bash
curl -fsSL https://tuiml.ai/install.sh | bash
```

Alternatives:

```bash
uv tool install tuiml        # isolated tool venv (recommended if uv is already installed)
pip install tuiml            # inside any existing Python environment
```

### Wire TuiML into your AI client

```bash
tuiml setup          # opens an Auto / Manual / Quit menu, default Auto
tuiml setup -y       # skip the menu and configure every detected client
tuiml setup --manual # prompt per-client (old one-by-one behaviour)
tuiml setup --list   # just show what's detected without writing anything
```

Detects Claude Desktop, Claude Code (skill file), OpenClaw, Cursor, ChatGPT Desktop, Perplexity Desktop, Codex CLI, Zed, Continue (VS Code), VS Code Copilot, Windsurf, and Goose.

### Update / uninstall

```bash
tuiml update              # CLI; the MCP twin is the `tuiml_self_update` tool,
                          # which tells the agent `restart_required: true`
uv tool install --reinstall --force tuiml   # from the shell
tuiml uninstall           # remove TuiML from every wired AI client
                          # (does NOT remove the Python package itself)
uv tool uninstall tuiml   # finally, remove the package
```

## 10. CLI

**Two rules cover almost every CLI mistake:**

1. **Everything is an option; there are no positional arguments.** Not
   `tuiml train RandomForestClassifier data.csv class` but
   `tuiml train -a RandomForestClassifier -d data.csv -t class`.
2. **Multi-word commands use hyphens, not underscores.** `tuiml select-features`,
   not `tuiml select_features`. (The MCP *tools* use underscores —
   `tuiml_select_features` — which is why this is easy to get backwards.)

```bash
# Train  (-a algorithm, -d data, -t target)
tuiml train -a RandomForestClassifier -d iris -t class --cv 10
tuiml train -a SVC -d data.csv -t label -p SimpleImputer -p StandardScaler
tuiml train -a NaiveBayesClassifier -d data.csv -t label -P '{"use_kernel_estimator": true}'
tuiml train -a LogisticRegression -d data.csv -t label -f SelectKBestSelector --save-path model.pkl
# also: --test-size, --preset, -m metrics, -o report.json, --random-seed, --stage

# List & search
tuiml list
tuiml list --type classifier --search "forest"
tuiml list --format json

# Describe
tuiml describe --name RandomForestClassifier

# Predict & evaluate  (identify the model by --model-path OR --model-id)
tuiml predict --model-path model.pkl -d data.csv -o predictions.csv
tuiml evaluate --model-path model.pkl -d data.csv -t class -m accuracy_score

# Benchmark  (-a repeats per model, -d repeats per dataset)
tuiml benchmark -a RandomForestClassifier -a SVC -a NaiveBayesClassifier -d iris -t class --cv 10
tuiml benchmark -a SVC -a RandomForestClassifier -d iris -d glass --baseline SVC
# also: --test-size, --repeats, -m metrics, -o out -f markdown|latex|csv|html, --plot

# Tuning
tuiml tune --algorithm RandomForestClassifier --data iris --target class \
           --method grid --param-grid '{"n_estimators": [50, 100]}'

# Preprocessing & feature selection
tuiml preprocess --data data.csv --steps SimpleImputer --steps StandardScaler
tuiml select-features --data data.csv --target class --method CFSSelector

# Plotting
tuiml plot --plot-type confusion_matrix --model-path model.pkl --data data.csv --target class

# Statistical tests
tuiml test-statistics --test friedman --results results.json

# Serve
tuiml serve --model-path model.pkl -p 8000
tuiml stop-server                       # omit --server-id to stop all
tuiml status

# Data tools
tuiml profile --data data.csv
tuiml read-data --data data.csv --n-rows 20
tuiml generate --generator Blobs --n-samples 1000 --n-clusters 5
tuiml upload --file-path data.csv --name sales
tuiml save --model-id a1b2c3d4 --destination model.pkl

# Agent-authored algorithms
tuiml get-skeleton --kind classifier
tuiml create-algorithm --name MyAlgo --kind classifier --code "$(cat algo.py)"
tuiml read-algorithm --name MyAlgo
tuiml edit-algorithm --name MyAlgo --old-string "..." --new-string "..." --bump-version
tuiml delete-algorithm --name MyAlgo
tuiml list-files
tuiml search-source --query "def fit"

# Setup / uninstall (MCP-client wiring)
tuiml setup                     # Auto/Manual menu
tuiml setup -y                  # configure all detected clients
tuiml uninstall                 # unwire every client

# System
tuiml info
tuiml update
tuiml restart
tuiml trace                     # follow MCP tool calls live
```

---


## Manual MCP client configuration
Let the CLI wire every detected client for you:

```bash
tuiml setup -y
```

Run the server manually (for debugging) with:

```bash
tuiml-mcp
```

If you prefer editing the client config by hand, add the following to any MCP client's `mcpServers` block (Claude Desktop, Cursor, Windsurf, ChatGPT Desktop, Perplexity Desktop, Continue):

```json
{
    "mcpServers": {
        "tuiml": { "command": "tuiml-mcp" }
    }
}
```

OpenClaw uses the key `mcp.servers`, Zed uses `context_servers`, Codex CLI uses a TOML `[mcp_servers.tuiml]` block, and Goose takes YAML. `tuiml setup` handles all of these, the JSON above is shown for reference only.


### Keeping this skill fresh

This `SKILL.md` is bundled with the `tuiml` package, its `version:` frontmatter matches the installed package version. Refresh workflow:

```bash
# 1. Upgrade the package (or call tuiml_self_update from the agent)
uv tool install --reinstall --force tuiml

# 2. Re-run setup to copy the bundled skill into each client's skills dir
tuiml setup -y
```

For pure MCP clients (Claude Desktop, Cursor, OpenClaw, …) the tool schemas are fetched live from the MCP server on every connection, no skill file to refresh.

