# TuiML MCP Tools Reference

Read this when calling `tuiml_*` tools. Arguments are FLAT (see the dialect
table in SKILL.md). Every tool returns `{"status": "success"|"error", ...}`;
on error read `error`, `suggestion`, `recovery_tool` and `recovery_params`.

## Contents
- Tool table by group (workflow, data, discovery, serving, system, export, authoring)
- Auto-discovery of custom components
- Tool call examples (train, benchmark, serve, tune, export)
- Programmatic execution from Python
- MCP resources (datasets, this skill)
- Error envelope
- Auto-research loop with agent-authored algorithms

### MCP Tools

30 tools total. All follow `tuiml_<verb>_<noun>` naming.

**Core workflow**

| Tool | Purpose |
|------|---------|
| `tuiml_train` | Train any model with preprocessing and CV |
| `tuiml_predict` | Predict using model_id or model path |
| `tuiml_evaluate` | Evaluate trained model with metrics |
| `tuiml_benchmark` | Compare multiple algorithms across datasets (Python twin: `tuiml.Benchmark`); forecasters get time-ordered expanding-window folds |
| `tuiml_tune` | Grid, random, or bayesian search over hyperparameters |
| `tuiml_test_statistics` | Friedman / Wilcoxon / Nemenyi / Quade / ANOVA / aligned-Friedman test on experiment results |

**Data & preparation**

| Tool | Purpose |
|------|---------|
| `tuiml_upload_data` | Upload CSV/ARFF/TSV/JSON content for other tools |
| `tuiml_read_data` | Preview rows from a dataset (head/tail/sample/indices) |
| `tuiml_profile_data` | Summary stats: shape, dtypes, missingness, cardinality |
| `tuiml_generate_data` | Generate synthetic datasets (blobs, moons, Friedman, …) |
| `tuiml_preprocess` | Apply preprocessors as a standalone step (with atomic stages: split/impute/balance/scale/encode/discretize) |
| `tuiml_select_features` | Feature selection as a standalone step |
| `tuiml_plot` | Standard plots (confusion matrix, ROC, PCA, feature importance, …) |

**Discovery**

| Tool | Purpose |
|------|---------|
| `tuiml_list` | List components by category (`algorithm`, `dataset`, `preprocessing`, `feature`, `splitting`, `custom`, `all`) |
| `tuiml_describe` | Get parameter schema for any component |

> `tuiml_list(search="forest")` filters by keyword.
> `tuiml_list(category="custom")` lists user-authored algorithms with versions and best scores.
> `tuiml_list(category="custom", include_runs=true)` adds full experiment run history.
> `tuiml_list` also accepts `type` (e.g. `"classifier"`, `"regressor"`) and pagination via `limit` / `offset`.

**Serving**

| Tool | Purpose |
|------|---------|
| `tuiml_save_model` | Save trained model to custom path |
| `tuiml_serve_model` | Start REST API for a model (params: `model_id`, `model_path`, `port`, `host`) |
| `tuiml_stop_server` | Stop a serving server (param: `server_id`; omit to stop all) |
| `tuiml_server_status` | Check server status |

**Self-introspection & upgrade**

| Tool | Purpose |
|------|---------|
| `tuiml_system_info` | Installed version, install method, package path, latest PyPI version, update_available flag |
| `tuiml_self_update` | Upgrade to the latest release (params: `dry_run`, `target_version`). Restart the client afterward. |
| `tuiml_restart` | Restart tuiml-mcp processes (param: `include_self`, default true) |

**Export**

| Tool | Purpose |
|------|---------|
| `tuiml_export_notebook` | Export the current MCP session as a Jupyter notebook (params: `path`, `title`) |

**Agent-authored algorithms (requires `TUIML_ALLOW_USER_ALGORITHMS=1`)**

| Tool | Purpose |
|------|---------|
| `tuiml_get_skeleton` | Return a fill-in-the-blanks algorithm template (classifier or regressor) |
| `tuiml_create_algorithm` | AST-validate and register new Python source as a named + versioned algorithm |
| `tuiml_edit_algorithm` | str_replace patch on a user algorithm, read first, then edit a unique string |
| `tuiml_read_algorithm` | Get full source of any algorithm (user or built-in) with line numbers |
| `tuiml_list_files` | List all algorithm source files (built-in and user) with paths |
| `tuiml_search_source` | Grep inside algorithm source files by regex pattern |
| `tuiml_delete_algorithm` | Remove a version (or all versions) from disk |

### Auto-Discovery

Any component registered with `@classifier`, `@regressor`, `@transformer`, etc. is automatically discoverable through all MCP tools. No tool definitions need updating. This is how agent-authored algorithms become first-class citizens of `tuiml_train` / `tuiml_benchmark` the moment they are registered.

### MCP Tool Examples

Train with preprocessing:

```json
{
    "tool": "tuiml_train",
    "arguments": {
        "algorithm": "SVC",
        "data": "data.csv",
        "target": "label",
        "preprocessing": [
            {"name": "SimpleImputer", "strategy": "median"},
            "StandardScaler"
        ],
        "feature_selection": {"name": "SelectKBestSelector", "k": 10},
        "cv": 10,
        "random_seed": 42
    }
}
```

Compare algorithms:

```json
{
    "tool": "tuiml_benchmark",
    "arguments": {
        "algorithms": ["RandomForestClassifier", "SVC", "NaiveBayesClassifier"],
        "data": "iris",
        "target": "class",
        "cv": 10,
        "metrics": ["accuracy_score", "f1_score"]
    }
}
```

Serve model as API:

```json
{
    "tool": "tuiml_serve_model",
    "arguments": {
        "model_id": "a1b2c3d4",
        "port": 8000,
        "host": "127.0.0.1"
    }
}
```

The response carries an `auth_token`. Every endpoint except `/` and `/health`
requires it as `Authorization: Bearer <auth_token>`; the returned
`example_curl` already includes the header. The token is generated per server
and appears only in that response, so keep it if you plan to call the API.

Hyperparameter tuning:

```json
{
    "tool": "tuiml_tune",
    "arguments": {
        "algorithm": "RandomForestClassifier",
        "data": "iris",
        "target": "class",
        "method": "grid",
        "param_grid": {"n_estimators": [50, 100, 200], "max_depth": [5, 10]},
        "cv": 5,
        "scoring": "f1_weighted"
    }
}
```

Export session as notebook:

```json
{
    "tool": "tuiml_export_notebook",
    "arguments": {
        "path": "experiment.ipynb",
        "title": "Iris Classification Experiment"
    }
}
```

### Programmatic Tool Execution

```python
from tuiml.agent import execute_tool, get_tools_for_llm, invoke, agent

result = execute_tool("tuiml_train", algorithm="RandomForestClassifier", data="iris", target="class")
tools = get_tools_for_llm(format="mcp")

# Pydantic-AI agent pre-loaded with all TuiML tools
ai_agent = agent()
```

### MCP Resources

- `tuiml://dataset/{name}` — built-in dataset info and a 5-row preview (e.g. `tuiml://dataset/iris`)
- `skill://tuiml/SKILL.md` and `skill://tuiml/reference/*.md`, `skill://tuiml/workflows/*.md` — this skill,
  served over MCP (SEP-2640 `skills/list` / `skills/get`), so clients without a skills directory can read it

### Error Handling

All tools return structured responses:

```json
{
    "status": "success",
    "error": "message (if status=error)",
    "suggestion": "recovery hint",
    "recovery_tool": "tool to call",
    "recovery_params": {}
}
```

---


### Auto-Research Loop (agent-authored algorithms)

**When to use:** the agent has an algorithmic idea ("what if I bag shallow decision trees with bootstrap weights proportional to label noise?") and wants to implement, run, and compare it against shipped algorithms, all without leaving the conversation.

**Prerequisite:** export `TUIML_ALLOW_USER_ALGORITHMS=1` in the shell that launches the MCP server, then restart the client. Source is AST-filtered, not sandboxed, the trust model is that the agent is local.

**The loop**

```
1. tuiml_list(category="custom")
   →  see what algorithms already exist + versions + best scores

2. tuiml_get_skeleton(kind="classifier")
   →  template source

3. <fill in fit() / predict() / __init__ hyperparams>

4. tuiml_create_algorithm(name="NoisyTreeBag",
                          kind="classifier",
                          code=<source>,
                          version="1.0.0")
   →  registered as NoisyTreeBag (latest)
                 and NoisyTreeBag_v1_0_0 (pinned)

5. tuiml_train(algorithm="NoisyTreeBag",
               data="iris", target="target", cv=5)
   →  baseline score

6. <iterate: fix a bug or tweak logic>
   tuiml_read_algorithm(name="NoisyTreeBag")         →  current source
   tuiml_search_source(query="def fit", name="NoisyTreeBag")  →  locate the line
   tuiml_edit_algorithm(name="NoisyTreeBag",
                        old_string="...",
                        new_string="...",
                        bump_version=True)            →  saved as v1.0.1, re-registered

7. tuiml_benchmark(
      algorithms=["NoisyTreeBag_v1_0_0",
                  "NoisyTreeBag_v1_0_1",
                  "RandomForestClassifier",
                  "XGBoostClassifier"],
      data=["iris", "glass", "breast-cancer"],
      target="target", cv=10,
      metrics=["accuracy_score", "f1_score"])
   →  ranked comparison with mean ± std per dataset

8. tuiml_test_statistics(
      results=<tuiml_benchmark result>,
      test="friedman", metric="accuracy_score")   # then test="nemenyi" for pairs
   →  tells you whether your variant is significantly better

9. tuiml_list(category="custom", include_runs=true)
   →  full history: all versions, best scores, run counts

10. tuiml_export_notebook(path="research.ipynb", title="NoisyTreeBag Study")
    →  reproducible Jupyter notebook of the full session
```

**Versioning rules**

- `name` must be a valid Python identifier, usually equal to the class name.
- `version` must be semver (`MAJOR.MINOR.PATCH`). Bump it on every change.
- Every version is kept on disk at `~/.tuiml/user_algorithms/<name>/<version>/algorithm.py` with a `metadata.json` next to it (class name, kind, source hash, description). Nothing is deleted until `tuiml_delete_algorithm` is called.
- The bare class name (`NoisyTreeBag`) always resolves to the most recently registered version. Pinned aliases (`NoisyTreeBag_v1_0_0`) resolve to exact versions, use these when comparing variants in one `tuiml_benchmark`.
- All versions are re-registered at MCP server startup, so agent work survives restarts.

**Guardrails enforced by `tuiml_create_algorithm`**

- Forbidden imports: `subprocess`, `socket`, `os`, `shutil`, `urllib`, `requests`, `httpx`, `http`, `ftplib`, `smtplib`, `ctypes`, `webbrowser`, `pty`.
- Forbidden calls: `eval`, `exec`, `compile`, `__import__`, `open`, `input`.
- Must contain at least one `@classifier` or `@regressor` decorated class.
- Declared `kind` must match the base class of the imported class.

Any rejection returns `status: error, error_type: UnsafeSource` with the specific rule that fired. Bump the version instead of overwriting when you want to keep the history.

