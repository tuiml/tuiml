# Changelog

All notable changes to TuiML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.8] - 2026-08-02

### Changed (breaking)
- **The MCP server now targets the 2.x MCP SDK** (`mcp>=2.0.0`, was
  `mcp>=1.0.0`). The 2.0 SDK replaced the low-level decorator API the server
  was built on — `@server.list_tools()`, `@server.call_tool()` and the
  resource decorators — with handlers registered on the `Server(...)`
  constructor, so the two are not source-compatible and there is no single
  code path that serves both. Because the old pin was unbounded, a fresh
  `pip install tuiml` / `uv tool install tuiml` resolved mcp 2.0.0 and
  `tuiml-mcp` died at startup with `AttributeError: 'Server' object has no
  attribute 'list_tools'`, which surfaced in Claude Desktop and Cursor as
  `connection closed: calling "initialize"`. If you pin `mcp` yourself, move
  to 2.x; an SDK that is too old is now reported at startup with an
  actionable message instead of an `AttributeError`.

### Fixed
- **Tuning and benchmark progress reaches the client again.** Progress was
  streamed as `notifications/message` log entries, which 2.x gates behind a
  per-request opt-in that no client sends by default. It now goes out as
  spec-correct `notifications/progress` against the client's `progressToken`,
  so `tuiml_tune` and `tuiml_benchmark` report live progress in clients that
  ask for it, with the iteration total attached for tuning runs.
- **Tool arguments are validated again.** Input validation against each
  tool's `inputSchema` was performed by the 1.x decorator; on 2.x it has to
  be done in the handler, without which a malformed argument reached the tool
  as a confusing `TypeError` instead of a clear validation error.
- **`tuiml-mcp --info` reported version `1.0.0`** regardless of the installed
  version. It now reports the package version, which is also what the server
  advertises in the initialize handshake.
- **`tuiml_plot(plot_type='tree')` failed for every agent-trained model** with
  "The tree model is not fitted yet." `tuiml_train` saves a `Workflow`, and
  the `tree` branch handed that wrapper to `plot_tree` instead of unwrapping
  `workflow.model_` the way `feature_importance` already did — so a `model_id`
  could never plot, however tree-based the model was. Asking for a tree plot
  of a model that genuinely has no tree now returns a clear message naming the
  model and pointing at `feature_importance`, instead of claiming it is
  unfitted. `NaiveBayesClassifier` is excluded correctly: it stores a
  list-of-lists of *probability* estimators under `estimators_`, which a
  looser check mistook for a tree ensemble and drew nonsense from.

## [0.1.7] - 2026-08-02

### Removed (breaking)
- **Flat agent adapter modules.** The five framework adapters moved under
  `tuiml.agent.adapters`. The old module paths are gone, with no compatibility
  shims — pre-1.0, one blessed path per concept:

  | Old | New |
  |-----|-----|
  | `from tuiml.agent import openai` | `from tuiml.agent.adapters import openai` |
  | `from tuiml.agent import anthropic` | `from tuiml.agent.adapters import anthropic` |
  | `from tuiml.agent.langchain import get_tools` | `from tuiml.agent.adapters.langchain import get_tools` |
  | `from tuiml.agent.crewai import get_tools` | `from tuiml.agent.adapters.crewai import get_tools` |
  | `from tuiml.agent.pydantic_ai import get_tools` | `from tuiml.agent.adapters.pydantic_ai import get_tools` |

  `tuiml.agent.registry` moved to `tuiml.agent.tools._components`, and
  `tuiml.agent.restart_util` to `tuiml.agent.tools.system.restart`.
- **`tuiml.agent()` as a top-level function.** Binding it at the root shadowed
  the `tuiml.agent` *package*, so `import tuiml.agent as x` handed back the
  function instead of the module. Use `from tuiml.agent import agent` (or
  `tuiml.agent.agent()`). The returned Pydantic-AI agent is unchanged.
- **`tuiml_search` MCP tool.** It was never registered in any tool table, so
  calling it always returned `Unknown tool`. Keyword search over components is
  `tuiml_list(search=...)`, which has always worked.

### Fixed
- **Fitting a tree on data with missing values never returned.**
  `DecisionTreeClassifier`, `DecisionTreeRegressor` and both random forests
  hung and then died with a `RecursionError` on any `X` containing `NaN` —
  including the built-in `vote` dataset — with a traceback pointing at
  impurity arithmetic several frames from the cause.

  In the split search, `np.argsort` sorts `NaN` last and the candidate mask
  `sorted_col[1:] != sorted_col[:-1]` counts a boundary at a `NaN` as valid,
  because `NaN != NaN` is True. The chosen threshold was then the midpoint of
  two values one of which was `NaN`, so the threshold itself was `NaN`; every
  `x <= NaN` is False, so the builder put all rows on one side and recursed on
  a subproblem identical to its parent forever. `build_classifier_tree`
  already routed `NaN` away from the C++ builder and into the Python one, so
  the Python path was the designated missing-value path — it just could not
  handle them.

  Candidate thresholds now come from observed values only, so a threshold is
  always finite. Impurities are normalised over that subset and the gain is
  scaled by the fraction observed — C4.5's correction, which stops a
  mostly-missing feature outranking a complete one by being scored on an
  easier subset; `j48` already dropped `NaN` the same way. Missing rows route
  right, matching `NaN <= t`, and a split that fails to partition falls back
  to a leaf. Verified a no-op on complete data: predictions are unchanged
  across `gini`/`entropy`/`gain_ratio`,
  `squared_error`/`friedman_mse`/`absolute_error` and `RandomForest` on
  diabetes, iris, glass and cpu. `vote` now trains at 0.961 ± 0.025 under
  10-fold CV.
- **A batch prediction and a single-sample prediction could disagree on the
  same row.** `predict_single_numpy` sent missing values to whichever child
  held more training samples, which matched neither the structure the tree was
  actually fitted with nor the flattened batch predictor that `predict()`
  uses. Fitting, batch prediction and single-sample prediction now share one
  rule. Previously unreachable, because fitting on such data crashed first.
- **`list_algorithms()` hid every native algorithm after a bare `import
  tuiml`.** `tuiml/__init__.py` never imported `tuiml.algorithms`, and the
  `@classifier` / `@regressor` decorators only register on module import — so
  discovery reported 103 of 189 components and `RandomForestClassifier` was
  absent from the catalog entirely until something else happened to import it.
  Machines with entries in `~/.tuiml/user_algorithms` never saw this, because
  loading those pulls the imports in as a side effect. `train()` was
  unaffected (it resolves lazily), as was the MCP `tuiml_list` tool.
- **Every plot logged a matplotlib warning.** The stylesheet set
  `font.weight: 'medium'`, which no bundled font advertises as a face, so each
  figure emitted `findfont: Failed to find font weight medium, now using 400`.
  Same rendering, without the noise. The comparison-table renderer set it too.
- **A quarter of the See Also links in the API docs were broken, and the site
  hid it.** 147 of 544 pointed at pages that do not exist. The resolver only
  indexed *classes*, so every `:func:` reference fell through to a fallback
  that guessed a path — producing URLs like
  `/docs/utils/serialization/load_model.html` for a function documented on
  its module's page — while `:mod:` references produced `/docs//train.html`
  or a dead `#`. Modules, packages and functions are now indexed too, with
  anchors, and a reference that resolves to nothing renders as plain text
  rather than a link that only fails once clicked.

  The reason this went unnoticed: `404.html` was a byte-for-byte copy of
  `getting_started.html`, so every broken link looked like a successful
  navigation to the Getting Started page. It is now a real not-found page.
- **The next release would have shipped two files on the old version.**
  `bump_version.py` still pointed at `tuiml/agent/SKILL.md`, which moved to
  `tuiml/agent/prompts/` in the agent reorg. The script printed `SKIP (not
  found)` and carried on, so the skill file's advertised version would have
  silently lagged. A pre-flight check now verifies every registered file
  exists and still matches its pattern, and aborts before writing anything —
  a rejected bump can no longer leave the tree half-updated. Verified by
  running a real bump against a throwaway clone.
- **Every version bump broke `test_schemas.py`.** It hardcoded the release
  version twice, but the registered regex matched only the constructor
  argument (`version="0.1.6"`), not the assertion (`== "0.1.6"`), so a bump
  rewrote one and left the other. The test now uses an arbitrary version
  string — the real endpoint passes `tuiml.__version__` in, so the test only
  ever checked that the field round-trips — and is no longer version-tracked.
- **The published MCP tutorial showed a stale version.** A stored output cell
  in `tutorials/llm_friendly/02_mcp_server.ipynb` prints `TuiML version:` and
  was never bumped, so readers on tuiml.ai saw whatever it last said. Now
  registered.
- **The Pages workflow ignored two of its own build inputs.** It fired only on
  `website/**`, but `build.py` also reads `tutorials/` and `CHANGELOG.md` from
  the repo root. A tutorial-only edit changed what the site would publish
  without triggering the build that publishes it, leaving the 19 `/tutorials/`
  pages behind the repo until an unrelated `website/` change fired a deploy.
- **`tuiml_self_update(dry_run=True)` failed in a dev checkout.** The
  editable-install guard ran before the dry-run branch, so a call that promises
  to change nothing and report what would happen returned an error instead. A
  dry run now always succeeds, reporting the refusal as its prediction.
- **The notebook-export smoke test had been dead since the `tuiml/agent/`
  reorg.** It imported `_SESSION_CALLS` and `execute_export_notebook` from
  `tuiml.agent.tools`, which moved to `tools/_state.py` and behind
  `execute_tool`, so the test errored on import rather than checking anything —
  leaving every generated notebook cell unverified. Fixed, and it again
  executes all 21 generated cells.
- **Eight `tuiml.sklearn` wrappers were unusable inside a pipeline.** The
  wrappers read `get_params()` off `__dict__`, which also picked up the
  attributes TuiML's own base classes set (`FeatureSelector` sets `k` and
  `threshold`, `FeatureExtractor` sets `n_components`). Cloning a step for each
  CV fold fed those back through the constructor, and scikit-learn rejected
  them: `sklearn.SelectKBest: unknown parameter(s) ['threshold']`. Affected
  `SelectKBest`, `SelectPercentile`, `SelectFdr`, `SelectFpr`, `SelectFwe`,
  `GenericUnivariateSelect`, `VarianceThreshold` and `AdditiveChi2Sampler`.
  `get_params()` now reports exactly the parameters the wrapper was built with.
- **`random_seed` never reached the `tuiml.sklearn` wrappers.** Seed injection
  looked for `random_state` in the constructor signature, and every wrapper
  takes `**params`, so `tuiml.train(..., "random_seed": 42)` left them
  unseeded and runs were not reproducible. The seed now also consults
  `get_parameter_schema()`, so it reaches wrappers whose backing estimator
  accepts a seed and still skips those that do not.
- **`tuiml uninstall` could not unwire OpenClaw or OpenCode.** Both kinds fell
  through the dispatcher to `unknown client kind`, so `tuiml setup` could add
  an entry that `tuiml uninstall` then refused to remove. OpenClaw is removed
  via `openclaw mcp remove` (falling back to a direct config edit) and OpenCode
  through its `mcp` key.
- **Docstring examples that could not run.** `tuiml/sklearn/__init__.py`
  documented `tuiml.train("sklearn.RandomForestClassifier", {...}, cv=5)` and
  `tuiml.benchmark(...)`, neither of which exists; the CapyMOA `partial_fit`
  examples showed no output for a call that returns `self`. Every example in
  both bridge packages is now a passing doctest.
- **Staged training raised `ImportError`.** `tuiml_train` with `stage="init"`,
  `stage="fit"` (new model), or `stage="partial_fit"` imported
  `_inject_seed_to_algorithm` from `tuiml.workflow`, which is named
  `_inject_seed`. All four staged paths work again.
- **Framework adapters were missing the discovery tools.** `get_tools()` for
  LangChain / OpenAI / Anthropic / CrewAI / Pydantic-AI exposed only the 24
  workflow tools, so agents had no `tuiml_list` or `tuiml_describe` and could
  only train an algorithm whose exact class name they already knew. All 30
  tools are now exposed, matching what an MCP client sees.
- **Agent-started and library-started model servers could not see each other.**
  The `tuiml_serve_model` / `tuiml_stop_server` / `tuiml_server_status` tools
  kept their own server registry, separate from `tuiml.serve()` /
  `tuiml.stop_server()` / `tuiml.server_status()`. They are now wrappers over
  the root serving API: one registry, one `"host:port"` server-id scheme, and
  the real readiness wait instead of a fixed one-second sleep (so a busy port
  now reports an error rather than handing back a URL for a dead server).
- **`model_id`s no longer survive only in memory.** The model index is
  rehydrated from `~/.tuiml/models/` at import, matching the dataset index, so
  a `model_id` still resolves after an MCP server restart.
- **MCP annotations understated several tools.** `tuiml_delete_algorithm`,
  `tuiml_edit_algorithm`, `tuiml_self_update` and `tuiml_restart` had no
  annotation entry and were advertised to clients as read-only and
  non-destructive. They are now marked destructive; `tuiml_create_algorithm` is
  no longer read-only and `tuiml_system_info` is marked open-world.

### Added
- **"View source on GitHub" links throughout the API docs.** Every module,
  class, function and method now carries a GitHub icon linking to its own
  implementation, anchored to the line it starts on and opening in a new tab.
  2501 links, each verified to point at a file that exists at a line that
  does.
- **Package overviews for every package.** All 65 package index pages now
  carry real documentation — what the package is for, what is in it, and when
  to reach for which piece. 31 were previously under 400 characters and
  `tuiml/agent/mcp/` had no docstring at all, so its page rendered blank.
- **Smoke test covering all 30 MCP tools** (`tests/test_agent/`). Each tool is
  dispatched through `execute_tool` — the entry point `server.py` uses — in one
  coherent session, so ids flow from the tool that produces them to the tools
  that consume them, and each result is recorded exactly as the server records
  it. A coverage test compares the registry against the plan, so a tool added
  without being smoke-tested fails the suite instead of shipping untested. One
  tool is deliberately not executed, with the reason stated in code:
  `tuiml_restart` kills every running `tuiml-mcp` process, so only its
  read-only discovery half runs. Fixtures redirect every `~/.tuiml` write
  target at a tmp dir, so a test run neither pollutes the user's home nor reads
  state left by a previous one.

### Removed
- **The `/docs/` prefix on hand-written pages.** `/docs/` is where the 350
  generated API pages live; a privacy policy is not documentation. The eight
  hand-written pages moved to the site root — `/getting_started.html`,
  `/api-reference.html`, `/benchmarks.html`, `/changelog.html`,
  `/contributing.html`, `/privacy.html`, `/terms.html`, `/about.html` — and
  every old URL redirects to its new home, so existing links and search
  results keep working. The generated API tree is unchanged. Redirects
  previously ran the other way (`/privacy.html` into `/docs/privacy.html`);
  those are now reversed.
- **`remote-mcp.html`.** Nothing linked to it from the nav, the footer, or any
  page, and it was absent from the sitemap. The MCP server printed its URL on
  startup, which is now the getting-started page.
- **Dead test code.** 102 unused imports across 91 test files (mostly an
  `import pickle` copied into ~50 algorithm test modules), 6 unused locals, and
  from `tests/conftest.py` 10 fixtures and 3 assertion helpers no test has ever
  referenced — shrinking conftest from 557 lines to 299. Test count is
  unchanged at 1647.
- **A test that asserted nothing.** `test_shrinkage_adds_noise` set up the
  minority rows, then asserted only that resampling produced *some* new rows,
  which is true with or without shrinkage — so it passed whether or not the
  feature worked. It now asserts what its comment always claimed: that no new
  sample is an exact copy of a minority row.

### Changed
- **The tutorials are now a book.** The ten notebooks are replaced by fifteen
  numbered chapters that run in reading order, each assuming the ones before
  it: chapters 0–8 build the Python arc (data, evaluation, pipelines,
  features, imbalance, tuning, benchmarking), 9 covers experiments-as-data,
  10–12 cover driving TuiML from an agent, and 13–14 serve and apply it. Every
  notebook ships executed, so the published pages carry real tables, plots and
  CD diagrams.

  The old set mirrored the API's own table of contents; this one is organised
  around decisions, and reports what the library actually does rather than
  what a tutorial would prefer it did — imputation barely moves a random
  forest on the diabetes data, every feature-engineering technique loses on
  eight curated columns, logistic regression beats a 200-tree forest, and
  tuning's reported gain is smaller than its own optimism. Two leakage
  demonstrations carry the fold-discipline argument: feature selection outside
  the fold scores 82% on pure noise, and SMOTE before splitting inflates by
  six points.

  Every previous tutorial URL redirects to the chapter that now covers it.
- **`tuiml setup` client registry brought up to date.** Three entries pointed
  at config files the vendors no longer read:
  - **ChatGPT Desktop** has no MCP config of its own. OpenAI put the desktop
    app, the Codex CLI and the IDE extension on one file, `~/.codex/config.toml`
    (`[mcp_servers.tuiml]`), so they are now a single client. `--client
    chatgpt-desktop` still resolves, as an alias.
  - **VS Code** moved MCP out of `settings.json` (`mcp.servers`) into a
    dedicated user-profile `mcp.json`, whose root key is `servers` and whose
    stdio entries need an explicit `"type": "stdio"`.
  - **Antigravity** now has a native global MCP config shared by the IDE, the
    `agy` CLI and the SDK at `~/.gemini/config/mcp_config.json`; wiring it no
    longer depends on which VS Code extension happens to be installed. Gemini
    CLI is kept as a legacy entry (retired for consumer tiers, still live for
    enterprise and API-key users) and is detected on its `settings.json` rather
    than on `~/.gemini/`, which Antigravity now also owns.

  An existing server entry is merged rather than replaced, so extra fields the
  user added (`env`, `args`) survive a re-run.
- **Package overviews now appear in the generated API docs.** `generate_docs.py`
  dropped every `__init__.py`, so a package's docstring — its overview, install
  notes and usage guide — was never rendered anywhere. Package index pages now
  lead with it. `tuiml.sklearn` and `tuiml.capymoa` gained full usage guides in
  the process: how to install the extra, the two equivalent ways to reach a
  wrapper (import the class, or name it `"sklearn.<ClassName>"` in a spec),
  mixing wrapped and native components in one pipeline, benchmarking a wrapper
  against its native counterpart, and discovering parameters via
  `get_parameter_schema()`.
- **`install.sh` warns when CapyMOA is selected without Java.** The wheel
  installs fine without a JVM, then every learner fails at fit time. The
  installer's client list also matched an older registry; it now names the
  clients `tuiml setup` actually detects and points at `tuiml setup --list`.
- **`tuiml/agent/` reorganized into packages.** The 6741-line
  `tuiml/agent/tools.py` is now a `tools/` package with one module per tool,
  grouped by domain (`workflow/`, `data/`, `analysis/`, `discovery/`,
  `authoring/`, `system/`, `notebook/`). Each tool declares a single `ToolSpec`
  next to its executor; the schema tables, dispatch table, MCP annotations and
  notebook-export skip list are all derived from those specs rather than
  maintained as parallel dicts. `user_algorithms.py` was likewise split into a
  package, and `SKILL.md` moved to `tuiml/agent/prompts/`. Public entry points
  (`tuiml.agent.execute_tool`, `get_workflow_tools`, and the whole `tuiml.cli`
  surface) are unchanged.
- **`train()` / `run()`: spec-based, consistent component API.** Every ML
  component is now described the same way — a spec dict `{"name": ..., **params}`
  — for the model, each preprocessing step, and the feature selector. The data
  is its own spec, `{"source": ..., "target": ..., "features": ...}` (or
  `{"X": ..., "y": ...}` for in-memory arrays), so the file/label pairing lives
  in one place. `run(spec)` takes one declarative object with `model` / `data` /
  `preprocessing` / `feature_selection` / run-option keys. The previous forms —
  string algorithm name, positional `data`/`target`, loose hyperparameter kwargs,
  configured instances, and the legacy `run()` config schema — all still work,
  so this is fully backward compatible.

### Fixed
- **`train()` ignored the target column for file paths.** When `data` was a file
  path, `Workflow` loaded it without forwarding `target`, silently falling back
  to the last column — training on the wrong label if the class column wasn't
  last. The target is now passed through to the loader (ARFF still uses its
  declared class attribute), and a warning is emitted when a tabular file falls
  back to the last column.

### Added
- **`features=` / data-spec `"features"`** on `train()` — restrict the feature
  matrix to a named subset of columns.
- **`CategoricalNBClassifier`** — Naive Bayes for nominal / integer-coded
  categorical features (per-feature categorical distributions with Laplace
  smoothing), the discrete-data analogue of the Gaussian `NaiveBayesClassifier`.
  Matches scikit-learn's `CategoricalNB`.

### Fixed
- **MCP server: dataset resources unreadable.** The MCP SDK passes resource
  URIs as `AnyUrl` objects, not strings, so `read_resource` crashed with
  `AttributeError` on any `tuiml://dataset/...` resource; the URI is now
  coerced to `str` first.
- **Decision trees / Random Forests: NaN predictions from degenerate splits.**
  When the two feature values flanking the best split are adjacent doubles,
  the midpoint threshold `(a+b)/2` can round up to `b`; the `<=` partition
  then sends every sample left, and the empty right child's mean / class
  distribution is `0/0 = NaN`, which poisons any prediction routed to that
  leaf (seen as rare `Input contains NaN` failures on high-precision
  features). The C++ splitters now clamp the threshold to the lower value
  (as scikit-learn does), and the builders keep a node a leaf if a partition
  comes back empty.
- **`LogisticRegression` under-convergence:** the fixed-step batch gradient
  descent solver (`learning_rate=1.0`, `max_iter=100`, loss-delta stopping)
  stopped far short of the optimum on large or high-dimensional datasets,
  and the default `ridge=1e-8` was effectively unregularized (overfitting
  when p >> n). The default solver is now **L-BFGS** (SciPy) with
  `max_iter=1000` and `ridge="auto"` (= `1/n_samples`, equivalent to
  scikit-learn's `C=1.0`). The legacy solver remains available via
  `solver="gd"`.
- **`RandomForestClassifier` / `RandomForestRegressor` memory:** parallel tree
  builds each materialize a full bootstrap copy of `X`, so with `n_jobs=-1` on
  a many-core machine peak memory reached `n_workers * X.nbytes` (8-16 GB on
  the largest datasets). The number of *concurrent* builds is now bounded to a
  memory budget (default 1 GB, override with `TUIML_RF_MEM_BUDGET_MB`), cutting
  peak RSS to ~2 GB with no change to accuracy or fit time. Bootstrap indices
  are also no longer retained unless `oob_score=True`.

## [0.1.6] - 2026-06-29

### Added
- **Optional scikit-learn backend** (`tuiml.sklearn`): a curated set of
  scikit-learn estimators registered into the hub under namespaced keys
  (`sklearn.RandomForestClassifier`, …) plus a generic `SklearnAdapter` /
  `wrap_sklearn()` that wraps *any* scikit-learn-compatible estimator
  (pipelines, `GridSearchCV`, third-party). `train()` / `experiment()`
  auto-wrap a passed estimator. Install with `pip install tuiml[sklearn]`.
- **Optional CapyMOA backend** (`tuiml.capymoa`): streaming/online learners
  under `capymoa.*` keys. Install with `pip install tuiml[capymoa]`.
- Both backends are optional and lazily imported — native algorithms keep
  working with no extra dependencies; a missing backend errors only at
  instantiation, never at `import tuiml`.
- Reproducible cross-library benchmark suite (`benchmarks/`): TuiML vs
  scikit-learn vs Weka across 51 TabArena datasets.

### Changed
- `NaiveBayesClassifier` gains a `var_smoothing` parameter (scale-relative
  variance floor, scikit-learn compatible).
- `SVC` / `SVR` default `max_iter` is now `-1` (auto-scales the SMO iteration
  cap to `max(10000, n_samples)`).

### Fixed
- **Decision trees and Random Forests (classification & regression):** feature
  values and split thresholds were truncated to `float32` in prediction and
  flattening, which rounded values across split boundaries and misrouted
  samples on high-precision features — producing catastrophic, seed-dependent
  trees. Now `float64` end-to-end; results match scikit-learn.
- **`NaiveBayesClassifier`:** a fixed absolute `1e-6` variance floor plus a
  `log(prob)` underflow clamp caused collapses on standardized / one-hot data.
  Replaced with a scale-relative floor and direct log-density.
- **`SVC` / `SVR`:** the old fixed `max_iter=1000` left the SMO solver
  under-converged (degenerate but fast) on datasets larger than a few thousand
  rows; the auto-scaling default fixes this.
- **`MultilayerPerceptronRegressor`:** the gradient was averaged by `n_samples`
  three times (near-zero updates → ~0 / negative R²); corrected the averaging
  and added global-norm gradient clipping.
- **`tuiml_export_notebook`:** exported predict / evaluate / plot cells now
  re-apply the fitted preprocessing / feature-selection pipeline (via the
  `WorkflowResult`) instead of running the bare model on raw data, so the
  notebook reproduces correct results.

## [0.1.5] - 2026-06-17

### Added
- `tuiml_export_notebook` MCP tool — export the current MCP session as a
  reproducible Jupyter notebook (`.ipynb`). Training, experiment, tuning,
  plotting, and data-prep steps are translated to equivalent Python API
  calls so the workflow can be re-run without the AI client.

### Fixed
- `tuiml_export_notebook` now embeds the effective random seed. The seed is
  resolved into the call *result* (not the args), so auto-resolved seeds were
  previously dropped; exported notebooks now fold it into train/experiment/
  tune cells and emit a `set_global_seed()` cell for full reproducibility.
- Silenced spurious "Component '...' is already registered. Overwriting."
  warnings on intentional user-algorithm re-registration (new version,
  restart bootstrap). Genuine name clashes still warn, now on stderr so the
  message can no longer corrupt the MCP stdio JSON-RPC stream.

## [0.1.4] - 2026-05-20

### Added
- C++ pybind11 kernels for SGD (classifier + regressor),
  agglomerative hierarchical clustering, and Gaussian Mixture EM —
  closing large performance gaps vs sklearn (see Performance).
- MCP setup support for 6 new clients: Gemini CLI, Cline, Roo Code,
  Kilo Code, OpenCode (custom `mcp.<name>` schema), and Antigravity.
- `path` field in `tuiml_plot` MCP response — plots are now persisted
  to `~/.tuiml/plots/` (override with `$TUIML_PLOT_DIR`) so agents
  can embed them in markdown reports via `![](/path/to/plot.png)`.

### Changed
- `plot_roc_curve` now handles multiclass input: draws per-class
  one-vs-rest curves with class labels in the legend, plus a
  dotted macro-average overlay. Previously silently took
  `probas[:, 1]` and plotted a single curve, giving misleading
  AUC=1.000 on 3-class problems like iris.
- AdaBoost: vectorised the prediction aggregation loop
  (241× → 14× slower vs sklearn on 10k×20 5-class data).
- RandomForest: default `n_jobs=-1` so all CPU cores are used.

### Fixed
- `np.trapz` → `np.trapezoid` in `evaluation.metrics.classification`;
  `roc_auc_score` was crashing on NumPy >= 2.0.

### Performance
- SGDRegressor: 278× → 2.4× slower vs sklearn (C++ kernel; was
  pure-Python mini-batch loop with per-sample allocations).
- Gaussian Mixture (EM, full covariance): 292× → 1.4× slower vs
  sklearn (C++ kernel with pre-allocated scratch + manual Cholesky).
- Hierarchical (ward, 1k samples): 96× → 7× slower vs sklearn
  (C++ kernel with min-heap + condensed distance matrix +
  Lance-Williams updates).

## [0.1.3] - 2026-05-20

### Added
- `tuiml uninstall` CLI command and Auto/Manual mode menu in `tuiml setup`.
- `tuiml_system_info` and `tuiml_self_update` MCP tools.
- Agent-authored algorithm MCP tools with versioned aliases.
- Lineage store and `tuiml_research_log` for agent-authored algorithm history.
- NemoClaw (NVIDIA) client and generic `instructions` kind for MCP registration.
- OpenClaw kind for CLI-based MCP registration in `tuiml setup`.
- Native Linux aarch64 wheel builds via ARM runners.

### Changed
- Renamed `tuiml.llm` → `tuiml.agent` with framework adapters.
- Clarified example prompts with "using TuiML" suffix.
- Agent module docs no longer hardcode tool count.

## [0.1.2] - 2026-04-22

### Added
- `tuiml setup` CLI wizard for configuring MCP clients (Claude Code, Claude Desktop, Cursor, OpenClaw, Perplexity Desktop, and more) with auto-detection of installed clients.
- Star History chart and download badge in README.

### Changed
- Repositioned tuiml as an agent-native ML runtime; slimmed README to focus on agent workflows over Python/CLI examples.
- Improved algorithm defaults and extended benchmarks to 10k samples.
- Updated SVM kernel handling and neighbor search internals.
- Enhanced workflow preprocessing and switched Python badge to show ≥3.10.

### Fixed
- XGBoost and LightGBM wrappers no longer require scikit-learn to be installed: both now use their native training APIs (`xgb.train` + `DMatrix`, `lgb.train` + `Dataset`) instead of the sklearn-compatible estimator wrappers.
- MCP server upload/train/serve path resolution issues.
- Author display on the PyPI sidebar.
- Logo image source in README.

## [0.1.1] - 2026-03-15

### Added
- 100+ machine learning algorithms across 13 categories: trees, linear, bayesian, clustering, ensemble, gradient boosting, SVM, neural, neighbors, rules, associations, time series, and anomaly detection.
- High-level API with one-liner `train()`, `predict()`, `experiment()`, and `run()` functions.
- Fluent `Workflow` builder for chainable ML pipelines with preprocessing, feature engineering, and evaluation.
- Comprehensive preprocessing module: scaling, encoding, imputation, discretization, outlier handling, SMOTE, and text vectorizers.
- Feature engineering: PCA, random projection, univariate selection, sequential selection, variance threshold, polynomial and mathematical feature generation.
- Built-in datasets: iris, diabetes, wine, breast cancer, glass, ionosphere, soybean, and segment challenge.
- Dataset generators for classification (Agrawal, Hyperplane, LED, Random RBF), clustering (blobs, circles, moons, swiss roll), and regression (Friedman, Mexican Hat, Sine).
- Data loaders for ARFF, CSV, JSON, Excel, Parquet, NumPy, and pandas formats.
- LLM integration via Model Context Protocol (MCP) server with 200+ tools for agentic ML workflows.
- CLI for training, prediction, evaluation, experiments, model serving, and hub operations.
- WekaHub community platform for algorithm discovery, publishing, and benchmarking.
- Dataset Hub for browsing, downloading, and sharing datasets with the community.
- 21 interactive Jupyter tutorials across 6 learning tracks: Quickstart, ML Simplified, LLM Friendly, Community, Deploy, and Case Studies.
- Full API documentation generated from NumPy-style docstrings with KaTeX math rendering.
- Base classes (`Classifier`, `Regressor`) with scikit-learn compatible fit/predict interface and `@classifier`/`@regressor` decorators.
- Model serialization via joblib with save/load utilities.
- Cross-validation, grid search, and hyperparameter tuning support.

[0.1.8]: https://github.com/tuiml/tuiml/releases/tag/v0.1.8
[0.1.7]: https://github.com/tuiml/tuiml/releases/tag/v0.1.7
[0.1.6]: https://github.com/tuiml/tuiml/releases/tag/v0.1.6
[0.1.5]: https://github.com/tuiml/tuiml/releases/tag/v0.1.5
[0.1.4]: https://github.com/tuiml/tuiml/releases/tag/v0.1.4
[0.1.3]: https://github.com/tuiml/tuiml/releases/tag/v0.1.3
[0.1.2]: https://github.com/tuiml/tuiml/releases/tag/v0.1.2
[0.1.1]: https://github.com/tuiml/tuiml/releases/tag/v0.1.1
