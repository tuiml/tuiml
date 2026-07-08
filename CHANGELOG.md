# Changelog

All notable changes to TuiML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
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

[0.1.6]: https://github.com/tuiml/tuiml/releases/tag/v0.1.6
[0.1.5]: https://github.com/tuiml/tuiml/releases/tag/v0.1.5
[0.1.4]: https://github.com/tuiml/tuiml/releases/tag/v0.1.4
[0.1.3]: https://github.com/tuiml/tuiml/releases/tag/v0.1.3
[0.1.2]: https://github.com/tuiml/tuiml/releases/tag/v0.1.2
[0.1.1]: https://github.com/tuiml/tuiml/releases/tag/v0.1.1
