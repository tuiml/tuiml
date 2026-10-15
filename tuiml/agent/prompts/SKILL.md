---
name: tuiml
description: Trains, evaluates, benchmarks and serves machine-learning models with TuiML (200+ native algorithms, preprocessors, datasets) through the tuiml_* MCP tools, the Python API or the CLI. Use for any ML task - classification, regression, clustering, anomaly detection, time series, churn or fraud models, hyperparameter tuning, statistical comparison of algorithms, autonomous ML research and paper writing, or business ML reports and dashboards.
version: 0.2.0
mcp_server: tuiml-mcp
---

# TuiML

TuiML is a Python ML framework: 200+ components, one `fit`/`predict`
interface, C++ hot paths, no scikit-learn dependency. Three ways in:

| Surface | When | Reference |
|---|---|---|
| `tuiml_*` MCP tools | inside an AI client (default) | [reference/mcp-tools.md](reference/mcp-tools.md) |
| Python API (`train`, `Benchmark`, `Workflow`) | scripts, notebooks, custom code | [reference/python-api.md](reference/python-api.md) |
| `tuiml` CLI | shell, install, client setup | [reference/cli.md](reference/cli.md) |

Task guides, read the one that matches the request:

- **Research** (hypothesis → experiments → statistics → paper → self-review): [workflows/research.md](workflows/research.md)
- **Business ML** (churn, fraud, forecasting → HTML report or dashboard): [workflows/industry.md](workflows/industry.md)
- **Before any tool call that failed once**: [reference/mistakes.md](reference/mistakes.md)

## The two dialects

MCP tool arguments are **flat**; Python specs are **nested and strict**.
Decide which one you are writing before every call.

```python
# MCP tool call: flat, params inline or in algorithm_params
tuiml_train(algorithm="SVC", algorithm_params={"C": 2.0}, data="data.csv",
            target="label", preprocessing=["SimpleImputer", "StandardScaler"], cv=10)

# Python: one spec dict, every component is {"name", "params"}
train({"model": {"name": "SVC", "params": {"C": 2.0}},
       "data": {"source": "data.csv", "target": "label"},
       "pipeline": [{"name": "SimpleImputer"}, {"name": "StandardScaler"}],
       "evaluation": {"cv": 10}})
```

## Exact names only

Names are exact class names, case-sensitive. When unsure, look up first:
`tuiml_list(search="forest")` or `tuiml_describe(name="RandomForestClassifier")`.

| You mean | Use | Not |
|---|---|---|
| random forest | `RandomForestClassifier` / `RandomForestRegressor` | `RandomForest` |
| SVM | `SVC` / `SVR` | `SVM` |
| k-NN | `KNearestNeighborsClassifier` / `KNearestNeighborsRegressor` | `KNN`, `KNeighborsClassifier` |
| naive Bayes | `NaiveBayesClassifier` | `GaussianNB` |
| k-means | `KMeansClusterer` | `KMeans` |
| boosting | `XGBoostClassifier`, `LightGBMClassifier`, `GradientBoostingRegressor` | `GBM` |

Pretrained foundation models use a namespaced key and need
`pip install 'tuiml[foundation]'` (pulls PyTorch; weights download on first
use, TuiML ships none): `foundation.TabICLClassifier` /
`foundation.TabICLRegressor` (tabular, in-context), `foundation.ChronosForecaster`
(`checkpoint` picks `amazon/chronos-bolt-{tiny,mini,small,base}` or
`amazon/chronos-2`) and `foundation.TimesFMForecaster` (zero-shot forecasting).
Constructing and describing them works without the extra; `fit` needs it.

Built-in datasets (hyphenated): `iris`, `glass`, `diabetes`, `breast-cancer`,
`credit-g`, `ionosphere`, `vote`, `soybean`, `labor`, `hypothyroid`,
`contact-lenses`, `cpu`. There is no `wine`. Built-ins define their own
target; `target` is only needed for files. For a file, `target` is the
column name, read it from `tuiml_profile_data` first.

Metrics are exact function names: `accuracy_score`, `f1_score`,
`precision_score`, `recall_score`, `roc_auc_score` (classification);
`r2_score`, `root_mean_squared_error`, `mean_absolute_error` (regression);
`silhouette_score`, `davies_bouldin_score` (clustering). Omit `metrics` to
get sensible defaults.

## Core loop

```
tuiml_profile_data(data=...)                       # shape, dtypes, target, missing values
tuiml_train(algorithm=..., data=..., target=..., cv=10, random_seed=42)   # → model_id
tuiml_plot(plot_type="confusion_matrix", model_id=..., data=..., target=...)
tuiml_benchmark(algorithms=[...], data=[...], target=..., cv=10)          # compare
tuiml_test_statistics(results=<benchmark result>, test="friedman", metric="accuracy_score")
tuiml_export_notebook(path="run.ipynb", title=...)  # reproducible record of the session
```

Rules that prevent most failed calls:

1. Pass `random_seed=42` on every train/benchmark/tune call. Results must be reproducible.
2. `cv` and `test_size` are exclusive; `cv=10` for reports, `test_size=0.2` for quick checks. Forecasters (`ARIMA`, `ThetaForecaster`, `foundation.ChronosForecaster`, …) are never shuffled: `cv` becomes expanding-window time-ordered folds and `test_size` a tail holdout, in `tuiml_train`, `tuiml_benchmark` and `Benchmark`. Classifiers and regressors on time-ordered rows still shuffle; see the business ML guide.
3. `model_id` comes from a `tuiml_train` result in **this** session. Never invent one.
4. Preprocessing order: impute → scale → encode → resample → select. Fit on train folds only, which the tools already do.
5. `random_seed` is the run seed; `random_state` is an algorithm constructor parameter. They are not interchangeable.
6. An error result carries `suggestion`, `recovery_tool` and `recovery_params`. Follow them, do not retry the same call.

## Custom algorithms

`tuiml_get_skeleton` → fill in `fit`/`predict` → `tuiml_create_algorithm(name, kind, code, version)`
→ it is immediately usable in `tuiml_train` and `tuiml_benchmark`, pinned as `Name_v1_0_0`.
Needs `TUIML_ALLOW_USER_ALGORITHMS=1` on the server. Full loop in
[reference/mcp-tools.md](reference/mcp-tools.md).
