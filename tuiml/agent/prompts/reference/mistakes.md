# Mistakes agents actually make with TuiML, and the fix

Every entry below comes from failed calls in real MCP sessions (the server's
trace log). Check the matching row before retrying a failed call.

## Contents
- Names (algorithms, datasets, targets, metrics)
- Argument shape (MCP vs Python, ids, seeds)
- Tool sequencing
- Plots
- Custom algorithms
- Reading an error result

## Names

| Symptom | Cause | Fix |
|---|---|---|
| `Unknown component 'RandomForest'` | family name, not class name | `RandomForestClassifier` or `RandomForestRegressor` |
| `Unknown component 'SVM'` | no class called SVM | `SVC` (classification) or `SVR` (regression) |
| `Unknown component 'KNN'` / `'KNeighborsClassifier'` | sklearn name | `KNearestNeighborsClassifier` |
| `Unknown component 'C45TreeClassifier'` | Weka name | `DecisionTreeClassifier`, or `weka.J48` with `tuiml[weka]` |
| `requires scikit-learn` for `sklearn.X` | optional backend not installed | use the native class (drop the `sklearn.` prefix) or `pip install tuiml[sklearn]` |
| `Unknown component 'ChronosForecaster'` / `'TimesFM'` | foundation models are namespaced | `foundation.ChronosForecaster`, `foundation.TimesFMForecaster`, `foundation.TabICLClassifier` |
| `... backed by the 'chronos' package, which is not installed` | optional extra missing; `fit` is the first call that needs it | `pip install 'tuiml[foundation]'` (pulls PyTorch; weights download on first use), or use a native forecaster such as `ThetaForecaster` |
| forecaster scores look random / `truth value of an array is ambiguous` | a forecaster was given a feature matrix instead of a series | pass the value column as `target`; forecasters consume one column and are split in time order automatically |
| `Tuning is not supported for forecasting benchmarks` | `tune` block on a forecaster | compare fixed `params` variants in one `tuiml_benchmark` instead |
| `Dataset 'breast_cancer' not found` | built-ins are hyphenated | `breast-cancer`, `credit-g`, `contact-lenses` |
| `Dataset 'wine' not found` | not shipped | pick from the built-in list in SKILL.md or upload a CSV with `tuiml_upload_data` |
| `Dataset 'credit' not found` | short name | `credit-g` |
| target `class` / `species` / `target` on a built-in | built-ins define their own target; the argument is ignored | omit `target`, or pass anything |
| result has `"metrics": {}` (empty) | a metric name was wrong (`precision`, `auc`, `rmse`) or does not fit the task | use exact function names: `precision_score`, `roc_auc_score`, `root_mean_squared_error`; classification metrics on a classifier |

When a name is wrong, the error now lists close matches ("Did you mean:
SVC, SVR?"). Take the first suggestion that matches the task type.

## Argument shape

| Symptom | Cause | Fix |
|---|---|---|
| `dataset=` rejected | argument is `data` | `data="iris"` |
| `train() takes 1 positional argument` | Python `train(spec, cv=5)` | put `cv` under `"evaluation"` in the spec |
| `bare string in a Python spec` | `{"model": "SVC"}` | `{"model": {"name": "SVC"}}` |
| loose key rejected | `{"name": "SVC", "C": 2}` in a Python spec | `{"name": "SVC", "params": {"C": 2}}` (MCP args accept inline keys) |
| `Model not found` for `model_id` | id invented or from another session | use the `model_id` returned by `tuiml_train` in this session; ids look like `b05c52a92ff2` |
| `random_state` passed to `tuiml_train` | wrong level | `random_seed=42` at the top level; `random_state` goes inside `algorithm_params` |
| `cv=2` warnings / unstable numbers | too few folds | `cv=5` minimum, `cv=10` for anything reported |
| random/bayesian search needs a range | ranges must be explicitly tagged | use `{"param": [low, high, "int"]}` or `[low, high, "float"]`; plain lists, including two numbers, are discrete choices |
| `experiment_results=` for comparison plots | wrong key | `tuiml_plot(plot_type="cd_diagram", benchmark_results=<tuiml_benchmark result>, metric=...)` |
| `could not convert string to float` in `tuiml_test_statistics` | a hand-built dict with the wrong nesting | pass the `tuiml_benchmark` result as-is with `metric=`, or `{algorithm: [fold scores]}` |
| `post_hoc="nemenyi"` rejected | no such argument | `test="nemenyi"` is its own test; `friedman` first, then `nemenyi` |

## Tool sequencing

- `tuiml_predict`, `tuiml_evaluate`, `tuiml_plot`, `tuiml_save_model`, `tuiml_serve_model` need a `model_id` from a **prior** `tuiml_train`. Train first.
- `tuiml_test_statistics` needs per-fold scores: run `tuiml_benchmark` with `cv` first and pass its result, with `metric=` when more than one metric was computed.
- `tuiml_export_notebook` exports **this session's** calls; call it last, and give a writable `path`.
- After `tuiml_self_update` the client must be restarted before any other call.
- `tuiml_create_algorithm` requires `TUIML_ALLOW_USER_ALGORITHMS=1`; if the tool is missing, the server was started without it.

## Plots

| plot_type | needs |
|---|---|
| `confusion_matrix`, `roc_curve`, `pr_curve` | `model_id` + `data` + `target` |
| `learning_curve` | `algorithm` + `data` + `target` |
| `tree` | `model_id` of a decision tree |
| `feature_importance` | `model_id` of a model that exposes importances (forests, boosting, linear). Single trees and SVMs do not |
| `cd_diagram`, `boxplot_comparison`, `heatmap`, `ranking_table` | `benchmark_results` from `tuiml_benchmark` |

`pr_curve` and `roc_curve` are binary by default; for multiclass use
`confusion_matrix`.

## Custom algorithms

- `name` must be a Python identifier, normally equal to the class name in `code`.
- `kind` must match the decorator in the source (`@classifier` ↔ `"classifier"`).
- Imports of `os`, `subprocess`, `socket`, `requests`, `open`, `eval`, `exec` are rejected (`UnsafeSource`). Use numpy only.
- `import sklearn` inside custom code fails when scikit-learn is absent; implement natively.
- Bump `version` on every edit; compare versions in one `tuiml_benchmark` with the pinned names `Name_v1_0_0`, `Name_v1_0_1`.

## Reading an error result

```json
{"status": "error", "error": "...", "error_type": "ValueError",
 "suggestion": "...", "recovery_tool": "tuiml_list", "recovery_params": {"search": "forest"}}
```

Do exactly one of: apply the fix from `suggestion`, or call `recovery_tool`
with `recovery_params`. Do not repeat the failed call unchanged, and do not
switch to a different dataset or algorithm silently; tell the user what
changed and why.
