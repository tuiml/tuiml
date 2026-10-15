# Business ML with TuiML: churn, fraud, forecasting, anomalies, and reports

Use this when the request comes from a business setting: predict churn or
default, detect fraud or anomalies, forecast demand, segment customers, or
"build a model and give me a report/dashboard". The output is a model whose
business value is stated in the client's units, plus a single-file HTML
report or dashboard.

## Contents
- Delivery checklist
- Step 1: understand the data and the decision
- Step 2: playbooks by problem type
- Step 3: business metrics, not just accuracy
- Step 4: HTML report
- Step 5: dashboard variant
- Step 6: hand-off (serve, save, monitor)

## Delivery checklist

```
Delivery progress:
- [ ] 1. Data profiled; target, leakage columns and business decision confirmed
- [ ] 2. Baseline and candidates evaluated with cv=10 (or a chronological holdout for time series), seed fixed
- [ ] 3. Business metric computed (lift, cost, expected value, MAPE...)
- [ ] 4. HTML report written and opened; every number traceable to a call
- [ ] 5. Model saved; serving or monitoring next step stated
```

## Step 1: understand the data and the decision

```
tuiml_profile_data(data="customers.csv")     # rows, dtypes, missing, cardinality, class balance
tuiml_read_data(data="customers.csv", n_rows=10)
```

Then answer, in the response, before modelling:

- What decision does the prediction drive, and at what threshold? (Contact the top 10 % churn risks; block transactions above p = 0.9.)
- Which columns are **leakage** (recorded after the outcome: `cancel_date`, `chargeback_flag`, `days_since_churn`)? Drop them with `features=[...]`.
- Which columns are identifiers (`customer_id`, `email`)? Drop them too.
- Is the class balance below 10 %? Then use `f1_score` or `roc_auc_score`, and consider `preprocessing=["SMOTESampler"]` or `preset="imbalanced"`.
- Is there a time order? For a *forecaster* (a univariate series) the tools already split chronologically, see the forecasting playbook. For a *classifier or regressor* on time-ordered rows, sort by time and explicitly reserve the final rows for evaluation with the Python API below. MCP `test_size` shuffles those rows; sorting before that call does not produce a chronological holdout.

## Step 2: playbooks by problem type

For non-temporal tasks, use `random_seed=42`, `cv=10`, and leakage-free
features where supported. For time series, use the chronological split below.

**Churn / default / conversion (binary classification)**

```
baseline:   tuiml_train(algorithm="LogisticRegression", preprocessing=["SimpleImputer", "StandardScaler", "OneHotEncoder"], ...)
candidates: tuiml_benchmark(algorithms=["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"],
                            data=..., target=..., cv=10, metrics=["roc_auc_score", "f1_score", "precision_score", "recall_score"])
tune the winner: tuiml_tune(algorithm="XGBoostClassifier", method="random", n_iter=20,
                            param_grid={"n_estimators": [100, 600, "int"], "max_depth": [2, 8, "int"], "learning_rate": [0.01, 0.3, "float"]})
explain: tuiml_plot(plot_type="feature_importance", model_id=...)
```

Prefer `LogisticRegression` when it is within two points of the best
model; it is what the business can explain.

**Fraud / rare events**: same as churn, plus `preset="imbalanced"`,
`pr_curve` instead of ROC, and report precision at the operating recall.
When there are no labels use anomaly detection:
`tuiml_train(algorithm="IsolationForestDetector", data=..., ...)` and rank
by score.

**Demand / sales forecasting (time series)**: the file holds one value
column sorted by time (plus optional date column); `target` names it.
Forecasters are never shuffled: `tuiml_train` holds out the final 20 %,
`tuiml_benchmark` with `cv=k` uses `k` expanding-window folds whose test
window always follows its training window.

```
baseline:   tuiml_train(algorithm="ThetaForecaster", data="sales.csv", target="units", random_seed=42)
candidates: tuiml_benchmark(algorithms=["ThetaForecaster", "ARIMA", "ExponentialSmoothing",
                                        "foundation.ChronosForecaster"],
                            data="sales.csv", target="units", cv=5,
                            metrics=["mean_absolute_error", "root_mean_squared_error"])
forecast:   tuiml_predict(model_id=..., steps=12)        # next 12 periods
```

`foundation.ChronosForecaster` and `foundation.TimesFMForecaster` are
pretrained zero-shot forecasters (`pip install 'tuiml[foundation]'`): no
tuning, strong on short series, and they return quantiles via
`predict_quantiles` in Python. Always include a naive baseline in the
report: the MAE of repeating the last observed value, computed in Python
with `np.repeat(history[-1], len(held_out))`. Report both MAEs in business
units. `tuiml_tune` uses shuffled folds, so do not use it on forecasters;
pick parameters via `algorithm_params` and compare them in one
`tuiml_benchmark`.

**Customer segmentation (clustering)**: `preprocessing=["SimpleImputer", "StandardScaler"]`,
`tuiml_benchmark(algorithms=["KMeansClusterer", "GaussianMixtureClusterer", "AgglomerativeClusterer"], ...)`
with `silhouette_score`; try k = 3 to 8 via `algorithm_params={"n_clusters": k}`
and describe each segment by its feature means from `tuiml_read_data`.

**Price / LTV / quantity (regression)**: `LinearRegression` baseline,
`RandomForestRegressor`, `XGBoostRegressor`; metric
`root_mean_squared_error` and `mean_absolute_error` in currency.

## Step 3: business metrics, not just accuracy

Compute these from the predictions (`tuiml_predict` returns them, or use
the confusion matrix) and state them in the client's units:

| Problem | Report | How |
|---|---|---|
| churn | lift in top decile, campaigns needed to save N customers | sort by predicted probability; decile rates vs base rate |
| fraud | fraud caught per 1,000 reviews at the chosen threshold, false-alarm cost | precision and recall at threshold × volumes × unit costs |
| forecast | MAE in units, MAPE, vs naive baseline | absolute errors on the held-out tail |
| segmentation | segment sizes and the two features that separate them | cluster means |
| any classifier | expected value = TP·gain − FP·cost − FN·loss | confusion matrix × the client's cost matrix (ask for it) |

State the model's calibration (`log_loss`, or predicted vs observed rate
per decile) before anyone acts on probabilities.

## Step 4: HTML report

One self-contained file, no external assets, opens in any browser.
Every `tuiml_plot` result carries `path`, a PNG on disk (the image is also
shown inline). Read that file, base64-encode it, and embed it as
`<img src="data:image/png;base64,...">`. If the client cannot read files,
use `<img src="file://{path}">` and say the report needs the plots folder.

Template (keep the section order; drop a section only if empty):

```html
<!doctype html><html><head><meta charset="utf-8"><title>{Project} — model report</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
 h1{font-size:1.6rem} h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #ddd}
 .kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem}
 .kpi div{background:#f5f6f8;border-radius:8px;padding:1rem} .kpi b{font-size:1.6rem;display:block}
 table{border-collapse:collapse;width:100%} td,th{padding:.4rem .6rem;border-bottom:1px solid #e5e5e5;text-align:left}
 img{max-width:100%} .note{color:#555;font-size:.9rem}
</style></head><body>
<h1>{Project}: {what is predicted}</h1>
<p class="note">Data: {file, rows × columns, date range}. Model: {algorithm}. Evaluation: {CV folds or chronological train/test date ranges}, seed {seed}. TuiML {version}. Generated {date}.</p>
<h2>Headline</h2>
<div class="kpi">
  <div><b>{primary metric}</b>{metric name}, held-out</div>
  <div><b>{business number}</b>{e.g. lift in top decile}</div>
  <div><b>{baseline}</b>{baseline model or naive rule}</div>
</div>
<h2>Recommendation</h2><p>{one paragraph: the decision, the threshold, expected value, the caveat}</p>
<h2>Data</h2><table>{profile: rows, target rate, missing values, dropped leakage columns and why}</table>
<h2>Model comparison</h2><table>{benchmark table: model, metric mean ± std}</table>
<h2>Performance</h2>{confusion matrix / ROC or PR / forecast plot images}
<h2>Drivers</h2>{feature importance image + three sentences on what they mean}
<h2>Limitations</h2><ul>{data window, drift, calibration, what was not tested}</ul>
<h2>Reproducibility</h2><p>Notebook: {path}. Model id: {model_id}. Saved model: {path}.</p>
</body></html>
```

Write it to a path the user can open, then state the path. Every number in
the report must come from a tool result in this session.

## Step 5: dashboard variant

Same file with a filter bar: add a `<select>` per segment (region, plan,
month) and a small script that filters a JSON array of per-row predictions
embedded in the page (`const rows = [...]` from `tuiml_predict` output)
and re-renders the KPI tiles and a table of the top-N risks. Keep it under
a few thousand rows; beyond that ship a CSV next to the page.

## Step 6: hand-off

```
tuiml_save_model(model_id=..., destination="churn_model.pkl")
tuiml_serve_model(model_id=..., port=8000)      # returns url + auth_token; quote both
tuiml_export_notebook(path="churn_study.ipynb", title=...)
```

State what to monitor: the target rate and the feature means per month,
and re-train when either drifts. Say when the model should be retrained
and which metric would trigger it.
