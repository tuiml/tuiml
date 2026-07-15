<p align="center">
  <img src="https://raw.githubusercontent.com/tuiml/tuiml/main/tuiml_logo.png" alt="TuiML Logo" width="320">
</p>
<p align="center"><strong>Machine Learning that agents can actually call.</strong></p>

<p align="center">
TuiML is an agent-native ML runtime. Install, connect to your AI agent, and start running real ML workflows &mdash; classification, regression, clustering, experiments &mdash; all from one structured interface.
</p>

<p align="center">
  <a href="https://pypi.org/project/tuiml/"><img src="https://img.shields.io/pypi/v/tuiml?style=for-the-badge" alt="PyPI version"></a>&nbsp;
  <a href="https://pypi.org/project/tuiml/"><img src="https://img.shields.io/badge/Python-≥3.10-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python versions"></a>&nbsp;
  <a href="https://tuiml.ai/docs/getting_started.html"><img src="https://img.shields.io/badge/Docs-tuiml.ai-blue?style=for-the-badge" alt="Documentation"></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-blue.svg?style=for-the-badge" alt="BSD-3-Clause License"></a>&nbsp;
  <a href="https://pepy.tech/projects/tuiml"><img src="https://img.shields.io/pepy/dt/tuiml?style=for-the-badge" alt="Downloads"></a>
</p>

<p align="center">
  <a href="#why-tuiml">Why TuiML</a> &nbsp;&bull;&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;&bull;&nbsp;
  <a href="#python-api">Python API</a> &nbsp;&bull;&nbsp;
  <a href="#whats-included">What's Included</a> &nbsp;&bull;&nbsp;
  <a href="#mcp-tools">MCP Tools</a> &nbsp;&bull;&nbsp;
  <a href="#documentation">Docs</a>
</p>

---

## Why TuiML

**Agents can call it** &mdash; Every algorithm, dataset, and metric ships with a JSON schema. Agents read the schema, call the tool, get structured results. No hallucinated parameters, no wrapper glue.

**Agents can discover it** &mdash; A queryable registry tagged by task, data shape, and benchmarks. Agents browse and pick instead of memorising class names.

**Agents can trust it** &mdash; Deterministic, typed, reproducible outputs. Every call is a loggable, replayable tool invocation you can audit, diff, and trust in production.

---

<a id="quickstart"></a>

## Get running in 3 steps

**1. Install** &mdash; one command, installs `uv` and `tuiml` globally:

```bash
curl -fsSL https://tuiml.ai/install.sh | bash
```

Already have Python? `pip install tuiml` works too.

**2. Connect your agent** &mdash; auto-detects Claude Desktop, Cursor, Claude Code, and more:

```bash
tuiml setup
```

**3. Ask your agent** &mdash; in any connected client:

> "Train a random forest on my sales data and report the accuracy."

Your agent discovers algorithms, sets parameters from the schema, trains, evaluates, and returns structured results. No glue code.

---

<a id="python-api"></a>

## Use it from Python

The same runtime agents call is a first-class Python library. Every component &mdash; the model, each preprocessing step, the feature selector &mdash; is described the same way: a **spec** of the form `{"name": ..., **params}`. The data is its own spec, `{"source": ..., "target": ...}`.

```python
import tuiml

# One call trains, evaluates, and returns metrics.
result = tuiml.train(
    {"name": "RandomForestClassifier", "n_estimators": 100},   # model spec
    {"source": "sales.csv", "target": "label"},                # data spec
    preprocessing=[{"name": "MinMaxScaler"}],
    cv=10,
)
print(result.metrics)          # {'accuracy_score': 0.97, 'f1_score': 0.96}
preds = result.model.predict(X_new)
```

Benchmark many algorithms across many datasets with `tuiml.experiment(...)`, and browse the same registry agents use with `tuiml.list_algorithms()` / `tuiml.search_algorithms(...)` / `tuiml.describe_algorithm(...)`. See the [tutorials](https://tuiml.ai/docs/tutorials.html) for the full tour.

---

<a id="whats-included"></a>

## What's Included

**13 algorithm families**, many originally from Weka, rewritten natively in Python with C++ acceleration for hot paths: trees and random forests, Bayesian methods, nearest neighbors, linear models, SVMs, neural networks, rule learners, ensembles, gradient boosting, clustering, association rules, anomaly detection, and time series.

Plus preprocessing (scaling, encoding, imputation, SMOTE, text vectorization), feature engineering (selection, extraction, generation), evaluation (metrics, cross-validation, tuning, statistical tests), and 15+ built-in datasets.

Optional backends live in their own extras and never affect the native core: `pip install tuiml[sklearn]` (scikit-learn estimators through the same interface) and `pip install tuiml[capymoa]` (CapyMOA streaming learners).

---

<a id="mcp-tools"></a>

## MCP Tools

The MCP server exposes 200+ tools agents can call directly. Key workflow tools include `tuiml_train`, `tuiml_predict`, `tuiml_evaluate`, `tuiml_experiment`, `tuiml_tune`, `tuiml_plot`, `tuiml_list`, `tuiml_describe`, and `tuiml_search`. Any component registered with `@classifier`, `@regressor`, or `@transformer` is automatically discoverable through these tools.

For manual setup, add this to your client's MCP config:

```json
{
    "mcpServers": {
        "tuiml": { "command": "tuiml-mcp" }
    }
}
```

---

<a id="documentation"></a>

## Documentation

Full documentation is available at [tuiml.ai/docs](https://tuiml.ai/docs/getting_started.html), including getting started guides, API reference, and tutorials. Want to contribute? Pick something from the [Build Board](https://tuiml.ai/projects) &mdash; algorithms, integrations, and good first issues.

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{tuiml2026,
    title={TuiML: Machine Learning that agents can actually call},
    author={Verma, Nilesh and Bifet, Albert and Pfahringer, Bernhard},
    year={2026},
    url={https://tuiml.ai}
}
```

## Links

| | | |
|---|---|---|
| 🌐 [Website](https://tuiml.ai) | 📚 [Documentation](https://tuiml.ai/docs/getting_started.html) | 🔧 [API Reference](https://tuiml.ai/docs/api-reference.html) |
| 💻 [GitHub](https://github.com/tuiml/tuiml) | 📦 [PyPI](https://pypi.org/project/tuiml) | 📝 [Changelog](https://tuiml.ai/docs/changelog.html) |

---

<p align="center">
  Built by the TuiML team &mdash; <a href="https://tuiml.ai">tuiml.ai</a><br>
  <sub>If TuiML is useful to you, consider leaving a ⭐ &mdash; it helps others find the project.</sub>
</p>
