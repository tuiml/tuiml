<p align="center">
  <img src="https://raw.githubusercontent.com/tuiml/tuiml/main/assets/tuiml_logo.png" alt="TuiML Logo" width="180">
</p>
<p align="center"><strong>TuiML - Machine Learning for AI Agents.</strong></p>

<p align="center">
Ask your agent to train a model, tune it, compare it to the last run, or find an algorithm that fits your data. It just does it. No code. No guesswork. No forgotten context.
</p>

<p align="center">
  <a href="https://pypi.org/project/tuiml/"><img src="https://img.shields.io/pypi/v/tuiml?style=flat-square" alt="PyPI version"></a>&nbsp;
  <a href="https://pypi.org/project/tuiml/"><img src="https://img.shields.io/badge/Python-≥3.10-blue?style=flat-square&logo=python&logoColor=white" alt="Python versions"></a>&nbsp;
  <a href="https://tuiml.ai/getting_started.html"><img src="https://img.shields.io/badge/Docs-tuiml.ai-blue?style=flat-square" alt="Documentation"></a>&nbsp;
  <a href="https://github.com/tuiml/tuiml/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-blue.svg?style=flat-square" alt="BSD-3-Clause License"></a>&nbsp;
  <a href="https://pepy.tech/projects/tuiml"><img src="https://img.shields.io/pepy/dt/tuiml?style=flat-square" alt="Downloads"></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &nbsp;&bull;&nbsp;
  <a href="#python-api">Python API</a> &nbsp;&bull;&nbsp;
  <a href="#mcp-tools">MCP Tools</a> &nbsp;&bull;&nbsp;
  <a href="#benchmarks">Benchmarks</a> &nbsp;&bull;&nbsp;
  <a href="#documentation">Docs</a>
</p>

---

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
model = tuiml.train({
    "model": {"name": "RandomForestClassifier", "params": {"n_estimators": 100}},
    "data": {"source": "iris", "target": "class"},
    "pipeline": [{"name": "MinMaxScaler"}],
    "evaluation": {"cv": 10, "metrics": ["accuracy_score", "f1_score"]},
})
print(model.metrics_)
# {'cv_accuracy_score_mean': 0.96, 'cv_accuracy_score_std': 0.053,
#  'cv_f1_score_mean': 0.96, 'cv_f1_score_std': 0.057}
preds = model.predict([[5.1, 3.5, 1.4, 0.2]])
```

Benchmark many algorithms across many datasets with `tuiml.experiment(...)`, and browse the same registry agents use with `tuiml.list_algorithms()` / `tuiml.search_algorithms(...)` / `tuiml.describe_algorithm(...)`. See the [tutorials](https://tuiml.ai/tutorials/hello_tuiml.html) for the full tour.

---

<a id="mcp-tools"></a>

## MCP Tools

Everything TuiML can do, your agent can do &mdash; the MCP server exposes **30 typed MCP tools** with JSON schemas the agent reads directly, addressing a registry of **300+ components** &mdash; algorithms, preprocessors, metrics and datasets &mdash; by name.

**Train &middot; Tune &middot; Compare** &mdash; fit a model, sweep hyperparameters, and rank runs in one conversation. No notebook, no glue code.

**Algorithm Discovery** &mdash; the agent searches the catalog by task, data shape, or constraint and gets ranked recommendations with rationale, not a flat list of names.

**Persistent Experiments** &mdash; every run is logged with lineage and metrics, so today's model can be compared against last week's without re-running anything.

**One-Call Serving** &mdash; deploy a trained model to a local HTTP endpoint with a single tool call. Stop it the same way.

**100% Local &amp; Private** &mdash; your data, your machine. No cloud, no API keys, no telemetry.

Key workflow tools: `tuiml_train`, `tuiml_predict`, `tuiml_evaluate`, `tuiml_benchmark`, `tuiml_tune`, `tuiml_plot`, `tuiml_list`, `tuiml_describe`.

Works with anything that speaks MCP &mdash; `tuiml setup` auto-detects Claude Desktop, Claude Code, Cursor, ChatGPT Desktop, Codex CLI, Zed, Continue, Windsurf, VS Code Copilot, Perplexity, Goose, and OpenClaw / NemoClaw. For manual setup, add this to your client's MCP config:

```json
{
    "mcpServers": {
        "tuiml": { "command": "tuiml-mcp" }
    }
}
```

---

<a id="benchmarks"></a>

## Benchmarks

Every framework on its own library defaults, across **13 algorithms × 51 real-world [TabArena](https://tabarena.ai) datasets**, 10-fold cross-validation, same data and folds for all three:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tuiml/tuiml/main/assets/benchmark_summary_dark.svg">
  <img src="https://raw.githubusercontent.com/tuiml/tuiml/main/assets/benchmark_summary_light.svg" alt="TuiML vs scikit-learn vs Weka: accuracy, training time, inference time, and peak memory averaged across 51 TabArena datasets">
</picture>

The harness also runs a **matched** arm, where hyperparameters are forced to agree across
frameworks rather than each library using its own defaults. TuiML matches scikit-learn exactly on
SVM, naive Bayes and k-NN there, and leads on Random Forest, but trails on logistic regression
(&minus;2.6 pt) and the MLP (&minus;1.7 pt) — two known gaps being tracked.

<sub>Weka memory includes its in-process JVM baseline. Every number above is recomputed from
[`benchmarks/summary_cv10.csv`](benchmarks/summary_cv10.csv) by
[`benchmarks/tools/headline.py`](benchmarks/tools/headline.py), and the figure is regenerated by
[`benchmarks/tools/figure.py`](benchmarks/tools/figure.py). Full per-algorithm and per-dataset
results: [tuiml.ai/benchmarks](https://tuiml.ai/benchmarks.html).</sub>

---

<a id="documentation"></a>

## Documentation

Full documentation is available at [tuiml.ai/docs](https://tuiml.ai/getting_started.html), including getting started guides, API reference, and tutorials. Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md), then pick something from the [Build Board](https://tuiml.ai/projects.html) &mdash; algorithms, integrations, and good first issues. Found a security issue? Please report it privately: see [SECURITY.md](SECURITY.md).

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{tuiml2026,
    title={TuiML: Machine Learning that agents can actually call},
    author={Verma, Nilesh and Bifet, Albert and Pfahringer, Bernhard and Lim, Nick},
    year={2026},
    url={https://tuiml.ai}
}
```

## Links

<div align="center">

| | | |
|---|---|---|
| 🌐 [Website](https://tuiml.ai) | 📚 [Documentation](https://tuiml.ai/getting_started.html) | 🔧 [API Reference](https://tuiml.ai/api-reference.html) |
| 💻 [GitHub](https://github.com/tuiml/tuiml) | 📦 [PyPI](https://pypi.org/project/tuiml) | 📝 [Changelog](https://tuiml.ai/changelog.html) |

</div>

---

<p align="center">
  Built by the TuiML team &mdash; <a href="https://tuiml.ai">tuiml.ai</a><br>
  <sub>If TuiML is useful to you, consider leaving a ⭐ &mdash; it helps others find the project.</sub>
</p>
