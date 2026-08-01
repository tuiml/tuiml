#!/usr/bin/env python3
"""Smoke test for tuiml_export_notebook code generation.

For every MCP tool that has a notebook-translation branch in
``_translate_call``, this records a realistic *successful* session call,
exports the notebook, then executes each generated code cell in a shared
namespace — catching cases where the generated Python is invalid or raises
(e.g. passing an MCP-only ``algorithm_params=`` kwarg to ``tuiml.train``).

Runs under pytest (collected via ``test_notebook_export_smoke``) or standalone:

    uv run python tests/test_integration/test_notebook_export_smoke.py
"""
import json
import os
import sys
import tempfile
import traceback

# Headless plotting so plot cells don't block or need a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None  # no-op so plot cells return cleanly

from tuiml.agent import tools as t

MODEL_ID = "smoke_model_1"

# A coherent session: train first (so predict/evaluate/plot/save resolve the
# model var), then everything else. Each entry is (tool, args, result).
SESSION = [
    ("tuiml_profile_data",
     {"data": "iris", "target": "class"},
     {"status": "success"}),
    ("tuiml_train",
     {"algorithm": "RandomForestClassifier", "data": "iris", "target": "class",
      "test_size": 0.2, "metrics": ["accuracy_score", "f1_score"],
      "algorithm_params": {"n_estimators": 50, "random_state": 42}},
     {"status": "success", "model_id": MODEL_ID, "random_seed": 42}),
    ("tuiml_predict",
     {"model_id": MODEL_ID, "data": "iris"},
     {"status": "success"}),
    ("tuiml_evaluate",
     {"model_id": MODEL_ID, "data": "iris", "target": "class",
      "metrics": ["accuracy_score"]},
     {"status": "success"}),
    ("tuiml_benchmark",
     {"algorithms": ["RandomForestClassifier", "DecisionTreeClassifier"],
      "data": "iris", "cv": 3, "metrics": ["accuracy_score"]},
     {"status": "success", "random_seed": 42}),
    ("tuiml_tune",
     {"algorithm": "DecisionTreeClassifier", "data": "iris", "method": "grid",
      "param_grid": {"max_depth": [3, 5]}, "cv": 3, "scoring": "accuracy_score"},
     {"status": "success", "random_seed": 42}),
    ("tuiml_plot",
     {"plot_type": "confusion_matrix", "model_id": MODEL_ID, "data": "iris",
      "target": "class"},
     {"status": "success"}),
    ("tuiml_plot",
     {"plot_type": "feature_importance", "model_id": MODEL_ID, "data": "iris"},
     {"status": "success"}),
    ("tuiml_plot",
     {"plot_type": "roc_curve", "model_id": MODEL_ID, "data": "iris", "target": "class"},
     {"status": "success"}),
    ("tuiml_plot",
     {"plot_type": "pr_curve", "model_id": MODEL_ID, "data": "iris", "target": "class"},
     {"status": "success"}),
    ("tuiml_plot",
     {"plot_type": "learning_curve", "algorithm": "DecisionTreeClassifier",
      "data": "iris", "target": "class"},
     {"status": "success"}),
    ("tuiml_plot",
     {"plot_type": "cd_diagram",
      "benchmark_results": {"A": [0.9, 0.91], "B": [0.8, 0.82], "C": [0.85, 0.86]}},
     {"status": "success"}),
    ("tuiml_plot",
     {"plot_type": "boxplot_comparison",
      "benchmark_results": {"A": [0.9, 0.91], "B": [0.8, 0.82]}},
     {"status": "success"}),
    ("tuiml_save_model",
     {"model_id": MODEL_ID,
      "destination": os.path.join(tempfile.gettempdir(), "smoke_model.joblib")},
     {"status": "success"}),
    ("tuiml_generate_data",
     {"generator": "Blobs", "n_samples": 100, "n_features": 4, "n_clusters": 3},
     {"status": "success"}),
    ("tuiml_preprocess",
     {"data": "iris", "steps": ["StandardScaler"], "target": "class"},
     {"status": "success"}),
    ("tuiml_select_features",
     {"data": "iris", "method": "SelectKBestSelector", "target": "class", "k": 2},
     {"status": "success"}),
    ("tuiml_test_statistics",
     {"test": "friedman",
      "results": {"A": [0.9, 0.91, 0.89], "B": [0.8, 0.82, 0.81]},
      "significance_level": 0.05},
     {"status": "success"}),
    ("tuiml_upload_data",
     {"file_path": "", "name": "my_inline_dataset"},
     {"status": "success"}),
]


def build_notebook():
    t._SESSION_CALLS.clear()
    t._MODEL_ID_TO_VAR.clear()
    t._TRAIN_CALL_SEQ.clear()
    for tool, args, result in SESSION:
        t.record_session_call(tool, args, result)
    out = os.path.join(tempfile.mkdtemp(), "smoke.ipynb")
    res = t.execute_export_notebook(path=out, title="Smoke Test")
    assert res["status"] == "success", res
    return json.load(open(out))


def run_cells(nb):
    """Exec each code cell in a shared namespace; return per-cell results.

    A failing cell is reported but its error doesn't abort the run — the
    namespace from prior successful cells is kept so independent cells still
    get exercised (NameError cascades are noted, not hidden).
    """
    ns = {}
    results = []
    last_header = "(setup)"
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            # Track the section header so failures are attributable to a tool.
            first = src.strip().splitlines()[0] if src.strip() else ""
            if first.startswith("##"):
                last_header = first.lstrip("# ").strip()
            continue
        if src.strip().startswith("!pip"):
            results.append((last_header, "SKIP", "(pip install cell)"))
            continue
        try:
            exec(src, ns)
            results.append((last_header, "PASS", ""))
        except Exception as e:
            tb = traceback.format_exc().strip().splitlines()[-1]
            results.append((last_header, "FAIL", tb))
    return results


def test_notebook_export_smoke():
    """Every generated code cell must execute without raising."""
    results = run_cells(build_notebook())
    fails = [(h, d) for h, s, d in results if s == "FAIL"]
    assert not fails, "generated notebook cells failed: " + "; ".join(
        f"{h} -> {d}" for h, d in fails
    )


def main():
    nb = build_notebook()
    results = run_cells(nb)
    print(f"\n{'='*72}\nNOTEBOOK EXPORT SMOKE TEST — {len(results)} code cells\n{'='*72}")
    fails = 0
    for header, status, detail in results:
        mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "·"}[status]
        line = f"  {mark} [{status}] {header}"
        if detail and status != "PASS":
            line += f"\n        → {detail}"
        print(line)
        if status == "FAIL":
            fails += 1
    print(f"{'='*72}")
    total_run = sum(1 for _, s, _ in results if s != "SKIP")
    print(f"{total_run - fails}/{total_run} executable cells passed, {fails} failed.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
