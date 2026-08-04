"""Agent MCP tools and notebook export.

Merged from: test_agent_mcp_tools_smoke.py, test_integration_notebook_export_smoke.py
"""

import json
import os
import socket
import pytest
from tuiml.agent import tools as agent_tools
from tuiml.agent.tools import SPECS, execute_tool, get_tool_annotations
import sys
import tempfile
import traceback
import matplotlib
import matplotlib.pyplot as plt
from tuiml.agent.tools import _state, execute_tool, record_session_call


# --------------------------------------------------------------------------
# End-to-end smoke test for every MCP tool TuiML exposes.
# --------------------------------------------------------------------------

NOT_EXECUTED = {
    "tuiml_restart": (
        "kills every running tuiml-mcp process on the machine, including the "
        "ones serving the developer's own editor"
    ),
}


def _free_port():
    """Return a TCP port that is free right now.

    Returns
    -------
    port : int
        A port number the OS just handed out and immediately released.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


USER_ALGO_SOURCE = '''
import numpy as np
from tuiml.base.algorithms import Classifier, classifier


@classifier(tags=["smoke"], version="1.0.0")
class SmokeTestClassifier(Classifier):
    """Predicts the majority class. Exists only to exercise the MCP tools."""

    def fit(self, X, y):
        """Record the most frequent label."""
        values, counts = np.unique(y, return_counts=True)
        self.majority_ = values[int(np.argmax(counts))]
        self._is_fitted = True
        return self

    def predict(self, X):
        """Return the majority label for every row."""
        return np.full(len(np.asarray(X)), self.majority_)
'''


EXECUTION_ORDER = [
    # Discovery and read-only inspection
    "tuiml_list",
    "tuiml_describe",
    "tuiml_system_info",
    "tuiml_list_files",
    "tuiml_search_source",
    "tuiml_read_algorithm",
    "tuiml_self_update",
    # Data
    "tuiml_generate_data",
    "tuiml_upload_data",
    "tuiml_read_data",
    "tuiml_profile_data",
    "tuiml_preprocess",
    "tuiml_select_features",
    # Modelling — train first, everything downstream needs its model_id
    "tuiml_train",
    "tuiml_predict",
    "tuiml_evaluate",
    "tuiml_save_model",
    "tuiml_plot",
    "tuiml_benchmark",
    "tuiml_tune",
    "tuiml_test_statistics",
    # Serving — serve, observe, stop
    "tuiml_serve_model",
    "tuiml_server_status",
    "tuiml_stop_server",
    # Authoring — skeleton, create, edit, then delete what was created
    "tuiml_get_skeleton",
    "tuiml_create_algorithm",
    "tuiml_edit_algorithm",
    "tuiml_delete_algorithm",
    # Export last: it turns the whole session above into a notebook
    "tuiml_export_notebook",
]


def _arguments_for(name, context, tmp_path):
    """Build a realistic argument dict for one tool.

    Parameters
    ----------
    name : str
        MCP tool name.
    context : dict
        Values produced by earlier calls (``model_id``, ``server_id``).
    tmp_path : pathlib.Path
        Per-test temporary directory for file outputs.

    Returns
    -------
    arguments : dict
        Keyword arguments to pass to :func:`execute_tool`.
    """
    model_id = context.get("model_id")
    return {
        "tuiml_list": {"category": "algorithms", "limit": 5},
        "tuiml_describe": {"name": "RandomForestClassifier"},
        "tuiml_system_info": {"check_latest": False},
        "tuiml_list_files": {"builtin": True, "user": False},
        "tuiml_search_source": {"query": "class RandomForest", "user": False},
        "tuiml_read_algorithm": {"name": "RandomForestClassifier", "builtin": True},
        # dry_run: the real path pip-installs a new tuiml over the running one.
        "tuiml_self_update": {"dry_run": True},

        "tuiml_generate_data": {
            "generator": "Blobs", "n_samples": 60, "n_features": 4, "n_clusters": 3,
        },
        "tuiml_upload_data": {
            "content": "a,b,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n",
            "format": "csv",
            "name": "smoke_inline",
        },
        "tuiml_read_data": {"data": "iris", "n_rows": 5, "mode": "head"},
        "tuiml_profile_data": {"data": "iris", "target": "class"},
        "tuiml_preprocess": {
            "data": "iris", "target": "class", "steps": ["StandardScaler"],
        },
        "tuiml_select_features": {
            "data": "iris", "target": "class",
            "method": "SelectKBestSelector", "k": 2,
        },

        "tuiml_train": {
            "algorithm": "RandomForestClassifier",
            "data": "iris",
            "target": "class",
            "test_size": 0.3,
            "metrics": ["accuracy_score"],
            "algorithm_params": {"n_estimators": 10},
            "random_seed": 42,
        },
        "tuiml_predict": {"model_id": model_id, "data": "iris"},
        "tuiml_evaluate": {
            "model_id": model_id, "data": "iris", "target": "class",
            "metrics": ["accuracy_score"],
        },
        "tuiml_save_model": {
            "model_id": model_id,
            "destination": str(tmp_path / "smoke_saved.joblib"),
        },
        "tuiml_plot": {
            "plot_type": "confusion_matrix", "model_id": model_id,
            "data": "iris", "target": "class",
        },
        "tuiml_benchmark": {
            "algorithms": ["DecisionTreeClassifier", "NaiveBayesClassifier"],
            "data": "iris", "target": "class", "cv": 3,
            "metrics": ["accuracy_score"],
        },
        "tuiml_tune": {
            "algorithm": "DecisionTreeClassifier", "data": "iris",
            "target": "class", "method": "grid",
            "param_grid": {"max_depth": [2, 4]}, "cv": 3,
            "scoring": "accuracy_score",
        },
        "tuiml_test_statistics": {
            "test": "friedman",
            "results": {"A": [0.90, 0.91, 0.89], "B": [0.80, 0.82, 0.81],
                        "C": [0.85, 0.86, 0.84]},
            "significance_level": 0.05,
        },

        "tuiml_serve_model": {
            "model_id": model_id, "port": context["port"], "host": "127.0.0.1",
        },
        "tuiml_server_status": {},
        "tuiml_stop_server": {"server_id": context.get("server_id")},

        "tuiml_get_skeleton": {
            "kind": "classifier", "class_name": "SmokeTestClassifier",
        },
        "tuiml_create_algorithm": {
            "name": "SmokeTestClassifier", "kind": "classifier",
            "code": USER_ALGO_SOURCE, "version": "1.0.0",
            "description": "Majority-class baseline used by the MCP smoke test.",
        },
        "tuiml_edit_algorithm": {
            "name": "SmokeTestClassifier",
            "old_string": "Exists only to exercise the MCP tools.",
            "new_string": "Exists only to exercise the MCP tools (edited).",
        },
        "tuiml_delete_algorithm": {"name": "SmokeTestClassifier"},

        "tuiml_export_notebook": {
            "path": str(tmp_path / "smoke_session.ipynb"),
            "title": "MCP Smoke Test",
        },
    }[name]


class TestSPECS:
    """End-to-end smoke test for every MCP tool TuiML exposes."""

    def test_every_tool_is_covered(self):
        """Every registered tool is either smoke-run or listed as not-executed.

        This is the test that keeps the rest honest: a tool added to ``SPECS``
        without a plan here fails immediately, instead of shipping untested.
        """
        registered = {spec.name for spec in SPECS}
        planned = set(EXECUTION_ORDER) | set(NOT_EXECUTED)

        assert registered == planned, (
            f"untested tools: {sorted(registered - planned)}; "
            f"stale entries: {sorted(planned - registered)}"
        )

    @pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
    def test_tool_spec_is_well_formed(self, spec):
        """Each tool advertises a name, a description and a valid input schema.

        Parameters
        ----------
        spec : ToolSpec
            One entry from the registry.
        """
        tool = spec.as_mcp_tool()

        assert spec.name.startswith("tuiml_"), "MCP tool names share the tuiml_ prefix"
        assert tool["description"].strip(), "an agent picks tools by description"

        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        for required in schema.get("required", []):
            assert required in schema["properties"], (
                f"{spec.name} requires '{required}' but never declares it"
            )

        # The schema is handed to clients as JSON; anything unserializable
        # (a class, a numpy default) breaks the handshake rather than one call.
        json.dumps(tool)

        annotations = get_tool_annotations(spec.name)
        assert set(annotations) == {
            "readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"
        }
        assert not (annotations["readOnlyHint"] and annotations["destructiveHint"]), (
            f"{spec.name} claims to be both read-only and destructive"
        )

    def test_restart_discovery_without_killing_anything(self):
        """``tuiml_restart``'s discovery half works; the killing half is not run.

        Executing the tool would terminate the developer's own MCP servers, so
        only :func:`find_mcp_processes` — its read-only half — is exercised.
        """
        from tuiml.agent.tools.system.restart import find_mcp_processes

        processes = find_mcp_processes(exclude_self=True)
        assert isinstance(processes, list)
        assert all({"pid", "ppid", "command"} <= set(p) for p in processes)
        assert all(os.getpid() != p["pid"] for p in processes), (
            "exclude_self must never return the calling process"
        )

        annotations = get_tool_annotations("tuiml_restart")
        assert annotations["destructiveHint"], (
            "tuiml_restart kills processes and must be advertised as destructive"
        )

    @pytest.mark.usefixtures("agent_home", "clean_session")
    def test_all_mcp_tools_execute(self, tmp_path):
        """Dispatch every executable tool and require a success status.

        The calls run in one session, in :data:`EXECUTION_ORDER`, so ids flow from
        the tool that produced them to the tools that consume them — the same way
        an agent's conversation works. Each result is passed to
        ``record_session_call`` exactly as ``server.py`` does after dispatching, so
        the final ``tuiml_export_notebook`` call has a real session to export.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Per-test temporary directory for model, notebook and plot outputs.
        """
        context = {"port": _free_port()}
        failures = []

        for name in EXECUTION_ORDER:
            arguments = _arguments_for(name, context, tmp_path)
            result = execute_tool(name, **arguments)

            if not isinstance(result, dict) or result.get("status") != "success":
                failures.append((name, result))
                continue

            agent_tools.record_session_call(name, arguments, result)

            # Carry forward the ids later tools depend on.
            if result.get("model_id"):
                context["model_id"] = result["model_id"]
            if result.get("server_id"):
                context["server_id"] = result["server_id"]

        assert not failures, "MCP tools failed:\n" + "\n".join(
            f"  {name}: {result!r}" for name, result in failures
        )

    @pytest.mark.usefixtures("agent_home", "clean_session")
    def test_unknown_tool_reports_an_error_rather_than_raising(self):
        """An unknown name comes back as an error dict.

        ``server.py`` turns an exception into a protocol-level failure, so the
        dispatcher must degrade to a status dict instead.
        """
        result = execute_tool("tuiml_does_not_exist")

        assert result["status"] == "error"
        assert "tuiml_does_not_exist" in result["error"]

    @pytest.mark.usefixtures("agent_home", "clean_session")
    def test_failed_calls_are_kept_out_of_the_notebook(self, tmp_path):
        """Only successful calls become notebook cells.

        A failed call recorded as a cell would make the exported notebook raise on
        re-run, which is the whole reason ``record_session_call`` filters on status.
        """
        failed = execute_tool("tuiml_train", algorithm="NoSuchAlgorithm", data="iris",
                              target="class")
        assert failed["status"] == "error"
        agent_tools.record_session_call("tuiml_train", {"algorithm": "NoSuchAlgorithm"},
                                        failed)

        ok = execute_tool("tuiml_profile_data", data="iris", target="class")
        agent_tools.record_session_call("tuiml_profile_data", {"data": "iris"}, ok)

        export = execute_tool(
            "tuiml_export_notebook",
            path=str(tmp_path / "filtered.ipynb"),
            title="Filtered",
        )
        assert export["status"] == "success"

        source = json.dumps(json.load(open(tmp_path / "filtered.ipynb")))
        assert "NoSuchAlgorithm" not in source


# --------------------------------------------------------------------------
# Smoke test for tuiml_export_notebook code generation.
# --------------------------------------------------------------------------

matplotlib.use("Agg")


plt.show = lambda *a, **k: None  # no-op so plot cells return cleanly


MODEL_ID = "smoke_model_1"


# Source of a user-authored algorithm, as tuiml_create_algorithm would record it.
# The exported notebook must inline this: the class lives in
# ~/.tuiml/user_algorithms/, so a notebook that only does `pip install tuiml`
# cannot resolve the name in the tuiml.train(...) cell below it.
# Distinct from USER_ALGO_SOURCE above, which the MCP dispatch test edits by
# exact string match — the two must not share a name or a docstring.
EXPORT_USER_ALGO_SOURCE = '''\
import numpy as np
from tuiml.base.algorithms import Classifier, classifier


@classifier(tags=["custom"], version="1.0.0")
class SmokeUserAlgo(Classifier):
    """Predicts the majority class seen during fit."""

    def fit(self, X, y):
        vals, counts = np.unique(y, return_counts=True)
        self.majority_ = vals[np.argmax(counts)]
        self.classes_ = vals
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)
'''


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
    ("tuiml_get_skeleton",
     {"kind": "classifier", "class_name": "SmokeUserAlgo"},
     {"status": "success"}),
    ("tuiml_create_algorithm",
     {"name": "SmokeUserAlgo", "kind": "classifier", "version": "1.0.0",
      "code": EXPORT_USER_ALGO_SOURCE, "description": "Majority-class baseline."},
     {"status": "success"}),
    ("tuiml_train",
     {"algorithm": "SmokeUserAlgo", "data": "iris", "target": "class", "cv": 3},
     {"status": "success", "model_id": "smoke_user_model", "random_seed": 42}),
]


def build_notebook():
    # The session log lives in tools/_state.py and is imported by value
    # elsewhere, so it is emptied in place rather than rebound.
    _state._SESSION_CALLS.clear()
    _state._MODEL_ID_TO_VAR.clear()
    _state._TRAIN_CALL_SEQ.clear()
    for tool, args, result in SESSION:
        record_session_call(tool, args, result)
    out = os.path.join(tempfile.mkdtemp(), "smoke.ipynb")
    res = execute_tool("tuiml_export_notebook", path=out, title="Smoke Test")
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
        except Exception:
            tb = traceback.format_exc().strip().splitlines()[-1]
            results.append((last_header, "FAIL", tb))
    return results


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


class TestIntegrationNotebookExportSmoke:
    """Smoke test for tuiml_export_notebook code generation."""

    def test_notebook_export_smoke(self):
        """Every generated code cell must execute without raising."""
        results = run_cells(build_notebook())
        fails = [(h, d) for h, s, d in results if s == "FAIL"]
        assert not fails, "generated notebook cells failed: " + "; ".join(
            f"{h} -> {d}" for h, d in fails
        )

    def test_user_algorithm_source_is_inlined(self):
        """A user-authored algorithm must be defined before it is trained.

        Regression test: tuiml_create_algorithm was recorded but had no branch
        in _translate_call, so it was silently dropped from the notebook and the
        following tuiml.train(...) cell referenced a name the registry could not
        resolve on a fresh machine.
        """
        nb = build_notebook()
        code = [
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
        ]
        define_at = next(
            (i for i, s in enumerate(code) if "class SmokeUserAlgo" in s), None
        )
        train_at = next(
            (i for i, s in enumerate(code) if "'SmokeUserAlgo'" in s and "tuiml.train" in s),
            None,
        )
        assert define_at is not None, "user algorithm source was not inlined"
        assert train_at is not None, "user algorithm training cell is missing"
        assert define_at < train_at, "algorithm is trained before it is defined"

    def test_scaffolding_tools_are_not_recorded(self):
        """Skeleton and delete produce no reproducible notebook Python."""
        assert not agent_tools.is_reproducible("tuiml_get_skeleton")
        assert not agent_tools.is_reproducible("tuiml_delete_algorithm")
        nb = build_notebook()
        source = json.dumps(nb)
        assert "tuiml_get_skeleton" not in source

    def test_repeated_edits_emit_one_definition(self):
        """Identical post-edit sources collapse to a single redefinition cell."""
        from tuiml.agent.tools.notebook.translate import _translate_call

        emitted = {}
        call = {
            "tool": "tuiml_create_algorithm",
            "args": {"name": "DedupAlgo", "kind": "classifier",
                     "code": EXPORT_USER_ALGO_SOURCE, "version": "1.0.0"},
        }
        first_md, first_code = _translate_call(call, [0], emitted)
        assert first_md is not None and "class SmokeUserAlgo" in "".join(first_code)
        # Same source again -> skipped rather than duplicated.
        again_md, _ = _translate_call(call, [0], emitted)
        assert again_md is None
