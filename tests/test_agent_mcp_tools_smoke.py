"""End-to-end smoke test for every MCP tool TuiML exposes.

The MCP surface is what agents actually call, and it is the part of the library
with no compiler and no type checker watching it: a tool is reachable only
through its name and a JSON schema, so a renamed executor or a schema that
disagrees with the executor's real signature fails at the agent, not at import.
This module calls all of them.

Two guarantees, in order of importance:

1. **Coverage is enforced.** :func:`test_every_tool_is_covered` compares the
   registry against the plans below, so adding a tool without deciding how it
   gets smoke-tested fails the suite rather than silently going untested.
2. **Every tool runs.** Each one is dispatched through
   :func:`tuiml.agent.tools.execute_tool` — the same entry point ``server.py``
   uses — and must report ``status == "success"``.

Three tools are deliberately not executed, each for a stated reason, and are
still checked for schema and annotation correctness. That list is short and
explicit on purpose: "it's hard to test" is not one of the reasons.
"""

import json
import os
import socket

import pytest

from tuiml.agent import tools as agent_tools
from tuiml.agent.tools import SPECS, execute_tool, get_tool_annotations


# ---------------------------------------------------------------------------
# Tools that must not actually run, and why
# ---------------------------------------------------------------------------

#: Tool name -> reason it is exercised without invoking its side effect.
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


#: Minimal source for an agent-authored algorithm, used by the authoring tools.
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


# ---------------------------------------------------------------------------
# Spec-level checks (no execution)
# ---------------------------------------------------------------------------

def test_every_tool_is_covered():
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
def test_tool_spec_is_well_formed(spec):
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


def test_restart_discovery_without_killing_anything():
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


# ---------------------------------------------------------------------------
# The smoke run
# ---------------------------------------------------------------------------

#: Tools that need state an earlier call produced (a model id, a server id, an
#: authored algorithm) and so cannot be ordered arbitrarily.
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


@pytest.mark.usefixtures("agent_home", "clean_session")
def test_all_mcp_tools_execute(tmp_path):
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
def test_unknown_tool_reports_an_error_rather_than_raising():
    """An unknown name comes back as an error dict.

    ``server.py`` turns an exception into a protocol-level failure, so the
    dispatcher must degrade to a status dict instead.
    """
    result = execute_tool("tuiml_does_not_exist")

    assert result["status"] == "error"
    assert "tuiml_does_not_exist" in result["error"]


@pytest.mark.usefixtures("agent_home", "clean_session")
def test_failed_calls_are_kept_out_of_the_notebook(tmp_path):
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
