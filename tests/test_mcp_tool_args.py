"""MCP tool arguments after the schema trim.

Runs the real server behind the SDK's in-memory client. ``target`` is no
longer required for built-in datasets, and ``tuiml_tune`` accepts JSON
``[low, high, "int"|"float"]`` ranges next to plain-list choices for every
search method.
"""

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp.client import Client

from tuiml.agent.mcp.server import create_server


def _run(coro):
    """Run ``async def coro(client)`` against a fresh in-memory server.

    Parameters
    ----------
    coro : callable
        Receives the connected client.

    Returns
    -------
    Any
        Whatever ``coro`` returns.
    """
    async def main():
        async with Client(create_server()) as client:
            return await coro(client)
    return asyncio.run(main())


def test_builtin_benchmark_needs_no_target():
    async def check(client):
        return await client.call_tool("tuiml_benchmark", {
            "algorithms": ["DecisionTreeClassifier", "NaiveBayesClassifier"],
            "data": "iris", "cv": 2, "random_seed": 42,
        })
    result = _run(check)
    assert not result.is_error, result.content
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "success", payload
    assert set(payload["results"]["iris"]) == {"DecisionTreeClassifier", "NaiveBayesClassifier"}


@pytest.mark.parametrize("method", ["grid", "random", "bayesian"])
def test_builtin_tuning_accepts_json_ranges_and_choices(method, monkeypatch, tmp_path):
    from tuiml.agent.tools import _shared
    monkeypatch.setattr(_shared, "_MODELS_DIR", str(tmp_path))
    space = {
        "max_depth": [2, 6, "int"],
        "min_impurity_decrease": [0.001, 0.03, "float"],
        "min_samples_leaf": [1, 3],
        "criterion": ["gini", "entropy"],
    } if method != "grid" else {"max_depth": [2, 6]}

    async def check(client):
        return await client.call_tool("tuiml_tune", {
            "algorithm": "DecisionTreeClassifier", "data": "iris",
            "method": method, "param_grid": space,
            "cv": 2, "n_iter": 3, "n_iterations": 3, "random_seed": 42,
        })
    result = _run(check)
    assert not result.is_error, result.content
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "success", payload
    candidates = [entry["params"] for entry in payload["cv_results"]["top_5"]]
    assert len(candidates) == (2 if method == "grid" else 3)
    assert all(set(params) == set(space) for params in candidates)
    if method == "grid":
        assert {params["max_depth"] for params in candidates} == {2, 6}
    else:
        for params in candidates:
            assert isinstance(params["max_depth"], int)
            assert 2 <= params["max_depth"] <= 6
            assert 0.001 <= params["min_impurity_decrease"] <= 0.03
            assert params["min_samples_leaf"] in [1, 3]
            assert params["criterion"] in ["gini", "entropy"]
        assert len({params["min_impurity_decrease"] for params in candidates}) > 1
