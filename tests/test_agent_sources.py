"""Built-in source discovery covers the wrapper packages, not just algorithms/.

``tuiml_read_algorithm``, ``tuiml_list_files`` and ``tuiml_search_source``
all route through :mod:`tuiml.agent.user_algorithms.sources`. These tests pin
the promise that a class an agent can see in ``tuiml_list`` -- including the
namespaced ``foundation.*``, ``sklearn.*`` and ``weka.*`` keys -- can also be
read, without any wrapper package being imported to find it.
"""

import pathlib
import sys

import pytest

from tuiml.agent.user_algorithms import sources


@pytest.mark.parametrize("name, expected_file", [
    ("RandomForestClassifier", "random_forest.py"),
    ("ChronosForecaster", "chronos.py"),
    ("foundation.ChronosForecaster", "chronos.py"),
    ("foundation.TimesFMForecaster", "timesfm.py"),
    ("TabICLClassifier", "tabicl.py"),
    ("sklearn.SVC", "svm.py"),
    # Same class name in two packages: the bare name is the native one, the
    # namespaced key pins the wrapper.
    ("sklearn.RandomForestClassifier", "sklearn/ensemble.py"),
])
def test_read_builtin_source_finds_wrapper_classes(name, expected_file):
    """A namespaced hub key or bare class name resolves to its source file."""
    result = sources.read_source(name, builtin=True)
    assert result["status"] == "success", result
    assert pathlib.Path(result["path"]).as_posix().endswith(expected_file)
    class_name = name.rsplit(".", 1)[-1]
    assert f"class {class_name}(" in result["source"]


def test_read_builtin_source_unknown_name_names_the_listing_tool():
    result = sources.read_source("NoSuchAlgorithmAnywhere", builtin=True)
    assert result["status"] == "error"
    assert "tuiml_list_files" in result["error"]


def test_list_algorithm_files_includes_every_builtin_package():
    files = sources.list_algorithm_files(builtin=True, user=False)["files"]
    packages = {f["package"] for f in files}
    assert {"algorithms", "foundation", "sklearn", "weka"} <= packages
    foundation = [f for f in files if f["package"] == "foundation"]
    assert {f["file"] for f in foundation} >= {"chronos.py", "tabicl.py", "timesfm.py"}
    assert all(f["category"] == "foundation" for f in foundation)
    # The algorithms package keeps its sub-folder categories unchanged.
    trees = [f for f in files if f["package"] == "algorithms" and f["file"] == "random_forest.py"]
    assert trees and trees[0]["category"] == "trees"


def test_listing_does_not_import_the_wrapper_packages():
    """Listing source must never start a JVM or pull an optional dependency."""
    for mod in ("tuiml.weka", "tuiml.capymoa"):
        sys.modules.pop(mod, None)
    sources.list_algorithm_files(builtin=True, user=False)
    sources.read_source("weka.J48", builtin=True)
    assert "tuiml.weka" not in sys.modules
    assert "tuiml.capymoa" not in sys.modules


def test_search_source_reaches_the_foundation_package():
    result = sources.search_source(r"fd_regressor\(", user=False)
    assert result["status"] == "success"
    paths = {m["path"] for m in result["matches"]}
    assert any(p.endswith("chronos.py") for p in paths)
    assert any(p.endswith("timesfm.py") for p in paths)
