"""Containment checks for the two agent-facing paths that write and execute.

The MCP server is driven by an LLM that may be relaying content it did not
author, so ``tuiml_upload_data`` (which chooses a path to write) and the user
algorithm loader (which executes Python) are the two places where a crafted
string turns into a filesystem or interpreter operation. Both were previously
exploitable and compose: an arbitrary write into the algorithms directory gains
code execution at the next server start.

These tests pin the containment properties rather than the implementation --
they assert that an escape does not happen, not how it is prevented -- so they
keep holding if the sanitiser is rewritten.
"""

import os
from pathlib import Path

import pytest

from tuiml.agent.tools.data.upload import (
    _dest_within,
    _safe_dataset_name,
    execute_upload_data,
)
from tuiml.agent.tools._state import _UPLOADS_DIR
from tuiml.agent.user_algorithms.validation import _ast_validate


# --------------------------------------------------------------------------
# Upload: a dataset name must not be able to describe a path.
# --------------------------------------------------------------------------

# Each of these, joined naively onto the uploads directory, writes somewhere
# else. The first is the original report: it lands on ~/.claude.json, the MCP
# client config that `tuiml setup` manages.
TRAVERSALS = [
    "../../.claude",
    "../../../etc/passwd",
    "..\\..\\windows_style",
    "....//doubled",
    "/absolute/path/dataset",
    "sub/dir/dataset",
]


@pytest.mark.parametrize("raw", TRAVERSALS)
def test_dataset_name_cannot_describe_a_path(raw):
    """A traversal attempt is reduced to a bare filename."""
    name = _safe_dataset_name(raw)
    assert os.sep not in name
    assert "/" not in name and "\\" not in name
    assert not name.startswith(".")
    assert ".." not in name


@pytest.mark.parametrize("raw", ["..", ".", "...", "   ", "///"])
def test_dataset_name_with_nothing_usable_is_rejected(raw):
    """A name that sanitises away raises rather than silently inventing one."""
    with pytest.raises(ValueError):
        _safe_dataset_name(raw)


def test_ordinary_dataset_names_survive_intact():
    """Sanitisation must not mangle the names people actually use."""
    for name in ("iris", "my_dataset", "sales-2026", "run.v2"):
        assert _safe_dataset_name(name) == name


@pytest.mark.parametrize("filename", ["../escape.csv", "../../.claude.json", "a/../../b.csv"])
def test_destination_outside_uploads_is_refused(filename):
    """Containment is enforced on the resolved path, not just the name."""
    with pytest.raises(ValueError):
        _dest_within(_UPLOADS_DIR, filename)


def test_destination_inside_uploads_is_allowed():
    """A bare filename resolves inside the uploads directory."""
    dest = _dest_within(_UPLOADS_DIR, "ok.csv")
    assert dest.startswith(os.path.realpath(_UPLOADS_DIR) + os.sep)


def test_upload_does_not_write_outside_uploads(tmp_path):
    """End-to-end: the original exploit leaves the target file untouched.

    The destination is ``<uploads>/<name>.<format>``, and both halves were
    caller-controlled, so the traversal has to name a victim whose extension
    the format supplies -- otherwise the write lands on ``victim.csv`` and a
    test asserting on ``victim`` passes while the escape still happened.
    """
    outside = tmp_path / "victim.csv"
    outside.write_text("original")

    # Hop out of the uploads directory and back down to the victim, leaving the
    # ".csv" for `format` to append -- exactly the reported exploit shape.
    depth = len(Path(_UPLOADS_DIR).resolve().parts)
    target = str(outside.resolve())[: -len(".csv")]
    escape = "../" * depth + target.lstrip("/")

    result = execute_upload_data(content="a,b\n1,2\n", format="csv", name=escape)

    assert outside.read_text() == "original", "upload escaped the uploads directory"
    if result.get("status") == "success":
        written = Path(result["file_path"]).resolve()
        assert str(written).startswith(str(Path(_UPLOADS_DIR).resolve()) + os.sep)


def test_upload_rejects_an_arbitrary_extension():
    """Content mode writes only known data extensions."""
    result = execute_upload_data(content="a,b\n1,2\n", format="py", name="payload")
    assert result["status"] == "error"
    assert "format" in result["error"].lower()


# --------------------------------------------------------------------------
# User algorithms: validation is an allowlist, and it runs before execution.
# --------------------------------------------------------------------------

MINIMAL_ALGORITHM = """
import numpy as np
from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["test"], version="1.0.0")
class Demo(Classifier):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)
"""

# Every one of these reaches the filesystem, the network or the interpreter
# while stepping around a denylist of module and attribute names.
#
# Each is appended to MINIMAL_ALGORITHM rather than checked alone: validation
# also requires a decorated class, so a bare snippet is rejected for the
# missing decorator and the test would pass without exercising the escape at
# all.
ESCAPES = {
    "importlib reaches any module": "import importlib\nimportlib.import_module('os')\n",
    "pathlib writes without open": "from pathlib import Path\nPath('/tmp/x').write_text('x')\n",
    "computed attribute name": "def f(o):\n    return getattr(o, 'sys' + 'tem')\n",
    "dunder attribute by string": "def f(o):\n    return getattr(o, '__class__')\n",
    "subclasses walk to builtins": "x = ().__class__.__bases__[0].__subclasses__()\n",
    "builtins via globals": "def f():\n    return globals()['__builtins__']\n",
    "builtin bound to a name": "g = eval\n",
    "direct os import": "import os\n",
    "socket": "import socket\n",
    "relative import": "from . import sibling\n",
}


@pytest.mark.parametrize("snippet", ESCAPES.values(), ids=list(ESCAPES))
def test_escape_routes_are_refused(snippet):
    """Source that reaches outside the numerical subset is rejected.

    The snippet is combined with an otherwise-valid algorithm so that the only
    thing wrong with the source is the escape itself.
    """
    source = MINIMAL_ALGORITHM + "\n" + snippet
    # Guard the guard: without the snippet this source must be accepted, or the
    # assertion below proves nothing.
    assert _ast_validate(MINIMAL_ALGORITHM)[0], "baseline algorithm must validate"

    ok, reason = _ast_validate(source)
    assert not ok, "validation accepted an escape route"
    assert reason


def test_a_normal_algorithm_is_accepted():
    """The shape the skeleton generates must pass."""
    ok, reason = _ast_validate(MINIMAL_ALGORITHM)
    assert ok, reason


def test_duck_typed_attribute_access_is_still_allowed():
    """``getattr(model, "classes_", None)`` is ordinary ensemble code.

    A blanket ban on ``getattr`` breaks real algorithms, so only a computed or
    dunder attribute name is refused.
    """
    source = MINIMAL_ALGORITHM + """
def probabilities(model):
    return getattr(model, "classes_", None)
"""
    ok, reason = _ast_validate(source)
    assert ok, reason


def test_wrapping_a_supported_library_is_allowed():
    """Wrapping scikit-learn is a normal thing for a user algorithm to do."""
    source = "from sklearn.ensemble import RandomForestClassifier\n" + MINIMAL_ALGORITHM
    ok, reason = _ast_validate(source)
    assert ok, reason


def test_stored_algorithm_is_validated_before_it_executes(tmp_path, monkeypatch):
    """A file already on disk is re-checked at load, not trusted.

    This is the half that matters: validating only when an algorithm is created
    protects nothing if a file can reach the directory by another route, because
    the loader executes whatever it finds at the next server start.
    """
    from tuiml.agent.user_algorithms import registration

    marker = tmp_path / "executed"
    alg_dir = tmp_path / "algs" / "Planted" / "1.0.0"
    alg_dir.mkdir(parents=True)
    (alg_dir / "algorithm.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
    )

    monkeypatch.setattr(registration, "USER_ALGS_DIR", tmp_path / "algs")
    result = registration.load_all()

    assert not marker.exists(), "planted algorithm executed without validation"
    assert result["loaded"] == 0
    assert any("Planted" in e["path"] for e in result["errors"])
