"""The gradient-boosting backends must stay *lazily imported*.

XGBoost, LightGBM and CatBoost ship with TuiML, but importing them eagerly is
what caused an interpreter segfault: each loads its own OpenMP runtime, and on
macOS a second runtime arriving with PyTorch kills the process on the first
parallel region. Because :mod:`tuiml.algorithms` imported all three
unconditionally, *every* session was one ``import torch`` away from that.

Being installed was never the problem; being imported at ``import tuiml`` time
was. That distinction is the whole point of this file: a stray module-scope
``import xgboost`` would bring the crash straight back while every other test
kept passing, so the absence of one is pinned two ways -- by a subprocess check
of what actually gets imported, and by an AST scan of the source.
"""

import ast
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from tuiml.algorithms import (
    CatBoostClassifier,
    LightGBMClassifier,
    XGBoostClassifier,
    XGBoostRegressor,
)
from tuiml.algorithms.gradient_boosting._backend import has_backend, require_backend

BACKENDS = ["xgboost", "lightgbm", "catboost"]
#: Each wrapper keeps its upstream library's own parameter name for the number
#: of boosting rounds -- CatBoost calls it ``iterations``, the other two
#: ``n_estimators`` -- so the count parameter is named per wrapper here rather
#: than assumed uniform.
WRAPPERS = [
    (XGBoostClassifier, "n_estimators"),
    (LightGBMClassifier, "n_estimators"),
    (CatBoostClassifier, "iterations"),
]

_PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "tuiml"


def test_importing_tuiml_does_not_import_any_backend():
    """The invariant the OpenMP fix rests on -- true even though all three
    are installed.

    Run in a subprocess: this test session has almost certainly imported the
    backends already, so checking ``sys.modules`` in-process proves nothing.
    """
    code = (
        "import sys; import tuiml.algorithms; "
        "print(','.join(m for m in ('xgboost','lightgbm','catboost') "
        "if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "", f"eagerly imported by tuiml.algorithms: {out}"


def test_no_module_scope_backend_import_in_the_package():
    """Catch a re-introduced top-level import by reading the source.


    The subprocess test above only covers what ``tuiml.algorithms`` pulls in
    today. This one fails the moment anybody writes the import, wherever it is.
    """
    offenders = []
    for path in _PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:  # module scope only -- nested imports are fine
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            hits = set(names) & set(BACKENDS)
            if hits:
                offenders.append(f"{path.relative_to(_PACKAGE)}:{node.lineno} {sorted(hits)}")
    assert offenders == [], "module-scope backend imports: " + "; ".join(offenders)


@pytest.mark.parametrize("cls, rounds_param", WRAPPERS)
def test_construction_never_needs_the_backend(cls, rounds_param):
    """Constructing records hyperparameters and must not touch the library.

    The availability check used to live in ``__init__``. Moving it to ``fit``
    keeps construction free of the import, which is what lets the schema be
    read without loading a second OpenMP runtime.
    """
    model = cls(**{rounds_param: 7})
    assert getattr(model, rounds_param) == 7


@pytest.mark.parametrize("cls, rounds_param", WRAPPERS)
def test_schema_is_readable_without_fitting(cls, rounds_param):
    """Agents and the CLI read schemas without ever touching the backend."""
    assert rounds_param in cls.get_parameter_schema()


def test_require_backend_error_names_the_class_and_the_extra():
    """The missing-dependency message has to be actionable."""
    with pytest.raises(ImportError) as excinfo:
        require_backend("definitely_not_a_real_backend", "XGBoostClassifier")
    message = str(excinfo.value)
    assert "XGBoostClassifier" in message
    assert "tuiml[boosting]" in message


@pytest.mark.parametrize("name", BACKENDS)
def test_has_backend_reports_a_bool(name):
    """Used by tests and callers that want to branch rather than raise."""
    assert isinstance(has_backend(name), bool)


@pytest.mark.skipif(not has_backend("xgboost"), reason="boosting libraries missing")
def test_wrappers_still_fit_when_the_backend_is_present():
    """Making the import lazy must not change what the wrappers do."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    assert (XGBoostClassifier(n_estimators=10).fit(X, y).predict(X) == y).mean() > 0.8

    y_reg = X[:, 0] * 2.0 - X[:, 1]
    pred = XGBoostRegressor(n_estimators=20).fit(X, y_reg).predict(X)
    assert np.corrcoef(pred, y_reg)[0, 1] > 0.9


@pytest.mark.skipif(sys.platform != "darwin", reason="OpenMP clash is macOS-specific")
@pytest.mark.skipif(not has_backend("xgboost"), reason="boosting libraries missing")
@pytest.mark.skipif(not has_backend("torch"), reason="needs uv sync --extra torch")
def test_boosting_and_neural_models_coexist_in_one_process():
    """Both orderings must survive, because the libomp clash is symmetric.

    Whichever runtime initialises second can crash, and the two directions need
    different mitigations, so both are exercised in a fresh interpreter.
    """
    order_a = (
        "import numpy as np;"
        "from tuiml.algorithms import XGBoostClassifier, FTTransformerClassifier;"
        "rng=np.random.default_rng(0); X=rng.normal(size=(40,3)); y=(X[:,0]>0).astype(int);"
        "XGBoostClassifier(n_estimators=5).fit(X,y);"
        "FTTransformerClassifier(random_state=0).fit(X,y);"
        "print('ok')"
    )
    order_b = order_a.replace(
        "XGBoostClassifier(n_estimators=5).fit(X,y);"
        "FTTransformerClassifier(random_state=0).fit(X,y);",
        "FTTransformerClassifier(random_state=0).fit(X,y);"
        "XGBoostClassifier(n_estimators=5).fit(X,y);",
    )
    for label, code in (("boosting-first", order_a), ("neural-first", order_b)):
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert done.returncode == 0, (
            f"{label} crashed with exit {done.returncode} "
            f"(139 = SIGSEGV, the duplicate-libomp clash): {done.stderr[-400:]}"
        )
