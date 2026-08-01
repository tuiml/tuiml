"""Fixtures that keep agent-tool tests out of the real ``~/.tuiml``.

The tool executors write models, uploads, plots and agent-authored algorithms
into ``~/.tuiml/`` — the user's actual working directory. Left alone, a test
run would add junk there and, worse, read state a previous run left behind, so
a passing test would depend on machine history. :func:`agent_home` redirects
every one of those write targets at a tmp dir for the duration of a test.

Two patterns matter here. Path constants are imported *by value* into several
modules (``from ._state import _MODELS_DIR``), so rebinding one definition is
not enough: every import site is patched, which is what ``_PATH_BINDINGS``
enumerates. The shared *containers* (the model index, the session log) have the
same problem but cannot be rebound at all — holders keep the original object —
so those are emptied and refilled in place.
"""

import matplotlib
import pytest

matplotlib.use("Agg")  # headless: plot tools must not need a display


#: ``(module path, attribute, subdirectory)`` for every binding that points
#: into ``~/.tuiml`` and has to be redirected. A new import site added without
#: being listed here shows up as a test writing to the real home directory.
_PATH_BINDINGS = [
    ("tuiml.agent.tools._state", "_MODELS_DIR", "models"),
    ("tuiml.agent.tools._shared", "_MODELS_DIR", "models"),
    ("tuiml.agent.tools._state", "_UPLOADS_DIR", "uploads"),
    ("tuiml.agent.tools.data.upload", "_UPLOADS_DIR", "uploads"),
    # Six modules re-import USER_ALGS_DIR from _paths. Patching fewer than all
    # of them splits the world: tuiml_create_algorithm writes to the tmp dir
    # while tuiml_edit_algorithm looks in the real one and reports "not found".
    ("tuiml.agent.user_algorithms._paths", "USER_ALGS_DIR", "user_algorithms"),
    ("tuiml.agent.user_algorithms", "USER_ALGS_DIR", "user_algorithms"),
    ("tuiml.agent.user_algorithms.storage", "USER_ALGS_DIR", "user_algorithms"),
    ("tuiml.agent.user_algorithms.sources", "USER_ALGS_DIR", "user_algorithms"),
    ("tuiml.agent.user_algorithms.research_log", "USER_ALGS_DIR", "user_algorithms"),
    ("tuiml.agent.user_algorithms.registration", "USER_ALGS_DIR", "user_algorithms"),
]


def _emptied(containers):
    """Context-manager-free helper: empty containers, return their contents.

    Parameters
    ----------
    containers : iterable of list or dict
        Live shared containers to clear.

    Returns
    -------
    saved : list
        A copy of each container's original contents, in the same order, for
        :func:`_refill`.
    """
    saved = [c.copy() for c in containers]
    for container in containers:
        container.clear()
    return saved


def _refill(containers, saved):
    """Restore contents captured by :func:`_emptied`.

    Parameters
    ----------
    containers : iterable of list or dict
        The same containers, in the same order.
    saved : list
        Contents returned by :func:`_emptied`.

    Returns
    -------
    None
    """
    for container, original in zip(containers, saved):
        container.clear()
        if isinstance(container, dict):
            container.update(original)
        else:
            container.extend(original)


@pytest.fixture
def agent_home(tmp_path, monkeypatch):
    """Point every ``~/.tuiml`` write target at a tmp directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's per-test temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Restores the path bindings and the env var when the test ends.

    Yields
    ------
    home : pathlib.Path
        The tmp directory standing in for ``~/.tuiml``.
    """
    import importlib
    from pathlib import Path

    home = tmp_path / "tuiml_home"
    for module_path, attribute, subdir in _PATH_BINDINGS:
        module = importlib.import_module(module_path)
        target = home / subdir
        target.mkdir(parents=True, exist_ok=True)
        # Path-typed bindings must stay Paths and str-typed ones must stay str:
        # consuming code uses os.path.join on some and / on others.
        current = getattr(module, attribute)
        monkeypatch.setattr(
            module, attribute, target if isinstance(current, Path) else str(target)
        )

    plots = home / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TUIML_PLOT_DIR", str(plots))

    # The in-memory indices map ids to files under the *real* home. Empty them
    # so a model id left there by the user can never resolve during a test.
    from tuiml.agent.tools import _state
    indices = (_state._MODEL_INDEX, _state._DATASET_INDEX)
    saved = _emptied(indices)

    yield home

    _refill(indices, saved)


@pytest.fixture
def clean_session():
    """Empty the notebook-export session log around a test.

    The log is process-global state shared by value across modules, so a test
    that records calls would otherwise leak them into the next test's exported
    notebook.

    Yields
    ------
    state : module
        ``tuiml.agent.tools._state``, with its session containers emptied.
    """
    from tuiml.agent.tools import _state

    containers = (
        _state._SESSION_CALLS,
        _state._MODEL_ID_TO_VAR,
        _state._TRAIN_CALL_SEQ,
    )
    saved = _emptied(containers)

    yield _state

    _refill(containers, saved)
