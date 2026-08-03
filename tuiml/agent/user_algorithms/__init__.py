"""Algorithms an agent wrote itself.

An agent that can only call built-in algorithms is limited to what shipped.
``tuiml_create_algorithm`` lets it submit raw Python for a new
``@classifier`` / ``@regressor`` class, which this package validates, stores
and registers — after which the new algorithm behaves like any other, and
every existing tool works on it unchanged.

The pipeline
------------
1. **Validate** — the source is AST-checked against a conservative denylist:
   no network, no subprocess, no filesystem escape hatches. Accepting
   arbitrary generated code is the risk here, so this runs before anything
   touches disk (:mod:`~tuiml.agent.user_algorithms.validation`).
2. **Store** — accepted source is written to
   ``~/.tuiml/user_algorithms/<Name>/<version>/algorithm.py``, versioned, so
   an edit never destroys the version that worked
   (:mod:`~tuiml.agent.user_algorithms.storage`).
3. **Register** — the file is imported and registered under both a versioned
   alias (``MyAlg_v1_0_0``) and a bare latest alias (``MyAlg``), which is what
   makes ``tuiml_train``, ``tuiml_benchmark`` and ``tuiml_describe`` work on
   it with no special-casing
   (:mod:`~tuiml.agent.user_algorithms.registration`).
4. **Reload** — ``load_all()`` rescans the directory at import and
   re-registers everything, so an agent's work survives an MCP server
   restart.

Also here
---------
- :mod:`~tuiml.agent.user_algorithms.research_log` — experiment history per
  algorithm version, so an agent can see what it already tried.
- :mod:`~tuiml.agent.user_algorithms.sources` — reading, searching and
  editing algorithm source, user-authored *and* built-in.

Notes
-----
The denylist is a guard against an agent's mistakes, not a sandbox. Source
that reaches here still runs in your interpreter with your permissions.
"""

from ._paths import USER_ALGS_DIR
from .registration import ensure_loaded, load_all
from .research_log import record_experiment_runs, research_log
from .sources import (
    edit_algorithm,
    list_algorithm_files,
    read_source,
    search_source,
)
from .storage import create, delete, list_all
from .templates import skeleton

__all__ = [
    "USER_ALGS_DIR",
    "create",
    "delete",
    "list_all",
    "ensure_loaded",
    "load_all",
    "skeleton",
    "read_source",
    "search_source",
    "list_algorithm_files",
    "edit_algorithm",
    "research_log",
    "record_experiment_runs",
]
