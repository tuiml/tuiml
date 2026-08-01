"""Agent-authored algorithms: persistence, safety checks, and registry bootstrap.

Agents can call ``tuiml_create_algorithm`` with raw Python source describing a
new ``@classifier`` / ``@regressor`` class. This package:

1. AST-validates the source against a conservative denylist (no network,
   subprocess, or filesystem escape hatches) -- :mod:`validation`.
2. Saves each accepted submission under
   ``~/.tuiml/user_algorithms/<Name>/<version>/algorithm.py`` -- :mod:`storage`,
   :mod:`_paths`.
3. Imports the file and registers both the *versioned* alias
   (``MyAlg_v1_0_0``) and the bare latest-alias (``MyAlg``) into the TuiML
   registry, so every existing MCP tool (``tuiml_train``, ``tuiml_benchmark``,
   ``tuiml_describe``) works on user algorithms unchanged -- :mod:`registration`.
4. On ``load_all()`` scans the directory and re-registers everything,
   preserving agent work across MCP server restarts -- :mod:`registration`.

Experiment history per algorithm version lives in :mod:`research_log`; reading,
searching and editing algorithm source (user-authored *and* built-in) lives in
:mod:`sources`.
"""

from ._paths import USER_ALGS_DIR
from .registration import load_all
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
    "load_all",
    "skeleton",
    "read_source",
    "search_source",
    "list_algorithm_files",
    "edit_algorithm",
    "research_log",
    "record_experiment_runs",
]
