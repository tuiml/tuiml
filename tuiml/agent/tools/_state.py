"""Process-wide state shared by the tool executors.

Everything here lives for the lifetime of one MCP server process. The on-disk
directories under ``~/.tuiml/`` outlive it, so the indexes are rehydrated from
disk at import time: an agent that trains a model, then hits a server restart,
still resolves its ``model_id``.
"""

import os
import random
import threading
from typing import Dict, List, Optional

# Persistent TuiML state directory (survives MCP server restarts, unlike /tmp).
_TUIML_HOME = os.path.join(os.path.expanduser('~'), '.tuiml')
_MODELS_DIR = os.path.join(_TUIML_HOME, 'models')
_UPLOADS_DIR = os.path.join(_TUIML_HOME, 'uploads')
os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_UPLOADS_DIR, exist_ok=True)


def _scan(directory: str) -> Dict[str, str]:
    """Index the files in a directory by their basename without extension.

    Parameters
    ----------
    directory : str
        Directory to scan; missing directories yield an empty index.

    Returns
    -------
    index : dict
        Mapping of file stem to absolute path.
    """
    index: Dict[str, str] = {}
    for entry in os.listdir(directory):
        full = os.path.join(directory, entry)
        if os.path.isfile(full):
            index[os.path.splitext(entry)[0]] = full
    return index


# Maps model_id -> file path on disk. Models are written as
# ``<model_id>.joblib``, so the id is recoverable from the filename.
_MODEL_INDEX: Dict[str, str] = _scan(_MODELS_DIR)

# Maps dataset_id (user-provided name or auto-generated) -> file path on disk.
_DATASET_INDEX: Dict[str, str] = _scan(_UPLOADS_DIR)

# Serving state lives in tuiml.serving.server._SERVERS, not here: the agent
# serving tools wrap tuiml.serve()/stop_server()/server_status() so that
# agent-started and library-started servers share one registry.

# Session call log, populated by record_session_call() after every successful
# tool invocation. Consumed by tuiml_export_notebook.
_SESSION_CALLS: List[Dict] = []          # [{tool, args}, ...]
_SESSION_LOCK = threading.Lock()
_MODEL_ID_TO_VAR: Dict[str, str] = {}    # model_id -> "result_N"
_TRAIN_CALL_SEQ: List[int] = []          # indices into _SESSION_CALLS for train calls

# The seed every tool call falls back to when it is given no explicit
# ``random_seed``. Fixed for the life of the process, which is what makes a
# conversation reproducible: re-running the same benchmark must return the same
# table, or comparing two runs measures the seed rather than the change. It used
# to be redrawn per call, so back-to-back identical calls disagreed and the only
# way to reproduce anything was to copy a seed out of an earlier response.
_SESSION_SEED: Optional[int] = None
_SEED_LOCK = threading.Lock()


def get_session_seed() -> int:
    """Return this session's default seed, drawing one on first use.

    Drawn lazily rather than at import so merely importing the package does
    not consume entropy, and so ``TUIML_SEED`` still applies when it is set
    after import. An unparseable ``TUIML_SEED`` is ignored in favour of a
    random draw: a typo in an environment variable should not stop the server.

    Returns
    -------
    seed : int
        The session seed, stable for the lifetime of the process unless
        :func:`set_session_seed` replaces it.
    """
    global _SESSION_SEED
    with _SEED_LOCK:
        if _SESSION_SEED is None:
            env = os.environ.get('TUIML_SEED', '').strip()
            if env:
                try:
                    _SESSION_SEED = int(env)
                except ValueError:
                    _SESSION_SEED = None
            if _SESSION_SEED is None:
                _SESSION_SEED = random.randint(0, 2 ** 31 - 1)
        return _SESSION_SEED


def set_session_seed(seed: Optional[int] = None) -> int:
    """Replace the session seed, or draw a fresh one.

    Parameters
    ----------
    seed : int or None, default=None
        The seed to pin the session to. ``None`` draws a new random one,
        which is how a caller asks for a clean slate without restarting the
        server.

    Returns
    -------
    seed : int
        The seed now in effect.
    """
    global _SESSION_SEED
    with _SEED_LOCK:
        _SESSION_SEED = random.randint(0, 2 ** 31 - 1) if seed is None else int(seed)
        return _SESSION_SEED
