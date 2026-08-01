"""Process-wide state shared by the tool executors.

Everything here lives for the lifetime of one MCP server process. The on-disk
directories under ``~/.tuiml/`` outlive it, so the indexes are rehydrated from
disk at import time: an agent that trains a model, then hits a server restart,
still resolves its ``model_id``.
"""

import os
import threading
from typing import Dict, List

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
