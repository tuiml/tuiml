"""Session -> Jupyter notebook export."""

import os
import uuid
from typing import Any, Dict, List

from .._spec import ToolSpec
from .translate import _translate_call
from .._state import _SESSION_CALLS, _SESSION_LOCK


def _nb_markdown(lines: List[str]) -> Dict:
    """Build a Jupyter markdown cell dict from source lines."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": lines}


def _nb_code(lines: List[str]) -> Dict:
    """Build a Jupyter code cell dict from source lines."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8],
            "execution_count": None, "metadata": {}, "outputs": [],
            "source": lines}


def execute_export_notebook(**kwargs) -> Dict[str, Any]:
    """Export the current MCP session as a reproducible Jupyter notebook.

    Backs the ``tuiml_export_notebook`` tool. Translates every recorded
    successful workflow call into paired markdown + code cells, pinning
    the session's random seed so the notebook reproduces the run.

    Parameters
    ----------
    path : str, default='~/tuiml_session.ipynb'
        Output path for the notebook file (arrives via ``**kwargs``,
        like all parameters below).
    title : str, default='TuiML Session, Exported Notebook'
        Title used in the notebook's header cell.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``path`` (absolute),
        ``cells``, ``workflow_calls`` and ``message``. On failure:
        ``status`` (``'error'``) and ``error`` (e.g. when the session has
        no recorded calls or the file cannot be written).
    """
    import json
    import datetime as _dt

    path = os.path.expanduser(kwargs.get('path') or '~/tuiml_session.ipynb')
    title = kwargs.get('title', 'TuiML Session, Exported Notebook')

    with _SESSION_LOCK:
        calls_snapshot = list(_SESSION_CALLS)

    if not calls_snapshot:
        return {
            'status': 'error',
            'error': (
                'No workflow calls have been recorded in this session yet. '
                'Run some tuiml_train / tuiml_benchmark / tuiml_plot calls first.'
            ),
        }

    cells = []

    # ── Header cell ──────────────────────────────────────────────────────────
    cells.append(_nb_markdown([
        f"# {title}\n",
        f"\n",
        f"Exported from MCP session · {_dt.date.today()}  \n",
        "Re-run each cell top-to-bottom to reproduce the full workflow.\n",
        "\n",
        "**Requirements:** `pip install tuiml`",
    ]))

    # ── Install cell ─────────────────────────────────────────────────────────
    # First executable cell installs tuiml (e.g. on Colab / a fresh kernel).
    cells.append(_nb_code([
        "!pip install tuiml",
    ]))

    # ── Imports cell ─────────────────────────────────────────────────────────
    cells.append(_nb_code([
        "import tuiml\n",
        "from tuiml.datasets import load_dataset\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "import numpy as np",
    ]))

    # ── Global seed cell ─────────────────────────────────────────────────────
    # Mirror the MCP session's reproducibility: execute_tool sets a process-wide
    # seed, which the workflow reads as a fallback for any step that doesn't take
    # an explicit seed (data generation, feature selection, CV splits, plots).
    # Pin the same seed here so the notebook reproduces those steps too.
    _session_seed = next(
        (c['args']['random_seed'] for c in calls_snapshot
         if c['args'].get('random_seed') is not None),
        None,
    )
    if _session_seed is not None:
        cells.append(_nb_markdown([
            "## Reproducibility\n",
            f"This session ran with random seed `{_session_seed}`. "
            "Setting it globally pins NumPy/Python RNG so results match the original run.",
        ]))
        cells.append(_nb_code([
            "from tuiml.utils.seed import set_global_seed\n",
            f"set_global_seed({repr(_session_seed)})",
        ]))

    train_counter = [0]
    skipped = 0
    # Tracks the last source emitted per user-authored algorithm so a session
    # with several edits doesn't repeat the same class definition.
    emitted_sources: Dict[str, str] = {}

    for call in calls_snapshot:
        md_lines, code_lines = _translate_call(call, train_counter, emitted_sources)
        if md_lines is None:
            skipped += 1
            continue
        cells.append(_nb_markdown(md_lines))
        cells.append(_nb_code(code_lines))

    # Build notebook JSON
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(nb, fh, indent=1)
    except Exception as e:
        return {'status': 'error', 'error': f'Could not write notebook: {e}'}

    abs_path = os.path.abspath(path)
    workflow_count = len(calls_snapshot) - skipped
    return {
        'status': 'success',
        'path': abs_path,
        'cells': len(cells),
        'workflow_calls': workflow_count,
        'message': (
            f'Notebook written to {abs_path} '
            f'({workflow_count} workflow steps → {len(cells)} cells). '
            f'Open with: jupyter notebook {abs_path}'
        ),
    }


SPEC = ToolSpec(
    name='tuiml_export_notebook',
    description="Export the current MCP chat session as a reproducible Jupyter notebook (.ipynb). "
        "Training, experiment, tuning, plotting, and data-prep steps performed in this "
        "session are translated to equivalent Python API calls so the user can re-run the "
        "full workflow without the AI client. "
        "Call this at the end of a session when the user wants to save their work as a notebook.",
    input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Destination file path for the notebook. "
                        "Defaults to ~/tuiml_session.ipynb if omitted."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Optional custom title for the notebook header cell.",
                },
            },
            "required": [],
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "path": {"type": "string", "description": "Absolute path to the written .ipynb file"},
                "cells": {"type": "integer", "description": "Total number of notebook cells"},
                "workflow_calls": {"type": "integer", "description": "Number of MCP calls translated"},
                "message": {"type": "string"},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_export_notebook,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=True, open_world=False,
    reproducible=False,
)
