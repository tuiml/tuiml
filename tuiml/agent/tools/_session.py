"""Recording of successful tool calls for notebook export."""

from ._state import (
    _MODEL_ID_TO_VAR,
    _SESSION_CALLS,
    _SESSION_LOCK,
    _TRAIN_CALL_SEQ,
)


def record_session_call(tool_name: str, args: dict, result: dict) -> None:
    """Record a *successful* MCP tool call for notebook export.

    Called by server.py's call_tool handler after every invocation. Only
    successful calls are stored: failed calls (bad algorithm name, schema
    mismatch, etc.) return ``{"status": "error"}`` rather than raising, and
    must not become notebook cells, otherwise the exported notebook would
    raise when re-run. Strips internal kwargs (_progress_callback, etc.)
    before storing so the notebook sees only user-visible arguments.

    Parameters
    ----------
    tool_name : str
        Name of the MCP tool that was invoked (e.g. ``"tuiml_train"``).
    args : dict
        Arguments the tool was called with. Keys starting with an
        underscore are stripped before recording.
    result : dict
        The result dict returned by the tool executor. Only calls whose
        result has ``status == "success"`` are recorded.

    Returns
    -------
    None
        The call is appended to the module-level session log as a side
        effect; nothing is returned.
    """
    from . import is_reproducible

    if not is_reproducible(tool_name):
        return
    # Skip anything that didn't succeed. Tool executors signal failure via a
    # status field instead of raising, so a missing/non-success status means
    # the call produced no reproducible result worth exporting.
    if not isinstance(result, dict) or result.get('status') != 'success':
        return
    clean_args = {k: v for k, v in args.items() if not k.startswith('_')}
    # Capture the effective random seed. execute_tool resolves the seed (explicit
    # arg → global seed → default) and writes it back into the *result*, not the
    # args. Fold it into the recorded args so the exported notebook reproduces the
    # exact run even when the seed was auto-resolved rather than passed explicitly.
    if isinstance(result, dict) and result.get('random_seed') is not None \
            and 'random_seed' not in clean_args:
        clean_args['random_seed'] = result['random_seed']
    with _SESSION_LOCK:
        idx = len(_SESSION_CALLS)
        _SESSION_CALLS.append({'tool': tool_name, 'args': clean_args})
        if tool_name == 'tuiml_train' and isinstance(result, dict) and result.get('model_id'):
            n = len(_TRAIN_CALL_SEQ) + 1
            _MODEL_ID_TO_VAR[result['model_id']] = f'result_{n}'
            _TRAIN_CALL_SEQ.append(idx)
