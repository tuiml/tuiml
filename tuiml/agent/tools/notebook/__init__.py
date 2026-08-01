"""Turning a finished agent session into a runnable notebook.

A conversation with an agent produces results but no artefact anyone can
re-run or review. This exports the session as a ``.ipynb`` of plain TuiML
Python — no MCP, no agent — so the work can be checked, edited and repeated.

Tools
-----
- **tuiml_export_notebook:** Write the session so far to a notebook, with a
  markdown heading per step and the equivalent Python beneath it.

Modules
-------
- :mod:`~tuiml.agent.tools.notebook.translate` turns one recorded tool call
  into the Python that reproduces it.
- :mod:`~tuiml.agent.tools.notebook.export` orders those cells and writes the
  notebook.

Notes
-----
Only *successful* calls are recorded, and only reproducible ones: a failed
call would make the exported notebook raise when re-run, and a lookup like
``tuiml_list`` produces no analysis worth exporting. The effective random
seed is folded into each cell even when it was auto-generated, so the
notebook reproduces the exact run rather than a similar one.
"""
