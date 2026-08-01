"""The tools for the installation itself, and for serving models.

Housekeeping rather than modelling: what version is running, how to update
it, and how to put a trained model behind an HTTP endpoint.

Tools
-----
- **tuiml_system_info:** Installed version, install method, Python and
  platform, optionally checking PyPI for a newer release.
- **tuiml_self_update:** Reinstall TuiML at the latest or a pinned version.
  Supports ``dry_run`` to report what it would do without doing it.
- **tuiml_restart:** Kill running ``tuiml-mcp`` processes so their parent
  clients respawn them on the freshly installed code.
- **tuiml_serve_model / tuiml_stop_server / tuiml_server_status:** Run a
  trained model as a REST API, and manage those servers.

Notes
-----
``tuiml_self_update`` refuses to upgrade an editable checkout, since the fix
there is ``git pull``. ``tuiml_restart`` is genuinely destructive: it
terminates processes, and it excludes the calling process so an agent does
not kill the server handling its own request. The serving tools are thin
wrappers over :mod:`tuiml.serving`, sharing one registry with it, so a server
started from Python can be stopped by an agent and vice versa.
"""
