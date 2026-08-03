"""The ``tuiml`` command-line interface.

The same operations the Python API and the MCP tools expose, driven from a
shell — train, evaluate, benchmark, tune, plot, serve — plus the commands
that manage the installation itself.

Commands
--------
- **Modelling:** ``train``, ``predict``, ``evaluate``, ``benchmark``,
  ``tune``, ``plot``, ``save``, ``test-statistics``.
- **Data:** ``upload``, ``read-data``, ``profile``, ``preprocess``,
  ``select-features``, ``generate``.
- **Discovery:** ``list``, ``describe``, ``search-source``, ``list-files``.
- **Authoring:** ``create-algorithm``, ``read-algorithm``,
  ``edit-algorithm``, ``delete-algorithm``, ``get-skeleton``.
- **Serving:** ``serve``, ``stop-server``, ``status``.
- **Agents:** ``setup``, ``uninstall``, ``mcp``, ``trace``.
- **Admin:** ``info``, ``update``, ``restart``.

Layout
------
Every command lives in its own flat module (``tuiml.cli.train``), exposing
exactly one ``click`` command whose docstring doubles as its ``--help`` text,
and is attached to the top-level group by :func:`cli`.

Examples
--------
Connect TuiML to your AI clients, then check what it found::

    tuiml setup
    tuiml setup --list

Train and serve a model::

    tuiml train --algorithm RandomForestClassifier --data iris --target class
    tuiml serve model.pkl --port 8000

See Also
--------
:mod:`tuiml.agent.mcp` : The same operations, for an AI agent.
"""

import importlib
import importlib.metadata
import inspect
import re
from typing import List

import click

# Read the version from installed metadata rather than importing the package.
# `from tuiml import __version__` costs ~1.8s, since tuiml/__init__ pulls in
# tuiml.algorithms and, through it, the plotting stack -- a price `tuiml
# --version` should not pay. The installer runs exactly that command right
# after installing, where the pause reads as a hang.
try:
    __version__ = importlib.metadata.version("tuiml")
except importlib.metadata.PackageNotFoundError:  # running from a source tree
    from tuiml import __version__


# Subcommand name -> "module:attribute". Names are not always derivable from
# the function (``list`` is ``list_cmd:list_algorithms``), so they are spelled
# out. A command added here is picked up by ``tuiml --help`` automatically.
_COMMANDS = {
    "benchmark": "tuiml.cli.benchmark:benchmark",
    "create-algorithm": "tuiml.cli.create_algorithm:create_algorithm",
    "delete-algorithm": "tuiml.cli.delete_algorithm:delete_algorithm",
    "describe": "tuiml.cli.describe:describe",
    "edit-algorithm": "tuiml.cli.edit_algorithm:edit_algorithm",
    "evaluate": "tuiml.cli.evaluate:evaluate",
    "generate": "tuiml.cli.generate:generate",
    "get-skeleton": "tuiml.cli.get_skeleton:get_skeleton",
    "info": "tuiml.cli.info:info",
    "list": "tuiml.cli.list_cmd:list_algorithms",
    "list-files": "tuiml.cli.list_files:list_files",
    "mcp": "tuiml.cli.mcp:mcp",
    "plot": "tuiml.cli.plot:plot",
    "predict": "tuiml.cli.predict:predict",
    "preprocess": "tuiml.cli.preprocess:preprocess",
    "profile": "tuiml.cli.profile:profile",
    "read-algorithm": "tuiml.cli.read_algorithm:read_algorithm",
    "read-data": "tuiml.cli.read_data:read_data",
    "restart": "tuiml.cli.restart:restart",
    "save": "tuiml.cli.save:save",
    "search-source": "tuiml.cli.search_source:search_source",
    "select-features": "tuiml.cli.select_features:select_features",
    "serve": "tuiml.cli.serve:serve",
    "setup": "tuiml.cli.setup:setup",
    "status": "tuiml.cli.status:status",
    "stop-server": "tuiml.cli.stop_server:stop_server",
    "test-statistics": "tuiml.cli.test_statistics:test_statistics",
    "trace": "tuiml.cli.trace:trace",
    "train": "tuiml.cli.train:train",
    "tune": "tuiml.cli.tune:tune",
    "uninstall": "tuiml.cli.uninstall:uninstall",
    "update": "tuiml.cli.update:update",
    "upload": "tuiml.cli.upload:upload",
}


class _LazyGroup(click.Group):
    """Import a subcommand's module only when that subcommand is reached.

    Registering the commands eagerly meant importing all 33 modules, and each
    of them imports the tuiml package, so every invocation paid for the whole
    library however little it needed. Resolving them on demand keeps
    ``--version`` and an unknown-command error free of that cost. ``--help``
    still loads every command, because it prints their summaries.
    """

    def list_commands(self, ctx) -> List[str]:
        return sorted(_COMMANDS)

    def get_command(self, ctx, name):
        target = _COMMANDS.get(name)
        if target is None:
            return None
        module_name, _, attr = target.partition(":")
        command = getattr(importlib.import_module(module_name), attr)
        # Reformat once: click keeps the same object across lookups.
        if not getattr(command, "_tuiml_help_formatted", False):
            command.help = _help_for_terminal(command.help)
            command._tuiml_help_formatted = True
        return command


@click.group(cls=_LazyGroup)
@click.option('--random-seed', type=int,
              help='Seed every random number generator TuiML uses, so the whole '
                   'command runs reproducibly.')
@click.version_option(version=__version__, prog_name="tuiml")
@click.pass_context
def cli(ctx, random_seed):
    """TuiML - modern machine learning from the command line.

    Train, evaluate, benchmark, tune, and serve models without writing any
    Python. Components are referred to by their exact class name everywhere
    (``RandomForestClassifier``, ``StandardScaler``, ...), so anything in the
    component registry is reachable from the CLI with no extra mapping.
    Run ``tuiml list`` to browse what is available.

    Examples
    --------
    Train a model on a built-in dataset:

    $ tuiml train -a RandomForestClassifier -d iris -t class

    Make every command in a session reproducible:

    $ tuiml --random-seed 42 benchmark -a SVC -a RandomForestClassifier -d iris

    See the options a subcommand accepts:

    $ tuiml train --help
    """
    # Register the user's own algorithms before any subcommand reads the
    # registry. This runs here rather than at import so that --version and
    # --help, which click handles eagerly and exits before this callback, do
    # not execute user code or report on it.
    from tuiml.agent.user_algorithms import ensure_loaded
    ensure_loaded()

    if random_seed is not None:
        from tuiml.utils.seed import set_global_seed
        set_global_seed(random_seed)
        click.echo(f"Global seed set to: {random_seed}")


def _help_for_terminal(text: str) -> str:
    """Render a NumPy-style docstring as readable ``--help`` text.

    Command docstrings are written in NumPy style so the HTML documentation
    generator can parse them. Click rewraps paragraphs, which would collapse
    a dash underline onto its header (``Examples --------``) and break shell
    examples across lines. This turns headers into ``Header:`` and marks
    example blocks with click's ``\\b`` no-rewrap escape.

    The HTML docs are generated from the source docstrings, not from this
    text, so reformatting here affects only terminal output.

    Parameters
    ----------
    text : str
        The raw command docstring.

    Returns
    -------
    help_text : str
        Help text formatted for click's terminal renderer.
    """
    if not text:
        return text

    # RST inline-code markers are for the HTML docs; plain text in a terminal.
    text = re.sub(r'``([^`]+)``', r'\1', inspect.cleandoc(text))

    lines = text.split('\n')
    out: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        following = lines[index + 1].strip() if index + 1 < len(lines) else ''

        # "Examples" + "--------" collapses under rewrapping; use a plain label
        # followed by a blank line so it stays its own paragraph.
        if stripped and len(following) >= 3 and set(following) == {'-'}:
            out.append(f'{stripped}:')
            out.append('')
            index += 2
            while index < len(lines) and not lines[index].strip():
                index += 1
            continue

        # Keep command lines verbatim so they stay copy-pasteable.
        if stripped.startswith(('$ ', '>>> ')) and (not out or not out[-1].strip()):
            out.append('\b')

        out.append(line)
        index += 1

    return '\n'.join(out)


# The subcommands are resolved lazily by _LazyGroup; only the group's own
# help is reformatted here, since each command's is done as it is loaded.
cli.help = _help_for_terminal(cli.help)


def main():
    """Run the ``tuiml`` command-line interface.

    This is the console-script entry point registered in ``pyproject.toml``.
    It invokes the top-level :func:`cli` group with an empty context object.

    Returns
    -------
    None
        The process exits with the status code chosen by ``click``.
    """
    cli(obj={})

if __name__ == "__main__":
    main()
