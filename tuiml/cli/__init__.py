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

import inspect
import re
from typing import List

import click
from tuiml import __version__

@click.group()
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


# Import commands
from tuiml.cli import (
    train, predict, evaluate, benchmark, list_cmd,
    serve, setup, uninstall, info, update, mcp,
    status, trace, restart,
    upload, save, stop_server, plot, profile, generate,
    preprocess, select_features, test_statistics, tune,
    read_data, get_skeleton, create_algorithm, delete_algorithm,
    describe, read_algorithm, list_files, search_source, edit_algorithm
)

# Register commands
cli.add_command(train.train)
cli.add_command(predict.predict)
cli.add_command(evaluate.evaluate)
cli.add_command(benchmark.benchmark)
cli.add_command(list_cmd.list_algorithms)
cli.add_command(serve.serve)
cli.add_command(setup.setup)
cli.add_command(uninstall.uninstall)
cli.add_command(info.info)
cli.add_command(update.update)
cli.add_command(mcp.mcp)
cli.add_command(status.status)
cli.add_command(trace.trace)
cli.add_command(restart.restart)
cli.add_command(upload.upload)
cli.add_command(save.save)
cli.add_command(stop_server.stop_server)
cli.add_command(plot.plot)
cli.add_command(profile.profile)
cli.add_command(generate.generate)
cli.add_command(preprocess.preprocess)
cli.add_command(select_features.select_features)
cli.add_command(test_statistics.test_statistics)
cli.add_command(tune.tune)
cli.add_command(read_data.read_data)
cli.add_command(get_skeleton.get_skeleton)
cli.add_command(create_algorithm.create_algorithm)
cli.add_command(delete_algorithm.delete_algorithm)
cli.add_command(describe.describe)
cli.add_command(read_algorithm.read_algorithm)
cli.add_command(list_files.list_files)
cli.add_command(search_source.search_source)
cli.add_command(edit_algorithm.edit_algorithm)

# Docstrings are authored in NumPy style for the HTML docs; reformat them for
# the terminal so section headers and shell examples survive click's wrapping.
cli.help = _help_for_terminal(cli.help)
for _command in cli.commands.values():
    _command.help = _help_for_terminal(_command.help)


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
