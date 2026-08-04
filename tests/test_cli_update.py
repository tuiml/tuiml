"""`tuiml update` progress reporting and dry-run output.

The upgrade blocks inside a captured pip/uv subprocess for up to five
minutes, so these cover the two things that made it look hung: that phase
progress is emitted before each blocking step, and that the CLI surfaces it.
"""

import contextlib
import json
import subprocess
from unittest import mock

from click.testing import CliRunner

from tuiml.agent.tools.system.self_update import execute_self_update
from tuiml.cli.update import _Status, update


@contextlib.contextmanager
def _stub_upgrade(method="pip"):
    """Patch out every side effect of a real upgrade.

    ``execute_system_info`` is stubbed alongside ``subprocess.run`` because
    it shells out too, and a blanket ``subprocess.run`` patch would feed it
    a response shaped for the installer.

    Parameters
    ----------
    method : str, default='pip'
        Install method the detector should report.

    Yields
    ------
    run : unittest.mock.MagicMock
        The patched ``subprocess.run``, for argv assertions.
    """
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="9.9.9\n", stderr="",
    )
    with mock.patch(
        "tuiml.agent.tools.system.self_update._detect_install_method",
        return_value={"method": method},
    ), mock.patch(
        "tuiml.agent.tools.system.self_update._detect_install_source",
        return_value={"kind": "pypi"},
    ), mock.patch(
        "tuiml.agent.tools.system.self_update.execute_system_info",
        return_value={"version": "0.0.1"},
    ), mock.patch(
        "subprocess.run", return_value=completed,
    ) as run:
        yield run


class TestSelfUpdateProgress:
    """execute_self_update announces each phase before blocking on it."""

    def _run_with_stub(self, events, method="pip"):
        """Run a fake upgrade, recording progress events.

        Parameters
        ----------
        events : list
            Mutable list the progress callback appends payloads to.
        method : str, default='pip'
            Install method the detector should report.

        Returns
        -------
        result : dict
            The result dict from ``execute_self_update``.
        """
        with _stub_upgrade(method):
            return execute_self_update(_progress_callback=events.append)

    def test_phases_are_emitted_in_order(self):
        """Every blocking step is announced before it runs."""
        events = []
        result = self._run_with_stub(events)
        assert result["status"] == "success", result
        phases = [e["phase"] for e in events]
        assert phases == ["detect", "read_version", "install", "verify"], phases
        assert all(e["type"] == "update_progress" for e in events)
        assert all(e["message"] for e in events), "a phase carried no message"

    def test_install_phase_names_the_command(self):
        """The long phase says what it is doing, not just 'working'."""
        events = []
        self._run_with_stub(events, method="uv-tool")
        install = next(e for e in events if e["phase"] == "install")
        assert "uv-tool" in install["message"]

    def test_callback_is_optional(self):
        """Omitting the hook must not break the MCP path."""
        with _stub_upgrade():
            assert execute_self_update()["status"] == "success"

    def test_callback_is_not_forwarded_to_the_installer(self):
        """The internal hook must not leak into the pip/uv argv."""
        events = []
        result = self._run_with_stub(events)
        assert not any("progress" in str(part) for part in result["command"])


class TestUpdateCli:
    """CLI-level behaviour of `tuiml update`."""

    def test_dry_run_on_editable_checkout_does_not_crash(self):
        """Regression: command is None here, and ' '.join(None) raised TypeError."""
        with mock.patch(
            "tuiml.agent.tools.system.self_update._detect_install_method",
            return_value={"method": "editable-dev"},
        ):
            result = CliRunner().invoke(update, ["--dry-run"])
        assert result.exit_code == 0, result.output
        assert "editable" in result.output
        assert result.exception is None

    def test_json_output_is_parseable(self):
        """--json emits JSON on stdout and no progress chatter."""
        with mock.patch(
            "tuiml.agent.tools.system.self_update._detect_install_method",
            return_value={"method": "editable-dev"},
        ):
            result = CliRunner().invoke(update, ["--dry-run", "--json"])
        payload = json.loads(result.output)
        assert payload["status"] == "success"

    def test_progress_reaches_the_user(self):
        """A non-TTY run still prints each phase, so it never looks hung."""
        with _stub_upgrade():
            result = CliRunner().invoke(update, [])
        assert result.exit_code == 0, result.output
        # The first line lands before the slow import and the subprocess.
        assert "Starting TuiML update" in result.output
        assert "Installing" in result.output


class TestMcpProgressNotifications:
    """The same phases reach MCP clients, not just the CLI."""

    def test_self_update_is_registered_for_progress(self):
        """Without this the client sees nothing while pip/uv runs."""
        from tuiml.agent.mcp.server import _PROGRESS_TOOLS

        assert "tuiml_self_update" in _PROGRESS_TOOLS

    def test_update_phase_formats_as_a_sentence(self):
        """update_progress must not fall through to the raw JSON dump."""
        from tuiml.agent.mcp.server import _format_progress

        msg = _format_progress({
            "type": "update_progress",
            "phase": "install",
            "message": "Installing tuiml via uv-tool, this can take a minute",
        })
        assert msg == "[Update] Installing tuiml via uv-tool, this can take a minute"

    def test_unknown_progress_type_still_falls_back(self):
        """The fallback other tools rely on must survive the new branch."""
        from tuiml.agent.mcp.server import _format_progress

        assert _format_progress({"type": "mystery", "a": 1}) == '{"type": "mystery", "a": 1}'

    def test_callback_survives_the_execute_tool_dispatch(self):
        """execute_tool must forward the hook through to the executor."""
        from tuiml.agent.tools import execute_tool

        events = []
        with _stub_upgrade("uv-tool"):
            result = execute_tool("tuiml_self_update", _progress_callback=events.append)
        assert result["status"] == "success", result
        assert [e["phase"] for e in events] == [
            "detect", "read_version", "install", "verify",
        ]


class TestStatusIndicator:
    """The status line must stay out of the way when it cannot animate."""

    def test_disabled_status_writes_nothing(self, capsys):
        """--json builds a disabled indicator; it must be silent."""
        status = _Status(enabled=False)
        status.start("hello")
        status.update("world")
        status.stop()
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_stop_is_idempotent(self):
        """Calling stop twice (finally + normal exit) must not raise."""
        status = _Status(enabled=False)
        status.start("hello")
        status.stop()
        status.stop()
