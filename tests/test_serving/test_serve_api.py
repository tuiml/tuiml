"""Tests for tuiml.serve() — the background-server entry point."""

import socket

import pytest

import tuiml


@pytest.fixture
def fitted_model():
    """A small fitted pipeline to serve."""
    return tuiml.train("NaiveBayesClassifier", {"source": "iris"}, random_seed=1)


@pytest.fixture(autouse=True)
def _stop_servers():
    """Make sure no test leaves a port bound."""
    yield
    tuiml.stop_server()


def _occupy_port():
    """Bind a free port and return (socket, port) so it reads as in use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock, sock.getsockname()[1]


def _free_port():
    """Return a port number that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestServeBackground:
    def test_serve_returns_reachable_server(self, fitted_model):
        """The reported URL must actually answer, not just look plausible."""
        import requests

        info = tuiml.serve(fitted_model, port=_free_port())
        response = requests.get(info["endpoints"]["health"], timeout=10)
        assert response.status_code == 200

    # uvicorn's worker thread calls sys.exit(1) when it cannot bind, which is
    # precisely the condition under test.
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_serve_raises_when_port_is_in_use(self, fitted_model):
        """A bind failure happens on a background thread; it must still surface.

        Previously this returned a success dict for a server that never
        started, so every later request failed for no visible reason.
        """
        blocker, port = _occupy_port()
        try:
            with pytest.raises(RuntimeError, match="failed to start"):
                tuiml.serve(fitted_model, port=port)
        finally:
            blocker.close()

    def test_serve_reports_the_expected_endpoints(self, fitted_model):
        info = tuiml.serve(fitted_model, port=_free_port())
        assert set(info["endpoints"]) == {"predict", "health", "docs"}
        assert info["url"].startswith("http://127.0.0.1:")

    def test_served_pipeline_applies_its_transformations(self):
        """Serving a Workflow must serve the whole pipeline, not the bare model."""
        import requests
        from tuiml.datasets import load_dataset

        model = tuiml.train(
            {"name": "NaiveBayesClassifier"},
            {"source": "iris"},
            pipeline=[{"name": "StandardScaler"}],
            random_seed=1,
        )
        info = tuiml.serve(model, port=_free_port())
        dataset = load_dataset("iris")

        response = requests.post(
            info["endpoints"]["predict"],
            json={"features": dataset.X[:3].tolist()},
            timeout=10,
        )
        assert response.status_code == 200, response.text
        predictions = response.json()["predictions"]
        # Raw (unscaled) input must still predict correctly, which only holds
        # if the fitted scaler travelled with the model.
        assert list(predictions) == list(model.predict(dataset.X[:3]))

    def test_server_status_and_stop(self, fitted_model):
        tuiml.serve(fitted_model, port=_free_port())
        assert len(tuiml.server_status()) == 1

        tuiml.stop_server()
        assert tuiml.server_status() == []
