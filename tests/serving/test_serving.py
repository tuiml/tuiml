"""Model manager, request/response schemas and the ``serve`` API.

Merged from: test_serving_model_manager.py, test_serving_schemas.py, test_serving_serve_api.py
"""

from unittest.mock import patch
import numpy as np
import pytest
from tuiml.serving.model_manager import ModelManager
from tuiml.serving.schemas import (
    PredictRequest,
    PredictResponse,
    PredictProbaResponse,
    LoadModelRequest,
    ModelInfoResponse,
    ModelListResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
    BatchPredictRequest,
    BatchPredictResponse,
)
import socket
import tuiml


# --------------------------------------------------------------------------
# Tests for tuiml.serving.model_manager.ModelManager.
# --------------------------------------------------------------------------

class MockModel:
    """A minimal mock model with predict and predict_proba support."""

    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.ones((len(X), 2)) * 0.5


class MockModelNoProba:
    """A mock model that does NOT support predict_proba."""

    def predict(self, X):
        return np.zeros(len(X))


MOCK_INFO = {
    "model_class": "MockModel",
    "model_module": "tests.test_serving.test_model_manager",
    "params": {"alpha": 1.0},
    "metadata": {},
    "saved_at": "2025-01-01T00:00:00",
    "format": "pickle",
}


def _load_mock(manager, model_id, tmp_path, model=None, metadata=None):
    """Create a dummy file and load a mock model into the manager."""
    model_file = tmp_path / f"{model_id}.pkl"
    model_file.touch()

    mock_model = model or MockModel()

    with patch("tuiml.serving.model_manager.load_model", return_value=mock_model):
        with patch("tuiml.serving.model_manager.load_model_info", return_value=MOCK_INFO.copy()):
            return manager.load(model_id, model_file, metadata=metadata)


class TestModelManagerInit:
    def test_init_default(self):
        manager = ModelManager()
        assert manager.max_models == 10
        assert manager.list_models() == []

    def test_init_custom(self):
        manager = ModelManager(max_models=5)
        assert manager.max_models == 5


class TestModelManagerLoadUnload:
    @patch("tuiml.serving.model_manager.load_model")
    @patch("tuiml.serving.model_manager.load_model_info")
    def test_load_model(self, mock_info, mock_load, tmp_path):
        mock_load.return_value = MockModel()
        mock_info.return_value = MOCK_INFO.copy()

        model_file = tmp_path / "model.pkl"
        model_file.touch()

        manager = ModelManager()
        info = manager.load("test_model", model_file)

        assert manager.is_loaded("test_model")
        assert "test_model" in manager.list_models()
        assert info["model_id"] == "test_model"
        assert info["model_class"] == "MockModel"

    def test_load_file_not_found(self):
        manager = ModelManager()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            manager.load("missing", "/nonexistent/path/model.pkl")

    def test_unload_model(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path)
        assert manager.is_loaded("m1")

        result = manager.unload("m1")
        assert result is True
        assert not manager.is_loaded("m1")

    def test_unload_nonexistent(self):
        manager = ModelManager()
        assert manager.unload("does_not_exist") is False


class TestModelManagerGetters:
    def test_get_model(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path)

        model = manager.get_model("m1")
        assert isinstance(model, MockModel)

    def test_get_model_not_loaded(self):
        manager = ModelManager()
        with pytest.raises(KeyError, match="not loaded"):
            manager.get_model("missing")

    def test_get_model_info(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path, metadata={"description": "test"})

        info = manager.get_model_info("m1")
        assert info["model_id"] == "m1"
        assert info["model_class"] == "MockModel"
        assert info["model_module"] == "tests.test_serving.test_model_manager"
        assert "params" in info
        assert "loaded_at" in info
        assert info["prediction_count"] == 0
        assert info["metadata"] == {"description": "test"}

    def test_get_model_info_not_loaded(self):
        manager = ModelManager()
        with pytest.raises(KeyError, match="not loaded"):
            manager.get_model_info("missing")

    def test_list_models(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "a", tmp_path)
        _load_mock(manager, "b", tmp_path)
        _load_mock(manager, "c", tmp_path)

        models = manager.list_models()
        assert set(models) == {"a", "b", "c"}

    def test_is_loaded_true_then_false(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path)
        assert manager.is_loaded("m1") is True

        manager.unload("m1")
        assert manager.is_loaded("m1") is False


class TestModelManagerPredict:
    def test_predict(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path)

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        predictions = manager.predict("m1", X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2
        np.testing.assert_array_equal(predictions, np.zeros(2))

    def test_predict_not_loaded(self):
        manager = ModelManager()
        with pytest.raises(KeyError, match="not loaded"):
            manager.predict("missing", np.array([[1, 2]]))

    def test_predict_proba(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path)

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        probas = manager.predict_proba("m1", X)

        assert isinstance(probas, np.ndarray)
        assert probas.shape == (2, 2)
        np.testing.assert_array_almost_equal(probas, np.ones((2, 2)) * 0.5)

    def test_predict_proba_not_loaded(self):
        manager = ModelManager()
        with pytest.raises(KeyError, match="not loaded"):
            manager.predict_proba("missing", np.array([[1, 2]]))

    def test_predict_proba_not_supported(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path, model=MockModelNoProba())

        X = np.array([[1.0, 2.0]])
        with pytest.raises(NotImplementedError, match="does not support probability"):
            manager.predict_proba("m1", X)

    def test_predict_with_list_input(self, tmp_path):
        """Predict should accept a plain Python list and convert to ndarray."""
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path)

        predictions = manager.predict("m1", [[1, 2], [3, 4]])
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2


class TestModelManagerEviction:
    def test_lru_eviction(self, tmp_path):
        max_models = 3
        manager = ModelManager(max_models=max_models)

        for i in range(max_models + 1):
            _load_mock(manager, f"model_{i}", tmp_path)

        # The first model should have been evicted
        assert not manager.is_loaded("model_0")
        # The rest should still be loaded
        assert manager.is_loaded("model_1")
        assert manager.is_loaded("model_2")
        assert manager.is_loaded("model_3")
        assert len(manager.list_models()) == max_models

    def test_lru_access_prevents_eviction(self, tmp_path):
        """Accessing a model via get_model moves it to end (LRU), preventing eviction."""
        manager = ModelManager(max_models=3)

        _load_mock(manager, "a", tmp_path)
        _load_mock(manager, "b", tmp_path)
        _load_mock(manager, "c", tmp_path)

        # Access "a" so it becomes most recently used
        manager.get_model("a")

        # Now load a 4th model — "b" (the oldest untouched) should be evicted
        _load_mock(manager, "d", tmp_path)

        assert manager.is_loaded("a"), "a was accessed and should survive"
        assert not manager.is_loaded("b"), "b should be evicted"
        assert manager.is_loaded("c")
        assert manager.is_loaded("d")


class TestModelManagerStats:
    def test_get_stats(self, tmp_path):
        manager = ModelManager(max_models=5)
        _load_mock(manager, "m1", tmp_path)
        _load_mock(manager, "m2", tmp_path)

        stats = manager.get_stats()
        assert stats["loaded_models"] == 2
        assert stats["max_models"] == 5
        assert stats["total_predictions"] == 0
        assert len(stats["models"]) == 2

    def test_prediction_count(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path)

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        manager.predict("m1", X)

        info = manager.get_model_info("m1")
        assert info["prediction_count"] == 3

        # Second batch
        manager.predict("m1", np.array([[7.0, 8.0]]))
        info = manager.get_model_info("m1")
        assert info["prediction_count"] == 4

    def test_prediction_count_in_stats(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path)

        manager.predict("m1", np.array([[1, 2], [3, 4]]))

        stats = manager.get_stats()
        assert stats["total_predictions"] == 2

    def test_predict_proba_increments_count(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path)

        manager.predict_proba("m1", np.array([[1, 2], [3, 4]]))
        info = manager.get_model_info("m1")
        assert info["prediction_count"] == 2


class TestModelManagerClear:
    def test_clear(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path)
        _load_mock(manager, "m2", tmp_path)
        assert len(manager.list_models()) == 2

        manager.clear()
        assert len(manager.list_models()) == 0
        assert not manager.is_loaded("m1")
        assert not manager.is_loaded("m2")

    def test_clear_resets_stats(self, tmp_path):
        manager = ModelManager()
        _load_mock(manager, "m1", tmp_path)
        manager.predict("m1", np.array([[1, 2]]))

        manager.clear()
        stats = manager.get_stats()
        assert stats["loaded_models"] == 0
        assert stats["total_predictions"] == 0


# --------------------------------------------------------------------------
# Tests for tuiml.serving.schemas (Pydantic request/response models).
# --------------------------------------------------------------------------

class TestPredictRequest:
    def test_valid_construction(self):
        req = PredictRequest(features=[[1, 2], [3, 4]])
        assert req.features == [[1, 2], [3, 4]]

    def test_single_sample(self):
        req = PredictRequest(features=[[5.1, 3.5, 1.4, 0.2]])
        assert len(req.features) == 1

    def test_float_values(self):
        req = PredictRequest(features=[[1.5, 2.7], [3.1, 4.9]])
        assert req.features[0][0] == 1.5

    def test_missing_features_raises(self):
        with pytest.raises(Exception):
            PredictRequest()


class TestPredictResponse:
    def test_valid_construction(self):
        resp = PredictResponse(predictions=[0, 1, 0], model_id="clf_1")
        assert resp.predictions == [0, 1, 0]
        assert resp.model_id == "clf_1"
        assert resp.model_class is None

    def test_with_model_class(self):
        resp = PredictResponse(
            predictions=[1, 0],
            model_id="test",
            model_class="NaiveBayes",
        )
        assert resp.model_class == "NaiveBayes"


class TestPredictProbaResponse:
    def test_valid_construction(self):
        resp = PredictProbaResponse(
            probabilities=[[0.8, 0.2], [0.3, 0.7]],
            model_id="clf_1",
        )
        assert len(resp.probabilities) == 2
        assert resp.classes is None

    def test_with_classes(self):
        resp = PredictProbaResponse(
            probabilities=[[0.9, 0.1]],
            classes=["cat", "dog"],
            model_id="clf",
        )
        assert resp.classes == ["cat", "dog"]


class TestLoadModelRequest:
    def test_valid_construction(self):
        req = LoadModelRequest(model_id="my_model", path="models/clf.pkl")
        assert req.model_id == "my_model"
        assert req.path == "models/clf.pkl"
        assert req.metadata is None

    def test_with_metadata(self):
        req = LoadModelRequest(
            model_id="m1",
            path="/tmp/model.pkl",
            metadata={"version": "1.0", "dataset": "iris"},
        )
        assert req.metadata["version"] == "1.0"


class TestModelInfoResponse:
    def test_valid_construction(self):
        resp = ModelInfoResponse(
            model_id="m1",
            model_class="RandomForest",
            model_module="tuiml.algorithms.ensemble",
            params={"n_trees": 100},
            path="/models/rf.pkl",
            loaded_at="2025-01-01T00:00:00",
            prediction_count=42,
            metadata={"tag": "production"},
        )
        assert resp.model_id == "m1"
        assert resp.model_class == "RandomForest"
        assert resp.prediction_count == 42

    def test_defaults(self):
        resp = ModelInfoResponse(model_id="m1")
        assert resp.model_class is None
        assert resp.model_module is None
        assert resp.params == {}
        assert resp.prediction_count == 0
        assert resp.metadata == {}


class TestModelListResponse:
    def test_valid_construction(self):
        resp = ModelListResponse(models=["m1", "m2", "m3"], count=3)
        assert resp.count == 3
        assert "m1" in resp.models

    def test_empty_list(self):
        resp = ModelListResponse(models=[], count=0)
        assert resp.count == 0
        assert resp.models == []


class TestHealthResponse:
    def test_valid_construction(self):
        # An arbitrary version string on purpose: the real endpoint passes
        # tuiml.__version__ in, so this only checks the field round-trips.
        # Hardcoding the release version here made every bump break the test,
        # because bump_version.py rewrote the constructor argument and left
        # the assertion on the old value.
        resp = HealthResponse(status="healthy", version="9.9.9", models_loaded=2)
        assert resp.status == "healthy"
        assert resp.version == "9.9.9"
        assert resp.models_loaded == 2


class TestStatsResponse:
    def test_valid_construction(self):
        resp = StatsResponse(
            loaded_models=2,
            max_models=10,
            total_predictions=500,
            models=[
                {"model_id": "m1", "model_class": "SVM", "prediction_count": 300},
                {"model_id": "m2", "model_class": "NB", "prediction_count": 200},
            ],
        )
        assert resp.loaded_models == 2
        assert resp.total_predictions == 500
        assert len(resp.models) == 2


class TestErrorResponse:
    def test_valid_construction(self):
        resp = ErrorResponse(error="Model not found")
        assert resp.error == "Model not found"
        assert resp.detail is None
        assert resp.model_id is None

    def test_with_all_fields(self):
        resp = ErrorResponse(
            error="Prediction failed",
            detail="Input shape mismatch",
            model_id="broken_model",
        )
        assert resp.detail == "Input shape mismatch"
        assert resp.model_id == "broken_model"


class TestBatchPredictRequest:
    def test_valid_construction(self):
        req = BatchPredictRequest(
            requests=[
                {"model_id": "m1", "features": [[1, 2]]},
                {"model_id": "m2", "features": [[3, 4]]},
            ]
        )
        assert len(req.requests) == 2


class TestBatchPredictResponse:
    def test_valid_construction(self):
        resp = BatchPredictResponse(
            results=[
                {"model_id": "m1", "predictions": [0]},
                {"model_id": "m2", "error": "not loaded"},
            ],
            successful=1,
            failed=1,
        )
        assert resp.successful == 1
        assert resp.failed == 1
        assert len(resp.results) == 2


# --------------------------------------------------------------------------
# Tests for tuiml.serve() — the background-server entry point.
# --------------------------------------------------------------------------

@pytest.fixture
def fitted_model():
    """A small fitted pipeline to serve."""
    return tuiml.train({"model": {"name": "NaiveBayesClassifier"},
                        "data": {"source": "iris"}, "random_seed": 1})


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

        model = tuiml.train({
            "model": {"name": "NaiveBayesClassifier"},
            "data": {"source": "iris"},
            "pipeline": [{"name": "StandardScaler"}],
            "random_seed": 1,
        })
        info = tuiml.serve(model, port=_free_port())
        dataset = load_dataset("iris")

        response = requests.post(
            info["endpoints"]["predict"],
            json={"features": dataset.X[:3].tolist()},
            headers={"Authorization": f"Bearer {info['auth_token']}"},
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
