"""Tests for tuiml.serving.server.ModelServer and REST endpoints."""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from tuiml.serving.model_manager import ModelManager
from tuiml.serving.server import ModelServer

# FastAPI TestClient — skip tests gracefully if fastapi is missing
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class MockModel:
    """A minimal mock model with predict and predict_proba."""

    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.ones((len(X), 2)) * 0.5

    def get_params(self):
        return {}


MOCK_INFO = {
    "model_class": "MockModel",
    "model_module": "test",
    "params": {},
    "metadata": {},
    "saved_at": "2025-01-01T00:00:00",
    "format": "pickle",
}


def _create_model_file(tmp_path, name="model.pkl"):
    """Create a valid serialized mock model file using tuiml serialization."""
    from tuiml.utils.serialization import save_model

    model = MockModel()
    path = tmp_path / name
    save_model(model, path)
    return path


@pytest.fixture
def server():
    """Create a fresh ModelServer."""
    return ModelServer(max_models=5)


@pytest.fixture
def client(server):
    """Create a TestClient from a fresh ModelServer."""
    app = server.create_app()
    return TestClient(app)


@pytest.fixture
def loaded_client(tmp_path):
    """Create a TestClient with one model already loaded."""
    model_path = _create_model_file(tmp_path)
    srv = ModelServer(max_models=5)
    srv.load_model("test_model", model_path)
    app = srv.create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# ModelServer class tests
# ---------------------------------------------------------------------------
class TestModelServerInit:
    def test_server_init_defaults(self):
        srv = ModelServer()
        assert srv.title == "TuiML Model Server"
        assert srv.version == "1.0.0"
        assert srv.manager.max_models == 10

    def test_server_init_custom(self):
        srv = ModelServer(max_models=3, title="My API", version="2.0")
        assert srv.manager.max_models == 3
        assert srv.title == "My API"
        assert srv.version == "2.0"

    def test_create_app_returns_fastapi(self, server):
        from fastapi import FastAPI

        app = server.create_app()
        assert isinstance(app, FastAPI)

    def test_app_property_creates_on_demand(self):
        srv = ModelServer()
        from fastapi import FastAPI

        assert isinstance(srv.app, FastAPI)

    def test_load_model_via_server(self, tmp_path):
        model_path = _create_model_file(tmp_path)
        srv = ModelServer()
        info = srv.load_model("m1", model_path)
        assert info["model_id"] == "m1"
        assert srv.manager.is_loaded("m1")

    def test_unload_model_via_server(self, tmp_path):
        model_path = _create_model_file(tmp_path)
        srv = ModelServer()
        srv.load_model("m1", model_path)
        assert srv.unload_model("m1") is True
        assert not srv.manager.is_loaded("m1")


# ---------------------------------------------------------------------------
# Health & status endpoints
# ---------------------------------------------------------------------------
class TestHealthEndpoints:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert data["models_loaded"] == 0

    def test_stats_endpoint_empty(self, client):
        response = client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["loaded_models"] == 0
        assert data["total_predictions"] == 0

    def test_stats_endpoint_with_model(self, loaded_client):
        response = loaded_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["loaded_models"] == 1


# ---------------------------------------------------------------------------
# Model management endpoints
# ---------------------------------------------------------------------------
class TestModelEndpoints:
    def test_list_models_empty(self, client):
        response = client.get("/models")
        assert response.status_code == 200

        data = response.json()
        assert data["models"] == []
        assert data["count"] == 0

    def test_list_models_with_loaded(self, loaded_client):
        response = loaded_client.get("/models")
        assert response.status_code == 200

        data = response.json()
        assert "test_model" in data["models"]
        assert data["count"] == 1

    def test_load_model_endpoint(self, client, tmp_path):
        model_path = _create_model_file(tmp_path)
        response = client.post(
            "/models",
            json={"model_id": "api_model", "path": str(model_path)},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["model_id"] == "api_model"

    def test_load_model_not_found(self, client):
        response = client.post(
            "/models",
            json={"model_id": "bad", "path": "/nonexistent/model.pkl"},
        )
        assert response.status_code == 404

    def test_get_model_info_endpoint(self, loaded_client):
        response = loaded_client.get("/models/test_model")
        assert response.status_code == 200

        data = response.json()
        assert data["model_id"] == "test_model"

    def test_get_model_info_not_found(self, client):
        response = client.get("/models/nonexistent")
        assert response.status_code == 404

    def test_unload_model_endpoint(self, loaded_client):
        response = loaded_client.delete("/models/test_model")
        assert response.status_code == 200

        # Verify it is gone
        response = loaded_client.get("/models")
        data = response.json()
        assert "test_model" not in data["models"]

    def test_unload_model_not_found(self, client):
        response = client.delete("/models/ghost")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------
class TestPredictionEndpoints:
    def test_predict_endpoint(self, loaded_client):
        response = loaded_client.post(
            "/models/test_model/predict",
            json={"features": [[1.0, 2.0], [3.0, 4.0]]},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["model_id"] == "test_model"
        assert len(data["predictions"]) == 2

    def test_predict_model_not_found(self, client):
        response = client.post(
            "/models/missing/predict",
            json={"features": [[1, 2]]},
        )
        assert response.status_code == 404

    def test_predict_proba_endpoint(self, loaded_client):
        response = loaded_client.post(
            "/models/test_model/predict_proba",
            json={"features": [[1.0, 2.0]]},
        )
        assert response.status_code == 200

        data = response.json()
        assert "probabilities" in data
        assert len(data["probabilities"]) == 1
        assert len(data["probabilities"][0]) == 2

    def test_predict_proba_model_not_found(self, client):
        response = client.post(
            "/models/missing/predict_proba",
            json={"features": [[1, 2]]},
        )
        assert response.status_code == 404

    def test_predict_default_endpoint(self, loaded_client):
        """POST /predict uses the first loaded model by default."""
        response = loaded_client.post(
            "/predict",
            json={"features": [[1.0, 2.0], [3.0, 4.0]]},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["model_id"] == "test_model"
        assert len(data["predictions"]) == 2

    def test_predict_default_no_models(self, client):
        response = client.post("/predict", json={"features": [[1, 2]]})
        assert response.status_code == 400

    def test_predict_default_with_model_id(self, loaded_client):
        response = loaded_client.post(
            "/predict?model_id=test_model",
            json={"features": [[1.0, 2.0]]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test_model"


# ---------------------------------------------------------------------------
# End-to-end: load then predict
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_load_and_predict(self, client, tmp_path):
        model_path = _create_model_file(tmp_path)

        # Load the model via API
        load_resp = client.post(
            "/models",
            json={"model_id": "e2e_model", "path": str(model_path)},
        )
        assert load_resp.status_code == 200

        # Predict
        pred_resp = client.post(
            "/models/e2e_model/predict",
            json={"features": [[1, 2], [3, 4], [5, 6]]},
        )
        assert pred_resp.status_code == 200
        data = pred_resp.json()
        assert len(data["predictions"]) == 3

    def test_load_predict_unload(self, client, tmp_path):
        model_path = _create_model_file(tmp_path)

        # Load
        client.post(
            "/models",
            json={"model_id": "lifecycle", "path": str(model_path)},
        )

        # Predict
        resp = client.post(
            "/models/lifecycle/predict",
            json={"features": [[1, 2]]},
        )
        assert resp.status_code == 200

        # Unload
        resp = client.delete("/models/lifecycle")
        assert resp.status_code == 200

        # Predict again should fail
        resp = client.post(
            "/models/lifecycle/predict",
            json={"features": [[1, 2]]},
        )
        assert resp.status_code == 404
