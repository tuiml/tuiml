"""HTTP prediction server."""

import numpy as np
import pytest
from tuiml.serving.server import ModelServer
from fastapi.testclient import TestClient


pytest.importorskip("fastapi")


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


# A fixed token keeps the fixtures deterministic. Production servers generate
# their own; see TestAuthentication for the behaviour that matters.
TEST_TOKEN = "test-token-not-a-secret"

AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}


@pytest.fixture
def server(tmp_path):
    """Create a fresh ModelServer.

    ``models_dir`` is the temp directory the model fixtures write into, since
    ``POST /models`` refuses paths outside it.
    """
    return ModelServer(max_models=5, auth_token=TEST_TOKEN, models_dir=tmp_path)


@pytest.fixture
def client(server):
    """Create an authenticated TestClient from a fresh ModelServer.

    The token is sent by default so these tests exercise the endpoints rather
    than the middleware; authentication itself is covered separately.
    """
    return TestClient(server.create_app(), headers=AUTH)


@pytest.fixture
def loaded_client(tmp_path):
    """Create an authenticated TestClient with one model already loaded."""
    model_path = _create_model_file(tmp_path)
    srv = ModelServer(max_models=5, auth_token=TEST_TOKEN, models_dir=tmp_path)
    srv.load_model("test_model", model_path)
    return TestClient(srv.create_app(), headers=AUTH)


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

    def test_load_model_not_found(self, client, tmp_path):
        """A missing file inside models_dir is a 404.

        The path must be inside models_dir to reach the filesystem check at
        all -- one outside is refused as 403 before the file is looked for, so
        the endpoint cannot be used to probe for files elsewhere on the disk.
        """
        response = client.post(
            "/models",
            json={"model_id": "bad", "path": str(tmp_path / "nonexistent.pkl")},
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


class TestAuthentication:
    """The API is authenticated by default.

    Loading a model unpickles a file and predicting runs it, so an unauthenticated
    server hands both to anyone who can reach the port -- including, when it is
    bound to loopback, any web page the operator happens to open.
    """

    def test_token_is_generated_by_default(self):
        srv = ModelServer()
        assert srv.auth_token
        assert len(srv.auth_token) >= 32

    def test_generated_tokens_differ_between_servers(self):
        assert ModelServer().auth_token != ModelServer().auth_token

    @pytest.mark.parametrize("path", ["/models", "/stats", "/models/test_model"])
    def test_endpoints_require_a_token(self, tmp_path, path):
        srv = ModelServer(auth_token=TEST_TOKEN, models_dir=tmp_path)
        srv.load_model("test_model", _create_model_file(tmp_path))
        anon = TestClient(srv.create_app())
        assert anon.get(path).status_code == 401

    def test_predict_requires_a_token(self, tmp_path):
        srv = ModelServer(auth_token=TEST_TOKEN, models_dir=tmp_path)
        srv.load_model("test_model", _create_model_file(tmp_path))
        anon = TestClient(srv.create_app())
        response = anon.post(
            "/models/test_model/predict", json={"features": [[1.0, 2.0]]}
        )
        assert response.status_code == 401

    def test_unload_requires_a_token(self, tmp_path):
        srv = ModelServer(auth_token=TEST_TOKEN, models_dir=tmp_path)
        srv.load_model("test_model", _create_model_file(tmp_path))
        assert TestClient(srv.create_app()).delete("/models/test_model").status_code == 401

    @pytest.mark.parametrize(
        "header",
        [None, "", "Bearer wrong", "Basic " + TEST_TOKEN, TEST_TOKEN],
        ids=["missing", "empty", "wrong-token", "wrong-scheme", "no-scheme"],
    )
    def test_bad_credentials_are_rejected(self, server, header):
        headers = {} if header is None else {"Authorization": header}
        anon = TestClient(server.create_app(), headers=headers)
        assert anon.get("/models").status_code == 401

    @pytest.mark.parametrize("path", ["/", "/health"])
    def test_liveness_endpoints_stay_public(self, server, path):
        """A health check should not need a credential."""
        assert TestClient(server.create_app()).get(path).status_code == 200

    def test_authentication_can_be_disabled(self, tmp_path):
        """Explicitly opting out is allowed, for deployment behind a proxy."""
        srv = ModelServer(auth_token=False, models_dir=tmp_path)
        assert srv.auth_token is None
        assert TestClient(srv.create_app()).get("/models").status_code == 200


class TestModelLoadingIsBounded:
    """``POST /models`` unpickles the path it is given, so it must be bounded."""

    def test_absolute_path_outside_models_dir_is_refused(self, client, tmp_path):
        outside = tmp_path.parent / "outside.pkl"
        _create_model_file(tmp_path.parent, "outside.pkl")
        response = client.post(
            "/models", json={"model_id": "x", "path": str(outside)}
        )
        assert response.status_code == 403

    @pytest.mark.parametrize(
        "path", ["../escape.pkl", "../../etc/passwd", "a/../../../escape.pkl"]
    )
    def test_traversal_out_of_models_dir_is_refused(self, client, path):
        response = client.post("/models", json={"model_id": "x", "path": path})
        assert response.status_code == 403

    def test_relative_path_inside_models_dir_is_allowed(self, client, tmp_path):
        _create_model_file(tmp_path, "inside.pkl")
        response = client.post(
            "/models", json={"model_id": "ok", "path": "inside.pkl"}
        )
        assert response.status_code == 200

    def test_loading_is_disabled_without_a_models_dir(self, tmp_path):
        """With no bound configured the endpoint refuses rather than guessing one."""
        srv = ModelServer(auth_token=TEST_TOKEN)
        assert srv.models_dir is None
        c = TestClient(srv.create_app(), headers=AUTH)
        response = c.post(
            "/models",
            json={"model_id": "x", "path": str(_create_model_file(tmp_path))},
        )
        assert response.status_code == 403
        assert "models_dir" in response.json()["detail"]

    def test_in_process_loading_is_unrestricted(self, tmp_path):
        """models_dir bounds the HTTP endpoint, not the trusted Python caller."""
        srv = ModelServer()
        info = srv.load_model("local", _create_model_file(tmp_path))
        assert info["model_id"] == "local"


class TestCrossOriginPolicy:
    def test_no_cors_headers_by_default(self, server):
        """A wildcard here lets any site the operator visits drive the server."""
        response = TestClient(server.create_app()).get(
            "/health", headers={"Origin": "https://evil.example"}
        )
        assert "access-control-allow-origin" not in {
            k.lower() for k in response.headers
        }

    def test_explicit_origins_are_honoured(self, tmp_path):
        srv = ModelServer(auth_token=False, models_dir=tmp_path,
                          allow_origins=["https://trusted.example"])
        response = TestClient(srv.create_app()).get(
            "/health", headers={"Origin": "https://trusted.example"}
        )
        assert response.headers["access-control-allow-origin"] == "https://trusted.example"
