"""Tests for tuiml.serving.model_manager.ModelManager."""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from tuiml.serving.model_manager import ModelManager


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


# ---------------------------------------------------------------------------
# Helper: load a model into the manager using mocked serialization
# ---------------------------------------------------------------------------
def _load_mock(manager, model_id, tmp_path, model=None, metadata=None):
    """Create a dummy file and load a mock model into the manager."""
    model_file = tmp_path / f"{model_id}.pkl"
    model_file.touch()

    mock_model = model or MockModel()

    with patch("tuiml.serving.model_manager.load_model", return_value=mock_model):
        with patch("tuiml.serving.model_manager.load_model_info", return_value=MOCK_INFO.copy()):
            return manager.load(model_id, model_file, metadata=metadata)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------
class TestModelManagerInit:
    def test_init_default(self):
        manager = ModelManager()
        assert manager.max_models == 10
        assert manager.list_models() == []

    def test_init_custom(self):
        manager = ModelManager(max_models=5)
        assert manager.max_models == 5


# ---------------------------------------------------------------------------
# Loading / unloading
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------
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
