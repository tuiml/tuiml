import numpy as np

from tuiml.algorithms.ensemble import EasyEnsembleClassifier


class TestEasyEnsembleClassifier:
    """Tests for EasyEnsembleClassifier."""

    @staticmethod
    def _make_imbalanced_dataset():
        """Create a small imbalanced binary classification dataset."""
        rng = np.random.RandomState(42)

        X_majority = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(30, 4),
        )
        X_minority = rng.normal(
            loc=2.0,
            scale=1.0,
            size=(10, 4),
        )

        X = np.vstack([X_majority, X_minority])
        y = np.array([0] * 30 + [1] * 10)

        return X, y

    def test_fit_and_predict(self):
        """EasyEnsemble should fit and produce predictions."""
        X, y = self._make_imbalanced_dataset()

        model = EasyEnsembleClassifier(
            n_estimators=3,
            adaboost_estimators=5,
            random_state=42,
        )

        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self):
        """EasyEnsemble should return valid probabilities."""
        X, y = self._make_imbalanced_dataset()

        model = EasyEnsembleClassifier(
            n_estimators=3,
            adaboost_estimators=5,
            random_state=42,
        )

        model.fit(X, y)

        probabilities = model.predict_proba(X)

        assert probabilities.shape == (len(X), 2)
        assert np.allclose(
            probabilities.sum(axis=1),
            1.0,
        )
        assert np.all(probabilities >= 0.0)
        assert np.all(probabilities <= 1.0)

    def test_balanced_subsets(self):
        """Each EasyEnsemble member should use a balanced subset."""
        X, y = self._make_imbalanced_dataset()

        model = EasyEnsembleClassifier(
            n_estimators=3,
            adaboost_estimators=5,
            random_state=42,
        )

        model.fit(X, y)

        for indices in model.bootstrap_indices_:
            sampled_y = y[indices]

            _, counts = np.unique(
                sampled_y,
                return_counts=True,
            )

            assert len(counts) == 2
            assert counts[0] == counts[1]

    def test_reproducibility(self):
        """Same random state should produce the same results."""
        X, y = self._make_imbalanced_dataset()

        model_1 = EasyEnsembleClassifier(
            n_estimators=3,
            adaboost_estimators=5,
            random_state=42,
        )

        model_2 = EasyEnsembleClassifier(
            n_estimators=3,
            adaboost_estimators=5,
            random_state=42,
        )

        model_1.fit(X, y)
        model_2.fit(X, y)

        predictions_1 = model_1.predict(X)
        predictions_2 = model_2.predict(X)

        assert np.array_equal(
            predictions_1,
            predictions_2,
        )

        for indices_1, indices_2 in zip(
            model_1.bootstrap_indices_,
            model_2.bootstrap_indices_,
        ):
            assert np.array_equal(
                indices_1,
                indices_2,
            )

    def test_different_random_states_create_different_subsets(self):
        """Different random states should create different subsets."""
        X, y = self._make_imbalanced_dataset()

        model_1 = EasyEnsembleClassifier(
            n_estimators=3,
            adaboost_estimators=5,
            random_state=42,
        )

        model_2 = EasyEnsembleClassifier(
            n_estimators=3,
            adaboost_estimators=5,
            random_state=123,
        )

        model_1.fit(X, y)
        model_2.fit(X, y)

        assert any(
            not np.array_equal(indices_1, indices_2)
            for indices_1, indices_2 in zip(
                model_1.bootstrap_indices_,
                model_2.bootstrap_indices_,
            )
        )

    def test_multiclass_balancing(self):
        """Each member should balance every class in a multiclass dataset."""
        rng = np.random.RandomState(42)

        X = rng.normal(size=(45, 4))
        y = np.array(
            [0] * 20
            + [1] * 15
            + [2] * 10
        )

        model = EasyEnsembleClassifier(
            n_estimators=3,
            adaboost_estimators=5,
            random_state=42,
        )

        model.fit(X, y)

        for indices in model.bootstrap_indices_:
            sampled_y = y[indices]

            _, counts = np.unique(
                sampled_y,
                return_counts=True,
            )

            assert len(counts) == 3
            assert np.all(counts == 10)
            
    def test_repr(self):
        """EasyEnsemble should have a useful representation."""
        model = EasyEnsembleClassifier(
            n_estimators=10,
            adaboost_estimators=20,
            random_state=42,
        )

        representation = repr(model)

        assert "EasyEnsembleClassifier" in representation
        assert "n_estimators=10" in representation
        assert "adaboost_estimators=20" in representation

    def __repr__(self) -> str:
        return (
            f"EasyEnsembleClassifier("
            f"n_estimators={self.n_estimators}, "
            f"adaboost_estimators={self.adaboost_estimators}"
            f")"
        )