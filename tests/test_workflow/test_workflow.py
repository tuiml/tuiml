"""Tests for tuiml.workflow (Workflow and On)."""

import numpy as np
import pytest

from tuiml.workflow import Workflow, On


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _TrackingScaler:
    """Transformer that records how many rows each fit saw."""

    fit_sizes = []

    def __init__(self):
        pass

    def get_params(self, deep=True):
        return {}

    def fit(self, X, y=None):
        self.__class__.fit_sizes.append(len(X))
        self.offset_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return np.asarray(X) - self.offset_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class _TrackingModel:
    """Minimal classifier that records the inputs it receives."""

    fit_sizes = []
    _estimator_type = "classifier"

    def __init__(self):
        pass

    def get_params(self, deep=True):
        return {}

    def fit(self, X, y):
        self.__class__.fit_sizes.append(len(X))
        self.last_predict_input_ = None
        return self

    def predict(self, X):
        self.last_predict_input_ = np.asarray(X)
        return np.zeros(len(X), dtype=int)


class _NotATransformer:
    """Has fit/predict but no transform — only valid as a final step."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _make_classification_data(n_samples=100, n_features=4, seed=42):
    """Create a simple binary classification dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ===========================================================================
# Construction — the three step notations
# ===========================================================================
class TestWorkflowConstruction:
    def test_empty(self):
        wf = Workflow()
        assert len(wf) == 0
        assert wf.model is None

    def test_from_strings(self):
        wf = Workflow(["StandardScaler", "NaiveBayesClassifier"])
        assert list(wf.named_steps) == ["standardscaler", "naivebayesclassifier"]
        assert type(wf.model).__name__ == "NaiveBayesClassifier"

    def test_from_spec_dicts(self):
        wf = Workflow([
            {"name": "PCAExtractor", "params": {"n_components": 2}},
            {"name": "NaiveBayesClassifier"},
        ])
        assert wf["pcaextractor"].n_components == 2

    def test_from_instances(self):
        from tuiml.preprocessing import StandardScaler
        from tuiml.algorithms.bayesian import NaiveBayesClassifier

        wf = Workflow([StandardScaler(), NaiveBayesClassifier()])
        assert type(wf.model).__name__ == "NaiveBayesClassifier"

    def test_mixed_notations(self):
        from tuiml.preprocessing import StandardScaler

        wf = Workflow([
            "SimpleImputer",
            StandardScaler(),
            {"name": "NaiveBayesClassifier"},
        ])
        assert len(wf) == 3

    def test_explicit_step_names(self):
        wf = Workflow([("scale", "StandardScaler"), ("clf", "NaiveBayesClassifier")])
        assert list(wf.named_steps) == ["scale", "clf"]

    def test_duplicate_class_names_get_suffixes(self):
        wf = Workflow(["StandardScaler", "StandardScaler", "NaiveBayesClassifier"])
        assert list(wf.named_steps)[:2] == ["standardscaler", "standardscaler-2"]

    def test_unknown_component_name(self):
        with pytest.raises(ValueError, match="Unknown component 'NoSuchThing'"):
            Workflow(["NoSuchThing", "NaiveBayesClassifier"])

    def test_spec_dict_without_name(self):
        with pytest.raises(ValueError, match='needs a "name" key'):
            Workflow([{"params": {"k": 1}}, "NaiveBayesClassifier"])

    def test_spec_dict_with_loose_params(self):
        with pytest.raises(ValueError, match="Unexpected keys"):
            Workflow([{"name": "PCAExtractor", "n_components": 2}, "NaiveBayesClassifier"])

    def test_final_step_must_predict(self):
        with pytest.raises(TypeError, match="must be a model"):
            Workflow(["StandardScaler", "MinMaxScaler"])

    def test_middle_step_must_transform(self):
        with pytest.raises(TypeError, match="cannot transform"):
            Workflow([_NotATransformer(), "NaiveBayesClassifier"])

    def test_indexing(self):
        wf = Workflow(["StandardScaler", "NaiveBayesClassifier"])
        assert type(wf[0]).__name__ == "StandardScaler"
        assert type(wf[-1]).__name__ == "NaiveBayesClassifier"
        assert type(wf["standardscaler"]).__name__ == "StandardScaler"

    def test_transformers_property(self):
        wf = Workflow(["SimpleImputer", "StandardScaler", "NaiveBayesClassifier"])
        assert [name for name, _ in wf.transformers] == ["simpleimputer", "standardscaler"]


# ===========================================================================
# Parameters — nested step__param addressing
# ===========================================================================
class TestWorkflowParams:
    def test_get_params_deep(self):
        wf = Workflow(["PCAExtractor", "NaiveBayesClassifier"])
        params = wf.get_params()
        assert "steps" in params
        assert "pcaextractor__n_components" in params

    def test_get_params_shallow(self):
        wf = Workflow(["PCAExtractor", "NaiveBayesClassifier"])
        assert set(wf.get_params(deep=False)) == {"steps"}

    def test_set_params_nested(self):
        wf = Workflow(["PCAExtractor", "NaiveBayesClassifier"])
        wf.set_params(pcaextractor__n_components=3)
        assert wf["pcaextractor"].n_components == 3

    def test_set_params_unknown_step(self):
        wf = Workflow(["PCAExtractor", "NaiveBayesClassifier"])
        with pytest.raises(ValueError, match="Unknown step 'nope'"):
            wf.set_params(nope__k=1)

    def test_set_params_requires_step_prefix(self):
        wf = Workflow(["PCAExtractor", "NaiveBayesClassifier"])
        with pytest.raises(ValueError, match="not a Workflow parameter"):
            wf.set_params(n_components=3)

    def test_set_params_replaces_steps(self):
        wf = Workflow(["StandardScaler", "NaiveBayesClassifier"])
        wf.set_params(steps=["MinMaxScaler", "NaiveBayesClassifier"])
        assert list(wf.named_steps)[0] == "minmaxscaler"


# ===========================================================================
# fit() — sklearn semantics plus optional evaluation
# ===========================================================================
class TestWorkflowFit:
    def test_fit_arrays_returns_self(self):
        X, y = _make_classification_data()
        wf = Workflow(["StandardScaler", "NaiveBayesClassifier"])
        assert wf.fit(X, y) is wf

    def test_fit_without_evaluation_leaves_metrics_none(self):
        X, y = _make_classification_data()
        wf = Workflow(["NaiveBayesClassifier"]).fit(X, y)
        assert wf.metrics_ is None
        assert wf.cv_results_ is None

    def test_fit_holdout_populates_metrics(self):
        X, y = _make_classification_data()
        wf = Workflow(["NaiveBayesClassifier"]).fit(X, y, test_size=0.25, random_seed=1)
        assert "accuracy_score" in wf.metrics_
        assert wf.predictions_ is not None
        assert wf.metadata_["evaluation_method"] == "holdout"

    def test_fit_cv_populates_cv_metrics(self):
        X, y = _make_classification_data()
        wf = Workflow(["NaiveBayesClassifier"]).fit(X, y, cv=3, random_seed=1)
        assert any("mean" in k for k in wf.metrics_)
        assert any("std" in k for k in wf.metrics_)
        assert wf.cv_results_ is not None
        assert wf.metadata_["evaluation_method"] == "cross_validate"

    def test_fit_builtin_dataset_by_name(self):
        wf = Workflow(["StandardScaler", "NaiveBayesClassifier"]).fit("iris", cv=3)
        assert wf._is_fitted

    def test_fit_custom_metrics(self):
        X, y = _make_classification_data()
        wf = Workflow(["NaiveBayesClassifier"]).fit(
            X, y, test_size=0.25, metrics=["accuracy_score"], random_seed=1
        )
        assert set(wf.metrics_) == {"accuracy_score"}

    def test_fit_no_data(self):
        wf = Workflow(["NaiveBayesClassifier"])
        with pytest.raises(ValueError, match="No data provided"):
            wf.fit()

    def test_fit_no_steps(self):
        with pytest.raises(ValueError, match="has no steps"):
            Workflow().fit("iris")

    def test_fit_is_deterministic_with_seed(self):
        wf1 = Workflow(["RandomForestClassifier"]).fit("iris", cv=3, random_seed=7)
        wf2 = Workflow(["RandomForestClassifier"]).fit("iris", cv=3, random_seed=7)
        assert wf1.metrics_ == wf2.metrics_

    def test_fit_leaves_prototype_unfitted(self):
        """The instances passed in must never be mutated by fitting."""
        X, y = _make_classification_data()
        wf = Workflow(["StandardScaler", "NaiveBayesClassifier"])
        prototype = wf.model
        wf.fit(X, y)
        assert prototype is not wf.model_

    def test_fit_refits_on_all_data_after_cv(self):
        """CV scores each fold, then the delivered pipeline sees every row."""
        X, y = _make_classification_data(n_samples=12, n_features=4, seed=1)
        _TrackingScaler.fit_sizes = []
        _TrackingModel.fit_sizes = []

        Workflow([_TrackingScaler(), _TrackingModel()]).fit(X, y, cv=3, random_seed=0)

        assert sorted(_TrackingScaler.fit_sizes) == [8, 8, 8, 12]
        assert sorted(_TrackingModel.fit_sizes) == [8, 8, 8, 12]

    def test_features_subset(self):
        wf = Workflow(["NaiveBayesClassifier"]).fit(
            "iris", features=["sepallength", "sepalwidth"], test_size=0.25
        )
        assert wf.feature_names_in_ == ["sepallength", "sepalwidth"]

    def test_features_unknown_column(self):
        with pytest.raises(ValueError, match="features not found"):
            Workflow(["NaiveBayesClassifier"]).fit("iris", features=["nope"])


# ===========================================================================
# Inference — the pipeline travels with the model
# ===========================================================================
class TestWorkflowInference:
    def test_predict_applies_fitted_steps(self):
        X, y = _make_classification_data(n_samples=20, n_features=4, seed=2)
        _TrackingScaler.fit_sizes = []
        _TrackingModel.fit_sizes = []

        wf = Workflow([_TrackingScaler(), _TrackingModel()]).fit(X, y)
        X_new = np.array([[10.0, 20.0, 30.0, 40.0]])
        predictions = wf.predict(X_new)

        assert predictions.shape == (1,)
        np.testing.assert_allclose(
            wf.model_.last_predict_input_, X_new - np.mean(X, axis=0)
        )

    def test_predict_before_fit(self):
        wf = Workflow(["NaiveBayesClassifier"])
        with pytest.raises(RuntimeError, match="not fitted yet"):
            wf.predict(np.array([[1, 2]]))

    def test_score_and_evaluate(self):
        X, y = _make_classification_data()
        wf = Workflow(["StandardScaler", "NaiveBayesClassifier"]).fit(X, y)
        assert 0.0 <= wf.score(X, y) <= 1.0
        assert "accuracy_score" in wf.evaluate(X, y)

    def test_predict_proba(self):
        X, y = _make_classification_data()
        wf = Workflow(["NaiveBayesClassifier"]).fit(X, y)
        assert wf.predict_proba(X[:5]).shape[0] == 5

    def test_predict_proba_unsupported(self):
        X, y = _make_classification_data()
        wf = Workflow([_TrackingModel()]).fit(X, y)
        with pytest.raises(AttributeError, match="does not support predict_proba"):
            wf.predict_proba(X[:5])

    def test_save_and_load_round_trip(self, tmp_path):
        X, y = _make_classification_data()
        wf = Workflow(["StandardScaler", "NaiveBayesClassifier"]).fit(X, y)
        path = tmp_path / "pipeline.pkl"
        wf.save(str(path))

        loaded = Workflow.load(str(path))
        assert isinstance(loaded, Workflow)
        np.testing.assert_array_equal(loaded.predict(X), wf.predict(X))


# ===========================================================================
# On — column routing
# ===========================================================================
class TestOn:
    def _mixed(self):
        """Two numeric columns and one categorical, as an object array."""
        X = np.array(
            [[1.0, 10.0, "a"], [2.0, 20.0, "b"], [3.0, 30.0, "a"], [4.0, 40.0, "b"]],
            dtype=object,
        )
        y = np.array([0, 1, 0, 1])
        return X, y

    def test_selects_numeric_columns(self):
        X, y = self._mixed()
        step = On("number", "StandardScaler", remainder="drop")
        out = step.fit_transform(X, y)
        assert step.columns_ == [0, 1]
        assert out.shape == (4, 2)

    def test_selects_categorical_columns(self):
        X, y = self._mixed()
        step = On("category", "OrdinalEncoder", remainder="drop")
        out = step.fit_transform(X, y)
        assert step.columns_ == [2]
        assert out.shape == (4, 1)

    def test_passthrough_keeps_other_columns(self):
        X, y = self._mixed()
        step = On("category", "OrdinalEncoder")
        out = step.fit_transform(X, y)
        assert out.shape == (4, 3)  # 1 encoded + 2 passed through

    def test_select_by_index(self):
        X, y = self._mixed()
        step = On([0], "StandardScaler", remainder="drop")
        assert step.fit_transform(X, y).shape == (4, 1)

    def test_select_by_name(self):
        X, y = self._mixed()
        step = On(["b"], "StandardScaler", remainder="drop")
        step._bind_feature_names(["a", "b", "c"])
        step.fit_transform(X, y)
        assert step.columns_ == [1]

    def test_select_by_name_without_names_bound(self):
        X, y = self._mixed()
        step = On(["b"], "StandardScaler")
        with pytest.raises(ValueError, match="by name"):
            step.fit_transform(X, y)

    def test_select_by_missing_name(self):
        X, y = self._mixed()
        step = On(["zzz"], "StandardScaler")
        step._bind_feature_names(["a", "b", "c"])
        with pytest.raises(ValueError, match="not found"):
            step.fit_transform(X, y)

    def test_transform_before_fit(self):
        X, _ = self._mixed()
        with pytest.raises(RuntimeError, match="not fitted"):
            On("number", "StandardScaler").transform(X)

    def test_invalid_remainder(self):
        with pytest.raises(ValueError, match="passthrough"):
            On("number", "StandardScaler", remainder="keep")

    def test_in_a_pipeline(self):
        X, y = self._mixed()
        wf = Workflow([
            On("category", "OrdinalEncoder"),
            "StandardScaler",
            "NaiveBayesClassifier",
        ]).fit(X, y)
        assert wf.predict(X).shape == (4,)

    def test_get_params_round_trip(self):
        step = On("number", "StandardScaler", remainder="drop")
        assert step.get_params() == {
            "columns": "number",
            "transformer": "StandardScaler",
            "remainder": "drop",
        }


# ===========================================================================
# to_config — export back to a train() spec
# ===========================================================================
class TestWorkflowConfig:
    def test_to_config(self):
        config = Workflow([
            {"name": "PCAExtractor", "params": {"n_components": 3}},
            {"name": "RandomForestClassifier", "params": {"n_estimators": 20}},
        ]).to_config()

        assert config["model"] == {
            "name": "RandomForestClassifier",
            "params": {"n_estimators": 20},
        }
        assert config["pipeline"][0]["name"] == "PCAExtractor"
        assert config["pipeline"][0]["params"]["n_components"] == 3

    def test_to_config_omits_defaults(self):
        config = Workflow(["StandardScaler", "NaiveBayesClassifier"]).to_config()
        assert config["pipeline"] == [{"name": "StandardScaler"}]

    def test_to_config_no_pipeline_key_when_model_only(self):
        config = Workflow(["NaiveBayesClassifier"]).to_config()
        assert "pipeline" not in config

    def test_to_config_replays_through_train(self):
        import tuiml

        wf = Workflow(["StandardScaler", "NaiveBayesClassifier"])
        spec = wf.to_config()
        spec["data"] = {"source": "iris"}
        spec["evaluation"] = {"cv": 3}
        spec["random_seed"] = 42

        replayed = tuiml.train(spec)
        direct = wf.fit("iris", cv=3, random_seed=42)
        assert replayed.metrics_ == direct.metrics_

    def test_to_config_survives_fitting(self):
        """Fitted state must not leak into the exported spec."""
        wf = Workflow(["StandardScaler", "RandomForestClassifier"]).fit(
            "iris", random_seed=1
        )
        config = wf.to_config()
        assert config["model"]["name"] == "RandomForestClassifier"
        assert "estimators_" not in config["model"].get("params", {})


# ===========================================================================
# Display
# ===========================================================================
class TestWorkflowDisplay:
    def test_repr_lists_steps(self):
        text = repr(Workflow(["StandardScaler", "NaiveBayesClassifier"]))
        assert "Workflow([" in text
        assert "StandardScaler" in text
        assert "NaiveBayesClassifier" in text

    def test_repr_empty(self):
        assert repr(Workflow()) == "Workflow([])"

    def test_repr_html_is_self_contained(self):
        html = Workflow(["StandardScaler", "NaiveBayesClassifier"])._repr_html_()
        assert html.startswith("<style>")
        assert "tuiml-serial" in html
        # Collapsible boxes must work without JavaScript.
        assert 'type="checkbox"' in html
        assert "<script" not in html

    def test_repr_html_shows_column_routing_in_parallel(self):
        html = Workflow([On("number", "StandardScaler"), "NaiveBayesClassifier"])._repr_html_()
        assert "tuiml-parallel" in html
        assert "passthrough" in html

    def test_repr_html_scopes_css_per_diagram(self):
        first = Workflow(["NaiveBayesClassifier"])._repr_html_()
        second = Workflow(["NaiveBayesClassifier"])._repr_html_()
        assert first != second  # different container ids


# ===========================================================================
# Task detection — each model family takes a different path through fit()
# ===========================================================================
class TestWorkflowTaskTypes:
    def test_classifier_metrics(self):
        wf = Workflow(["StandardScaler", "NaiveBayesClassifier"]).fit(
            "iris", test_size=0.2, random_seed=1
        )
        assert set(wf.metrics_) == {"accuracy_score", "f1_score"}

    def test_regressor_metrics(self):
        wf = Workflow(["StandardScaler", "LinearRegression"]).fit(
            "iris", test_size=0.2, random_seed=1
        )
        assert set(wf.metrics_) == {
            "r2_score", "mean_squared_error", "mean_absolute_error",
        }

    def test_clusterer_scores_on_all_data(self):
        """Clusterers have no held-out notion of correctness."""
        wf = Workflow(["StandardScaler", "KMeansClusterer"]).fit("iris", random_seed=1)
        assert set(wf.metrics_) == {"silhouette_score", "calinski_harabasz_score"}
        assert wf.metadata_["evaluation_method"] == "clusterer"

    def test_anomaly_detector_reports_counts(self):
        wf = Workflow(["IsolationForestDetector"]).fit("iris", random_seed=1)
        assert {"n_anomalies", "n_normal", "anomaly_ratio"} <= set(wf.metrics_)
        assert wf.metadata_["evaluation_method"] == "anomaly"

    def test_timeseries_forecasts_the_tail(self):
        series = np.cumsum(np.random.RandomState(0).randn(60)) + 50
        wf = Workflow(["AR"]).fit(series.reshape(-1, 1), series, random_seed=1)
        assert "r2_score" in wf.metrics_
        assert wf.metadata_["evaluation_method"] == "timeseries"
        # The delivered model is refitted on the whole series.
        assert len(wf.model_.predict(3)) == 3


# ===========================================================================
# to_config / experiment — regressions found while porting the tutorials
# ===========================================================================
class TestConfigRoundTripRegressions:
    def test_to_config_is_json_writable(self):
        """The spec must survive a JSON round trip to be worth exporting."""
        import json

        wf = Workflow([
            {"name": "SelectKBestSelector", "params": {"k": 2}},
            "NaiveBayesClassifier",
        ])
        config = wf.to_config()
        assert json.loads(json.dumps(config)) == config

    def test_to_config_json_writable_after_fitting(self):
        wf = Workflow([
            {"name": "SelectKBestSelector", "params": {"k": 2}},
            "RandomForestClassifier",
        ]).fit("iris", random_seed=1)
        import json

        json.dumps(wf.to_config())  # must not raise

    def test_to_config_omits_params_the_constructor_rejects(self):
        """get_params() may report derived values __init__ would refuse."""
        import tuiml

        wf = Workflow([
            {"name": "SelectKBestSelector", "params": {"k": 2}},
            "NaiveBayesClassifier",
        ])
        spec = wf.to_config()
        spec["data"] = {"source": "iris"}
        tuiml.train(spec)  # replaying the spec must not raise

    def test_experiment_accepts_a_pipeline(self):
        import tuiml

        exp = tuiml.experiment(
            algorithms=["NaiveBayesClassifier"],
            datasets=["iris"],
            pipeline=[{"name": "StandardScaler"}],
            cv=3,
        )
        assert exp is not None

    def test_experiment_accepts_a_preset_with_a_resampler(self):
        """The "imbalanced" preset ends in a sampler, which reshapes X and y."""
        import tuiml

        exp = tuiml.experiment(
            algorithms=["NaiveBayesClassifier"],
            datasets=["iris"],
            pipeline="imbalanced",
            cv=3,
        )
        assert exp is not None

    def test_experiment_rejects_unknown_preset(self):
        import tuiml

        with pytest.raises(ValueError, match="Unknown pipeline preset"):
            tuiml.experiment(
                algorithms=["NaiveBayesClassifier"], datasets=["iris"],
                pipeline="nope", cv=2,
            )


# ===========================================================================
# Review fixes — fold leakage, task inference, metric averaging, exports
# ===========================================================================
class _CountingScaler:
    """Records the size of every dataset it is fitted on."""

    fit_sizes = []

    def __init__(self):
        pass

    def get_params(self, deep=True):
        return {}

    def fit_transform(self, X, y=None):
        self.__class__.fit_sizes.append(len(X))
        self.mean_ = np.mean(X, axis=0)
        return X - self.mean_

    def transform(self, X):
        return X - self.mean_


class _FitPlusTransformOnly:
    """A transformer with the plain fit/transform pair — no fit_transform."""

    def __init__(self):
        pass

    def get_params(self, deep=True):
        return {}

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return np.asarray(X) - self.mean_


class TestReviewFixes:
    def test_experiment_pipeline_fits_inside_each_fold(self):
        """The shared pipeline must never see a validation fold during fit."""
        import tuiml
        from tuiml import registry

        _CountingScaler.fit_sizes = []
        tuiml.experiment(
            algorithms={"NB": registry.create("NaiveBayesClassifier")},
            datasets=["iris"],
            pipeline=[_CountingScaler()],
            cv=3,
        )
        # One fit per fold, each on that fold's training portion only —
        # not a single fit on the full 150 rows.
        assert len(_CountingScaler.fit_sizes) == 3
        assert all(size < 150 for size in _CountingScaler.fit_sizes)

    def test_experiment_infers_regression(self):
        import tuiml

        exp = tuiml.experiment(
            algorithms=["LinearRegression", "DecisionTreeRegressor"],
            datasets=["iris"],
            cv=3,
        )
        assert exp.experiment_type.value == "regression"
        assert "r2_score" in exp.metrics

    def test_experiment_accepts_component_spec_dicts(self):
        import tuiml

        exp = tuiml.experiment(
            algorithms=[
                {"name": "RandomForestClassifier", "params": {"n_estimators": 5}},
                "NaiveBayesClassifier",
            ],
            datasets=["iris"],
            cv=2,
        )
        assert exp.experiment_type.value == "classification"

    def test_experiment_rejects_loose_spec_keys(self):
        import tuiml

        with pytest.raises(ValueError, match="Unexpected keys"):
            tuiml.experiment(
                algorithms=[{"name": "RandomForestClassifier", "n_estimators": 5}],
                datasets=["iris"],
                cv=2,
            )

    def test_multiclass_auto_f1_uses_macro_averaging(self):
        """f1_score's binary default silently scores only class 1 on
        multiclass labels; auto metrics must use macro averaging instead."""
        import tuiml
        from tuiml.datasets import load_dataset
        from tuiml.evaluation.metrics import f1_score

        ds = load_dataset("iris")  # three classes
        model = tuiml.train("NaiveBayesClassifier", {"source": "iris"}, random_seed=42)
        predictions = model.predict(ds.X)

        auto = model.evaluate(ds.X, ds.y)["f1_score"]
        assert auto == pytest.approx(f1_score(ds.y, predictions, average="macro"))
        # And it genuinely differs from the binary default on this data.
        assert auto != pytest.approx(f1_score(ds.y, predictions))

    def test_fit_plus_transform_transformer_is_supported(self):
        """_validate accepts fit+transform, so fitting must support it too."""
        X, y = _make_classification_data()

        wf = Workflow([_FitPlusTransformOnly(), "NaiveBayesClassifier"]).fit(
            X, y, test_size=0.25, random_seed=1
        )
        assert "accuracy_score" in wf.metrics_

    def test_fit_plus_transform_transformer_in_on(self):
        X, y = _make_classification_data()
        step = On([0, 1], _FitPlusTransformOnly(), remainder="drop")
        assert step.fit_transform(X, y).shape == (len(X), 2)

    def test_train_rejects_removed_options(self):
        import tuiml

        for kwarg in ("return_model", "return_predictions",
                      "return_probabilities", "verbose"):
            with pytest.raises(TypeError):
                tuiml.train("NaiveBayesClassifier", {"source": "iris"}, **{kwarg: True})

    def test_star_import_exports_are_all_defined(self):
        import tuiml

        namespace = {}
        exec("from tuiml import *", namespace)
        missing = [name for name in tuiml.__all__ if name not in namespace]
        assert missing == []
        assert "registry" in namespace

    def test_workflow_reports_final_models_estimator_type(self):
        assert Workflow(["NaiveBayesClassifier"])._estimator_type == "classifier"
        assert Workflow(["LinearRegression"])._estimator_type == "regressor"
        assert Workflow(["KMeansClusterer"])._estimator_type == "clusterer"
        assert Workflow()._estimator_type is None


# ===========================================================================
# Second review round — override forwarding, tuple labels, class union
# ===========================================================================
class TestSecondReviewFixes:
    def test_experiment_type_override_reaches_the_run(self):
        """Mixed model collections need the explicit override to resolve."""
        import tuiml

        exp = tuiml.experiment(
            algorithms=["NaiveBayesClassifier"],
            datasets=["iris"],
            cv=2,
            experiment_type="regression",
        )
        assert exp.experiment_type.value == "regression"

    def test_experiment_tuple_gives_a_display_label(self):
        """('RF', component) names the entry; the component says what to build."""
        import tuiml

        exp = tuiml.experiment(
            algorithms=[
                ("RF", {"name": "RandomForestClassifier",
                        "params": {"n_estimators": 5}}),
                ("NB", "NaiveBayesClassifier"),
            ],
            datasets=["iris"],
            cv=2,
        )
        assert exp is not None

    def test_experiment_tuple_with_bare_params_dict_is_rejected(self):
        """The old (label, params) form looked the LABEL up in the registry."""
        import tuiml

        with pytest.raises(ValueError, match="which algorithm to build"):
            tuiml.experiment(
                algorithms=[("RF", {"n_estimators": 5})],
                datasets=["iris"],
                cv=2,
            )

    def test_call_metric_counts_predicted_classes_too(self):
        """Two true classes + a third predicted class is still multiclass."""
        from tuiml.base.algorithms import call_metric
        from tuiml.evaluation.metrics import f1_score

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 2, 1, 2])   # class 2 appears only in predictions

        result = call_metric(f1_score, y_true, y_pred)
        assert result == pytest.approx(f1_score(y_true, y_pred, average="macro"))


# ===========================================================================
# String columns must survive the whole path: loader -> On -> encoder
# ===========================================================================
class TestCategoricalCsvPath:
    @pytest.fixture
    def mixed_csv(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({
            "age": [25, 32, None, 41, 29, 55, 38, 47],
            "city": ["ny", "sf", "ny", "la", "sf", "la", "ny", "sf"],
            "label": [0, 1, 0, 1, 1, 1, 0, 1],
        })
        path = tmp_path / "mixed.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_loader_preserves_string_columns(self, mixed_csv):
        """String cells must load as strings, not silently become NaN."""
        from tuiml.datasets import load

        ds = load(mixed_csv, target_column="label")
        city = ds.X[:, ds.feature_names.index("city")]
        assert set(city) == {"ny", "sf", "la"}

    def test_loader_keeps_float_dtype_for_numeric_files(self, tmp_path):
        import pandas as pd
        from tuiml.datasets import load

        path = tmp_path / "numeric.csv"
        pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [0, 1]}).to_csv(
            path, index=False
        )
        assert load(str(path), target_column="y").X.dtype == np.float64

    def test_onehot_encodes_string_categories(self):
        from tuiml.preprocessing import OneHotEncoder

        X = np.array([["ny"], ["sf"], ["ny"], ["la"]], dtype=object)
        out = OneHotEncoder().fit_transform(X)
        assert out.shape == (4, 3)
        assert out.dtype == np.float64
        assert (out.sum(axis=1) == 1).all()   # exactly one hot per row

    def test_full_mixed_type_pipeline_from_csv(self, mixed_csv):
        wf = Workflow([
            On("number", "SimpleImputer"),
            On("category", "OneHotEncoder"),
            "StandardScaler",
            "NaiveBayesClassifier",
        ]).fit(mixed_csv, target="label", test_size=0.25, random_seed=7)

        assert "accuracy_score" in wf.metrics_
        # Raw, unencoded rows must predict through the fitted pipeline.
        row = np.array([[30, "ny"]], dtype=object)
        assert wf.predict(row).shape == (1,)
