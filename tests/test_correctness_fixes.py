"""Defects where the wrong behaviour was silent.

Each of these failed by producing a plausible result rather than an error: an
unstratified fold that quietly omits a class, a registry that reports empty, a
mistyped component that registers into a bucket nothing lists. Tests exist here
because none of them announce themselves, so a regression would look normal.
"""

import warnings

import numpy as np
import pytest

import tuiml
from tuiml.algorithms.bayesian import NaiveBayesClassifier
from tuiml.base.algorithms import Clusterer
from tuiml.evaluation.splitting import KFold, StratifiedKFold
from tuiml.registry import ComponentType, registry
from tuiml.workflow import Workflow


# --------------------------------------------------------------------------
# Cross-validation folds are stratified for classification
# --------------------------------------------------------------------------

def _workflow():
    """Return a Workflow with a cheap classifier, for splitter selection."""
    return Workflow([NaiveBayesClassifier()])


def test_plain_kfold_really_does_drop_a_rare_class():
    """The premise: unstratified folds can omit a class entirely.

    Without this, the stratification tests below assert a difference that may
    not exist for the data they use.
    """
    y = np.zeros(100, dtype=int)
    y[:5] = 1
    X = np.zeros((100, 2))

    classes_per_fold = [
        len(np.unique(y[val]))
        for _, val in KFold(n_splits=5, shuffle=True, random_state=0).split(X, y)
    ]
    assert 1 in classes_per_fold, "expected plain k-fold to omit the rare class"

    stratified = [
        len(np.unique(y[val]))
        for _, val in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y)
    ]
    assert all(n == 2 for n in stratified)


def test_classification_folds_are_stratified():
    y = np.zeros(200, dtype=int)
    y[:40] = 1
    assert isinstance(
        _workflow()._make_folds(y, 5, 0, True, "classifier"), StratifiedKFold
    )


def test_regression_folds_are_not_stratified():
    """There are no classes to balance."""
    y = np.linspace(0, 1, 200)
    assert isinstance(_workflow()._make_folds(y, 5, 0, True, "regressor"), KFold)


def test_stratify_false_is_honoured():
    y = np.zeros(200, dtype=int)
    y[:40] = 1
    assert isinstance(_workflow()._make_folds(y, 5, 0, False, "classifier"), KFold)


def test_falls_back_and_warns_when_a_class_is_too_rare():
    """Stratification needs cv members per class; say so rather than raising."""
    y = np.zeros(200, dtype=int)
    y[:3] = 1
    with pytest.warns(UserWarning, match="Cannot stratify"):
        splitter = _workflow()._make_folds(y, 5, 0, True, "classifier")
    assert isinstance(splitter, KFold)


def test_cross_validation_runs_on_imbalanced_data():
    """End to end, through fit() rather than the helper."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = np.zeros(200, dtype=int)
    y[:20] = 1

    wf = _workflow()
    wf.fit(X, y, cv=5, metrics=["accuracy_score"], random_seed=0)
    assert "cv_accuracy_score_mean" in wf.metrics_


# --------------------------------------------------------------------------
# registry.clear() is reversible
# --------------------------------------------------------------------------

def test_clear_then_read_repopulates():
    """clear() is documented as being for testing, where a one-way trip is
    exactly wrong: it left every later read in the process seeing nothing."""
    before = len(registry.list_names())
    assert before > 50

    registry.clear()
    assert len(registry.list_names()) == before


def test_clear_is_idempotent():
    before = len(registry.list_names())
    registry.clear()
    registry.clear()
    assert len(registry.list_names()) == before


# --------------------------------------------------------------------------
# A mistyped component type is refused
# --------------------------------------------------------------------------

def test_unknown_component_type_raises():
    """It used to fall back to ALGORITHM, filing the component where nothing
    would list it while reporting success."""
    with pytest.raises(ValueError, match="Unknown component type"):
        @registry.register("classifer")          # deliberate typo
        class Typo:
            pass


def test_known_component_type_still_registers():
    @registry.register("classifier")
    class WellTyped:
        pass

    assert "WellTyped" in registry
    registry.unregister("WellTyped")


@pytest.mark.parametrize("kind", [t.value for t in ComponentType])
def test_every_declared_type_is_accepted(kind):
    """The vocabulary the error message advertises must actually work."""
    cls = type(f"Probe_{kind}", (), {})
    registry.register(kind)(cls)
    assert cls.__name__ in registry
    registry.unregister(cls.__name__)


# --------------------------------------------------------------------------
# Clusterer.fit is abstract
# --------------------------------------------------------------------------

def test_clusterer_without_fit_cannot_be_instantiated():
    """The override existed only to document that y is ignored, but its plain
    body cancelled Algorithm.fit's @abstractmethod."""
    class Forgetful(Clusterer):
        def predict(self, X):
            return None

    with pytest.raises(TypeError, match="abstract"):
        Forgetful()


def test_a_real_clusterer_still_fits():
    from tuiml.algorithms.clustering import KMeansClusterer

    rng = np.random.default_rng(0)
    model = KMeansClusterer(n_clusters=2).fit(rng.normal(size=(30, 2)))
    assert model.labels_ is not None


# --------------------------------------------------------------------------
# A foreign estimator instance is rejected with the useful message
# --------------------------------------------------------------------------

def test_foreign_estimator_names_the_wrapper_convention():
    """The check ran after the one that only accepts dicts, so it could never
    fire; the caller got a generic shape complaint instead."""
    class ForeignSVC:
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return X

    with pytest.raises(TypeError, match="not a TuiML algorithm"):
        tuiml.train({
            "model": ForeignSVC(),
            "data": {"source": "iris", "target": "class"},
        })


def test_a_valid_spec_is_unaffected():
    model = tuiml.train({
        "model": {"name": "NaiveBayesClassifier"},
        "data": {"source": "iris", "target": "class"},
        "evaluation": {"cv": 3, "metrics": ["accuracy_score"]},
    })
    assert model.metrics_["cv_accuracy_score_mean"] > 0.5


# --------------------------------------------------------------------------
# The agent skill file is found where it actually lives
# --------------------------------------------------------------------------

def test_skill_file_installs(tmp_path):
    """It was read from tuiml.agent rather than tuiml.agent.prompts, so it was
    never installed and the CLI reported that as a non-change."""
    from tuiml.cli.setup import install_skill_file

    installed, reason = install_skill_file(tmp_path)
    assert installed, reason

    written = tmp_path / "tuiml" / "SKILL.md"
    assert written.is_file()
    assert written.stat().st_size > 1000

    assert install_skill_file(tmp_path) == (False, "already up to date")
