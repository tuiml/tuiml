"""Successive halving and Hyperband hyperparameter search."""

import numpy as np
import pytest

from tuiml.algorithms.trees import RandomForestClassifier
from tuiml.base.tuning import ParameterDistribution
from tuiml.datasets import load_iris
from tuiml.evaluation.tuning import (
    HyperbandSearchCV,
    RandomSearchCV,
    SuccessiveHalvingSearchCV,
)

SPACE = {"max_depth": (2, 12, "int"), "n_estimators": (5, 30, "int")}


@pytest.fixture
def iris():
    """Return the iris features and labels."""
    data = load_iris()
    return data.X, data.y


# --------------------------------------------------------------------------
# The halving schedule
# --------------------------------------------------------------------------

def test_halving_schedule_shrinks_pool_and_grows_resource(iris):
    """Candidates fall by ``factor`` per round while the resource rises by it."""
    X, y = iris
    search = SuccessiveHalvingSearchCV(
        RandomForestClassifier(), SPACE, n_candidates=9, factor=3,
        cv=3, random_seed=0,
    ).fit(X, y)

    assert search.n_candidates_per_round_ == [9, 3, 1]
    assert search.resources_per_round_ == sorted(search.resources_per_round_)
    assert len(search.resources_per_round_) == len(search.n_candidates_per_round_)


def test_final_round_reaches_the_full_resource(iris):
    """The last survivor must be scored on all the data.

    An earlier version stopped as soon as one candidate remained, so
    ``best_score_`` came from a partial-resource round and did not mean what
    the attribute claims.
    """
    X, y = iris
    search = SuccessiveHalvingSearchCV(
        RandomForestClassifier(), SPACE, n_candidates=9, factor=3,
        cv=3, random_seed=0,
    ).fit(X, y)

    assert search.resources_per_round_[-1] == len(X)
    final_round = search.n_rounds_ - 1
    assert final_round in search.cv_results_["round"]

    # best_score_ is one of the final-round scores, not an earlier cheap one.
    final_scores = [
        score
        for score, round_index in zip(
            search.cv_results_["mean_test_score"], search.cv_results_["round"]
        )
        if round_index == final_round
    ]
    assert search.best_score_ == max(final_scores)


def test_halving_evaluates_far_fewer_full_budget_fits(iris):
    """Most candidates are eliminated before ever seeing the full data."""
    X, y = iris
    search = SuccessiveHalvingSearchCV(
        RandomForestClassifier(), SPACE, n_candidates=27, factor=3,
        cv=3, random_seed=0,
    ).fit(X, y)

    resources = np.asarray(search.cv_results_["resource"])
    at_full = int((resources == len(X)).sum())
    # 27 candidates, but only the last survivors reach full resource.
    assert at_full < 5
    assert len(search.cv_results_["params"]) > at_full


def test_larger_factor_eliminates_faster(iris):
    """A bigger factor means fewer, steeper rounds."""
    X, y = iris
    gentle = SuccessiveHalvingSearchCV(
        RandomForestClassifier(), SPACE, n_candidates=16, factor=2,
        cv=3, random_seed=0,
    ).fit(X, y)
    steep = SuccessiveHalvingSearchCV(
        RandomForestClassifier(), SPACE, n_candidates=16, factor=4,
        cv=3, random_seed=0,
    ).fit(X, y)
    assert steep.n_rounds_ < gentle.n_rounds_


def test_parameter_resource_grows_an_estimator_knob(iris):
    """A named integer parameter can be the budget instead of sample count."""
    X, y = iris
    search = SuccessiveHalvingSearchCV(
        RandomForestClassifier(n_estimators=27),
        {"max_depth": (2, 12, "int")},
        n_candidates=9, factor=3, resource="n_estimators",
        cv=3, random_seed=0,
    ).fit(X, y)

    assert search.resources_per_round_ == [1, 3, 9, 27][-search.n_rounds_:]
    assert search.resources_per_round_[-1] == 27


def test_unknown_parameter_resource_is_rejected(iris):
    """Naming a parameter the estimator lacks fails loudly."""
    X, y = iris
    with pytest.raises(ValueError, match="no parameter"):
        SuccessiveHalvingSearchCV(
            RandomForestClassifier(), SPACE, n_candidates=4,
            resource="nonexistent_knob", cv=3, random_seed=0,
        ).fit(X, y)


def test_subsampling_keeps_every_class(iris):
    """A small early-round slice must not lose a class entirely.

    Unstratified subsampling of an imbalanced problem can drop a class, which
    would score candidates on a different task than the one being tuned for.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = np.zeros(300, dtype=int)
    y[:12] = 1  # 4% minority

    search = SuccessiveHalvingSearchCV(
        RandomForestClassifier(), {"max_depth": (2, 8, "int")},
        n_candidates=4, cv=3, random_seed=0,
    )
    subsample_X, subsample_y = search._subsample(X, y, 30, np.random.RandomState(0))

    assert set(np.unique(subsample_y)) == {0, 1}
    # Every class keeps at least cv rows, so no fold can come up empty.
    assert (subsample_y == 1).sum() >= search.cv


# --------------------------------------------------------------------------
# Search behaviour
# --------------------------------------------------------------------------

def test_halving_finds_a_reasonable_configuration(iris):
    """The search returns usable parameters and a refitted estimator."""
    X, y = iris
    search = SuccessiveHalvingSearchCV(
        RandomForestClassifier(), SPACE, n_candidates=9, cv=3, random_seed=0
    ).fit(X, y)

    assert search.best_score_ > 0.8
    assert set(search.best_params_) == {"max_depth", "n_estimators"}
    assert search.best_estimator_ is not None
    assert (search.best_estimator_.predict(X) == y).mean() > 0.9


def test_halving_spends_less_full_budget_than_random_search(iris):
    """Fewer candidates reach full resource than random search evaluates.

    Asserted in fits rather than seconds: on a dataset small enough to test
    quickly the bookkeeping can outweigh the saving, which is exactly the case
    the docstring warns halving is pointless for. The invariant that always
    holds is the number of full-budget evaluations.
    """
    X, y = iris
    random_search = RandomSearchCV(
        RandomForestClassifier(), SPACE, n_iter=27, cv=3, random_seed=0
    ).fit(X, y)
    halving = SuccessiveHalvingSearchCV(
        RandomForestClassifier(), SPACE, n_candidates=27, factor=3,
        cv=3, random_seed=0,
    ).fit(X, y)

    resources = np.asarray(halving.cv_results_["resource"])
    at_full_budget = int((resources == len(X)).sum())
    assert at_full_budget < len(random_search.cv_results_["params"]) / 4


def test_accepts_a_prepared_parameter_distribution(iris):
    """A ParameterDistribution works as well as a plain dict."""
    X, y = iris
    search = SuccessiveHalvingSearchCV(
        RandomForestClassifier(), ParameterDistribution(SPACE),
        n_candidates=4, cv=3, random_seed=0,
    ).fit(X, y)
    assert search.best_params_ is not None


def test_halving_rejects_a_degenerate_factor():
    """A factor below 2 eliminates nobody."""
    with pytest.raises(ValueError, match="factor"):
        SuccessiveHalvingSearchCV(RandomForestClassifier(), SPACE, factor=1)
    with pytest.raises(ValueError, match="n_candidates"):
        SuccessiveHalvingSearchCV(RandomForestClassifier(), SPACE, n_candidates=0)


# --------------------------------------------------------------------------
# Hyperband
# --------------------------------------------------------------------------

def test_hyperband_runs_several_brackets(iris):
    """Brackets trade pool size against starting resource."""
    X, y = iris
    search = HyperbandSearchCV(
        RandomForestClassifier(), SPACE, factor=3, cv=3, random_seed=0
    ).fit(X, y)

    assert len(search.brackets_) >= 2
    pools = [b["n_candidates"] for b in search.brackets_]
    resources = [b["min_resource"] for b in search.brackets_]
    # Most aggressive first: the largest pool starts at the smallest resource.
    assert pools == sorted(pools, reverse=True)
    assert resources == sorted(resources)


def test_hyperband_min_resource_does_not_collapse(iris):
    """The bracket range must not degenerate to a single full-resource run.

    Hyperband inherits a min-resource rule keyed on ``n_candidates``, which it
    has no single value for. Left inherited, the range collapsed and only five
    configurations were ever tried.
    """
    X, y = iris
    search = HyperbandSearchCV(
        RandomForestClassifier(), SPACE, factor=3, cv=3, random_seed=0
    )
    minimum, maximum = search._resource_bounds(len(X))

    assert minimum < maximum
    assert maximum // minimum >= search.factor

    search.fit(X, y)
    # The collapse showed up as a single bracket; several is the fix.
    assert len(search.brackets_) >= 2
    assert search.brackets_[0]["n_candidates"] > search.brackets_[-1]["n_candidates"]


def test_hyperband_records_every_bracket(iris):
    """cv_results_ carries the bracket each evaluation belongs to."""
    X, y = iris
    search = HyperbandSearchCV(
        RandomForestClassifier(), SPACE, factor=3, n_brackets=2,
        cv=3, random_seed=0,
    ).fit(X, y)

    brackets = set(search.cv_results_["bracket"])
    assert brackets == {0, 1}
    assert len(search.cv_results_["params"]) == len(search.cv_results_["bracket"])
    assert search.best_score_ == pytest.approx(
        max(b["best_score"] for b in search.brackets_)
    )


def test_hyperband_brackets_sample_different_candidates(iris):
    """A shared seed would make the brackets near-duplicates of each other."""
    X, y = iris
    search = HyperbandSearchCV(
        RandomForestClassifier(), SPACE, factor=3, n_brackets=2,
        cv=3, random_seed=0,
    ).fit(X, y)

    by_bracket = {}
    for params, bracket in zip(
        search.cv_results_["params"], search.cv_results_["bracket"]
    ):
        by_bracket.setdefault(bracket, []).append(tuple(sorted(params.items())))

    assert set(by_bracket[0]) != set(by_bracket[1])


def test_hyperband_refits_the_overall_best(iris):
    """The refitted estimator uses the winner across all brackets."""
    X, y = iris
    search = HyperbandSearchCV(
        RandomForestClassifier(), SPACE, factor=3, cv=3, random_seed=0
    ).fit(X, y)

    assert search.best_estimator_ is not None
    for key, value in search.best_params_.items():
        assert getattr(search.best_estimator_, key) == value
    assert (search.best_estimator_.predict(X) == y).mean() > 0.9


def test_hyperband_respects_an_explicit_min_resource(iris):
    """An explicit min_resource overrides the automatic rule."""
    X, y = iris
    search = HyperbandSearchCV(
        RandomForestClassifier(), SPACE, factor=3, min_resource=50,
        cv=3, random_seed=0,
    )
    minimum, _ = search._resource_bounds(len(X))
    assert minimum == 50


def test_all_searchers_are_exported_from_evaluation():
    """The three search strategies plus both multi-fidelity ones are public.

    BayesianSearchCV existed in the tuning subpackage but was never surfaced
    on tuiml.evaluation alongside the others.
    """
    import tuiml.evaluation as evaluation

    for name in (
        "GridSearchCV",
        "RandomSearchCV",
        "BayesianSearchCV",
        "SuccessiveHalvingSearchCV",
        "HyperbandSearchCV",
    ):
        assert name in evaluation.__all__
        assert hasattr(evaluation, name)
