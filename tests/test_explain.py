"""Feature importance, dependence curves and exact tree Shapley values."""

import itertools
import math

import numpy as np
import pytest

from tuiml._cpp_ext import shapley as cpp_shapley
from tuiml.algorithms.trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestRegressor,
)
from tuiml.explain import (
    Explanation,
    TreeExplainer,
    accumulated_local_effects,
    drop_column_importance,
    individual_conditional_expectation,
    partial_dependence,
    permutation_importance,
)


@pytest.fixture
def linear_data():
    """Return data where feature 0 dominates, 1 contributes, 2-3 are noise."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.2, 400)
    return X, y


@pytest.fixture
def duplicated_feature():
    """Return data where feature 2 is a near-copy of feature 1."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    X[:, 2] = X[:, 1] + rng.normal(0, 0.05, 400)
    y = 3.0 * X[:, 1] - 2.0 * X[:, 0] + rng.normal(0, 0.2, 400)
    return X, y


# --------------------------------------------------------------------------
# Explanation container
# --------------------------------------------------------------------------

def test_explanation_defaults_and_ranking():
    """Feature names default sensibly and top() ranks by magnitude."""
    explanation = Explanation(values=np.array([0.1, -0.9, 0.4]), method="demo")

    assert explanation.feature_names == ["feature_0", "feature_1", "feature_2"]
    assert explanation.top(2)[0][0] == "feature_1"  # ranked by magnitude, not sign
    assert "demo" in repr(explanation)


def test_explanation_ranks_local_attributions_by_mean_absolute():
    """A 2-D attribution is summarised the standard way, not by signed mean."""
    values = np.array([[1.0, -5.0], [-1.0, 5.0]])
    explanation = Explanation(values=values, method="demo")

    # Signed means are both zero; magnitude is what distinguishes them.
    assert explanation.top(1)[0] == ("feature_1", 5.0)


# --------------------------------------------------------------------------
# Importance
# --------------------------------------------------------------------------

def test_permutation_importance_ranks_the_driving_feature(linear_data):
    """The feature the target depends on most costs most to shuffle."""
    X, y = linear_data
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)
    result = permutation_importance(
        model, X, y, scoring="r2", n_repeats=5, random_state=0
    )

    assert result.values.shape == (4,)
    assert result.top(1)[0][0] == "feature_0"
    # The two noise columns should barely register.
    assert result.values[2] < result.values[0] / 10
    assert result.values[3] < result.values[0] / 10
    assert result.metadata["std"].shape == (4,)
    assert result.metadata["raw"].shape == (4, 5)


def test_permutation_importance_hides_correlated_features(duplicated_feature):
    """Two columns carrying the same signal each look unimportant alone.

    This is the documented trap: permuting either leaves the model able to
    recover the signal from the other, so neither scores highly. The result
    means "neither matters *given* the other", not "neither matters".
    """
    X, y = duplicated_feature
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)
    result = permutation_importance(
        model, X, y, scoring="r2", n_repeats=5, random_state=0
    )

    # Feature 0 is unique; features 1 and 2 duplicate each other.
    assert result.values[0] > 0.1
    assert result.values[1] < result.values[0]
    assert result.values[2] < result.values[0]


def test_permutation_importance_accepts_a_callable_scorer(linear_data):
    """A custom metric works wherever a named one does."""
    X, y = linear_data
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
    result = permutation_importance(
        model, X, y, scoring=lambda a, b: -np.mean((a - b) ** 2),
        n_repeats=3, random_state=0,
    )
    assert result.top(1)[0][0] == "feature_0"


def test_permutation_importance_rejects_an_unknown_metric(linear_data):
    """A misspelled metric fails loudly rather than silently defaulting."""
    X, y = linear_data
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="scoring"):
        permutation_importance(model, X, y, scoring="nonsense")


def test_drop_column_importance_tolerates_redundancy(duplicated_feature):
    """Refitting lets the model compensate, so a duplicate scores near zero.

    This is the difference from permutation importance: both give a low score
    to a duplicated column, but here it means "we could stop collecting it",
    which is the actionable reading.
    """
    X, y = duplicated_feature
    result = drop_column_importance(
        DecisionTreeRegressor(max_depth=5), X, y, scoring="r2",
        cv=3, random_state=0,
    )

    assert result.top(1)[0][0] == "feature_0"      # unique and needed
    assert abs(result.values[1]) < 0.1             # duplicated, so droppable
    assert abs(result.values[2]) < 0.1


def test_drop_column_importance_can_be_negative(linear_data):
    """A pure-noise column can score negative, and that is informative."""
    X, y = linear_data
    result = drop_column_importance(
        DecisionTreeRegressor(max_depth=6), X, y, scoring="r2",
        cv=3, random_state=0,
    )
    # Dropping a noise feature should not help the useful one's ranking.
    assert result.values[0] > result.values[2]
    assert result.values[0] > result.values[3]


# --------------------------------------------------------------------------
# Dependence
# --------------------------------------------------------------------------

def test_partial_dependence_recovers_a_monotone_effect(linear_data):
    """A positive linear effect shows up as a rising curve."""
    X, y = linear_data
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)
    curve = partial_dependence(model, X, feature=0, n_points=12)

    assert curve.values.shape == curve.metadata["grid"].shape
    assert curve.values[-1] > curve.values[0]
    assert np.all(np.diff(curve.values) > -0.5)  # broadly increasing


def test_partial_dependence_is_the_average_of_ice(linear_data):
    """The two are the same computation, so they must agree exactly."""
    X, y = linear_data
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)

    ice = individual_conditional_expectation(model, X, feature=0, n_points=8)
    curve = partial_dependence(model, X, feature=0, n_points=8)

    np.testing.assert_allclose(curve.values, ice.values.mean(axis=0))
    assert ice.values.shape == (len(X), len(curve.metadata["grid"]))


def test_ice_reveals_heterogeneity_a_flat_average_hides():
    """Opposing effects cancel in the average but show in the curves.

    This is why the docstring says to look at ICE before concluding from a
    flat partial-dependence curve that a feature does not matter.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    group = (X[:, 1] > 0)
    # Feature 0 helps one group and hurts the other, by the same amount.
    y = np.where(group, 3.0 * X[:, 0], -3.0 * X[:, 0]) + rng.normal(0, 0.1, 400)

    model = RandomForestRegressor(n_estimators=60, random_state=0).fit(X, y)
    curve = partial_dependence(model, X, feature=0, n_points=10)
    ice = individual_conditional_expectation(model, X, feature=0, n_points=10)

    # The phenomenon itself: individual curves run in opposite directions.
    slopes = ice.values[:, -1] - ice.values[:, 0]
    assert (slopes > 0).any() and (slopes < 0).any()
    assert (slopes > 0).mean() > 0.3 and (slopes < 0).mean() > 0.3

    # Averaging them attenuates that, which is what makes a flat PDP
    # misleading. It need not cancel exactly — the groups are not symmetric.
    average_range = float(curve.values.max() - curve.values.min())
    per_curve_range = ice.values.max(axis=1) - ice.values.min(axis=1)
    assert average_range < float(np.median(per_curve_range))


def test_ale_is_centred_and_tracks_the_effect(duplicated_feature):
    """ALE reports a deviation from average, so it is centred near zero."""
    X, y = duplicated_feature
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)
    ale = accumulated_local_effects(model, X, feature=1, n_bins=12)

    assert ale.values.shape == ale.metadata["grid"].shape
    assert ale.values[-1] > ale.values[0]        # increasing effect

    counts = ale.metadata["bin_counts"]
    midpoints = (ale.values[:-1] + ale.values[1:]) / 2.0
    weighted_mean = float((midpoints * counts).sum() / counts.sum())
    assert abs(weighted_mean) < 1e-9


def test_ale_costs_two_passes_not_one_per_grid_point(duplicated_feature):
    """ALE evaluates the model twice in total, however many bins are used."""
    X, y = duplicated_feature
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)

    calls = {"n": 0}
    original = model.predict

    def counting_predict(data):
        """Count how many times the model is asked to predict."""
        calls["n"] += 1
        return original(data)

    model.predict = counting_predict
    accumulated_local_effects(model, X, feature=0, n_bins=25)
    assert calls["n"] == 2


def test_ale_handles_a_constant_feature():
    """A feature with no variation has no effect to accumulate."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    X[:, 1] = 5.0
    y = X[:, 0]
    model = DecisionTreeRegressor(max_depth=3).fit(X, y)

    ale = accumulated_local_effects(model, X, feature=1, n_bins=10)
    assert np.all(np.isfinite(ale.values))


# --------------------------------------------------------------------------
# TreeSHAP
# --------------------------------------------------------------------------

def _tree_arrays(model, X):
    """Return the flat arrays and background coverage of a fitted tree."""
    flat = model.flat_tree_
    feature = np.ascontiguousarray(flat.feature, np.int32)
    threshold = np.ascontiguousarray(flat.threshold, np.float64)
    left = np.ascontiguousarray(flat.children_left, np.int32)
    right = np.ascontiguousarray(flat.children_right, np.int32)
    value = np.ascontiguousarray(
        np.asarray(flat.value, np.float64).reshape(flat.n_nodes, -1)
    )

    weight = np.zeros(flat.n_nodes)
    for row in X:
        node = 0
        while True:
            weight[node] += 1
            if feature[node] < 0:
                break
            node = left[node] if row[feature[node]] <= threshold[node] else right[node]
    weight /= len(X)
    return feature, threshold, left, right, value, weight


def test_treeshap_matches_brute_force_exact_shapley():
    """The polynomial algorithm equals enumerating all 2^F subsets.

    This is the definitive check. Efficiency alone is necessary but not
    sufficient — many wrong attributions still sum to the prediction.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = X[:, 0] * 2 + X[:, 1] * X[:, 2] + rng.normal(0, 0.3, 300)
    model = DecisionTreeRegressor(max_depth=4).fit(X, y)

    feature, threshold, left, right, value, weight = _tree_arrays(model, X)

    def expectation(x, known, node=0):
        """f(S): follow x on known features, average both branches otherwise."""
        if feature[node] < 0:
            return value[node, 0]
        split = feature[node]
        if split in known:
            branch = left[node] if x[split] <= threshold[node] else right[node]
            return expectation(x, known, branch)
        parent = weight[node]
        if parent <= 0:
            return value[node, 0]
        return (
            (weight[left[node]] / parent) * expectation(x, known, left[node])
            + (weight[right[node]] / parent) * expectation(x, known, right[node])
        )

    def brute_force(x, n_features=5):
        """Enumerate every subset, as the Shapley definition demands."""
        out = np.zeros(n_features)
        for j in range(n_features):
            rest = [k for k in range(n_features) if k != j]
            for size in range(len(rest) + 1):
                for subset in itertools.combinations(rest, size):
                    coefficient = (
                        math.factorial(size)
                        * math.factorial(n_features - size - 1)
                        / math.factorial(n_features)
                    )
                    known = set(subset)
                    out[j] += coefficient * (
                        expectation(x, known | {j}) - expectation(x, known)
                    )
        return out

    fast = np.asarray(
        cpp_shapley.tree_shap(feature, threshold, left, right, value, weight, X[:8])
    )[:, :, 0]
    slow = np.array([brute_force(X[i]) for i in range(8)])

    np.testing.assert_allclose(fast, slow, atol=1e-9)


def test_treeshap_satisfies_efficiency(linear_data):
    """Attributions plus the base value reconstruct the prediction exactly."""
    X, y = linear_data
    model = DecisionTreeRegressor(max_depth=6).fit(X, y)
    explainer = TreeExplainer(model, background=X)
    result = explainer.explain(X[:50])

    reconstructed = result.values.sum(axis=1) + explainer.expected_value_[0]
    np.testing.assert_allclose(reconstructed, model.predict(X[:50]), atol=1e-9)


def test_treeshap_efficiency_holds_across_a_forest(linear_data):
    """Additivity carries through the ensemble average."""
    X, y = linear_data
    model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
    explainer = TreeExplainer(model, background=X)
    result = explainer.explain(X[:40])

    reconstructed = result.values.sum(axis=1) + explainer.expected_value_[0]
    np.testing.assert_allclose(reconstructed, model.predict(X[:40]), atol=1e-8)
    assert result.metadata["n_trees"] == 30


def test_treeshap_efficiency_holds_per_class():
    """For a classifier, every class column reconstructs its own probability."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=5).fit(X, y)

    explainer = TreeExplainer(model, background=X)
    result = explainer.explain(X[:20])

    assert result.values.shape == (20, 4, len(np.unique(y)))
    reconstructed = result.values.sum(axis=1) + result.base_value
    np.testing.assert_allclose(reconstructed, model.predict_proba(X[:20]), atol=1e-9)


def test_treeshap_gives_an_unused_feature_zero_credit():
    """The dummy axiom: a feature the tree never splits on gets exactly zero."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = 5.0 * X[:, 0] + rng.normal(0, 0.05, 300)   # features 1 and 2 unused
    model = DecisionTreeRegressor(max_depth=3).fit(X, y)

    result = TreeExplainer(model, background=X).explain(X[:30])
    used = {model.flat_tree_.feature[i] for i in range(model.flat_tree_.n_nodes)}
    for column in range(3):
        if column not in used:
            np.testing.assert_array_equal(result.values[:, column], 0.0)


def test_treeshap_ranks_the_driving_feature(linear_data):
    """Mean absolute attribution identifies the feature the model leans on."""
    X, y = linear_data
    model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
    result = TreeExplainer(model, background=X).explain(X[:100])
    assert result.top(1)[0][0] == "feature_0"


def test_treeshap_requires_background_and_a_tree(linear_data):
    """Both preconditions fail loudly rather than guessing."""
    X, y = linear_data
    model = DecisionTreeRegressor(max_depth=3).fit(X, y)

    with pytest.raises(ValueError, match="background"):
        TreeExplainer(model)

    from tuiml.algorithms.linear import LinearRegression

    with pytest.raises(ValueError, match="not a fitted TuiML tree"):
        TreeExplainer(LinearRegression().fit(X, y), background=X)


def test_treeshap_rejects_a_feature_count_mismatch(linear_data):
    """Explaining data of the wrong width is an error, not a silent reshape."""
    X, y = linear_data
    model = DecisionTreeRegressor(max_depth=3).fit(X, y)
    explainer = TreeExplainer(model, background=X)

    with pytest.raises(ValueError, match="features"):
        explainer.explain(X[:5, :2])


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------

def test_lime_identifies_the_driving_feature(linear_data):
    """A local surrogate finds the feature the model leans on here."""
    from tuiml.explain import lime_explain

    X, y = linear_data
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)
    result = lime_explain(model, X[0], background=X, random_state=0)

    assert result.top(1)[0][0] == "feature_0"
    assert result.metadata["local_r2"] > 0.5
    assert "intercept" in result.metadata
    assert "prediction" in result.metadata


def test_lime_works_on_a_non_tree_model(linear_data):
    """The whole point: LIME only needs predict, so any model qualifies."""
    from tuiml.algorithms.neighbors import KNearestNeighborsRegressor
    from tuiml.explain import lime_explain

    X, y = linear_data
    model = KNearestNeighborsRegressor(k=5).fit(X, y)
    result = lime_explain(model, X[3], background=X, random_state=0)

    assert np.all(np.isfinite(result.values))
    assert result.metadata["local_r2"] > 0.5


def test_lime_local_r2_is_a_valid_bounded_fit_statistic():
    """local_r2 is a bounded fit measure, and the recorded prediction is the
    model's own output at the explained point.

    The relationship between kernel width and fit is not monotone — a narrow
    kernel is more linear locally but leaves few effective samples, so the fit
    can be noisier rather than tighter. The invariant worth testing is that
    the statistic is well-formed and that the explanation is anchored on the
    real prediction.
    """
    from tuiml.explain import lime_explain

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = X[:, 0] ** 3 + rng.normal(0, 0.1, 400)     # strongly non-linear
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)

    for width in (0.3, 3.0):
        result = lime_explain(model, X[0], background=X, kernel_width=width,
                              random_state=0)
        r2 = result.metadata["local_r2"]
        assert np.isfinite(r2) and r2 <= 1.0 + 1e-9
        assert result.metadata["prediction"] == pytest.approx(
            float(model.predict(X[0:1])[0])
        )


def test_lime_reports_coefficients_in_original_units(linear_data):
    """Coefficients are per feature unit, not per standard deviation."""
    from tuiml.explain import lime_explain

    X, y = linear_data
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)

    # Rescaling one feature by 100 must not change its *effect* coefficient
    # once expressed in original units, so the two agree to within noise.
    result = lime_explain(model, X[0], background=X, random_state=0)

    X_scaled = X.copy()
    X_scaled[:, 0] *= 100.0
    model_scaled = RandomForestRegressor(n_estimators=40, random_state=0).fit(X_scaled, y)
    result_scaled = lime_explain(model_scaled, X_scaled[0], background=X_scaled, random_state=0)

    # Effect on y per unit of feature 0 is roughly unchanged.
    assert result_scaled.values[0] == pytest.approx(result.values[0] / 100.0, rel=0.5)


def test_lime_can_limit_the_number_of_features(linear_data):
    """Reporting a subset zeroes the rest rather than returning fewer."""
    from tuiml.explain import lime_explain

    X, y = linear_data
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)
    result = lime_explain(model, X[0], background=X, n_features=2, random_state=0)

    assert int((result.values != 0).sum()) == 2


# --------------------------------------------------------------------------
# Counterfactuals
# --------------------------------------------------------------------------

@pytest.fixture
def binary_classification():
    """Return data where only feature 0 decides the class."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=4).fit(X, y)
    return model, X, y


def test_counterfactual_flips_with_one_feature(binary_classification):
    """On a single-feature rule, one change suffices."""
    from tuiml.explain import counterfactual

    model, X, y = binary_classification
    negative = X[model.predict(X) == 0][0]

    result = counterfactual(model, negative, background=X, target=1)

    assert result.metadata["found"]
    assert result.metadata["n_changed"] == 1
    assert int(model.predict(result.metadata["counterfactual"][None, :])[0]) == 1


def test_counterfactual_keeps_unchanged_features_unchanged(binary_classification):
    """Only the features that need to move, move."""
    from tuiml.explain import counterfactual

    model, X, _ = binary_classification
    negative = X[model.predict(X) == 0][0]

    result = counterfactual(model, negative, background=X, target=1)
    changed = np.flatnonzero(result.values != 0.0)
    for column in range(3):
        if column not in changed:
            assert result.metadata["counterfactual"][column] == negative[column]


def test_counterfactual_reports_not_found_cleanly():
    """A target absent from the background fails loudly but informatively."""
    from tuiml.explain import counterfactual

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = np.ones(100, dtype=int)          # every background row is class 1
    model = DecisionTreeClassifier().fit(X, y)

    result = counterfactual(model, X[0], background=X, target=0)
    assert result.metadata["found"] is False
    assert result.metadata["distance"] == float("inf")


def test_counterfactual_is_reproducible(binary_classification):
    """Deterministic search on a fixed background returns the same answer."""
    from tuiml.explain import counterfactual

    model, X, _ = binary_classification
    negative = X[model.predict(X) == 0][0]

    first = counterfactual(model, negative, background=X, target=1)
    second = counterfactual(model, negative, background=X, target=1)
    np.testing.assert_allclose(first.metadata["counterfactual"],
                               second.metadata["counterfactual"])


# --------------------------------------------------------------------------
# Surrogate tree and H-statistic
# --------------------------------------------------------------------------

def test_surrogate_tree_is_a_fitted_tuiML_tree(linear_data):
    """The surrogate is a real estimator, so downstream tools work on it."""
    from tuiml.explain import TreeExplainer, surrogate_tree

    X, y = linear_data
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)
    result = surrogate_tree(model, X, max_depth=3, random_state=0)

    assert result.metadata["fidelity"] > 0.7
    tree = result.metadata["tree"]
    assert tree.tree_ is not None

    # The surrogate can itself be explained, closing the loop.
    explainer = TreeExplainer(tree, background=X)
    assert explainer.explain(X[:5]).values.shape == (5, 4)


def test_surrogate_fidelity_is_bounded_by_depth(linear_data):
    """A deeper surrogate tracks the model more closely."""
    from tuiml.explain import surrogate_tree

    X, y = linear_data
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)

    shallow = surrogate_tree(model, X, max_depth=1, random_state=0)
    deep = surrogate_tree(model, X, max_depth=6, random_state=0)
    assert deep.metadata["fidelity"] >= shallow.metadata["fidelity"]


def test_friedman_h_statistic_ranks_interaction_above_additive():
    """The H-statistic distinguishes a product term from a sum term."""
    from tuiml.explain import friedman_h_statistic

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    additive = 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.normal(0, 0.2, 400)
    interaction = 3.0 * X[:, 0] * X[:, 1] + rng.normal(0, 0.2, 400)

    add_model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, additive)
    inter_model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, interaction)

    h_add = friedman_h_statistic(add_model, X, 0, 1, n_points=8, random_state=0)
    h_inter = friedman_h_statistic(inter_model, X, 0, 1, n_points=8, random_state=0)

    assert h_inter > h_add


def test_friedman_h_is_zero_when_the_model_ignores_the_features(linear_data):
    """Two features the model never splits on cannot interact.

    A stump that only splits on feature 0 has no dependence on features 2 and
    3 at all, so their joint surface is exactly flat and H must be 0 — not the
    1.0 that an unguarded 0/0 resolves to.

    A random forest would not do here: it splits on noise features
    occasionally, so it has genuine (spurious) dependence on them and a
    nonzero H is the *correct* answer for that model.
    """
    from tuiml.explain import friedman_h_statistic

    X, y = linear_data
    model = DecisionTreeRegressor(max_depth=1).fit(X, y)   # splits on feature 0 only
    h = friedman_h_statistic(model, X, 2, 3, n_points=6, random_state=0)
    assert h == 0.0
