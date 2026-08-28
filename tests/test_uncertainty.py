"""Conformal prediction and probability calibration."""

import numpy as np
import pytest

from tuiml.algorithms.linear import LinearRegression
from tuiml.algorithms.trees import DecisionTreeClassifier, DecisionTreeRegressor
from tuiml.uncertainty import (
    APSConformalClassifier,
    ConformalizedQuantileRegressor,
    CVPlusRegressor,
    IsotonicCalibrator,
    JackknifePlusRegressor,
    MondrianConformalClassifier,
    RAPSConformalClassifier,
    VennAbersCalibrator,
    PlattCalibrator,
    SplitConformalClassifier,
    SplitConformalRegressor,
    TemperatureScaler,
    VectorScaler,
    average_set_size,
    brier_score,
    coverage_score,
    expected_calibration_error,
    interval_width,
    maximum_calibration_error,
    reliability_curve,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Return a separable-ish binary classification set."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 4))
    y = (X[:, 0] + X[:, 1] + rng.normal(0, 0.5, 800) > 0).astype(int)
    return X, y


@pytest.fixture
def regression_data():
    """Return a linear regression set with Gaussian noise."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 3))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.5, 800)
    return X, y


# --------------------------------------------------------------------------
# Conformal prediction
# --------------------------------------------------------------------------

def test_split_conformal_classifier_covers(binary_data):
    """Classification sets reach the requested coverage on held-out data."""
    X, y = binary_data
    cp = SplitConformalClassifier(
        DecisionTreeClassifier(max_depth=4), alpha=0.1, random_state=0
    )
    cp.fit(X[:600], y[:600])
    sets = cp.predict_set(X[600:])

    assert sets.shape == (200, 2)
    assert coverage_score(y[600:], sets, classes=cp.classes_) >= 0.85
    assert 1.0 <= average_set_size(sets) <= 2.0


def test_split_conformal_classifier_margin_score(binary_data):
    """The margin score yields valid sets just as the LAC score does."""
    X, y = binary_data
    cp = SplitConformalClassifier(
        DecisionTreeClassifier(max_depth=4), alpha=0.2, score="margin", random_state=0
    )
    cp.fit(X[:600], y[:600])
    sets = cp.predict_set(X[600:])
    assert coverage_score(y[600:], sets, classes=cp.classes_) >= 0.7


def test_smaller_alpha_gives_larger_sets(binary_data):
    """Tightening alpha can only grow the prediction sets."""
    X, y = binary_data
    sizes = []
    for alpha in (0.3, 0.05):
        cp = SplitConformalClassifier(
            DecisionTreeClassifier(max_depth=4), alpha=alpha, random_state=0
        )
        cp.fit(X[:600], y[:600])
        sizes.append(average_set_size(cp.predict_set(X[600:])))
    assert sizes[1] >= sizes[0]


def test_split_conformal_regressor_covers_on_average(regression_data):
    """Marginal coverage holds when averaged over calibration splits."""
    X, y = regression_data
    coverages = []
    for seed in range(10):
        cp = SplitConformalRegressor(
            DecisionTreeRegressor(max_depth=5), alpha=0.1, random_state=seed
        )
        cp.fit(X[:600], y[:600])
        coverages.append(coverage_score(y[600:], cp.predict_interval(X[600:])))
    assert np.mean(coverages) >= 0.88


def test_normalized_intervals_vary_in_width(regression_data):
    """Difficulty normalisation produces sample-dependent interval widths."""
    X, y = regression_data
    cp = SplitConformalRegressor(
        DecisionTreeRegressor(max_depth=5), alpha=0.1, normalize=True, random_state=0
    )
    cp.fit(X[:600], y[:600])
    intervals = cp.predict_interval(X[600:])
    widths = intervals[:, 1] - intervals[:, 0]
    assert widths.std() > 0.0
    assert np.all(widths > 0.0)


def test_fit_calibrated_accepts_explicit_split(binary_data):
    """An explicit calibration set produces a usable predictor."""
    X, y = binary_data
    cp = SplitConformalClassifier(DecisionTreeClassifier(max_depth=4), alpha=0.1)
    cp.fit_calibrated(X[:400], y[:400], X[400:600], y[400:600])
    assert cp.predict_set(X[600:]).shape == (200, 2)


def test_conformal_quantile_finite_sample_correction():
    """The corrected quantile is infinite when calibration data is too small."""
    scores = np.arange(5, dtype=float)
    # ceil((5+1) * 0.99) / 5 > 1, so no finite threshold certifies alpha=0.01.
    assert np.isinf(SplitConformalRegressor.conformal_quantile(scores, 0.01))
    # ceil((5+1) * 0.5) / 5 = 0.6, which rounds up to the 4th order statistic
    # rather than the plain median the uncorrected quantile would give.
    assert SplitConformalRegressor.conformal_quantile(scores, 0.5) == 3.0
    assert np.quantile(scores, 0.5) == 2.0


def test_predict_before_fit_raises(binary_data):
    """Predicting before fitting is an explicit error."""
    X, _ = binary_data
    cp = SplitConformalClassifier(DecisionTreeClassifier())
    with pytest.raises(RuntimeError, match="not fitted"):
        cp.predict_set(X)


def test_invalid_alpha_rejected():
    """alpha outside (0, 1) is rejected at construction time."""
    with pytest.raises(ValueError, match="alpha"):
        SplitConformalClassifier(DecisionTreeClassifier(), alpha=1.5)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_platt_is_monotone_in_the_score():
    """Platt scaling preserves the ordering of the raw scores."""
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(-1, 1, 300), rng.normal(1, 1, 300)])
    y = np.concatenate([np.zeros(300), np.ones(300)])
    cal = PlattCalibrator().fit(scores, y)
    grid = np.linspace(-3, 3, 50)
    assert np.all(np.diff(cal.transform(grid)) >= -1e-12)


def test_platt_improves_calibration():
    """Calibrating an over-confident score reduces the calibration error."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    # Scores are pushed towards the extremes, so the raw model over-states
    # its confidence relative to its accuracy.
    raw = np.clip(0.5 + (y - 0.5) * rng.uniform(0.4, 1.4, 600), 0.01, 0.99)
    before = expected_calibration_error(y, raw)
    after = expected_calibration_error(y, PlattCalibrator().fit_transform(raw, y))
    assert after <= before + 1e-9


def test_isotonic_is_monotone_and_bounded():
    """Isotonic output is non-decreasing and stays inside [0, 1]."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < scores).astype(int)
    cal = IsotonicCalibrator().fit(scores, y)
    grid = np.linspace(0, 1, 100)
    proba = cal.transform(grid)
    assert np.all(np.diff(proba) >= -1e-12)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_isotonic_matches_pava_reference():
    """The C++ PAVA kernel reproduces the textbook isotonic fit."""
    from tuiml._cpp_ext import stats

    y = np.array([1.0, 3.0, 2.0, 4.0])
    w = np.ones(4)
    fitted = stats.pool_adjacent_violators(y, w, True)
    np.testing.assert_allclose(fitted, [1.0, 2.5, 2.5, 4.0])


def test_isotonic_rejects_multiclass():
    """Isotonic calibration is binary-only and says so."""
    scores = np.linspace(0, 1, 30)
    y = np.repeat([0, 1, 2], 10)
    with pytest.raises(ValueError, match="binary"):
        IsotonicCalibrator().fit(scores, y)


def test_temperature_preserves_accuracy():
    """Temperature scaling never changes the arg-max prediction."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    noisy = np.where(rng.random(400) < 0.3, (y + 1) % 3, y)
    logits = np.eye(3)[noisy] * 6.0 + rng.normal(0, 1.0, (400, 3))

    scaler = TemperatureScaler().fit(logits, y)
    assert np.array_equal(logits.argmax(1), scaler.transform(logits).argmax(1))
    assert scaler.temperature_ > 1.0


def test_temperature_reduces_calibration_error():
    """Scaling an over-confident model lowers the expected calibration error.

    Only ECE is asserted. MCE is a worst-bin statistic, and softening the
    probabilities spreads samples into sparsely populated bins whose noisy
    accuracy estimate can widen the worst gap even as the average shrinks.
    """
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    noisy = np.where(rng.random(400) < 0.3, (y + 1) % 3, y)
    logits = np.eye(3)[noisy] * 6.0 + rng.normal(0, 1.0, (400, 3))
    raw = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    calibrated = TemperatureScaler().fit(logits, y).transform(logits)
    assert expected_calibration_error(y, calibrated) < expected_calibration_error(y, raw)
    assert np.isfinite(maximum_calibration_error(y, calibrated))


def test_vector_scaler_returns_distribution():
    """Vector scaling returns rows that sum to one."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    logits = np.eye(3)[y] * 4.0 + rng.normal(0, 1.5, (300, 3))
    proba = VectorScaler(max_iter=200).fit(logits, y).transform(logits)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0.0)


def test_calibrators_accept_probability_input():
    """Two-column probabilities are accepted wherever scores are."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    positive = np.clip(rng.uniform(0, 1, 200), 0.01, 0.99)
    proba = np.column_stack([1 - positive, positive])
    assert PlattCalibrator().fit_transform(proba, y).shape == (200,)
    assert IsotonicCalibrator().fit_transform(proba, y).shape == (200,)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_coverage_score_handles_non_index_labels():
    """String labels are resolved against the classifier's class order."""
    y = np.array(["a", "b", "a"])
    sets = np.array([[True, False], [False, True], [False, True]])
    assert coverage_score(y, sets, classes=np.array(["a", "b"])) == pytest.approx(2 / 3)


def test_coverage_score_rejects_bad_shape():
    """A malformed prediction-set argument raises rather than guessing."""
    with pytest.raises(ValueError, match="prediction_sets"):
        coverage_score(np.array([1.0]), np.zeros((1, 3)))


def test_brier_score_perfect_and_worst():
    """The Brier score spans 0 for perfect and 1 for maximally wrong."""
    y = np.array([0, 1])
    assert brier_score(y, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert brier_score(y, np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_ece_zero_when_perfectly_calibrated():
    """A perfectly confident and perfectly correct model has zero ECE."""
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, proba) == pytest.approx(0.0)


def test_reliability_curve_partitions_all_samples():
    """Every sample lands in exactly one reliability bin."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    proba = rng.uniform(0, 1, 200)
    _, _, counts = reliability_curve(y, proba, n_bins=10)
    assert counts.sum() == 200


def test_quantile_binning_is_supported():
    """Quantile bins are accepted and produce a finite ECE."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    proba = rng.uniform(0.9, 1.0, 200)
    ece = expected_calibration_error(y, proba, n_bins=5, strategy="quantile")
    assert np.isfinite(ece)


def test_interval_width_and_average_set_size():
    """The sharpness metrics report the arithmetic mean."""
    assert interval_width(np.array([[0.0, 2.0], [1.0, 2.0]])) == pytest.approx(1.5)
    assert average_set_size(np.array([[True, False], [True, True]])) == pytest.approx(1.5)


# --------------------------------------------------------------------------
# CV+ and jackknife+
# --------------------------------------------------------------------------

def test_cv_plus_covers_on_average(regression_data):
    """CV+ reaches nominal coverage without a held-out calibration split."""
    X, y = regression_data
    coverages = []
    for seed in range(5):
        cp = CVPlusRegressor(
            DecisionTreeRegressor(max_depth=4), cv=5, alpha=0.1, random_state=seed
        )
        cp.fit(X[:600], y[:600])
        coverages.append(coverage_score(y[600:], cp.predict_interval(X[600:])))
    assert np.mean(coverages) >= 0.88


def test_cv_plus_fits_one_model_per_fold(regression_data):
    """Every fold contributes exactly one model and one residual per sample."""
    X, y = regression_data
    cp = CVPlusRegressor(DecisionTreeRegressor(max_depth=3), cv=4, random_state=0)
    cp.fit(X[:200], y[:200])
    assert len(cp.estimators_) == 4
    assert cp.scores_.shape == (200,)
    assert set(np.unique(cp.fold_index_)) == {0, 1, 2, 3}


def test_cv_plus_does_not_mutate_the_passed_estimator(regression_data):
    """The estimator handed in is deep-copied, never fitted in place."""
    X, y = regression_data
    estimator = DecisionTreeRegressor(max_depth=3)
    CVPlusRegressor(estimator, cv=3, random_state=0).fit(X[:120], y[:120])
    with pytest.raises(Exception):
        estimator.predict(X[:5])


def test_jackknife_plus_fits_one_model_per_sample():
    """Jackknife+ is the leave-one-out limit: n models for n samples."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.3, 40)
    cp = JackknifePlusRegressor(LinearRegression(), alpha=0.1).fit(X, y)
    assert len(cp.estimators_) == 40
    assert cp.predict_interval(X[:5]).shape == (5, 2)


def test_cv_requires_at_least_two_folds():
    """A single fold cannot cross-fit and is rejected."""
    with pytest.raises(ValueError, match="cv"):
        CVPlusRegressor(DecisionTreeRegressor(), cv=1)


# --------------------------------------------------------------------------
# Adaptive prediction sets
# --------------------------------------------------------------------------

@pytest.fixture
def noisy_multiclass():
    """Return a 4-class problem with irreducible label noise."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1200, 6))
    logits = X[:, :4] * 1.2
    y = np.array(
        [rng.choice(4, p=np.exp(row) / np.exp(row).sum()) for row in logits]
    )
    return X, y


def test_aps_covers_and_never_returns_an_empty_set(noisy_multiclass):
    """APS reaches nominal coverage and always keeps the top-ranked class."""
    X, y = noisy_multiclass
    cp = APSConformalClassifier(
        DecisionTreeClassifier(max_depth=5), alpha=0.1, random_state=0
    )
    cp.fit(X[:900], y[:900])
    sets = cp.predict_set(X[900:])
    assert coverage_score(y[900:], sets, classes=cp.classes_) >= 0.85
    assert np.all(sets.sum(axis=1) >= 1)


def test_aps_improves_worst_class_coverage_over_lac(noisy_multiclass):
    """APS trades average set size for better conditional coverage."""
    X, y = noisy_multiclass
    worst, sizes = {}, {}
    for name, cls in (("lac", SplitConformalClassifier), ("aps", APSConformalClassifier)):
        cp = cls(DecisionTreeClassifier(max_depth=5), alpha=0.1, random_state=0)
        cp.fit(X[:900], y[:900])
        sets = cp.predict_set(X[900:])
        y_test = y[900:]
        worst[name] = min(
            coverage_score(y_test[y_test == c], sets[y_test == c], classes=cp.classes_)
            for c in cp.classes_
        )
        sizes[name] = average_set_size(sets)

    assert worst["aps"] >= worst["lac"]
    assert sizes["aps"] >= sizes["lac"]


def test_raps_penalty_is_applied_to_both_sides(noisy_multiclass):
    """RAPS stays valid once the rank penalty is added."""
    X, y = noisy_multiclass
    cp = RAPSConformalClassifier(
        DecisionTreeClassifier(max_depth=5),
        alpha=0.1,
        lambda_penalty=0.1,
        k_reg=2,
        random_state=0,
    )
    cp.fit(X[:900], y[:900])
    sets = cp.predict_set(X[900:])
    assert coverage_score(y[900:], sets, classes=cp.classes_) >= 0.85


def test_raps_rejects_negative_penalty():
    """A negative penalty is meaningless and rejected."""
    with pytest.raises(ValueError, match="lambda_penalty"):
        RAPSConformalClassifier(DecisionTreeClassifier(), lambda_penalty=-1.0)


def test_aps_deterministic_without_randomization(noisy_multiclass):
    """Disabling randomisation makes the sets reproducible."""
    X, y = noisy_multiclass
    kwargs = dict(alpha=0.1, randomized=False, random_state=0)
    first = APSConformalClassifier(DecisionTreeClassifier(max_depth=4), **kwargs)
    second = APSConformalClassifier(DecisionTreeClassifier(max_depth=4), **kwargs)
    first.fit(X[:900], y[:900])
    second.fit(X[:900], y[:900])
    assert np.array_equal(first.predict_set(X[900:]), second.predict_set(X[900:]))


# --------------------------------------------------------------------------
# Mondrian
# --------------------------------------------------------------------------

def test_mondrian_fixes_rare_class_coverage():
    """Class-conditional calibration rescues a class the marginal one abandons."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 4))
    p = 1.0 / (1.0 + np.exp(-X[:, 0] * 2.0))
    y = (rng.uniform(size=2000) < p * 0.15).astype(int)
    train, test = slice(0, 1400), slice(1400, None)

    def worst_class_coverage(model):
        """Return the lowest per-class coverage achieved by a fitted model."""
        sets = model.predict_set(X[test])
        y_test = y[test]
        return min(
            coverage_score(y_test[y_test == c], sets[y_test == c], classes=model.classes_)
            for c in model.classes_
        )

    marginal = SplitConformalClassifier(
        DecisionTreeClassifier(max_depth=5), alpha=0.1, random_state=0
    ).fit(X[train], y[train])
    conditional = MondrianConformalClassifier(
        DecisionTreeClassifier(max_depth=5), alpha=0.1, random_state=0
    ).fit(X[train], y[train])

    assert worst_class_coverage(marginal) < 0.5
    assert worst_class_coverage(conditional) >= 0.8


def test_mondrian_records_one_threshold_per_group(binary_data):
    """Every calibration group gets its own threshold and size."""
    X, y = binary_data
    cp = MondrianConformalClassifier(
        DecisionTreeClassifier(max_depth=4), alpha=0.1, random_state=0
    ).fit(X, y)
    assert set(cp.group_quantiles_) == set(np.unique(y))
    assert sum(cp.group_sizes_.values()) == cp.scores_.size


def test_mondrian_accepts_an_external_taxonomy(binary_data):
    """Groups need not be the class label."""
    X, y = binary_data
    groups = (X[:, 2] > 0).astype(int)
    cp = MondrianConformalClassifier(
        DecisionTreeClassifier(max_depth=4), alpha=0.1, random_state=0
    ).fit(X[:600], y[:600], groups=groups[:600])

    sets = cp.predict_set_for_groups(X[600:], groups[600:])
    assert sets.shape == (200, 2)
    assert set(cp.group_quantiles_) == {0, 1}


# --------------------------------------------------------------------------
# Conformalized quantile regression
# --------------------------------------------------------------------------

def test_cqr_produces_input_dependent_widths():
    """CQR intervals widen where the noise widens."""
    # Guard on scikit-learn itself: tuiml.sklearn.ensemble imports fine without
    # it -- the wrapper only raises at construction -- so naming the wrapper
    # module here never skips.
    pytest.importorskip("sklearn")
    sklearn_ensemble = pytest.importorskip("tuiml.sklearn.ensemble")
    gbr = sklearn_ensemble.GradientBoostingRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 4, size=(1200, 1))
    y = X[:, 0] + rng.normal(0, 0.2 + 0.5 * X[:, 0], 1200)
    train, test = slice(0, 900), slice(900, None)

    cqr = ConformalizedQuantileRegressor(
        gbr(loss="quantile", alpha=0.05, n_estimators=60),
        gbr(loss="quantile", alpha=0.95, n_estimators=60),
        alpha=0.1,
        random_state=0,
    ).fit(X[train], y[train])

    intervals = cqr.predict_interval(X[test])
    widths = intervals[:, 1] - intervals[:, 0]
    assert coverage_score(y[test], intervals) >= 0.85
    # Width tracks the input, which is the whole point of CQR.
    assert np.corrcoef(X[test][:, 0], widths)[0, 1] > 0.8


def test_cqr_intervals_are_well_formed():
    """Lower never exceeds upper, even when the correction is negative."""
    # Guard on scikit-learn itself: tuiml.sklearn.ensemble imports fine without
    # it -- the wrapper only raises at construction -- so naming the wrapper
    # module here never skips.
    pytest.importorskip("sklearn")
    sklearn_ensemble = pytest.importorskip("tuiml.sklearn.ensemble")
    gbr = sklearn_ensemble.GradientBoostingRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = X[:, 0] + rng.normal(0, 0.3, 300)
    cqr = ConformalizedQuantileRegressor(
        gbr(loss="quantile", alpha=0.05, n_estimators=40),
        gbr(loss="quantile", alpha=0.95, n_estimators=40),
        alpha=0.1,
        random_state=0,
    ).fit(X, y)
    intervals = cqr.predict_interval(X)
    assert np.all(intervals[:, 0] <= intervals[:, 1])


# --------------------------------------------------------------------------
# Venn-Abers
# --------------------------------------------------------------------------

def test_venn_abers_brackets_and_narrows_with_more_data():
    """The probability interval is ordered and shrinks as calibration grows."""
    rng = np.random.default_rng(0)
    widths = []
    for n in (200, 2000):
        scores = rng.uniform(0, 1, n + 200)
        y = (rng.uniform(0, 1, n + 200) < scores).astype(int)
        va = VennAbersCalibrator().fit(scores[:n], y[:n])
        p0, p1 = va.predict_proba_interval(scores[n:])
        assert np.all(p0 <= p1)
        widths.append(float((p1 - p0).mean()))

    # O(1/n) shrinkage: ten times the calibration data must narrow the interval.
    assert widths[1] < widths[0] / 2.0


def test_venn_abers_summary_is_a_probability():
    """The p1 / (1 - p0 + p1) summary stays inside [0, 1] and is monotone."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 600)
    y = (rng.uniform(0, 1, 600) < scores).astype(int)
    va = VennAbersCalibrator().fit(scores, y)
    grid = np.linspace(0.05, 0.95, 40)
    proba = va.transform(grid)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert np.all(np.diff(proba) >= -1e-9)


def test_venn_abers_is_not_inflated_by_batching():
    """Predicting one point at a time matches predicting the whole batch."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 400)
    y = (rng.uniform(0, 1, 400) < scores).astype(int)
    va = VennAbersCalibrator().fit(scores, y)

    query = np.array([0.15, 0.5, 0.85])
    batch_p0, batch_p1 = va.predict_proba_interval(query)
    for i, point in enumerate(query):
        single_p0, single_p1 = va.predict_proba_interval(np.array([point]))
        assert single_p0[0] == pytest.approx(batch_p0[i])
        assert single_p1[0] == pytest.approx(batch_p1[i])
