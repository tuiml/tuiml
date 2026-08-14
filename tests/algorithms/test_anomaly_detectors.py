"""ECOD, COPOD, HBOS, kNN and ABOD outlier detectors."""

import numpy as np
import pytest

from tuiml._cpp_ext import stats as cpp_stats
from tuiml.algorithms.anomaly import (
    ABODDetector,
    COPODDetector,
    ECODDetector,
    HBOSDetector,
    KNNDetector,
    LSCPDetector,
)
from tuiml.evaluation.metrics import roc_auc_score

MARGINAL_DETECTORS = [ECODDetector, COPODDetector, HBOSDetector]
ALL_DETECTORS = MARGINAL_DETECTORS + [KNNDetector, ABODDetector]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def marginal_outliers():
    """Return data whose outliers are extreme in every individual feature."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (400, 6)), rng.normal(7, 1, (20, 6))])
    y = np.zeros(420, dtype=int)
    y[-20:] = 1
    return X, y


@pytest.fixture
def joint_outliers():
    """Return data whose outliers are ordinary per feature but off-manifold."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=(600, 1))
    X = np.hstack([z + rng.normal(0, 0.1, (600, 1)), z + rng.normal(0, 0.1, (600, 1))])
    y = np.zeros(600, dtype=int)
    # Flip the correlation: each coordinate stays in range, the pair does not.
    X[-30:, 1] = -X[-30:, 0]
    y[-30:] = 1
    return X, y


# --------------------------------------------------------------------------
# Shared detector contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("detector_cls", ALL_DETECTORS)
def test_detector_follows_the_anomaly_contract(detector_cls, marginal_outliers):
    """Every detector honours the -1/1 prediction and score conventions."""
    X, _ = marginal_outliers
    detector = detector_cls(contamination=0.05).fit(X)

    predictions = detector.predict(X)
    assert set(np.unique(predictions)).issubset({-1, 1})

    scores = detector.decision_function(X)
    assert scores.shape == (len(X),)
    assert np.all(np.isfinite(scores))
    # score_samples is documented as an alias.
    np.testing.assert_allclose(scores, detector.score_samples(X))
    # Lower scores are more anomalous, so flagged points sit below the rest.
    assert scores[predictions == -1].max() <= scores[predictions == 1].min()


# ABOD is excluded here on purpose: the fixture's anomalies form a tight
# group, which ABOD masks by design. That behaviour is pinned down separately
# in test_abod_masks_clustered_anomalies.
@pytest.mark.parametrize("detector_cls", MARGINAL_DETECTORS + [KNNDetector])
def test_detector_finds_marginal_outliers(detector_cls, marginal_outliers):
    """The four non-angle detectors separate per-feature extreme outliers."""
    X, y = marginal_outliers
    detector = detector_cls(contamination=0.05).fit(X)
    assert roc_auc_score(y, -detector.decision_function(X)) > 0.95


@pytest.mark.parametrize("detector_cls", ALL_DETECTORS)
def test_predict_before_fit_raises(detector_cls, marginal_outliers):
    """Scoring before fitting is an explicit error."""
    X, _ = marginal_outliers
    with pytest.raises(Exception):
        detector_cls().decision_function(X)


@pytest.mark.parametrize("detector_cls", ALL_DETECTORS)
def test_contamination_sets_the_flag_rate(detector_cls, marginal_outliers):
    """The proportion flagged tracks the contamination parameter."""
    X, _ = marginal_outliers
    for contamination in (0.05, 0.2):
        detector = detector_cls(contamination=contamination).fit(X)
        flagged = (detector.predict(X) == -1).mean()
        assert abs(flagged - contamination) < 0.05


# --------------------------------------------------------------------------
# The marginal / joint split, which decides which detector to reach for
# --------------------------------------------------------------------------

def test_knn_beats_marginal_detectors_on_joint_outliers(joint_outliers):
    """Distance-based detection sees structure the per-feature methods cannot.

    This is the documented blind spot: ECOD and HBOS score each feature
    independently, so a point whose coordinates are individually ordinary is
    invisible to them however far off the manifold it lies.
    """
    X, y = joint_outliers

    knn_auc = roc_auc_score(
        y, -KNNDetector(n_neighbors=10).fit(X).decision_function(X)
    )
    ecod_auc = roc_auc_score(y, -ECODDetector().fit(X).decision_function(X))

    assert knn_auc > 0.9
    assert ecod_auc < 0.5
    assert knn_auc > ecod_auc


def test_ecod_is_invariant_to_monotone_rescaling(marginal_outliers):
    """Rank-based scoring is untouched by any monotone per-feature transform."""
    X, _ = marginal_outliers
    baseline = ECODDetector().fit(X).decision_function(X)

    rescaled = X.copy()
    rescaled[:, 0] = rescaled[:, 0] * 1000.0 + 7.0
    rescaled[:, 1] = np.exp(rescaled[:, 1] / 5.0)
    shifted = ECODDetector().fit(rescaled).decision_function(rescaled)

    np.testing.assert_allclose(baseline, shifted, rtol=1e-9)


# --------------------------------------------------------------------------
# ECOD specifics
# --------------------------------------------------------------------------

def test_ecod_feature_contributions_explain_the_score(marginal_outliers):
    """Contributions are per-feature and sum to the skewness-guided score."""
    X, _ = marginal_outliers
    detector = ECODDetector().fit(X)

    contributions = detector.feature_contributions(X)
    assert contributions.shape == X.shape
    assert np.all(contributions >= 0.0)
    # The outlier rows carry more total surprise than the inliers.
    assert contributions[-20:].sum(axis=1).mean() > contributions[:400].sum(axis=1).mean()


def test_ecod_attributes_the_right_feature():
    """The feature carrying the anomaly is the one with the largest share."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, 4))
    # Feature 2 alone is extreme for the final row.
    X[-1] = [0.0, 0.1, 9.0, -0.2]

    contributions = ECODDetector().fit(X).feature_contributions(X[-1:])
    assert int(contributions[0].argmax()) == 2


def test_ecod_and_copod_agree_closely(marginal_outliers):
    """The two siblings rank points near-identically, as documented."""
    X, y = marginal_outliers
    ecod = -ECODDetector().fit(X).decision_function(X)
    copod = -COPODDetector().fit(X).decision_function(X)
    assert roc_auc_score(y, ecod) == pytest.approx(roc_auc_score(y, copod), abs=0.05)


# --------------------------------------------------------------------------
# HBOS specifics
# --------------------------------------------------------------------------

def test_hbos_auto_bins_grow_with_sample_size():
    """The 'auto' rule scales the bin count with n and stays inside its bounds."""
    rng = np.random.default_rng(0)
    small = HBOSDetector().fit(rng.normal(size=(100, 3)))
    large = HBOSDetector().fit(rng.normal(size=(50_000, 3)))
    assert 5 <= small.n_bins_ <= large.n_bins_ <= 100


def test_hbos_model_size_is_independent_of_n():
    """Unlike ECOD, HBOS does not retain the training data."""
    rng = np.random.default_rng(0)
    detector = HBOSDetector(n_bins=10).fit(rng.normal(size=(5000, 4)))
    assert not hasattr(detector, "X_train_")
    assert detector.density_.shape == (4, 10)
    assert detector.edges_.shape == (4, 11)


@pytest.mark.parametrize("strategy", ["equal_width", "equal_frequency"])
def test_hbos_both_binning_strategies_work(strategy, marginal_outliers):
    """Both binning strategies produce valid, finite scores."""
    X, y = marginal_outliers
    detector = HBOSDetector(strategy=strategy, n_bins=12).fit(X)
    scores = detector.decision_function(X)
    assert np.all(np.isfinite(scores))
    assert roc_auc_score(y, -scores) > 0.9


def test_hbos_handles_a_constant_feature():
    """A zero-variance column must not produce an infinite or NaN score."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    X[:, 1] = 4.0
    scores = HBOSDetector().fit(X).decision_function(X)
    assert np.all(np.isfinite(scores))


def test_hbos_rejects_too_few_bins():
    """A single bin carries no information and is rejected."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_bins"):
        HBOSDetector(n_bins=1).fit(rng.normal(size=(50, 2)))


# --------------------------------------------------------------------------
# kNN and ABOD specifics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["largest", "mean", "median"])
def test_knn_reduction_methods(method, marginal_outliers):
    """All three distance reductions detect clear outliers."""
    X, y = marginal_outliers
    detector = KNNDetector(n_neighbors=5, method=method).fit(X)
    assert roc_auc_score(y, -detector.decision_function(X)) > 0.95


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_knn_metrics(metric, marginal_outliers):
    """Every supported metric produces finite scores."""
    X, _ = marginal_outliers
    scores = KNNDetector(metric=metric).fit(X).decision_function(X)
    assert np.all(np.isfinite(scores))


def test_knn_rejects_k_larger_than_the_training_set():
    """Asking for more neighbours than exist is an error, not a silent clamp."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_neighbors"):
        KNNDetector(n_neighbors=50).fit(rng.normal(size=(20, 2)))


def test_knn_invalid_arguments_rejected():
    """Unknown method and metric names are caught at construction."""
    with pytest.raises(ValueError, match="method"):
        KNNDetector(method="nonsense")
    with pytest.raises(ValueError, match="metric"):
        KNNDetector(metric="nonsense")


def test_abod_finds_isolated_outliers_in_high_dimension():
    """ABOD's design case: a few isolated anomalies in many dimensions."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (300, 100)), rng.normal(6, 1, (3, 100))])
    y = np.zeros(303, dtype=int)
    y[-3:] = 1
    detector = ABODDetector(n_neighbors=15).fit(X)
    assert roc_auc_score(y, -detector.decision_function(X)) > 0.95


def test_abod_masks_clustered_anomalies():
    """Pin down ABOD's documented failure mode so it cannot regress silently.

    A tight group of anomalies are each other's nearest neighbours, and the
    inverse-square weighting rewards their tight neighbourhood, so the group
    scores as *more* normal than genuine inliers. kNN is unaffected, which is
    why the docstring sends bursty-anomaly users there.
    """
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (300, 50)), rng.normal(6, 0.05, (15, 50))])
    y = np.zeros(315, dtype=int)
    y[-15:] = 1

    abod_auc = roc_auc_score(y, -ABODDetector(n_neighbors=15).fit(X).decision_function(X))
    knn_auc = roc_auc_score(y, -KNNDetector(n_neighbors=15).fit(X).decision_function(X))

    assert abod_auc < 0.2   # inverted, as documented
    assert knn_auc > 0.95   # unaffected


def test_abod_matches_the_published_factor():
    """FastABOD reproduces a brute-force implementation of the ABOF formula."""
    from itertools import combinations

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))

    def reference(query, k):
        """Compute the ABOF of one point directly from the definition."""
        x = X[query]
        order = np.argsort(np.linalg.norm(X - x, axis=1))
        neighbours = [i for i in order if i != query][:k]
        return np.var([
            np.dot(X[i] - x, X[j] - x)
            / (np.dot(X[i] - x, X[i] - x) * np.dot(X[j] - x, X[j] - x))
            for i, j in combinations(neighbours, 2)
        ])

    scores = -ABODDetector(n_neighbors=8).fit(X).decision_function(X)
    expected = -np.array([reference(i, 8) for i in range(60)])
    np.testing.assert_allclose(scores, expected, rtol=1e-9)


def test_knn_matches_the_published_definition():
    """Each reduction reproduces a direct leave-self-out neighbour computation."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))

    for method, reduce_fn in (
        ("largest", lambda d: d[-1]),
        ("mean", lambda d: d.mean()),
        ("median", lambda d: np.median(d)),
    ):
        scores = -KNNDetector(n_neighbors=5, method=method).fit(X).decision_function(X)
        expected = np.array([
            reduce_fn(np.sort(np.delete(np.linalg.norm(X - X[i], axis=1), i))[:5])
            for i in range(60)
        ])
        np.testing.assert_allclose(scores, expected, rtol=1e-9)


def test_self_match_is_excluded_when_scoring_training_data():
    """Scoring a training point must not count its own zero distance.

    Without the exclusion, 'mean' and 'median' would average in a zero and
    'largest' would report the (k-1)-th neighbour, so decision_function on the
    training set would disagree with the threshold fit computed.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    detector = KNNDetector(n_neighbors=4, method="mean").fit(X)

    scores = -detector.decision_function(X)
    expected = np.array([
        np.sort(np.delete(np.linalg.norm(X - X[i], axis=1), i))[:4].mean()
        for i in range(80)
    ])
    np.testing.assert_allclose(scores, expected, rtol=1e-9)
    assert np.all(scores > 0.0)


def test_abod_handles_duplicate_points():
    """A duplicated training point has no direction and must not divide by zero."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    X[5] = X[4]  # exact duplicate
    scores = ABODDetector(n_neighbors=8).fit(X).decision_function(X)
    assert np.all(np.isfinite(scores))


def test_abod_needs_at_least_two_neighbours():
    """A single neighbour forms no pair and therefore no angle."""
    with pytest.raises(ValueError, match="n_neighbors"):
        ABODDetector(n_neighbors=1)


# --------------------------------------------------------------------------
# Shared C++ kernels
# --------------------------------------------------------------------------

def test_tail_probabilities_match_a_numpy_reference():
    """The ECDF kernel reproduces a direct numpy count, ties included."""
    rng = np.random.default_rng(0)
    X = np.round(rng.normal(size=(300, 3)), 1)  # rounding forces ties
    left, right = cpp_stats.tail_probabilities(X, X[:20])

    expected_left = np.array(
        [[(X[:, j] <= X[i, j]).mean() for j in range(3)] for i in range(20)]
    )
    expected_right = np.array(
        [[(X[:, j] >= X[i, j]).mean() for j in range(3)] for i in range(20)]
    )
    np.testing.assert_allclose(left, expected_left)
    np.testing.assert_allclose(right, expected_right)


def test_tail_probabilities_never_reach_zero():
    """Probabilities are floored so -log() stays finite for unseen extremes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    query = np.array([[-50.0, 50.0]])
    left, right = cpp_stats.tail_probabilities(X, query)
    assert np.all(left > 0.0) and np.all(right > 0.0)
    assert np.all(np.isfinite(-np.log(left))) and np.all(np.isfinite(-np.log(right)))


def test_skewness_matches_scipy():
    """The kernel implements the adjusted Fisher-Pearson definition."""
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=500), rng.exponential(size=500)])
    np.testing.assert_allclose(
        cpp_stats.skewness(X), scipy_stats.skew(X, axis=0, bias=False), rtol=1e-10
    )


def test_skewness_of_a_constant_column_is_zero():
    """A column with no spread has no skew and must not divide by zero."""
    X = np.column_stack([np.ones(50), np.arange(50, dtype=float)])
    assert cpp_stats.skewness(X)[0] == 0.0


@pytest.mark.parametrize(
    "histogram", [cpp_stats.equal_width_histogram, cpp_stats.equal_frequency_histogram]
)
def test_histograms_integrate_to_one(histogram):
    """Both binning strategies return a normalised density."""
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=800), rng.exponential(size=800)])
    edges, density = histogram(X, 10)
    mass = (np.asarray(density) * np.diff(np.asarray(edges), axis=1)).sum(axis=1)
    np.testing.assert_allclose(mass, 1.0, atol=1e-9)


def test_histogram_density_clamps_out_of_range_queries():
    """Values beyond the outermost edges take the nearest bin, not zero."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    edges, density = cpp_stats.equal_width_histogram(X, 10)
    far = cpp_stats.histogram_density(edges, density, np.array([[-99.0, 99.0]]))
    assert np.all(np.isfinite(far))


# --------------------------------------------------------------------------
# LSCP ensemble
# --------------------------------------------------------------------------

@pytest.fixture
def mixed_density():
    """Return two clusters of very different density, with anomalies between.

    No single neighbourhood size wins here: a small k suits the tight cluster,
    a large k the diffuse one. That is the situation LSCP exists for.
    """
    rng = np.random.default_rng(0)
    tight = rng.normal([0, 0, 0, 0], 0.3, (250, 4))
    diffuse = rng.normal([9, 9, 9, 9], 2.5, (250, 4))
    between = rng.normal([4.5, 4.5, 4.5, 4.5], 0.2, (12, 4))
    X = np.vstack([tight, diffuse, between])
    y = np.zeros(len(X), dtype=int)
    y[-12:] = 1
    return X, y


def test_lscp_follows_the_anomaly_contract(marginal_outliers):
    """LSCP honours the same prediction and score conventions as the rest."""
    X, _ = marginal_outliers
    detector = LSCPDetector(contamination=0.05, random_state=0).fit(X)

    predictions = detector.predict(X)
    assert set(np.unique(predictions)).issubset({-1, 1})

    scores = detector.decision_function(X)
    assert scores.shape == (len(X),)
    assert np.all(np.isfinite(scores))
    np.testing.assert_allclose(scores, detector.score_samples(X))


def test_lscp_beats_averaging_the_same_pool(mixed_density):
    """Local selection recovers what a global average of the pool loses.

    The comparison that matters is against a plain average of the *same*
    detectors, not against a single detector — that is the baseline LSCP has
    to justify its extra cost against.
    """
    X, y = mixed_density
    neighbourhoods = (5, 10, 20, 35)

    standardised = []
    for k in neighbourhoods:
        raw = -KNNDetector(n_neighbors=k).fit(X).decision_function(X)
        standardised.append((raw - raw.mean()) / raw.std())
    average_auc = roc_auc_score(y, np.mean(standardised, axis=0))

    lscp_auc = roc_auc_score(
        y, -LSCPDetector(random_state=0).fit(X).decision_function(X)
    )

    assert average_auc < 0.8    # the global average is dragged down
    assert lscp_auc > 0.9       # local selection is not
    assert lscp_auc > average_auc


@pytest.mark.parametrize("method", ["average", "maximum"])
def test_lscp_both_combination_rules(method, mixed_density):
    """LSCP_A and LSCP_M both work and both beat the pool average."""
    X, y = mixed_density
    detector = LSCPDetector(method=method, random_state=0).fit(X)
    assert roc_auc_score(y, -detector.decision_function(X)) > 0.9


def test_lscp_accepts_a_heterogeneous_pool(marginal_outliers):
    """The pool may mix detector families, not just hyperparameters."""
    X, y = marginal_outliers
    pool = [ECODDetector(), HBOSDetector(), KNNDetector(n_neighbors=10)]
    detector = LSCPDetector(detectors=pool, random_state=0).fit(X)

    assert len(detector.detectors_) == 3
    assert roc_auc_score(y, -detector.decision_function(X)) > 0.9


def test_lscp_does_not_mutate_the_passed_pool(marginal_outliers):
    """Base detectors are deep-copied, so the caller's instances stay unfitted."""
    X, _ = marginal_outliers
    pool = [KNNDetector(n_neighbors=5), KNNDetector(n_neighbors=15)]
    LSCPDetector(detectors=pool, random_state=0).fit(X)
    for detector in pool:
        assert detector.X_train_ is None


def test_lscp_local_competence_varies_across_points(mixed_density):
    """Different neighbourhoods trust different detectors — the whole premise."""
    X, _ = mixed_density
    competence = LSCPDetector(random_state=0).fit(X).local_competence(X)

    assert competence.shape == (len(X), 4)
    assert np.all(np.isfinite(competence))
    assert np.all(np.abs(competence) <= 1.0 + 1e-9)
    # If the same detector won everywhere, local selection would be pointless.
    assert len(set(competence.argmax(axis=1).tolist())) > 1


def test_lscp_is_reproducible(marginal_outliers):
    """The same seed gives the same subspaces and therefore the same scores."""
    X, _ = marginal_outliers
    first = LSCPDetector(random_state=7).fit(X).decision_function(X)
    second = LSCPDetector(random_state=7).fit(X).decision_function(X)
    np.testing.assert_allclose(first, second)


def test_lscp_rejects_a_degenerate_pool():
    """A pool of one leaves nothing to select between."""
    with pytest.raises(ValueError, match="at least 2"):
        LSCPDetector(detectors=[KNNDetector()])
    with pytest.raises(ValueError, match="method"):
        LSCPDetector(method="nonsense")
