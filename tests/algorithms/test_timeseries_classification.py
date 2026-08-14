"""DTW distances and nearest-neighbour time-series classification."""

import numpy as np
import pytest

from tuiml._cpp_ext import timeseries as cpp_ts
from tuiml.algorithms.timeseries.classification import (
    DTWNeighborsClassifier,
    as_panel,
    dtw_distance,
    dtw_pairwise,
    lb_keogh,
    lb_keogh_envelope,
)


# --------------------------------------------------------------------------
# Reference implementations, used to pin the C++ kernel down
# --------------------------------------------------------------------------

def reference_dtw(a, b, window=None):
    """Compute DTW directly from the recurrence, with no optimisation."""
    n, m = len(a), len(b)
    band = max(window if window is not None else max(n, m), abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            cost[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    return np.sqrt(cost[n, m])


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def warped_shapes():
    """Return classes that differ by shape, randomly time-warped and shifted.

    This is DTW's design case: the same shape played at different speeds and
    starting at different times must still be recognised as one class.
    """
    length = 100
    grid = np.linspace(0, 1, length)

    def shape(u, cls):
        """Return a broad hump for class 0, twin sharp spikes for class 1."""
        if cls == 0:
            return np.exp(-((u - 0.5) ** 2) / 0.02)
        return np.exp(-((u - 0.42) ** 2) / 0.0012) + np.exp(
            -((u - 0.58) ** 2) / 0.0012
        )

    def build(n, seed):
        """Generate n warped, shifted, noisy series with alternating labels."""
        rng = np.random.default_rng(seed)
        X, y = [], []
        for i in range(n):
            cls = i % 2
            warp = grid ** rng.uniform(0.5, 2.0)
            series = shape(warp / warp.max(), cls)
            series = np.roll(series, rng.integers(-25, 26))
            X.append(series + rng.normal(0, 0.04, length))
            y.append(cls)
        return np.array(X), np.array(y)

    return build(120, 0), build(120, 1)


# --------------------------------------------------------------------------
# The DTW kernel
# --------------------------------------------------------------------------

def test_dtw_matches_the_recurrence():
    """The C++ kernel reproduces a direct implementation of the recurrence."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        n, m = int(rng.integers(3, 25)), int(rng.integers(3, 25))
        a, b = rng.normal(size=n), rng.normal(size=m)
        window = int(rng.choice([3, 8]))
        assert dtw_distance(a, b, window) == pytest.approx(
            reference_dtw(a, b, window), rel=1e-12
        )
        assert dtw_distance(a, b, None) == pytest.approx(
            reference_dtw(a, b, None), rel=1e-12
        )


def test_dtw_is_a_metric_where_it_should_be():
    """Identity and symmetry hold; unequal lengths are accepted."""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=40), rng.normal(size=55)

    assert dtw_distance(a, a) == 0.0
    assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a))
    assert dtw_distance(a, b) > 0.0


def test_dtw_tolerates_a_shift_that_euclidean_does_not():
    """The point of an elastic distance: a shifted copy stays close."""
    base = np.zeros(50)
    base[20:25] = 1.0
    shifted = np.roll(base, 4)

    assert dtw_distance(base, shifted) < np.linalg.norm(base - shifted)


def test_narrower_bands_give_larger_or_equal_distances():
    """Constraining the warping path can only make the optimum worse."""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=60), rng.normal(size=60)
    distances = [dtw_distance(a, b, w) for w in (2, 5, 15, 40)]
    assert all(
        earlier >= later - 1e-9 for earlier, later in zip(distances, distances[1:])
    )


def test_lb_keogh_never_exceeds_the_true_distance():
    """The bound must be a genuine lower bound, or pruning would be unsound."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        a, b = rng.normal(size=40), rng.normal(size=40)
        window = int(rng.choice([2, 5, 10]))
        assert lb_keogh(a, b, window) <= dtw_distance(a, b, window) + 1e-9


def test_lb_keogh_envelope_brackets_the_series():
    """The envelope contains the series it was built from."""
    rng = np.random.default_rng(0)
    series = rng.normal(size=60)
    lower, upper = lb_keogh_envelope(series, 4)

    assert lower.shape == upper.shape == series.shape
    assert np.all(lower <= series + 1e-12)
    assert np.all(series <= upper + 1e-12)


def test_pairwise_matrix_is_symmetric_with_zero_diagonal():
    """A self-distance matrix has the structure it should."""
    rng = np.random.default_rng(0)
    panel = rng.normal(size=(6, 40))
    distances = dtw_pairwise(panel, window=5)

    assert distances.shape == (6, 6)
    np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-12)
    np.testing.assert_allclose(distances, distances.T, rtol=1e-12)


def test_pairwise_accepts_unequal_lengths_and_multivariate():
    """Panels need not share a length, and channels are supported."""
    rng = np.random.default_rng(0)
    assert dtw_pairwise(rng.normal(size=(3, 50)), rng.normal(size=(4, 70))).shape == (3, 4)
    assert dtw_pairwise(rng.normal(size=(3, 2, 40)), rng.normal(size=(5, 2, 40))).shape == (3, 5)


def test_pruned_search_returns_exactly_the_brute_force_neighbours():
    """LB_Keogh pruning is an optimisation, not an approximation.

    If this ever diverges, the bound has stopped being a valid lower bound and
    the classifier is silently returning wrong neighbours.
    """
    rng = np.random.default_rng(0)
    train = rng.normal(size=(120, 1, 60)).cumsum(axis=2)
    query = rng.normal(size=(20, 1, 60)).cumsum(axis=2)

    full = np.asarray(cpp_ts.dtw_pairwise(query, train, 6))
    distances, indices = cpp_ts.dtw_knn(query, train, 5, 6)

    expected_indices = np.argsort(full, axis=1)[:, :5]
    np.testing.assert_array_equal(np.asarray(indices), expected_indices)
    np.testing.assert_allclose(
        np.asarray(distances), np.take_along_axis(full, expected_indices, axis=1)
    )


# --------------------------------------------------------------------------
# Panel handling
# --------------------------------------------------------------------------

def test_as_panel_promotes_2d_to_univariate():
    """A 2-D array is read as one univariate series per row."""
    assert as_panel(np.zeros((5, 100))).shape == (5, 1, 100)
    assert as_panel(np.zeros((5, 3, 100))).shape == (5, 3, 100)


def test_as_panel_rejects_wrong_rank():
    """A 1-D or 4-D input is ambiguous and rejected."""
    with pytest.raises(ValueError, match="n_samples"):
        as_panel(np.zeros(100))


def test_classifier_rejects_a_channel_mismatch(warped_shapes):
    """Predicting with a different channel count is an error, not a guess."""
    (X, y), _ = warped_shapes
    model = DTWNeighborsClassifier().fit(X, y)
    with pytest.raises(ValueError, match="channels"):
        model.predict(np.zeros((3, 2, X.shape[1])))


def test_classifier_rejects_mismatched_lengths():
    """X and y must describe the same number of series."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="labels"):
        DTWNeighborsClassifier().fit(rng.normal(size=(10, 30)), np.zeros(9))


def test_classifier_rejects_k_larger_than_the_training_set():
    """Asking for more neighbours than exist is an error."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_neighbors"):
        DTWNeighborsClassifier(n_neighbors=20).fit(rng.normal(size=(10, 30)), np.arange(10) % 2)


# --------------------------------------------------------------------------
# Classification behaviour
# --------------------------------------------------------------------------

def test_dtw_knn_classifies_warped_shapes(warped_shapes):
    """DTW's design case: same shape, different speed and start time."""
    (X_train, y_train), (X_test, y_test) = warped_shapes
    model = DTWNeighborsClassifier(n_neighbors=1, window=0.1).fit(X_train, y_train)
    assert (model.predict(X_test) == y_test).mean() > 0.95


def test_band_tracks_unconstrained_accuracy_far_more_cheaply(warped_shapes):
    """A 10% band stays within a hair of unconstrained DTW for a fraction of
    the compute, which is why it is the default.

    The claim is *comparable*, not *never worse*: a band forbids some optimal
    warping paths, so a single borderline series can flip either way.
    """
    import time

    (X_train, y_train), (X_test, y_test) = warped_shapes

    banded = DTWNeighborsClassifier(window=0.1).fit(X_train, y_train)
    start = time.perf_counter()
    banded_accuracy = (banded.predict(X_test) == y_test).mean()
    banded_seconds = time.perf_counter() - start

    full = DTWNeighborsClassifier(window=None).fit(X_train, y_train)
    start = time.perf_counter()
    full_accuracy = (full.predict(X_test) == y_test).mean()
    full_seconds = time.perf_counter() - start

    assert banded_accuracy >= full_accuracy - 0.02
    assert banded_seconds < full_seconds


def test_warping_invariance_hurts_when_timing_is_the_label():
    """Pin down the documented caveat: DTW loses when the label *is* timing.

    Both classes are the same two-peak shape; only the peak positions differ.
    DTW normalises that away by design, so a plain Euclidean neighbour wins.
    """
    from tuiml.algorithms.neighbors import KNearestNeighborsClassifier

    length = 100
    grid = np.linspace(0, 1, length)

    def build(n, seed):
        """Generate series whose class is encoded purely in peak position."""
        rng = np.random.default_rng(seed)
        X, y = [], []
        for i in range(n):
            cls = i % 2
            warp = grid ** rng.uniform(0.35, 2.8)
            u = warp / warp.max()
            centres = (0.25, 0.65) if cls == 0 else (0.35, 0.80)
            series = sum(np.exp(-((u - c) ** 2) / 0.006) for c in centres)
            X.append(series + rng.normal(0, 0.05, length))
            y.append(cls)
        return np.array(X), np.array(y)

    X_train, y_train = build(120, 0)
    X_test, y_test = build(120, 1)

    dtw_acc = (
        DTWNeighborsClassifier(window=0.1).fit(X_train, y_train).predict(X_test)
        == y_test
    ).mean()
    euclidean_acc = (
        KNearestNeighborsClassifier(k=1).fit(X_train, y_train).predict(X_test)
        == y_test
    ).mean()

    assert euclidean_acc > dtw_acc


def test_kneighbors_returns_sorted_valid_neighbours(warped_shapes):
    """Neighbours come back nearest-first with in-range indices."""
    (X, y), _ = warped_shapes
    model = DTWNeighborsClassifier(n_neighbors=3, window=0.1).fit(X, y)
    distances, indices = model.kneighbors(X[:10])

    assert distances.shape == indices.shape == (10, 3)
    assert np.all(np.diff(distances, axis=1) >= -1e-12)
    assert np.all((indices >= 0) & (indices < len(X)))
    # Every series is its own nearest neighbour at distance zero.
    np.testing.assert_allclose(distances[:, 0], 0.0, atol=1e-12)
    np.testing.assert_array_equal(indices[:, 0], np.arange(10))


def test_predict_proba_is_a_distribution(warped_shapes):
    """Vote shares are non-negative and sum to one."""
    (X, y), _ = warped_shapes
    model = DTWNeighborsClassifier(n_neighbors=5, window=0.1).fit(X, y)
    proba = model.predict_proba(X[:20])

    assert proba.shape == (20, len(model.classes_))
    assert np.all(proba >= 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_both_weightings_classify(weights, warped_shapes):
    """Uniform and distance weighting both work end to end."""
    (X_train, y_train), (X_test, y_test) = warped_shapes
    model = DTWNeighborsClassifier(
        n_neighbors=3, window=0.1, weights=weights
    ).fit(X_train, y_train)
    assert (model.predict(X_test) == y_test).mean() > 0.9


def test_distance_weighting_handles_an_exact_match(warped_shapes):
    """A zero distance must not divide by zero; it takes the whole vote."""
    (X, y), _ = warped_shapes
    model = DTWNeighborsClassifier(
        n_neighbors=3, window=0.1, weights="distance"
    ).fit(X, y)
    proba = model.predict_proba(X[:5])
    assert np.all(np.isfinite(proba))
    np.testing.assert_array_equal(model.predict(X[:5]), y[:5])


def test_multivariate_panels_are_supported():
    """Two synchronised channels classify with dependent DTW."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, 60)

    def build(n, cls, seed):
        """Generate n two-channel series of one class."""
        local = np.random.default_rng(seed)
        shift = local.uniform(0, 2 * np.pi, n)
        first = np.sin(t + shift[:, None]) if cls == 0 else np.sign(
            np.sin(t + shift[:, None])
        )
        second = np.cos(t + shift[:, None]) * (1 if cls == 0 else -1)
        return np.stack([first, second], axis=1) + local.normal(0, 0.1, (n, 2, 60))

    X = np.concatenate([build(30, 0, 1), build(30, 1, 2)])
    y = np.array([0] * 30 + [1] * 30)

    model = DTWNeighborsClassifier(window=0.1).fit(X, y)
    assert X.shape == (60, 2, 60)
    assert (model.predict(X) == y).mean() == 1.0


def test_classifier_is_registered_from_a_clean_import():
    """The hub finds the classifier without the subpackage being imported first.

    The registry discovers components by importing ``tuiml.algorithms``, so a
    new subpackage is invisible unless a parent ``__init__`` imports it. This
    runs in a subprocess because the registry is a process-wide singleton and
    this test file has already imported the subpackage directly.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tuiml import list_algorithms;"
            "print('DTWNeighborsClassifier' in {a['name'] for a in list_algorithms()})",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "True"


# --------------------------------------------------------------------------
# MINIROCKET
# --------------------------------------------------------------------------

@pytest.fixture
def burst_frequency():
    """Return a panel whose class is the frequency of a hidden burst.

    Amplitude, phase, sign and position are all randomised and each series is
    z-normalised, so no cue except the burst's frequency survives. This is
    MINIROCKET's territory and DTW's weakness: warping invariance smears
    exactly the information that defines the classes.
    """
    length = 256

    def build(n, seed):
        """Generate n series with alternating burst frequencies."""
        rng = np.random.default_rng(seed)
        X, y = [], []
        for i in range(n):
            cls = i % 2
            series = rng.normal(0, 1.0, length)
            start = rng.integers(0, length - 50)
            span = np.arange(50)
            frequency = 0.40 if cls == 0 else 0.22
            amplitude = rng.uniform(0.8, 2.2) * rng.choice([-1, 1])
            series[start : start + 50] += amplitude * np.sin(
                2 * np.pi * frequency * span + rng.uniform(0, 2 * np.pi)
            )
            X.append((series - series.mean()) / series.std())
            y.append(cls)
        return np.array(X), np.array(y)

    return build(300, 0), build(300, 1)


def test_minirocket_kernels_are_the_84_fixed_combinations():
    """The kernel table is C(9,3) distinct, ascending index triples."""
    kernels = np.asarray(cpp_ts.minirocket_kernel_indices())

    assert kernels.shape == (84, 3)
    assert len({tuple(row) for row in kernels}) == 84
    assert np.all(np.diff(kernels, axis=1) > 0)
    assert kernels.min() == 0 and kernels.max() == 8


def test_ppv_matches_a_direct_dilated_convolution():
    """The alpha/gamma decomposition equals convolving each kernel directly.

    MINIROCKET's speed comes from computing the all -1 convolution once per
    dilation and caching the +3 corrections, rather than convolving 84 times.
    This checks that shortcut is exact.
    """
    kernels = np.asarray(cpp_ts.minirocket_kernel_indices())
    rng = np.random.default_rng(0)
    series = rng.normal(size=100)

    dilations = np.array([1, 2, 4, 8], dtype=np.int32)
    per_dilation = np.ones(4, dtype=np.int32)

    def direct(gamma_positions, dilation):
        """Convolve directly from the kernel definition, no shortcuts."""
        weights = np.full(9, -1.0)
        weights[list(gamma_positions)] = 2.0
        out = np.zeros(len(series))
        for t in range(len(series)):
            total = 0.0
            for position in range(9):
                source = t + (position - 4) * dilation
                if 0 <= source < len(series):
                    total += weights[position] * series[source]
            out[t] = total
        return out

    for dilation_index, dilation in enumerate(dilations):
        for kernel_index in (0, 17, 41, 83):
            expected_output = direct(kernels[kernel_index], int(dilation))

            padding = ((9 - 1) * int(dilation)) // 2
            if (dilation_index + kernel_index) % 2 == 0:
                segment = expected_output
            else:
                lo, hi = min(padding, len(expected_output)), len(expected_output) - padding
                segment = expected_output if hi <= lo else expected_output[lo:hi]

            threshold = float(np.median(segment))
            # Isolate one slot: every other bias is set so its PPV is zero.
            biases = np.full(84 * len(dilations), 1e18)
            slot = dilation_index * 84 + kernel_index
            biases[slot] = threshold

            features = np.asarray(
                cpp_ts.minirocket_transform(
                    series[None, :], dilations, per_dilation, biases
                )
            )
            assert features[0, slot] == pytest.approx(
                float((segment > threshold).mean()), abs=1e-12
            )


def test_minirocket_features_are_proportions(burst_frequency):
    """Every feature is a PPV, so it lies in [0, 1]."""
    from tuiml.algorithms.timeseries.classification import MiniRocketTransformer

    (X, _), _ = burst_frequency
    features = MiniRocketTransformer(n_features=840, random_state=0).fit_transform(X)

    assert features.shape[0] == len(X)
    assert np.all((features >= 0.0) & (features <= 1.0))
    assert np.all(np.isfinite(features))


def test_minirocket_beats_the_baselines_on_frequency(burst_frequency):
    """MINIROCKET's design case, where DTW's warping invariance hurts."""
    from tuiml.algorithms.neighbors import KNearestNeighborsClassifier
    from tuiml.algorithms.timeseries.classification import MiniRocketClassifier

    (X_train, y_train), (X_test, y_test) = burst_frequency

    rocket = MiniRocketClassifier(n_features=9996, random_state=0).fit(X_train, y_train)
    euclidean = KNearestNeighborsClassifier(k=1).fit(X_train, y_train)

    rocket_accuracy = (rocket.predict(X_test) == y_test).mean()
    euclidean_accuracy = (euclidean.predict(X_test) == y_test).mean()

    assert rocket_accuracy > 0.9
    assert rocket_accuracy > euclidean_accuracy


def test_minirocket_transform_is_deterministic_given_a_seed(burst_frequency):
    """Only the bias quantiles are sampled, and the seed fixes those."""
    from tuiml.algorithms.timeseries.classification import MiniRocketTransformer

    (X, _), _ = burst_frequency
    first = MiniRocketTransformer(n_features=840, random_state=7).fit_transform(X)
    second = MiniRocketTransformer(n_features=840, random_state=7).fit_transform(X)
    np.testing.assert_allclose(first, second)


def test_valid_region_features_ignore_a_constant_offset():
    """Offset invariance holds where the whole kernel fits, and only there.

    The kernel weights sum to zero, so a constant cancels in the valid centre.
    The convolution is zero-padded at the edges, where only part of the kernel
    overlaps real data and the weights no longer cancel — so the padded half of
    the features is *not* offset-invariant. Both halves are asserted here so
    the distinction cannot quietly drift.
    """
    from tuiml.algorithms.timeseries.classification import MiniRocketTransformer

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 128))

    transformer = MiniRocketTransformer(n_features=840, random_state=0).fit(X)
    baseline = transformer.transform(X)
    shifted = transformer.transform(X + 17.0)

    # Features are laid out dilation-major, then kernel, then bias. A slot uses
    # padding when (dilation_index + kernel_index) is even.
    padded = []
    for dilation_index, count in enumerate(transformer.features_per_dilation_):
        for kernel_index in range(84):
            padded.extend([(dilation_index + kernel_index) % 2 == 0] * int(count))
    padded = np.array(padded)

    np.testing.assert_allclose(
        baseline[:, ~padded], shifted[:, ~padded], atol=1e-9
    )
    assert not np.allclose(baseline[:, padded], shifted[:, padded])


def test_minirocket_accepts_a_custom_head(burst_frequency):
    """The transform is the method; the classification head is a choice."""
    from tuiml.algorithms.timeseries.classification import MiniRocketClassifier
    from tuiml.algorithms.trees import RandomForestClassifier

    (X_train, y_train), (X_test, y_test) = burst_frequency
    model = MiniRocketClassifier(
        n_features=840,
        estimator=RandomForestClassifier(n_estimators=50, random_state=0),
        random_state=0,
    ).fit(X_train, y_train)

    assert model.estimator_.__class__.__name__ == "RandomForestClassifier"
    assert (model.predict(X_test) == y_test).mean() > 0.7


def test_minirocket_handles_multivariate_by_concatenating_channels():
    """Each channel is transformed and the features concatenated."""
    from tuiml.algorithms.timeseries.classification import MiniRocketTransformer

    rng = np.random.default_rng(0)
    univariate = MiniRocketTransformer(n_features=840, random_state=0).fit(
        rng.normal(size=(10, 128))
    )
    multivariate = MiniRocketTransformer(n_features=840, random_state=0).fit(
        rng.normal(size=(10, 3, 128))
    )

    assert multivariate.n_features_ == 3 * univariate.n_features_


def test_minirocket_dilations_fit_inside_the_series():
    """No dilation may be so wide the kernel cannot fit the series."""
    from tuiml.algorithms.timeseries.classification import MiniRocketTransformer

    rng = np.random.default_rng(0)
    for length in (32, 64, 256, 1000):
        model = MiniRocketTransformer(n_features=9996, random_state=0).fit(
            rng.normal(size=(6, length))
        )
        assert model.dilations_.min() >= 1
        assert (model.dilations_ * 8).max() <= max(length - 1, 8)


def test_minirocket_rejects_series_shorter_than_the_kernel():
    """A series shorter than 9 points cannot be convolved at all."""
    from tuiml.algorithms.timeseries.classification import MiniRocketTransformer

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="shorter than the kernel"):
        MiniRocketTransformer().fit(rng.normal(size=(5, 6)))


def test_minirocket_rejects_too_few_features():
    """Fewer features than kernels leaves nothing for most kernels to say."""
    from tuiml.algorithms.timeseries.classification import MiniRocketClassifier

    with pytest.raises(ValueError, match="n_features"):
        MiniRocketClassifier(n_features=10)


def test_transformer_refuses_to_predict():
    """The transformer has no head and says so rather than guessing."""
    from tuiml.algorithms.timeseries.classification import MiniRocketTransformer

    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 64))
    transformer = MiniRocketTransformer(n_features=840, random_state=0).fit(X)

    with pytest.raises(NotImplementedError, match="MiniRocketClassifier"):
        transformer.predict(X)


def test_minirocket_predict_cost_is_flat_in_training_size():
    """The claim that makes MINIROCKET the default: transform cost ignores n.

    DTW must compare each query against every training series, so its
    prediction cost grows with the training set. MINIROCKET's does not.
    """
    import time

    from tuiml.algorithms.timeseries.classification import MiniRocketClassifier

    rng = np.random.default_rng(0)
    query = rng.normal(size=(100, 128))

    timings = []
    for n_train in (100, 800):
        X = rng.normal(size=(n_train, 128))
        y = np.arange(n_train) % 2
        model = MiniRocketClassifier(n_features=840, random_state=0).fit(X, y)
        start = time.perf_counter()
        model.predict(query)
        timings.append(time.perf_counter() - start)

    # An eightfold larger training set must not cost more than 2x to predict.
    assert timings[1] < timings[0] * 2.0 + 0.05


# --------------------------------------------------------------------------
# Shapelets
# --------------------------------------------------------------------------

SPIKE = np.concatenate([np.linspace(0, 3, 10), np.linspace(3, 0, 10)])


@pytest.fixture
def planted_motif():
    """Return data where class 1 hides a triangular spike at a random position.

    The discriminating pattern is local and its location varies, which is the
    shapelet method's design case — and it gives a known ground truth to check
    the recovered shapelets against.
    """

    def build(n, seed):
        """Generate n noisy series, planting the spike in the odd-indexed ones.

        Returns the planted positions too, so tests can check against ground
        truth rather than eyeballing the recovered shapelets.
        """
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 0.35, (n, 150))
        y = np.arange(n) % 2
        positions = {}
        for i in np.flatnonzero(y == 1):
            start = int(rng.integers(0, 130))
            X[i, start : start + 20] += SPIKE
            positions[i] = start
        return X, y, positions

    return build(160, 0), build(160, 1)


def test_shapelet_distance_matches_brute_force():
    """The C++ kernel reproduces an explicit z-normalising implementation."""
    def reference(series, shapelet):
        """Slide the shapelet, z-normalising each window the obvious way."""
        m = len(shapelet)
        best = np.inf
        for start in range(len(series) - m + 1):
            window = series[start : start + m]
            spread = window.std()
            z = (
                np.zeros(m)
                if spread <= 1e-6
                else (window - window.mean()) / spread
            )
            best = min(best, float(((z - shapelet) ** 2).sum()))
        return np.sqrt(best / m)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 80))

    shapelets = []
    for length in (8, 15, 30):
        candidate = rng.normal(size=length)
        shapelets.append((candidate - candidate.mean()) / candidate.std())

    flat = np.concatenate(shapelets)
    lengths = np.array([len(s) for s in shapelets], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int32)

    actual = np.asarray(cpp_ts.shapelet_distances(X, flat, offsets, lengths))
    expected = np.array(
        [[reference(X[i], s) for s in shapelets] for i in range(6)]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_shapelet_distance_is_zero_for_a_self_match():
    """A subsequence taken from a series matches that series exactly."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 80))

    window = X[2, 10:28]
    shapelet = (window - window.mean()) / window.std()

    distances = np.asarray(
        cpp_ts.shapelet_distances(
            X, shapelet, np.array([0], np.int32), np.array([18], np.int32)
        )
    )
    assert distances[2, 0] == pytest.approx(0.0, abs=1e-12)


def test_shapelet_distance_survives_a_large_offset():
    """Window z-normalisation must not lose precision on unnormalised data.

    The running variance is E[x^2] - mean^2, which cancels catastrophically
    when values sit far from zero. The kernel centres each series first; without
    that, a series offset by 1e6 drifted by ~4e-2 from the same series at zero.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 80))
    window = X[2, 10:28]
    shapelet = (window - window.mean()) / window.std()
    offsets, lengths = np.array([0], np.int32), np.array([18], np.int32)

    baseline = np.asarray(cpp_ts.shapelet_distances(X, shapelet, offsets, lengths))
    for scale, offset in ((1.0, 1e3), (1.0, 1e6), (3.0, 7.0)):
        shifted = np.asarray(
            cpp_ts.shapelet_distances(scale * X + offset, shapelet, offsets, lengths)
        )
        np.testing.assert_allclose(shifted, baseline, atol=1e-6)


def test_shapelets_recover_the_planted_motif(planted_motif):
    """The distinguishing claim: the fitted shapelets are readable evidence.

    Checked against ground truth by *location* — each top shapelet should be
    cut from a series that contains the motif, at a window overlapping where
    the motif was actually planted. Comparing shapes directly would be the
    wrong test: a correct shapelet is typically offset from the motif and a
    little longer, so it captures the pattern without matching it pointwise.
    """
    from tuiml.algorithms.timeseries.classification import ShapeletTransformClassifier

    (X, y, positions), _ = planted_motif
    model = ShapeletTransformClassifier(
        n_shapelets=20, n_candidates=400, random_state=0
    ).fit(X, y)

    for info in model.shapelet_info_[:5]:
        series = info["series"]
        assert y[series] == 1, "shapelet drawn from a series with no motif"

        planted = positions[series]
        overlaps = (
            info["start"] < planted + len(SPIKE)
            and planted < info["start"] + info["length"]
        )
        assert overlaps, "shapelet does not cover where the motif was planted"


def test_shapelet_classifier_learns_a_local_motif(planted_motif):
    """Accuracy on the problem shapelets are built for."""
    from tuiml.algorithms.timeseries.classification import ShapeletTransformClassifier

    (X_train, y_train, _), (X_test, y_test, _) = planted_motif
    model = ShapeletTransformClassifier(
        n_shapelets=20, n_candidates=400, random_state=0
    ).fit(X_train, y_train)
    assert (model.predict(X_test) == y_test).mean() > 0.9


def test_shapelets_are_z_normalised_and_within_bounds(planted_motif):
    """Kept shapelets obey the configured length range and are normalised."""
    from tuiml.algorithms.timeseries.classification import ShapeletTransformClassifier

    (X, y, _), _ = planted_motif
    model = ShapeletTransformClassifier(
        n_shapelets=15, n_candidates=200, min_length=0.1, max_length=0.3,
        random_state=0,
    ).fit(X, y)

    assert len(model.shapelets_) == 15
    assert len(model.shapelet_info_) == 15
    for shapelet, info in zip(model.shapelets_, model.shapelet_info_):
        assert len(shapelet) == info["length"]
        assert 15 <= info["length"] <= 45  # 10% to 30% of 150
        assert abs(float(shapelet.mean())) < 1e-9
        assert float(shapelet.std()) == pytest.approx(1.0, abs=1e-9)


def test_shapelet_quality_is_sorted_best_first(planted_motif):
    """shapelet_info_ is ordered by quality, so entry 0 is the exhibit."""
    from tuiml.algorithms.timeseries.classification import ShapeletTransformClassifier

    (X, y, _), _ = planted_motif
    model = ShapeletTransformClassifier(
        n_shapelets=10, n_candidates=200, random_state=0
    ).fit(X, y)
    qualities = [info["quality"] for info in model.shapelet_info_]
    assert qualities == sorted(qualities, reverse=True)


@pytest.mark.parametrize("quality", ["f_stat", "information_gain"])
def test_both_quality_measures_work(quality, planted_motif):
    """The classical information gain and the faster F-statistic both select."""
    from tuiml.algorithms.timeseries.classification import ShapeletTransformClassifier

    (X_train, y_train, _), (X_test, y_test, _) = planted_motif
    model = ShapeletTransformClassifier(
        n_shapelets=10, n_candidates=100, quality=quality, random_state=0
    ).fit(X_train, y_train)
    assert (model.predict(X_test) == y_test).mean() > 0.85


def test_similarity_filter_spreads_shapelets_out(planted_motif):
    """Overlap filtering stops the selection filling with near duplicates."""
    from tuiml.algorithms.timeseries.classification import ShapeletTransformClassifier

    X, y, _ = planted_motif[0]
    kwargs = dict(n_shapelets=15, n_candidates=400, random_state=0)

    filtered = ShapeletTransformClassifier(remove_similar=True, **kwargs).fit(X, y)
    unfiltered = ShapeletTransformClassifier(remove_similar=False, **kwargs).fit(X, y)

    def distinct_sources(model):
        """Count how many distinct training series the shapelets came from."""
        return len({info["series"] for info in model.shapelet_info_})

    assert distinct_sources(filtered) >= distinct_sources(unfiltered)
    assert len(filtered.shapelets_) == 15


def test_shapelet_transform_output_shape(planted_motif):
    """transform() gives one distance column per kept shapelet."""
    from tuiml.algorithms.timeseries.classification import ShapeletTransformClassifier

    (X, y, _), (X_test, _, _) = planted_motif
    model = ShapeletTransformClassifier(
        n_shapelets=12, n_candidates=100, random_state=0
    ).fit(X, y)

    features = model.transform(X_test)
    assert features.shape == (len(X_test), 12)
    assert np.all(features >= 0.0)
    assert np.all(np.isfinite(features))


def test_shapelets_record_their_channel_on_multivariate_input():
    """Each shapelet remembers which channel it came from."""
    from tuiml.algorithms.timeseries.classification import ShapeletTransformClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.4, (60, 3, 100))
    y = np.arange(60) % 2
    # Plant the motif in channel 2 only.
    for i in np.flatnonzero(y == 1):
        start = rng.integers(0, 80)
        X[i, 2, start : start + 20] += SPIKE

    model = ShapeletTransformClassifier(
        n_shapelets=10, n_candidates=300, random_state=0
    ).fit(X, y)

    channels = {info["channel"] for info in model.shapelet_info_}
    assert channels.issubset({0, 1, 2})
    # The planted channel should dominate the selection.
    assert model.shapelet_info_[0]["channel"] == 2


def test_shapelet_invalid_arguments_rejected():
    """Contradictory configuration is caught at construction."""
    from tuiml.algorithms.timeseries.classification import ShapeletTransformClassifier

    with pytest.raises(ValueError, match="quality"):
        ShapeletTransformClassifier(quality="nonsense")
    with pytest.raises(ValueError, match="n_candidates"):
        ShapeletTransformClassifier(n_shapelets=100, n_candidates=10)


# --------------------------------------------------------------------------
# BOSS dictionary classification
# --------------------------------------------------------------------------

@pytest.fixture
def motif_count():
    """Return data whose class is *how many times* a motif repeats.

    Positions are random, so the signal is position-invariant content — what a
    bag-of-words representation is built to capture and what a pointwise
    comparison cannot.
    """
    length = 400
    motif = np.concatenate([np.linspace(0, 3, 12), np.linspace(3, 0, 12)])

    def build(n, seed):
        """Generate n series with 2 or 6 copies of the motif."""
        rng = np.random.default_rng(seed)
        X, y = [], []
        for i in range(n):
            cls = i % 2
            series = rng.normal(0, 0.5, length)
            for _ in range(2 if cls == 0 else 6):
                start = rng.integers(0, length - len(motif))
                series[start : start + len(motif)] += motif
            X.append(series)
            y.append(cls)
        return np.array(X), np.array(y)

    return build(120, 0), build(120, 1)


def test_sfa_matches_a_direct_dft():
    """The incremental sliding DFT equals recomputing each window from scratch.

    Both the normalised and unnormalised paths are checked: the kernel centres
    the series only in the normalised case, since that is the one where the DC
    coefficient is discarded and a constant shift is therefore irrelevant.
    """
    def reference(series, window, word_length, norm_mean):
        """Take a fresh DFT of every window, with no incremental update."""
        n_coefficients = (word_length + 1) // 2 + (1 if norm_mean else 0)
        rows = []
        for start in range(len(series) - window + 1):
            values = series[start : start + window]
            spectrum = np.fft.fft(values)[:n_coefficients]
            scale = 1.0
            if norm_mean:
                spread = values.std()
                scale = 1.0 / spread if spread > 1e-6 else 1.0
            first = 1 if norm_mean else 0
            rows.append([
                (
                    spectrum[first + f // 2].real
                    if f % 2 == 0
                    else spectrum[first + f // 2].imag
                )
                * scale
                / window
                for f in range(word_length)
            ])
        return np.array(rows)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 120))

    for window, word_length, norm_mean in (
        (16, 8, True), (32, 6, True), (20, 4, False), (60, 10, True), (12, 2, False)
    ):
        actual = np.asarray(cpp_ts.sfa_transform(X, window, word_length, norm_mean))
        expected = np.stack(
            [reference(X[i], window, word_length, norm_mean) for i in range(4)]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_sfa_normalised_output_survives_a_large_offset():
    """Window normalisation makes the output shift-invariant, precisely."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 120))
    baseline = np.asarray(cpp_ts.sfa_transform(X, 32, 8, True))
    for offset in (1e3, 1e6):
        shifted = np.asarray(cpp_ts.sfa_transform(X + offset, 32, 8, True))
        np.testing.assert_allclose(shifted, baseline, atol=1e-8)


def test_boss_beats_a_pointwise_comparison_on_motif_counts(motif_count):
    """Position-invariant content is where a bag-of-words earns its place."""
    from tuiml.algorithms.neighbors import KNearestNeighborsClassifier
    from tuiml.algorithms.timeseries.classification import BOSSClassifier

    (X_train, y_train), (X_test, y_test) = motif_count

    boss = BOSSClassifier(window_size=40, word_length=6).fit(X_train, y_train)
    euclidean = KNearestNeighborsClassifier(k=1).fit(X_train, y_train)

    boss_accuracy = (boss.predict(X_test) == y_test).mean()
    euclidean_accuracy = (euclidean.predict(X_test) == y_test).mean()

    assert boss_accuracy > 0.75
    assert boss_accuracy > euclidean_accuracy


def test_boss_distance_is_asymmetric():
    """The measure only counts words the *query* has, which breaks symmetry."""
    from tuiml.algorithms.timeseries.classification.dictionary import _boss_distance

    query = np.array([2.0, 0.0, 1.0])
    reference = np.array([[2.0, 9.0, 1.0]])

    # The reference's extra word (index 1) is invisible from the query's side.
    assert _boss_distance(query, reference)[0] == pytest.approx(0.0)
    # From the other direction it dominates.
    assert _boss_distance(reference[0], query[None, :])[0] == pytest.approx(81.0)


def test_boss_applies_numerosity_reduction():
    """A constant stretch must not swamp the histogram by sheer duration."""
    from tuiml.algorithms.timeseries.classification import BOSSClassifier

    rng = np.random.default_rng(0)
    varied = rng.normal(size=(4, 200))
    # A long flat run produces the same word over and over.
    flat = np.concatenate([rng.normal(size=(4, 40)), np.zeros((4, 160))], axis=1)

    X = np.vstack([varied, flat])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = BOSSClassifier(window_size=30, word_length=4).fit(X, y)

    histograms = model.transform(X)
    n_windows = X.shape[1] - model.window_size_ + 1
    # Without numerosity reduction each series would contribute n_windows
    # counts; collapsing runs must leave the flat series well below that.
    assert histograms[4:].sum(axis=1).max() < n_windows
    assert histograms.sum() > 0


def test_boss_breakpoints_are_strictly_increasing():
    """Tied quantiles would silently shrink the alphabet; they are nudged apart."""
    from tuiml.algorithms.timeseries.classification import BOSSClassifier

    rng = np.random.default_rng(0)
    # A largely constant panel produces heavily tied coefficients.
    X = np.zeros((10, 120))
    X[:, ::20] = rng.normal(size=(10, 6))
    y = np.arange(10) % 2

    model = BOSSClassifier(window_size=30, word_length=4, alphabet_size=4).fit(X, y)
    assert model.breakpoints_.shape == (4, 3)
    assert np.all(np.diff(model.breakpoints_, axis=1) > 0)


def test_boss_histograms_align_with_the_vocabulary(motif_count):
    """transform() returns counts over the fitted vocabulary, unseen words dropped."""
    from tuiml.algorithms.timeseries.classification import BOSSClassifier

    (X_train, y_train), (X_test, _) = motif_count
    model = BOSSClassifier(window_size=40, word_length=6).fit(X_train, y_train)

    histograms = model.transform(X_test)
    assert histograms.shape == (len(X_test), len(model.vocabulary_))
    assert np.all(histograms >= 0)
    assert np.all(np.isfinite(histograms))
    assert np.all(np.diff(model.vocabulary_) > 0)  # sorted, for searchsorted


def test_boss_predict_proba_is_a_distribution(motif_count):
    """Vote shares are non-negative and sum to one."""
    from tuiml.algorithms.timeseries.classification import BOSSClassifier

    (X_train, y_train), (X_test, _) = motif_count
    model = BOSSClassifier(window_size=40, word_length=6, n_neighbors=3).fit(
        X_train, y_train
    )
    proba = model.predict_proba(X_test[:20])

    assert proba.shape == (20, len(model.classes_))
    assert np.all(proba >= 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_boss_handles_multivariate_input():
    """Channels are pooled into one bag of patterns."""
    from tuiml.algorithms.timeseries.classification import BOSSClassifier

    rng = np.random.default_rng(0)
    t = np.linspace(0, 8 * np.pi, 160)
    slow = np.stack([np.sin(t), np.cos(t)])[None].repeat(20, 0)
    fast = np.stack([np.sin(4 * t), np.cos(4 * t)])[None].repeat(20, 0)
    X = np.concatenate([slow, fast]) + rng.normal(0, 0.2, (40, 2, 160))
    y = np.array([0] * 20 + [1] * 20)

    model = BOSSClassifier(window_size=40, word_length=6).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.9


def test_boss_invalid_arguments_rejected():
    """Degenerate configuration is caught at construction."""
    from tuiml.algorithms.timeseries.classification import BOSSClassifier

    with pytest.raises(ValueError, match="alphabet_size"):
        BOSSClassifier(alphabet_size=1)
    with pytest.raises(ValueError, match="word_length"):
        BOSSClassifier(word_length=0)
    with pytest.raises(ValueError, match="n_neighbors"):
        BOSSClassifier(n_neighbors=0)


# --------------------------------------------------------------------------
# Interval features and the time series forest
# --------------------------------------------------------------------------

def test_interval_features_match_numpy():
    """Prefix-sum statistics equal a direct per-interval computation.

    Widths on both sides of the direct/prefix-sum threshold are covered, plus
    a width-1 interval — the case where differencing prefix sums left a ~1e-14
    variance residue that sqrt turned into a ~1e-7 error in the standard
    deviation.
    """
    rng = np.random.default_rng(0)
    for scale, offset in ((1.0, 0.0), (3.0, 2.0), (1.0, 1e6)):
        X = rng.normal(size=(6, 300)) * scale + offset
        starts = np.array([0, 10, 50, 0, 297, 100, 5], dtype=np.int32)
        ends = np.array([300, 40, 51, 2, 300, 280, 6], dtype=np.int32)

        actual = np.asarray(cpp_ts.interval_features(X, starts, ends))
        assert actual.shape == (6, len(starts) * 3)

        for i in range(6):
            for k, (a, b) in enumerate(zip(starts, ends)):
                segment = X[i, a:b].astype(np.float64)
                time = np.arange(a, b, dtype=np.float64)

                assert actual[i, k * 3] == pytest.approx(segment.mean(), abs=1e-6)
                assert actual[i, k * 3 + 1] == pytest.approx(segment.std(), abs=1e-9)

                if len(segment) > 1:
                    centred_time = time - time.mean()
                    slope = float(
                        (centred_time * (segment - segment.mean())).sum()
                        / (centred_time ** 2).sum()
                    )
                else:
                    slope = 0.0
                assert actual[i, k * 3 + 2] == pytest.approx(slope, abs=1e-9)


def test_interval_std_of_a_single_point_is_exactly_zero():
    """A width-1 interval has zero spread and must report exactly that."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 200)) * 5 + 100
    features = np.asarray(
        cpp_ts.interval_features(
            X, np.array([7], np.int32), np.array([8], np.int32)
        )
    )
    np.testing.assert_array_equal(features[:, 1], 0.0)
    np.testing.assert_array_equal(features[:, 2], 0.0)


@pytest.fixture
def localised_trend():
    """Return data whose classes differ only in the middle of the series."""

    def build(n, seed):
        """Generate n series, adding a ramp to one class in a fixed window."""
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1.0, (n, 200))
        y = np.arange(n) % 2
        X[y == 1, 60:120] += np.linspace(0, 3, 60)
        return X, y

    return build(120, 0), build(120, 1)


def test_time_series_forest_finds_a_localised_trend(localised_trend):
    """The interval view's design case: a difference confined to one stretch."""
    from tuiml.algorithms.timeseries.classification import TimeSeriesForestClassifier

    (X_train, y_train), (X_test, y_test) = localised_trend
    model = TimeSeriesForestClassifier(n_estimators=100, random_state=0).fit(
        X_train, y_train
    )
    assert (model.predict(X_test) == y_test).mean() > 0.95


def test_time_series_forest_intervals_are_well_formed(localised_trend):
    """Sampled intervals lie inside the series and meet the minimum width."""
    from tuiml.algorithms.timeseries.classification import TimeSeriesForestClassifier

    (X, y), _ = localised_trend
    model = TimeSeriesForestClassifier(
        n_intervals=25, min_interval=5, n_estimators=20, random_state=0
    ).fit(X, y)

    assert model.intervals_.shape == (25, 2)
    assert np.all(model.intervals_[:, 0] >= 0)
    assert np.all(model.intervals_[:, 1] <= X.shape[1])
    assert np.all(model.intervals_[:, 1] - model.intervals_[:, 0] >= 5)
    assert model.transform(X).shape == (len(X), 25 * 3)


def test_time_series_forest_multivariate_concatenates_channels():
    """Every channel contributes the same intervals."""
    from tuiml.algorithms.timeseries.classification import TimeSeriesForestClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3, 120))
    y = np.arange(30) % 2
    model = TimeSeriesForestClassifier(
        n_intervals=10, n_estimators=20, random_state=0
    ).fit(X, y)
    assert model.transform(X).shape == (30, 10 * 3 * 3)


# --------------------------------------------------------------------------
# HIVE-COTE
# --------------------------------------------------------------------------

def test_hive_cote_weights_reflect_component_competence(localised_trend):
    """A component that cross-validates worse must receive a smaller weight."""
    from tuiml.algorithms.timeseries.classification import (
        BOSSClassifier,
        HIVECOTEClassifier,
        MiniRocketClassifier,
    )

    (X, y), _ = localised_trend
    model = HIVECOTEClassifier(
        components=[
            ("rocket", MiniRocketClassifier(n_features=840, random_state=0)),
            ("dictionary", BOSSClassifier(window_size=40, word_length=4)),
        ],
        cv=2,
        random_state=0,
    ).fit(X, y)

    assert set(model.component_accuracy_) == {"rocket", "dictionary"}
    assert np.isclose(model.weights_.sum(), 1.0)

    accuracies = [model.component_accuracy_[n] for n, _ in model.components_]
    # Weight order must follow accuracy order, which is the whole mechanism.
    assert np.argmax(model.weights_) == int(np.argmax(accuracies))


def test_hive_cote_tracks_its_best_component(localised_trend):
    """The ensemble should land at or near the best member without being told."""
    from tuiml.algorithms.timeseries.classification import (
        HIVECOTEClassifier,
        MiniRocketClassifier,
        TimeSeriesForestClassifier,
    )

    (X_train, y_train), (X_test, y_test) = localised_trend
    specification = [
        ("rocket", MiniRocketClassifier(n_features=840, random_state=0)),
        ("interval", TimeSeriesForestClassifier(n_estimators=50, random_state=0)),
    ]

    individual = []
    for _, component in specification:
        import copy as copy_module

        fitted = copy_module.deepcopy(component).fit(X_train, y_train)
        individual.append((fitted.predict(X_test) == y_test).mean())

    ensemble = HIVECOTEClassifier(
        components=specification, cv=2, random_state=0
    ).fit(X_train, y_train)
    accuracy = (ensemble.predict(X_test) == y_test).mean()

    assert accuracy >= max(individual) - 0.05


def test_hive_cote_predict_proba_is_a_distribution(localised_trend):
    """The weighted combination stays a probability distribution."""
    from tuiml.algorithms.timeseries.classification import (
        HIVECOTEClassifier,
        MiniRocketClassifier,
        TimeSeriesForestClassifier,
    )

    (X, y), (X_test, _) = localised_trend
    model = HIVECOTEClassifier(
        components=[
            ("rocket", MiniRocketClassifier(n_features=840, random_state=0)),
            ("interval", TimeSeriesForestClassifier(n_estimators=50, random_state=0)),
        ],
        cv=2,
        random_state=0,
    ).fit(X, y)

    proba = model.predict_proba(X_test[:20])
    assert proba.shape == (20, len(model.classes_))
    assert np.all(proba >= 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_hive_cote_aligns_components_that_saw_fewer_classes():
    """A component missing a class must not shift the other columns.

    Cross-validation folds can leave a component without every class, so its
    predict_proba columns cannot be assumed to line up with the ensemble's.
    """
    from tuiml.algorithms.timeseries.classification import (
        HIVECOTEClassifier,
        MiniRocketClassifier,
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 80))
    y = np.arange(40) % 3
    X[y == 1, 20:40] += 3.0
    X[y == 2, 50:70] -= 3.0

    model = HIVECOTEClassifier(
        components=[
            ("a", MiniRocketClassifier(n_features=840, random_state=0)),
            ("b", MiniRocketClassifier(n_features=840, random_state=1)),
        ],
        cv=2,
        random_state=0,
    ).fit(X, y)

    class Partial:
        """A stand-in component that only ever saw two of the three classes."""

        classes_ = np.array([0, 2])

        def predict_proba(self, panel):
            """Return confident predictions over its two known classes."""
            return np.tile([0.25, 0.75], (len(panel), 1))

    aligned = model._aligned_proba(Partial(), np.zeros((5, 1, 80)))
    assert aligned.shape == (5, 3)
    np.testing.assert_allclose(aligned[:, 0], 0.25)
    np.testing.assert_allclose(aligned[:, 1], 0.0)   # the unseen class
    np.testing.assert_allclose(aligned[:, 2], 0.75)


def test_hive_cote_default_components_cover_every_representation():
    """The default ensemble is one member per view, which is the whole point."""
    from tuiml.algorithms.timeseries.classification import HIVECOTEClassifier

    names = [name for name, _ in HIVECOTEClassifier()._resolve_components()]
    assert names == ["rocket", "dictionary", "interval", "distance"]


def test_hive_cote_invalid_arguments_rejected():
    """A single component is not an ensemble."""
    from tuiml.algorithms.timeseries.classification import (
        HIVECOTEClassifier,
        MiniRocketClassifier,
    )

    with pytest.raises(ValueError, match="at least 2 components"):
        HIVECOTEClassifier(components=[("only", MiniRocketClassifier())])
    with pytest.raises(ValueError, match="cv"):
        HIVECOTEClassifier(cv=1)
    with pytest.raises(ValueError, match="alpha"):
        HIVECOTEClassifier(alpha=-1.0)
