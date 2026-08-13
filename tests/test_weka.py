"""Tests for the Weka bridge (:mod:`tuiml.weka`).

The whole module is skipped unless ``python-weka-wrapper3`` is importable and a
JVM can actually be started, so a checkout without the optional extra (or
without Java) still runs a green suite.

Notes
-----
The JVM cannot be restarted once stopped, so every test in this module shares
the single process-wide JVM that :func:`tuiml.weka.ensure_jvm` starts.
"""

import numpy as np
import pytest

pytest.importorskip("weka.core.jvm", reason="requires tuiml[weka]")

from tuiml.weka import ensure_jvm  # noqa: E402
from tuiml.weka._base import fmt_num, rows_to_instances, to_instances  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _jvm():
    """Start the shared JVM once, skipping the module if Java is unavailable."""
    try:
        ensure_jvm()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"cannot start the Weka JVM: {exc}")


@pytest.fixture
def iris():
    """Return the iris features and labels."""
    from tuiml.datasets import load_iris

    data = load_iris()
    return data.X, data.y


@pytest.fixture
def regression_data():
    """Return a small, strongly linear regression problem."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=80)
    return X, y


@pytest.fixture
def nominal_data():
    """Return a categorical problem whose target is determined by column 0."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, size=(200, 4)).astype(float)
    y = (X[:, 0] > 1).astype(int)
    return X, y


# =============================================================================
# Option formatting
# =============================================================================

class TestFmtNum:
    """Tests for :func:`~tuiml.weka._base.fmt_num`."""

    def test_whole_float_drops_decimal(self):
        """Whole floats render without ``.0`` so Integer.parseInt accepts them."""
        assert fmt_num(2.0) == "2"

    def test_int_unchanged(self):
        """Integers render as-is."""
        assert fmt_num(2) == "2"

    def test_fractional_preserved(self):
        """Fractional values keep their decimal part."""
        assert fmt_num(0.25) == "0.25"

    def test_negative_whole(self):
        """Negative whole values also drop the decimal."""
        assert fmt_num(-1.0) == "-1"

    def test_bool_is_not_treated_as_float(self):
        """``bool`` is an ``int`` subclass but must render as 0/1."""
        assert fmt_num(True) == "1"
        assert fmt_num(False) == "0"

    def test_reptree_accepts_formatted_min_num(self, iris):
        """REPTree's ``-M`` is Integer-parsed and rejects ``"2.0"``."""
        from tuiml.weka import REPTree

        X, y = iris
        REPTree(min_num=2.0).fit(X, y)  # would raise NumberFormatException


# =============================================================================
# Array <-> Instances conversion
# =============================================================================

class TestInstanceConversion:
    """Tests for the numpy/Weka data bridge."""

    def test_to_instances_sets_class_last(self, iris):
        """A supervised table carries a class attribute in the last column."""
        X, y = iris
        _, codes = np.unique(y, return_inverse=True)
        data = to_instances(X, codes.astype(float), nominal_target=True)
        assert data.num_instances == 150
        assert data.num_attributes == 5
        assert data.class_index == data.num_attributes - 1
        assert data.class_attribute.is_nominal

    def test_to_instances_without_y_has_no_class(self, iris):
        """Passing ``y=None`` builds a table with no class attribute at all."""
        X, _ = iris
        data = to_instances(X, None)
        assert data.num_attributes == 4
        assert data.class_index < 0

    def test_nominal_features_declared_nominal(self, nominal_data):
        """Declared columns become nominal attributes."""
        X, y = nominal_data
        data = to_instances(X, y.astype(float), nominal_features=[0, 1],
                            nominal_target=True)
        assert data.attribute(0).is_nominal
        assert data.attribute(1).is_nominal
        assert data.attribute(2).is_numeric

    def test_nominal_roundtrip_preserves_category(self, nominal_data):
        """A category survives the header round-trip as the same index.

        Regression test: the header stores a label built from a float column as
        ``"3.0"``, so matching it against ``str(int(3.0))`` == ``"3"`` misses
        every time and silently turns the column into missing values.
        """
        X, y = nominal_data
        header = to_instances(X, y.astype(float), nominal_features=[0, 1, 2, 3],
                              nominal_target=True)
        test = rows_to_instances(header, X[:5])
        got = [test.get_instance(i).get_value(0) for i in range(5)]
        assert got == pytest.approx(X[:5, 0])

    def test_unseen_category_becomes_missing(self, nominal_data):
        """A category absent from training is written as missing, not remapped."""
        X, y = nominal_data
        header = to_instances(X, y.astype(float), nominal_features=[0],
                              nominal_target=True)
        unseen = X[:3].copy()
        unseen[:, 0] = 99.0
        test = rows_to_instances(header, unseen)
        assert all(np.isnan(test.get_instance(i).get_value(0)) for i in range(3))

    def test_numeric_columns_pass_through(self, iris):
        """Numeric columns keep their exact values."""
        X, y = iris
        _, codes = np.unique(y, return_inverse=True)
        header = to_instances(X, codes.astype(float), nominal_target=True)
        test = rows_to_instances(header, X[:4])
        got = np.array([[test.get_instance(i).get_value(j) for j in range(4)]
                        for i in range(4)])
        assert got == pytest.approx(X[:4])


# =============================================================================
# Classifiers
# =============================================================================

CLASSIFIERS = [
    "J48", "REPTree", "RandomTree", "RandomForest", "DecisionStump", "LMT",
    "ZeroR", "OneR", "JRip", "PART", "DecisionTable", "IBk", "KStar",
    "NaiveBayes", "SMO", "Logistic", "SimpleLogistic", "AdaBoostM1", "Bagging",
    "LogitBoost", "RandomCommittee", "RandomSubSpace", "MultiClassClassifier",
    "FilteredClassifier", "Vote",
]


class TestClassifiers:
    """Every classifier wrapper fits, predicts and yields valid probabilities."""

    @pytest.mark.parametrize("name", CLASSIFIERS)
    def test_fit_predict_shape(self, name, iris):
        """Predictions have one entry per row, in the original label space."""
        import tuiml.weka as W

        X, y = iris
        model = getattr(W, name)().fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (150,)
        assert set(np.unique(pred)).issubset(set(np.unique(y)))

    @pytest.mark.parametrize("name", CLASSIFIERS)
    def test_predict_proba_is_a_distribution(self, name, iris):
        """Each probability row is non-negative and sums to one."""
        import tuiml.weka as W

        X, y = iris
        model = getattr(W, name)().fit(X, y)
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 3)
        assert (proba >= 0).all()
        assert proba.sum(axis=1) == pytest.approx(1.0)

    def test_beats_majority_baseline(self, iris):
        """A real learner clears the ZeroR floor by a wide margin."""
        from tuiml.weka import J48, ZeroR

        X, y = iris
        floor = (ZeroR().fit(X, y).predict(X) == y).mean()
        tree = (J48().fit(X, y).predict(X) == y).mean()
        assert floor == pytest.approx(1 / 3, abs=0.02)
        assert tree > 0.9

    def test_string_labels_round_trip(self):
        """Non-integer labels come back in their original form."""
        from tuiml.weka import J48

        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 2))
        y = np.where(X[:, 0] > 0, "yes", "no")
        pred = J48().fit(X, y).predict(X)
        assert set(np.unique(pred)).issubset({"yes", "no"})

    def test_proba_columns_align_with_classes(self, iris):
        """``predict_proba`` columns follow ``classes_``, matching ``predict``."""
        from tuiml.weka import J48

        X, y = iris
        model = J48().fit(X, y)
        proba = model.predict_proba(X)
        assert np.array_equal(model.classes_[proba.argmax(axis=1)], model.predict(X))

    def test_nominal_features_are_honoured(self, nominal_data):
        """Declaring the categorical columns recovers a deterministic target.

        Regression test for the label-lookup bug: with the columns silently
        turned into missing values this scored ~0.57 instead of 1.0.
        """
        from tuiml.weka import J48

        X, y = nominal_data
        acc = (J48(nominal_features=[0, 1, 2, 3]).fit(X, y).predict(X) == y).mean()
        assert acc > 0.95

    def test_bayesnet_on_nominal_data(self, nominal_data):
        """BayesNet needs nominal attributes and learns the dependency."""
        from tuiml.weka import BayesNet

        X, y = nominal_data
        acc = (BayesNet(nominal_features=[0, 1, 2, 3]).fit(X, y).predict(X) == y).mean()
        assert acc > 0.95

    def test_stacking_default_folds(self, iris):
        """Stacking at its default fold count performs well."""
        from tuiml.weka import Stacking

        X, y = iris
        assert (Stacking().fit(X, y).predict(X) == y).mean() > 0.9

    def test_simple_logistic_is_not_degenerate(self, iris):
        """SimpleLogistic must not collapse to the majority class.

        Regression test: Weka's ``-W`` sets the weight-trimming beta, not the
        early-stopping heuristic. Emitting ``-W 50`` trimmed the model away and
        left accuracy at the ZeroR floor.
        """
        from tuiml.weka import SimpleLogistic

        X, y = iris
        assert (SimpleLogistic().fit(X, y).predict(X) == y).mean() > 0.9


# =============================================================================
# Regressors
# =============================================================================

REGRESSORS = [
    "M5P", "M5Rules", "LWL", "SMOreg", "LinearRegression",
    "SimpleLinearRegression", "GaussianProcesses", "AdditiveRegression",
    "RegressionByDiscretization",
]


class TestRegressors:
    """Every regressor wrapper fits and tracks a linear signal."""

    @pytest.mark.parametrize("name", REGRESSORS)
    def test_fit_predict(self, name, regression_data):
        """Predictions are one float per row and correlate with the target."""
        import tuiml.weka as W

        X, y = regression_data
        pred = getattr(W, name)().fit(X, y).predict(X)
        assert pred.shape == (80,)
        assert pred.dtype.kind == "f"
        assert np.corrcoef(pred, y)[0, 1] > 0.7

    def test_predict_proba_rejected(self, regression_data):
        """A regressor has no ``predict_proba``."""
        from tuiml.weka import M5P

        X, y = regression_data
        model = M5P().fit(X, y)
        with pytest.raises(AttributeError, match="regressor"):
            model.predict_proba(X)


# =============================================================================
# Clusterers
# =============================================================================

CLUSTERERS = ["SimpleKMeans", "EM", "Canopy", "FarthestFirst",
              "HierarchicalClusterer"]


class TestClusterers:
    """Every clusterer wrapper assigns labels to train and unseen rows."""

    @pytest.mark.parametrize("name", CLUSTERERS)
    def test_fit_labels(self, name, iris):
        """``fit`` records one label per training row."""
        import tuiml.weka as W

        X, _ = iris
        model = getattr(W, name)(n_clusters=3).fit(X)
        assert model.labels_.shape == (150,)
        assert model.n_clusters_ == 3
        assert model.labels_.min() >= 0

    @pytest.mark.parametrize("name", CLUSTERERS)
    def test_predict_unseen(self, name, iris):
        """``predict`` assigns clusters to new rows."""
        import tuiml.weka as W

        X, _ = iris
        model = getattr(W, name)(n_clusters=3).fit(X)
        assert model.predict(X[:10]).shape == (10,)

    def test_cobweb_finds_its_own_k(self, iris):
        """Cobweb derives the cluster count from the data."""
        from tuiml.weka import Cobweb

        X, _ = iris
        model = Cobweb().fit(X)
        assert model.labels_.shape == (150,)
        assert model.n_clusters_ >= 1


# =============================================================================
# Hub registration
# =============================================================================

class TestHubRegistration:
    """The wrappers are addressable by ``weka.<ClassName>`` hub key."""

    def test_keys_are_namespaced(self):
        """Every wrapper registers under the ``weka.`` prefix."""
        import tuiml

        keys = [a["name"] for a in tuiml.list_algorithms()
                if a["name"].startswith("weka.")]
        assert len(keys) >= 40
        assert "weka.J48" in keys
        assert "weka.SimpleKMeans" in keys

    def test_no_collision_with_native_names(self):
        """``RandomForest`` exists in both namespaces without clashing."""
        import tuiml

        names = {a["name"] for a in tuiml.list_algorithms()}
        assert "weka.RandomForest" in names
        assert "RandomForestClassifier" in names

    def test_train_by_hub_key(self):
        """:func:`tuiml.train` accepts a namespaced Weka key."""
        import tuiml

        model = tuiml.train({
            "model": {"name": "weka.J48"},
            "data": {"source": "iris", "target": "class"},
            "evaluation": {"test_size": 0.3},
        })
        assert model.metrics_["accuracy_score"] > 0.8

    def test_train_with_params(self):
        """Constructor parameters travel through the spec."""
        import tuiml

        model = tuiml.train({
            "model": {"name": "weka.RandomForest",
                      "params": {"num_iterations": 5, "seed": 42}},
            "data": {"source": "iris", "target": "class"},
            "evaluation": {"test_size": 0.3},
        })
        assert model.metrics_["accuracy_score"] > 0.8


# =============================================================================
# Model inspection and options
# =============================================================================

class TestModelAccess:
    """The escape hatches onto Weka itself."""

    def test_to_weka_string(self, iris):
        """``to_weka_string`` returns Weka's own model dump."""
        from tuiml.weka import J48

        X, y = iris
        text = J48().fit(X, y).to_weka_string()
        assert "J48" in text
        assert "Number of Leaves" in text

    def test_raw_options_are_appended(self, iris):
        """The ``options`` escape hatch reaches the backing learner."""
        from tuiml.weka import J48

        X, y = iris
        model = J48(options=["-A"])
        assert "-A" in model._resolved_options()
        model.fit(X, y)

    def test_kernel_options_reach_the_kernel(self, iris):
        """``kernel_options`` are embedded in the ``-K`` value."""
        from tuiml.weka import SMO

        X, y = iris
        model = SMO(kernel="weka.classifiers.functions.supportVector.RBFKernel",
                    kernel_options=["-G", "0.05"])
        assert any("RBFKernel -G 0.05" in tok for tok in model._resolved_options())
        model.fit(X, y)

    def test_base_learner_options_are_nested(self):
        """A meta-learner routes base options after ``--``."""
        from tuiml.weka.meta import _base_spec

        assert _base_spec("weka.classifiers.trees.J48", ["-C", "0.1"]) == [
            "-W", "weka.classifiers.trees.J48", "--", "-C", "0.1"]

    def test_meta_base_classifier_is_used(self, iris):
        """A meta-learner accepts a non-default base learner."""
        from tuiml.weka import Bagging

        X, y = iris
        model = Bagging(base_classifier="weka.classifiers.trees.J48",
                        num_iterations=5).fit(X, y)
        assert (model.predict(X) == y).mean() > 0.9

    def test_repr_names_the_weka_class(self):
        """``repr`` identifies the backing Weka class and fit state."""
        from tuiml.weka import J48

        assert "weka.classifiers.trees.J48" in repr(J48())
        assert "not fitted" in repr(J48())


class TestInvalidOptions:
    """Invalid parameter values fail with a clear Python error."""

    def test_bad_distance_weighting(self, iris):
        """IBk rejects an unknown weighting scheme."""
        from tuiml.weka import IBk

        X, y = iris
        with pytest.raises(ValueError, match="distance_weighting"):
            IBk(distance_weighting="nope").fit(X, y)

    def test_bad_missing_mode(self, iris):
        """KStar rejects an unknown missing-value mode."""
        from tuiml.weka import KStar

        X, y = iris
        with pytest.raises(ValueError, match="missing_mode"):
            KStar(missing_mode="nope").fit(X, y)

    def test_bad_combination_rule(self, iris):
        """Vote rejects an unknown combination rule."""
        from tuiml.weka import Vote

        X, y = iris
        with pytest.raises(ValueError, match="combination_rule"):
            Vote(combination_rule="nope").fit(X, y)

    def test_naive_bayes_mutually_exclusive_options(self, iris):
        """NaiveBayes rejects asking for both numeric strategies at once."""
        from tuiml.weka import NaiveBayes

        X, y = iris
        with pytest.raises(ValueError, match="mutually exclusive"):
            NaiveBayes(use_kernel_estimator=True,
                       use_supervised_discretization=True).fit(X, y)

    def test_bad_attribute_selection(self, regression_data):
        """LinearRegression rejects an unknown selection method."""
        from tuiml.weka import LinearRegression

        X, y = regression_data
        with pytest.raises(ValueError, match="attribute_selection"):
            LinearRegression(attribute_selection="nope").fit(X, y)
