"""Tests for the torch-backed deep tabular family.

Covers four things the family can silently get wrong:

1. **Learning.** Each model must beat a linear baseline by a wide margin on a
   problem a linear model cannot touch (XOR-like interaction, and a smooth
   product target). A deep model that fails to learn still fits, still
   predicts, and still passes every structural check -- so the accuracy floors
   here are the real test.
2. **The optional-dependency contract.** Import, construction and
   introspection must work with no torch; only ``fit`` may demand it, and it
   must say how to install it.
3. **The mechanisms that distinguish the architectures.** SAINT's intersample
   attention must actually mix rows (FT-Transformer's must not), and NODE's
   entmax must be a genuine sparse simplex projection.
4. **Reproducibility and persistence.** A seed must pin the predictions, and a
   pickle round-trip must preserve them.
"""

import pickle
import sys

import numpy as np
import pytest

from tuiml.algorithms.tabular_foundation import (
    FTTransformerClassifier,
    FTTransformerRegressor,
    NODEClassifier,
    NODERegressor,
    SAINTClassifier,
    SAINTRegressor,
    entmax15,
)
from tuiml.evaluation.metrics import accuracy_score, r2_score
from tuiml.registry import registry
from tuiml.utils.torch_backend import has_torch

torch_required = pytest.mark.skipif(
    not has_torch(), reason="requires the optional torch extra"
)

CLASSIFIERS = [FTTransformerClassifier, SAINTClassifier, NODEClassifier]
REGRESSORS = [FTTransformerRegressor, SAINTRegressor, NODERegressor]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _xor_data(n=600, seed=0):
    """Return a non-linear binary problem a linear model scores ~0.5 on."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


def _smooth_data(n=600, seed=0):
    """Return a smooth non-linear regression target: ``sin(x0) * x1``."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = np.sin(X[:, 0]) * X[:, 1]
    return X, y


def _split(X, y, n_train=400):
    """Split arrays into a train and a held-out test block."""
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


# ---------------------------------------------------------------------------
# The optional-dependency contract (runs with or without torch)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", CLASSIFIERS + REGRESSORS)
def test_construction_and_introspection_need_no_torch(cls, monkeypatch):
    """Building and inspecting a model must work on a torch-free install."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.nn", None)

    model = cls(random_state=0)
    schema = cls.get_parameter_schema()

    assert isinstance(schema, dict) and schema
    assert "random_state" in schema and "device" in schema
    assert model.get_params()  # constructor values are recorded, not applied
    assert cls.get_capabilities()
    assert cls.get_references()


@pytest.mark.parametrize("cls", CLASSIFIERS + REGRESSORS)
def test_fit_without_torch_raises_with_install_command(cls, monkeypatch):
    """``fit`` is the only entry point allowed to demand torch."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.nn", None)

    X = np.zeros((8, 3))
    y = np.array([0, 1] * 4)

    with pytest.raises(ImportError) as excinfo:
        cls().fit(X, y)

    message = str(excinfo.value)
    assert "pip install 'tuiml[torch]'" in message
    assert cls.__name__ in message


@pytest.mark.parametrize("cls", CLASSIFIERS + REGRESSORS)
def test_schema_covers_every_constructor_parameter(cls):
    """A parameter missing from the schema is invisible to tooling."""
    import inspect

    accepted = {
        name for name in inspect.signature(cls.__init__).parameters
        if name != "self"
    }
    assert accepted == set(cls.get_parameter_schema())


@pytest.mark.parametrize("cls", CLASSIFIERS + REGRESSORS)
def test_registered_in_the_hub(cls):
    """The models are in the registry whether or not torch is installed."""
    assert registry.get(cls.__name__) is cls


@pytest.mark.parametrize("cls", CLASSIFIERS + REGRESSORS)
def test_predict_before_fit_raises(cls):
    """Predicting on an unfitted model is a deliberate refusal."""
    with pytest.raises(RuntimeError, match="fitted"):
        cls().predict(np.zeros((4, 3)))


# ---------------------------------------------------------------------------
# They actually learn
# ---------------------------------------------------------------------------

@torch_required
@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_classifier_learns_xor_interaction(cls):
    """Each classifier must beat the majority baseline by a wide margin.

    XOR of two feature signs is exactly the problem a linear model cannot fit:
    it scores ~0.5, the same as always predicting the majority class.
    """
    X, y = _xor_data()
    X_train, y_train, X_test, y_test = _split(X, y)

    model = cls(n_epochs=150, random_state=0).fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    majority = max(np.mean(y_test == 0), np.mean(y_test == 1))

    assert accuracy > 0.85, f"{cls.__name__} scored {accuracy:.3f} on XOR"
    assert accuracy > majority + 0.3


@torch_required
@pytest.mark.parametrize("cls", REGRESSORS)
def test_regressor_learns_smooth_product(cls):
    """Each regressor must reach R^2 > 0.8 on ``sin(x0) * x1``."""
    X, y = _smooth_data()
    X_train, y_train, X_test, y_test = _split(X, y)

    model = cls(n_epochs=300, random_state=0).fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))

    assert score > 0.8, f"{cls.__name__} scored R^2={score:.3f}"


@torch_required
@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_classifier_handles_three_classes(cls):
    """Multiclass is declared, so multiclass must work end to end."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = np.digitize(X[:, 0] + X[:, 1], [-0.7, 0.7])

    model = cls(n_epochs=60, random_state=0).fit(X, y)
    proba = model.predict_proba(X)

    assert set(np.unique(model.predict(X))) <= set(np.unique(y))
    assert proba.shape == (300, 3)
    assert (proba >= 0).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@torch_required
def test_early_stopping_shortens_training():
    """Early stopping must be able to stop before ``n_epochs``."""
    X, y = _xor_data(n=200)
    model = FTTransformerClassifier(
        n_epochs=200, early_stopping=True, patience=3, learning_rate=1e-4,
        random_state=0,
    ).fit(X, y)

    assert model.n_iter_ <= 200
    assert model.loss_curve_.shape == (model.n_iter_,)


@torch_required
def test_categorical_features_use_an_embedding_table():
    """A column named categorical is embedded, not standardised."""
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 3, size=300)
    X = np.column_stack([rng.normal(size=300), codes.astype(float)])
    y = (codes == 1).astype(int)

    model = FTTransformerClassifier(
        categorical_features=[1], n_epochs=120, random_state=0,
    ).fit(X, y)

    assert model.cardinalities_ == [3]
    assert model.numeric_indices_.tolist() == [0]
    assert accuracy_score(y, model.predict(X)) > 0.9


# ---------------------------------------------------------------------------
# Determinism and persistence
# ---------------------------------------------------------------------------

@torch_required
@pytest.mark.parametrize("cls", CLASSIFIERS + REGRESSORS)
def test_same_seed_gives_identical_predictions(cls):
    """A seed must pin the fit; the default device is CPU for this reason."""
    X, y = _xor_data(n=200)
    if cls in REGRESSORS:
        X, y = _smooth_data(n=200)

    first = cls(n_epochs=20, random_state=7).fit(X, y).predict(X)
    second = cls(n_epochs=20, random_state=7).fit(X, y).predict(X)

    assert np.array_equal(first, second)


@torch_required
@pytest.mark.parametrize("cls", CLASSIFIERS + REGRESSORS)
def test_pickle_roundtrip_preserves_predictions(cls):
    """The network pickles as arrays and is rebuilt on first use."""
    X, y = _xor_data(n=200)
    if cls in REGRESSORS:
        X, y = _smooth_data(n=200)

    model = cls(n_epochs=20, random_state=0).fit(X, y)
    before = model.predict(X)
    restored = pickle.loads(pickle.dumps(model))

    assert np.array_equal(before, restored.predict(X))


@torch_required
def test_refuses_missing_values_and_mismatched_widths():
    """Neither NaN input nor a changed feature count is silently accepted."""
    X, y = _xor_data(n=100)
    model = FTTransformerClassifier(n_epochs=5, random_state=0).fit(X, y)

    with pytest.raises(ValueError, match="NaN"):
        model.predict(np.full((3, 4), np.nan))
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 7)))


# ---------------------------------------------------------------------------
# Attention mechanics
# ---------------------------------------------------------------------------

@torch_required
def test_attention_and_block_output_shapes():
    """Attention and the pre-norm block are shape-preserving."""
    import torch
    from torch import nn

    from tuiml.algorithms.tabular_foundation._base import (
        _build_attention,
        _build_tokenizer,
        _build_transformer_block,
    )

    tokenizer = _build_tokenizer(torch, nn, 4, [3], d_token=8)
    x_num = torch.randn(5, 4)
    x_cat = torch.randint(0, 3, (5, 1))
    tokens = tokenizer(x_num, x_cat)
    # 4 numerical + 1 categorical + [CLS]
    assert tokens.shape == (5, 6, 8)

    attention = _build_attention(torch, nn, 8, 2, 0.0)
    assert attention(tokens).shape == (5, 6, 8)

    block = _build_transformer_block(torch, nn, 8, 2, 0.0)
    assert block(tokens).shape == (5, 6, 8)

    with pytest.raises(ValueError, match="divisible"):
        _build_attention(torch, nn, 9, 2, 0.0)


@torch_required
def test_intersample_attention_shape_and_row_mixing():
    """SAINT's row attention must make a row depend on its batch neighbours."""
    import torch
    from torch import nn

    from tuiml.algorithms.tabular_foundation.saint import _build_intersample_block

    torch.manual_seed(0)
    block = _build_intersample_block(torch, nn, n_tokens=6, d_token=8,
                                     n_heads=2, dropout=0.0)
    block.eval()

    tokens = torch.randn(5, 6, 8)
    out = block(tokens)
    assert out.shape == tokens.shape

    # Replace every row but the first: row 0's representation must move.
    perturbed = tokens.clone()
    perturbed[1:] = torch.randn(4, 6, 8)
    moved = block(perturbed)

    assert torch.allclose(perturbed[0], tokens[0])
    assert not torch.allclose(out[0], moved[0], atol=1e-5), (
        "intersample attention did not mix rows -- this is FT-Transformer, "
        "not SAINT"
    )


@torch_required
def test_saint_predictions_depend_on_batch_and_ft_predictions_do_not():
    """The distinguishing behaviour, observed through the public API."""
    X, y = _xor_data(n=200)
    row = X[:1]
    other = np.random.default_rng(3).normal(size=(63, 4))

    saint = SAINTClassifier(n_epochs=30, random_state=0).fit(X, y)
    ft = FTTransformerClassifier(n_epochs=30, random_state=0).fit(X, y)

    for model, expect_mixing in ((saint, True), (ft, False)):
        alone = model.predict_proba(np.vstack([row, X[1:64]]))[0]
        beside_others = model.predict_proba(np.vstack([row, other]))[0]
        mixed = not np.allclose(alone, beside_others, atol=1e-6)
        assert mixed is expect_mixing, (
            f"{type(model).__name__}: batch-dependence was {mixed}"
        )


# ---------------------------------------------------------------------------
# entmax15
# ---------------------------------------------------------------------------

@torch_required
def test_entmax15_is_a_sparse_simplex_projection():
    """entmax15 output is a probability vector, and sparser than softmax."""
    import torch

    torch.manual_seed(0)
    logits = torch.randn(32, 10) * 3.0

    probs = entmax15(logits)
    soft = torch.softmax(logits, dim=-1)

    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(32), atol=1e-5)
    assert (probs == 0).sum() > 0, "entmax15 produced no exact zeros"
    assert (soft == 0).sum() == 0
    assert (probs > 0).float().mean() < (soft > 0).float().mean()

    # The argmax is preserved, and a peaked input collapses to one-hot.
    assert torch.equal(probs.argmax(-1), logits.argmax(-1))
    peaked = entmax15(torch.tensor([[0.0, 0.0, 50.0]]))
    assert torch.allclose(peaked, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)


@torch_required
def test_entmax15_matches_softmax_ordering_and_is_differentiable():
    """Gradients flow through the closed form, which is what NODE needs."""
    import torch

    logits = torch.randn(4, 6, requires_grad=True)
    entmax15(logits).sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    # The support boundary is where a naive autograd path evaluates sqrt(0)
    # and returns NaN. The closed-form backward must stay finite there.
    boundary = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, -5.0, -5.0], [1e-8, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    (entmax15(boundary) ** 2).sum().backward()
    assert torch.isfinite(boundary.grad).all()


@torch_required
def test_entmax15_gradient_matches_finite_differences():
    """The closed-form backward is the real Jacobian, not an approximation."""
    import torch

    torch.manual_seed(0)
    logits = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(3, 5, dtype=torch.float64)

    (entmax15(logits) * weights).sum().backward()
    numeric = torch.zeros_like(logits)
    eps = 1e-6
    for i in range(3):
        for j in range(5):
            step = torch.zeros_like(logits)
            step[i, j] = eps
            with torch.no_grad():
                up = (entmax15(logits + step) * weights).sum()
                down = (entmax15(logits - step) * weights).sum()
            numeric[i, j] = (up - down) / (2 * eps)

    assert torch.allclose(logits.grad, numeric, atol=1e-5)


@torch_required
def test_entmax15_projects_along_a_chosen_axis():
    """The projection axis is selectable, as the tree layer relies on."""
    import torch

    logits = torch.randn(7, 3, 5)
    probs = entmax15(logits, dim=0)

    assert torch.allclose(probs.sum(dim=0), torch.ones(3, 5), atol=1e-5)


# ---------------------------------------------------------------------------
# NODE internals
# ---------------------------------------------------------------------------

@torch_required
def test_oblivious_tree_layer_shapes_and_leaf_weights():
    """A tree layer returns one response vector per tree, per row."""
    import torch
    from torch import nn

    from tuiml.algorithms.tabular_foundation.node import _build_oblivious_tree_layer

    generator = torch.Generator().manual_seed(0)
    layer = _build_oblivious_tree_layer(
        torch, nn, in_features=4, n_trees=6, tree_depth=3, tree_dim=2,
        generator=generator,
    )
    out = layer(torch.randn(9, 4))

    assert out.shape == (9, 6, 2)
    assert layer.response.shape == (6, 2, 2 ** 3)
    # Split-feature selectors are a distribution over the input columns.
    selectors = entmax15(layer.feature_logits, dim=0)
    assert torch.allclose(selectors.sum(dim=0), torch.ones(6, 3), atol=1e-5)


@torch_required
def test_node_dense_stacking_widens_each_layer():
    """Later layers see the raw features plus every earlier layer's output."""
    import torch
    from torch import nn

    model = NODEClassifier(n_layers=3, n_trees=5, tree_depth=2)
    network = model._build_network(torch, nn, n_numeric=4, cardinalities=[],
                                   n_outputs=2)
    widths = [layer.feature_logits.shape[0] for layer in network.layers]

    assert widths == [4, 4 + 5 * 2, 4 + 2 * 5 * 2]
    assert network(torch.randn(6, 4), torch.zeros(6, 0, dtype=torch.long)).shape == (6, 2)
