"""NODE: Neural Oblivious Decision Ensembles.

Implements the architecture of Popov et al. (ICLR 2020), *Neural Oblivious
Decision Ensembles for Deep Learning on Tabular Data*. A NODE layer is an
ensemble of **oblivious** decision trees -- trees that use the same split
feature and threshold at every node of a given depth, so a tree of depth
:math:`D` is a lookup table with :math:`2^D` leaves -- made differentiable by
replacing hard splits with :math:`\\alpha`-entmax. Layers are stacked
DenseNet-style, each seeing the raw features plus every earlier layer's output.

This module also contains a dependency-free implementation of
:func:`entmax15`, the sparse alternative to softmax that gives the trees their
near-hard splits and near-hard feature choices.

PyTorch is an optional dependency -- ``pip install 'tuiml[torch]'``. Nothing in
this module imports torch until it is called.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.algorithms.tabular_deep._base import (
    _DeepTabularClassifierMixin,
    _DeepTabularRegressorMixin,
    _training_schema,
)


def entmax15(inputs, dim: int = -1):
    """Project logits onto the simplex with 1.5-entmax: a **sparse** softmax.

    :math:`\\alpha`-entmax interpolates between softmax (:math:`\\alpha = 1`,
    always dense) and sparsemax (:math:`\\alpha = 2`, aggressively sparse).
    At :math:`\\alpha = 1.5` the solution has a closed form found by sorting,
    and -- unlike softmax -- it assigns **exactly zero** to low-scoring
    coordinates, which is what lets a NODE tree commit to one feature and one
    side of a split while staying differentiable.

    Parameters
    ----------
    inputs : torch.Tensor
        Logits. Any shape; the projection is applied along ``dim``.
    dim : int, default=-1
        Axis to project over.

    Returns
    -------
    probabilities : torch.Tensor
        Same shape as ``inputs``. Non-negative and summing to one along
        ``dim``, with exact zeros outside the support.

    Notes
    -----
    Solves

    .. math::
        \\mathrm{entmax}_{1.5}(z) =
        \\big[(\\alpha - 1) z - \\tau \\mathbf{1}\\big]_{+}^{1/(\\alpha - 1)}
        \\Big|_{\\alpha = 1.5}
        = \\big[z/2 - \\tau\\big]_{+}^{2}

    where the threshold :math:`\\tau` is chosen so the result sums to one. The
    exact algorithm sorts the scores and walks the candidate support sizes,
    costing :math:`O(k \\log k)` for :math:`k` coordinates.

    The backward pass uses the closed form rather than autograd through the
    sort. That is not an optimisation: at the edge of the support the
    threshold search evaluates :math:`\\sqrt{0}`, whose derivative is
    infinite, so differentiating the search itself produces NaN weights a few
    hundred steps into training. The exact Jacobian-vector product is

    .. math::
        \\nabla_z \\mathcal{L} = s \\odot
        \\left(g - \\frac{\\langle g, s \\rangle}
        {\\langle s, \\mathbf{1} \\rangle} \\mathbf{1}\\right),
        \\quad s = \\sqrt{p},

    for upstream gradient :math:`g`, which is finite everywhere.

    Requires PyTorch: ``pip install 'tuiml[torch]'``.

    References
    ----------
    .. [Peters2019] Peters, B., Niculae, V., & Martins, A. F. T. (2019).
       Sparse Sequence-to-Sequence Models. *Proceedings of ACL 2019*,
       1504-1519. :doi:`10.18653/v1/P19-1146`

    Examples
    --------
    >>> from tuiml.algorithms.tabular_deep.node import entmax15
    >>> from tuiml.utils.torch_backend import has_torch
    >>> if has_torch():
    ...     import torch
    ...     p = entmax15(torch.tensor([[1.0, 2.0, 9.0]]))
    ...     print(bool(torch.allclose(p.sum(-1), torch.ones(1))), bool(p[0, 0] == 0))
    ... else:
    ...     print(True, True)
    True True
    """
    from tuiml.utils.torch_backend import require_torch

    torch, _nn = require_torch("entmax15")
    return _entmax15_autograd_function(torch).apply(inputs, dim)


def _entmax15_project(torch, inputs, dim: int):
    """Project pre-scaled logits onto the simplex by the sorting algorithm.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.
    inputs : torch.Tensor
        Logits, already halved and shifted by their maximum.
    dim : int
        Axis to project over.

    Returns
    -------
    probabilities : torch.Tensor
        The 1.5-entmax projection of ``inputs``.
    """
    sorted_inputs, _ = torch.sort(inputs, dim=dim, descending=True)
    k = inputs.shape[dim]
    shape = [1] * inputs.dim()
    shape[dim] = k
    rho = torch.arange(
        1, k + 1, device=inputs.device, dtype=inputs.dtype
    ).view(shape)

    mean = sorted_inputs.cumsum(dim) / rho
    mean_sq = (sorted_inputs ** 2).cumsum(dim) / rho
    # Feasibility test for each candidate support size, from the variance of
    # the top-k scores.
    ss = rho * (mean_sq - mean ** 2)
    delta = torch.clamp((1.0 - ss) / rho, min=0.0)
    tau = mean - torch.sqrt(delta)

    # At least one coordinate is always in the support; the clamp guards the
    # gather against a degenerate (all-equal or non-finite) row.
    support_size = torch.clamp((tau <= sorted_inputs).sum(dim=dim, keepdim=True), min=1)
    tau_star = tau.gather(dim, support_size - 1)
    return torch.clamp(inputs - tau_star, min=0.0) ** 2


#: Built once per process; ``torch.autograd.Function`` cannot be subclassed at
#: module scope because torch may not be importable there.
_entmax15_function = None


def _entmax15_autograd_function(torch):
    """Return the autograd Function implementing 1.5-entmax.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.

    Returns
    -------
    function : type
        A ``torch.autograd.Function`` subclass with the exact entmax backward.
    """
    global _entmax15_function
    if _entmax15_function is not None:
        return _entmax15_function

    class _Entmax15(torch.autograd.Function):
        """1.5-entmax with the closed-form (finite) backward pass."""

        @staticmethod
        def forward(ctx, inputs, dim):
            """Project ``inputs`` onto the simplex along ``dim``."""
            ctx.dim = dim
            shifted = (inputs - inputs.max(dim=dim, keepdim=True).values) / 2.0
            output = _entmax15_project(torch, shifted, dim)
            ctx.save_for_backward(output)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """Apply the exact Jacobian-vector product of the projection."""
            (output,) = ctx.saved_tensors
            dim = ctx.dim
            root = output.sqrt()
            grad = grad_output * root
            correction = grad.sum(dim) / root.sum(dim)
            grad = grad - correction.unsqueeze(dim) * root
            return grad, None

    _entmax15_function = _Entmax15
    return _entmax15_function


def _build_oblivious_tree_layer(torch, nn, in_features: int, n_trees: int,
                                tree_depth: int, tree_dim: int,
                                generator=None):
    """Build one differentiable ensemble of oblivious decision trees.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.
    nn : module
        ``torch.nn``.
    in_features : int
        Width of the layer input.
    n_trees : int
        Trees in this layer.
    tree_depth : int
        Depth of every tree; each has ``2 ** tree_depth`` leaves.
    tree_dim : int
        Response width per leaf.
    generator : torch.Generator, optional
        Seeded generator for parameter initialisation.

    Returns
    -------
    layer : torch.nn.Module
        Module mapping ``(batch, in_features)`` to
        ``(batch, n_trees, tree_dim)``.
    """

    class _ObliviousTreeEnsemble(nn.Module):
        """``n_trees`` oblivious trees evaluated as soft lookup tables."""

        def __init__(self):
            super().__init__()
            self.n_trees = n_trees
            self.tree_depth = tree_depth
            self.tree_dim = tree_dim

            def randn(*size, std=1.0):
                return torch.randn(*size, generator=generator) * std

            # Which feature each depth level of each tree splits on.
            self.feature_logits = nn.Parameter(
                randn(in_features, n_trees, tree_depth, std=1.0)
            )
            # Threshold and sharpness of each split.
            self.thresholds = nn.Parameter(randn(n_trees, tree_depth, std=1.0))
            self.log_temperatures = nn.Parameter(
                torch.zeros(n_trees, tree_depth)
            )
            # Leaf responses: one vector per leaf of every tree.
            self.response = nn.Parameter(
                randn(n_trees, tree_dim, 2 ** tree_depth, std=0.1)
            )

        def forward(self, x):
            """Route a batch through every tree and read the leaf responses."""
            # Near-one-hot choice of split feature per (tree, depth).
            selectors = entmax15(self.feature_logits, dim=0)
            values = torch.einsum("bi,itd->btd", x, selectors)

            logits = (values - self.thresholds) * torch.exp(-self.log_temperatures)
            # Soft binary decision: [go right, go left] per (tree, depth).
            choices = entmax15(torch.stack([logits, -logits], dim=-1), dim=-1)

            # Outer product across depths gives the leaf membership weights.
            weights = choices[:, :, 0, :]
            for depth in range(1, self.tree_depth):
                weights = (
                    weights.unsqueeze(-1) * choices[:, :, depth, :].unsqueeze(-2)
                ).flatten(start_dim=-2)

            return torch.einsum("btl,tol->bto", weights, self.response)

    return _ObliviousTreeEnsemble()


class _NODECore:
    """Constructor, schema and architecture shared by the two NODE models.

    Internal mixin: not registered, no task behaviour of its own.
    """

    def __init__(
        self,
        n_layers: int = 1,
        n_trees: int = 32,
        tree_depth: int = 4,
        learning_rate: float = 1e-2,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        n_epochs: int = 60,
        early_stopping: bool = False,
        validation_fraction: float = 0.15,
        patience: int = 10,
        device: str = "cpu",
        random_state: Optional[int] = 0,
    ):
        """Record hyperparameters. Never imports torch (see the class docs)."""
        super().__init__()
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.device = device
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        schema: Dict[str, Dict[str, Any]] = {
            "n_layers": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 16,
                "description": "Number of densely connected tree-ensemble layers",
            },
            "n_trees": {
                "type": "integer",
                "default": 32,
                "minimum": 1,
                "maximum": 4096,
                "description": "Oblivious trees per layer",
            },
            "tree_depth": {
                "type": "integer",
                "default": 4,
                "minimum": 1,
                "maximum": 12,
                "description": "Depth of each tree; leaves are 2 ** tree_depth",
            },
        }
        schema.update(_training_schema())
        # NODE trains well with a larger step size than the attention models.
        schema["learning_rate"]["default"] = 1e-2
        schema["weight_decay"]["default"] = 0.0
        return schema

    def _build_network(self, torch, nn, n_numeric: int,
                       cardinalities: Sequence[int], n_outputs: int):
        """Stack the tree-ensemble layers with dense (concatenating) links.

        Parameters
        ----------
        torch : module
            The imported ``torch`` module.
        nn : module
            ``torch.nn``.
        n_numeric : int
            Number of input columns; NODE treats them all as numerical.
        cardinalities : sequence of int
            Unused; NODE has no categorical embedding path.
        n_outputs : int
            Number of output units, used as the per-leaf response width.

        Returns
        -------
        network : torch.nn.Module
            Module mapping ``(x_num, x_cat)`` to ``(batch, n_outputs)``.
        """
        seed = 0 if self.random_state is None else int(self.random_state)
        generator = torch.Generator().manual_seed(seed)

        n_layers = int(self.n_layers)
        n_trees = int(self.n_trees)
        tree_dim = int(n_outputs)

        layers = nn.ModuleList()
        width = n_numeric
        for _ in range(n_layers):
            layers.append(
                _build_oblivious_tree_layer(
                    torch, nn, width, n_trees, int(self.tree_depth), tree_dim,
                    generator=generator,
                )
            )
            width += n_trees * tree_dim  # DenseNet-style concatenation

        class _NODENet(nn.Module):
            """Densely stacked oblivious tree ensembles, averaged at the end."""

            def __init__(self):
                super().__init__()
                self.layers = layers
                self.bias = nn.Parameter(torch.zeros(tree_dim))

            def forward(self, x_num, x_cat):
                """Map one batch of rows to output units."""
                features = x_num
                outputs = []
                for layer in self.layers:
                    out = layer(features)                      # (b, trees, dim)
                    outputs.append(out)
                    features = torch.cat(
                        [features, out.flatten(start_dim=1)], dim=1
                    )
                stacked = torch.cat(outputs, dim=1)            # (b, L*trees, dim)
                return stacked.mean(dim=1) + self.bias

        return _NODENet()


@classifier(
    tags=["deep-learning", "tree", "ensemble", "tabular", "torch"],
    version="1.0.0",
)
class NODEClassifier(_NODECore, _DeepTabularClassifierMixin, Classifier):
    """NODE: **differentiable oblivious decision trees**, stacked and dense.

    Gradient boosting wins on tabular data because axis-aligned splits fit
    tables; deep learning wins because layers compose. NODE takes both: it
    replaces the hard split :math:`\\mathbb{1}[x_f > b]` with an entmax
    relaxation, so a whole ensemble of trees becomes one differentiable layer,
    and then stacks those layers DenseNet-style so later trees split on
    earlier trees' outputs -- something a boosted ensemble cannot do.

    The trees are **oblivious**: every node at a given depth shares one split
    feature and one threshold. A depth-:math:`D` tree is therefore a lookup
    table with :math:`2^{D}` leaves, evaluated for the whole batch with two
    einsums and no branching.

    Overview
    --------
    1. Each of ``n_trees`` trees picks, per depth level, a split feature via
       :func:`entmax15` over the input columns -- near one-hot, but
       differentiable.
    2. The chosen value is compared to a learned threshold and squashed by a
       learned temperature, then :func:`entmax15` over
       ``[go right, go left]`` gives a soft, often exactly-hard, decision.
    3. The outer product of the ``tree_depth`` decisions gives a distribution
       over the :math:`2^{D}` leaves; the response table is read with it.
    4. Layers are concatenated to their own input (dense connectivity) and the
       responses of all trees in all layers are averaged into class logits.

    Theory
    ------
    For tree :math:`t` at depth level :math:`d`, the split score is

    .. math::
        h_{td}(x) = \\frac{\\langle x, \\mathrm{entmax}_{1.5}(\\theta_{td})
        \\rangle - b_{td}}{\\tau_{td}}

    and the leaf-membership weight of leaf :math:`\\ell = (c_1, \\dots, c_D)`
    is the product of the per-level choices

    .. math::
        w_{t\\ell}(x) = \\prod_{d=1}^{D}
        \\mathrm{entmax}_{1.5}\\big([h_{td}, -h_{td}]\\big)_{c_d} .

    The layer output is :math:`\\sum_{\\ell} w_{t\\ell}(x) R_{t\\ell}` with a
    learned response table :math:`R`. Because entmax returns exact zeros, most
    leaves receive weight zero: the relaxation is soft enough to train and
    sharp enough to behave like a tree.

    Parameters
    ----------
    n_layers : int, default=1
        Densely connected tree-ensemble layers. Real use wants 2-8; the
        default keeps a fit on toy data well under a second.
    n_trees : int, default=32
        Oblivious trees per layer. Real use wants 128-2048.
    tree_depth : int, default=4
        Depth of each tree, so :math:`2^{\\text{tree\\_depth}}` leaves. Cost
        grows exponentially in it; 6 is the practical ceiling.
    learning_rate : float, default=1e-2
        AdamW step size. Higher than the attention models want, because the
        threshold and response parameters start far from useful values.
    weight_decay : float, default=0.0
        AdamW decoupled weight decay; NODE is usually trained without it.
    batch_size : int, default=64
        Mini-batch size for training and inference.
    n_epochs : int, default=60
        Passes over the training set; real use wants several hundred.
    early_stopping : bool, default=False
        Hold out a validation split and restore the best weights.
    validation_fraction : float, default=0.15
        Fraction held out when ``early_stopping`` is enabled.
    patience : int, default=10
        Epochs without validation improvement before training stops.
    device : {"cpu", "auto", "cuda", "mps"}, default="cpu"
        Compute device; ``"auto"`` trades reproducibility for speed.
    random_state : int, optional, default=0
        Seed for initialisation, shuffling and the validation split.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted class labels seen during :meth:`fit`.
    network_ : torch.nn.Module
        The fitted network, rebuilt lazily after unpickling.
    feature_mean_, feature_scale_ : np.ndarray
        Feature standardisation statistics; the thresholds are initialised on
        the standardised scale, so this is not optional here.
    loss_curve_ : np.ndarray of shape (n_iter_,)
        Mean training loss per epoch.
    n_iter_ : int
        Epochs actually run.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    **Requires PyTorch.** Install with ``pip install 'tuiml[torch]'``. The
    class constructs and introspects without torch; :meth:`fit` raises
    ``ImportError`` naming the install command.

    **Complexity.** Per epoch,
    :math:`O(n L T (m D + 2^{D} k))` for :math:`L` layers, :math:`T` trees,
    depth :math:`D` and :math:`k` outputs -- exponential in the depth, linear
    in everything else. Memory is :math:`O(b T 2^{D})` for the leaf weights.

    **When to use.** NODE is the deep model to try when the problem looks like
    one gradient boosting would win: axis-aligned structure, thresholds,
    moderate feature counts. It keeps that inductive bias while remaining
    differentiable, so it can be trained jointly with other neural components
    -- which is the reason to prefer it over an actual boosted ensemble.

    References
    ----------
    .. [Popov2020] Popov, S., Morozov, S., & Babenko, A. (2020). Neural
       Oblivious Decision Ensembles for Deep Learning on Tabular Data.
       *International Conference on Learning Representations (ICLR)*.
       :doi:`10.48550/arXiv.1909.06312`
    .. [Peters2019] Peters, B., Niculae, V., & Martins, A. F. T. (2019).
       Sparse Sequence-to-Sequence Models. *Proceedings of ACL 2019*,
       1504-1519. :doi:`10.18653/v1/P19-1146`

    See Also
    --------
    :class:`~tuiml.algorithms.tabular_deep.FTTransformerClassifier` : Attention over features instead of trees.
    :class:`~tuiml.algorithms.tabular_deep.SAINTClassifier` : Attention over features and rows.
    :class:`~tuiml.algorithms.tabular_deep.NODERegressor` : The regression counterpart.

    Examples
    --------
    Constructing and inspecting a model needs no torch:

    >>> from tuiml.algorithms.tabular_deep import NODEClassifier
    >>> model = NODEClassifier(n_trees=16, tree_depth=3, random_state=0)
    >>> model.tree_depth
    3
    >>> NODEClassifier.get_parameter_schema()["n_trees"]["default"]
    32

    Fitting requires ``pip install 'tuiml[torch]'``; the example below is a
    no-op on an install without it:

    >>> import numpy as np
    >>> from tuiml.utils.torch_backend import has_torch
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 4))
    >>> y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    >>> if has_torch():
    ...     model = NODEClassifier(n_epochs=150, random_state=0).fit(X, y)
    ...     print(float(model.score(X, y)) > 0.85)
    ... else:
    ...     print(True)
    True
    """

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "binary_class", "multiclass", "probabilistic",
                "ensemble", "tree", "non_linear"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return ("Training: O(epochs * n * L * T * (m*D + 2^D * k)), "
                "Prediction: O(n * L * T * (m*D + 2^D * k)), "
                "L=n_layers, T=n_trees, D=tree_depth")

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Popov, S., Morozov, S. and Babenko, A., 2020. Neural oblivious "
            "decision ensembles for deep learning on tabular data. ICLR.",
            "Peters, B., Niculae, V. and Martins, A.F.T., 2019. Sparse "
            "sequence-to-sequence models. ACL.",
        ]


@regressor(
    tags=["deep-learning", "tree", "ensemble", "tabular", "torch"],
    version="1.0.0",
)
class NODERegressor(_NODECore, _DeepTabularRegressorMixin, Regressor):
    """NODE for regression: differentiable oblivious tree ensembles.

    The regression counterpart of
    :class:`~tuiml.algorithms.tabular_deep.NODEClassifier`. The architecture
    is unchanged -- entmax-relaxed oblivious trees, densely stacked -- and only
    the objective changes: a single response dimension trained with mean
    squared error against a standardised target.

    Overview
    --------
    1. Standardise features and target.
    2. Each tree picks split features with :func:`entmax15`, compares them to
       learned thresholds, and reads a leaf response.
    3. Layers concatenate their outputs to their input, so later trees can
       split on earlier trees' responses.
    4. Average every tree's response and undo the target standardisation.

    Theory
    ------
    The prediction is an average over all :math:`LT` trees of soft leaf
    lookups,

    .. math::
        \\hat{y}(x) = \\frac{1}{LT} \\sum_{t} \\sum_{\\ell}
        w_{t\\ell}(x) \\, R_{t\\ell},
        \\quad
        w_{t\\ell}(x) = \\prod_{d=1}^{D}
        \\mathrm{entmax}_{1.5}\\big([h_{td}, -h_{td}]\\big)_{c_d}

    trained by minimising :math:`\\|\\hat{y} - \\tilde{y}\\|^2` on the
    standardised target. Because :math:`w` is piecewise-smooth rather than
    piecewise-constant, the fitted surface is continuous -- unlike a tree
    ensemble's staircase, which is often the practical difference on smooth
    targets.

    Parameters
    ----------
    n_layers : int, default=1
        Densely connected tree-ensemble layers; real use wants 2-8.
    n_trees : int, default=32
        Oblivious trees per layer; real use wants 128-2048.
    tree_depth : int, default=4
        Depth of each tree, so :math:`2^{\\text{tree\\_depth}}` leaves.
    learning_rate : float, default=1e-2
        AdamW step size.
    weight_decay : float, default=0.0
        AdamW decoupled weight decay.
    batch_size : int, default=64
        Mini-batch size for training and inference.
    n_epochs : int, default=60
        Passes over the training set; real use wants several hundred.
    early_stopping : bool, default=False
        Hold out a validation split and restore the best weights.
    validation_fraction : float, default=0.15
        Fraction held out when ``early_stopping`` is enabled.
    patience : int, default=10
        Epochs without validation improvement before training stops.
    device : {"cpu", "auto", "cuda", "mps"}, default="cpu"
        Compute device; ``"auto"`` trades reproducibility for speed.
    random_state : int, optional, default=0
        Seed for initialisation, shuffling and the validation split.

    Attributes
    ----------
    network_ : torch.nn.Module
        The fitted network, rebuilt lazily after unpickling.
    target_mean_, target_scale_ : float
        Target standardisation statistics.
    feature_mean_, feature_scale_ : np.ndarray
        Feature standardisation statistics.
    loss_curve_ : np.ndarray of shape (n_iter_,)
        Mean training loss per epoch.
    n_iter_ : int
        Epochs actually run.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    **Requires PyTorch.** Install with ``pip install 'tuiml[torch]'``.

    **Complexity.** :math:`O(n L T (m D + 2^{D}))` per epoch, exponential in
    ``tree_depth`` and linear in everything else.

    **When to use.** Reach for NODE when the target is a smooth function of
    threshold-like structure: it keeps the axis-aligned bias of a boosted
    ensemble but produces a continuous surface, and it can be trained jointly
    with other neural components.

    References
    ----------
    .. [Popov2020] Popov, S., Morozov, S., & Babenko, A. (2020). Neural
       Oblivious Decision Ensembles for Deep Learning on Tabular Data.
       *International Conference on Learning Representations (ICLR)*.
       :doi:`10.48550/arXiv.1909.06312`

    See Also
    --------
    :class:`~tuiml.algorithms.tabular_deep.NODEClassifier` : The classification counterpart.
    :class:`~tuiml.algorithms.tabular_deep.FTTransformerRegressor` : Attention over features instead of trees.
    :class:`~tuiml.algorithms.tabular_deep.SAINTRegressor` : Attention over features and rows.

    Examples
    --------
    >>> from tuiml.algorithms.tabular_deep import NODERegressor
    >>> model = NODERegressor(n_layers=2, n_trees=16)
    >>> model.n_layers
    2
    >>> "tree" in NODERegressor.get_parameter_schema()["tree_depth"]["description"]
    True

    Fitting requires ``pip install 'tuiml[torch]'``; the example below is a
    no-op on an install without it:

    >>> import numpy as np
    >>> from tuiml.utils.torch_backend import has_torch
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 4))
    >>> y = np.sin(X[:, 0]) * X[:, 1]
    >>> if has_torch():
    ...     model = NODERegressor(n_epochs=300, random_state=0).fit(X, y)
    ...     print(float(model.score(X, y)) > 0.8)
    ... else:
    ...     print(True)
    True
    """

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "numeric_class", "regression", "ensemble", "tree",
                "non_linear"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return ("Training: O(epochs * n * L * T * (m*D + 2^D)), "
                "Prediction: O(n * L * T * (m*D + 2^D)), "
                "L=n_layers, T=n_trees, D=tree_depth")

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Popov, S., Morozov, S. and Babenko, A., 2020. Neural oblivious "
            "decision ensembles for deep learning on tabular data. ICLR.",
        ]


__all__ = ["entmax15", "NODEClassifier", "NODERegressor"]
