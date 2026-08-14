"""SAINT: self-attention over features **and** over rows.

Implements the architecture of Somepalli et al. (2021), *SAINT: Improved
Neural Networks for Tabular Data via Row Attention and Contrastive
Pre-Training*. The distinguishing mechanism is **intersample attention**:
after attending across the features of a row, the network attends across the
*rows of the batch*, so a prediction can borrow evidence from neighbouring
samples. Without that second stage SAINT would be FT-Transformer.

PyTorch is an optional dependency -- ``pip install 'tuiml[torch]'``. Nothing
in this module imports torch until :meth:`fit` is called.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.algorithms.tabular_foundation._base import (
    _DeepTabularClassifierMixin,
    _DeepTabularRegressorMixin,
    _build_head,
    _build_tokenizer,
    _build_transformer_block,
    _categorical_schema,
    _training_schema,
)


def _build_intersample_block(torch, nn, n_tokens: int, d_token: int,
                             n_heads: int, dropout: float):
    """Build a Transformer block that attends **across rows** of a batch.

    The batch of ``b`` rows, each already tokenized into ``n_tokens`` feature
    tokens, is flattened to one sequence of ``b`` tokens of width
    ``n_tokens * d_token``. Self-attention over that sequence relates whole
    rows to one another; the result is reshaped back to per-feature tokens.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.
    nn : module
        ``torch.nn``.
    n_tokens : int
        Tokens per row, including the ``[CLS]`` token.
    d_token : int
        Width of one feature token.
    n_heads : int
        Attention heads.
    dropout : float
        Dropout rate.

    Returns
    -------
    block : torch.nn.Module
        Module mapping ``(batch, n_tokens, d_token)`` to the same shape, with
        every row's representation a function of the whole batch.
    """
    d_row = n_tokens * d_token
    inner = _build_transformer_block(torch, nn, d_row, n_heads, dropout)

    class _IntersampleBlock(nn.Module):
        """Row-axis attention wrapped in the per-feature token layout."""

        def __init__(self):
            super().__init__()
            self.block = inner

        def forward(self, tokens):
            """Attend across the rows of the batch."""
            b, t, d = tokens.shape
            flat = tokens.reshape(1, b, t * d)
            flat = self.block(flat)
            return flat.reshape(b, t, d)

    return _IntersampleBlock()


class _SAINTCore:
    """Constructor, schema and architecture shared by the two SAINT models.

    Internal mixin: not registered, no task behaviour of its own.
    """

    def __init__(
        self,
        d_token: int = 16,
        n_blocks: int = 1,
        n_heads: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        n_epochs: int = 60,
        early_stopping: bool = False,
        validation_fraction: float = 0.15,
        patience: int = 10,
        categorical_features: Optional[Sequence[int]] = None,
        device: str = "cpu",
        random_state: Optional[int] = 0,
    ):
        """Record hyperparameters. Never imports torch (see the class docs)."""
        super().__init__()
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.categorical_features = categorical_features
        self.device = device
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        schema: Dict[str, Dict[str, Any]] = {
            "d_token": {
                "type": "integer",
                "default": 16,
                "minimum": 2,
                "maximum": 1024,
                "description": "Width of each feature token; must divide by n_heads",
            },
            "n_blocks": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 32,
                "description": "Number of (feature attention, row attention) stages",
            },
            "n_heads": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "maximum": 32,
                "description": "Number of attention heads per block",
            },
            "dropout": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 0.9,
                "description": "Dropout applied in attention and the FFN",
            },
        }
        schema.update(_training_schema())
        schema.update(_categorical_schema())
        return schema

    def _build_network(self, torch, nn, n_numeric: int,
                       cardinalities: Sequence[int], n_outputs: int):
        """Assemble the alternating feature/row attention stack.

        Parameters
        ----------
        torch : module
            The imported ``torch`` module.
        nn : module
            ``torch.nn``.
        n_numeric : int
            Number of numerical input columns.
        cardinalities : sequence of int
            Category count per categorical column.
        n_outputs : int
            Number of output units.

        Returns
        -------
        network : torch.nn.Module
            Module mapping ``(x_num, x_cat)`` to ``(batch, n_outputs)``.
        """
        d_token = int(self.d_token)
        n_heads = int(self.n_heads)
        dropout = float(self.dropout)
        n_tokens = n_numeric + len(cardinalities) + 1  # + [CLS]

        tokenizer = _build_tokenizer(
            torch, nn, n_numeric, cardinalities, d_token, prepend_cls=True
        )
        feature_blocks = nn.ModuleList([
            _build_transformer_block(torch, nn, d_token, n_heads, dropout)
            for _ in range(int(self.n_blocks))
        ])
        row_blocks = nn.ModuleList([
            _build_intersample_block(torch, nn, n_tokens, d_token, n_heads, dropout)
            for _ in range(int(self.n_blocks))
        ])
        head = _build_head(torch, nn, d_token, n_outputs)

        class _SAINTNet(nn.Module):
            """Alternating feature attention and intersample (row) attention."""

            def __init__(self):
                super().__init__()
                self.tokenizer = tokenizer
                self.feature_blocks = feature_blocks
                self.row_blocks = row_blocks
                self.head = head

            def forward(self, x_num, x_cat):
                """Map one batch of rows to output units."""
                tokens = self.tokenizer(x_num, x_cat)
                for feature_block, row_block in zip(self.feature_blocks, self.row_blocks):
                    tokens = feature_block(tokens)
                    tokens = row_block(tokens)
                return self.head(tokens[:, 0])

        return _SAINTNet()


@classifier(
    tags=["deep-learning", "transformer", "attention", "tabular", "torch"],
    version="1.0.0",
)
class SAINTClassifier(_SAINTCore, _DeepTabularClassifierMixin, Classifier):
    """SAINT: attention across features **and across rows of the batch**.

    SAINT starts where FT-Transformer stops. Attending across the features of
    a row lets the model see interactions; SAINT adds a second attention stage
    that runs across the **rows of the batch**, so the representation of a
    sample is a function of its neighbours as well as of itself. That is a
    learned, end-to-end analogue of a nearest-neighbour lookup, and it is what
    makes SAINT more than a second copy of FT-Transformer.

    Overview
    --------
    1. Tokenize every feature into a ``d_token`` embedding and prepend
       ``[CLS]`` (the same tokenizer FT-Transformer uses).
    2. **Feature attention:** a pre-norm Transformer block over the token
       axis, relating the columns of one row.
    3. **Intersample attention:** flatten each row's tokens into a single
       vector of width ``n_tokens * d_token``, treat the batch as a sequence
       of those vectors, and attend over it -- relating whole rows to each
       other. Reshape back.
    4. Repeat for ``n_blocks`` stages and read the ``[CLS]`` token.

    Theory
    ------
    Write :math:`T^{(i)} \\in \\mathbb{R}^{t \\times d}` for the tokens of row
    :math:`i` in a batch of size :math:`b`. Feature attention acts within a
    row,

    .. math::
        T^{(i)} \\leftarrow T^{(i)} +
        \\mathrm{MHSA}\\big(\\mathrm{LN}(T^{(i)})\\big),

    while intersample attention acts on the flattened matrix
    :math:`Z \\in \\mathbb{R}^{b \\times td}` whose rows are
    :math:`\\mathrm{vec}(T^{(i)})`:

    .. math::
        Z \\leftarrow Z + \\mathrm{MHSA}\\big(\\mathrm{LN}(Z)\\big),
        \\quad
        \\mathrm{Attn}(Q, K, V) = \\mathrm{softmax}\\!
        \\left(\\frac{Q K^{\\top}}{\\sqrt{d_h}}\\right) V .

    The two stages differ only in which axis the softmax runs over, and that
    difference is observable: permuting the *other* rows of a batch changes a
    given row's representation under intersample attention, and cannot change
    it under feature attention alone.

    Parameters
    ----------
    d_token : int, default=16
        Width of each feature token; must be divisible by ``n_heads``. The
        default is tuned for sub-second fits on toy data; real problems want
        32-64 (intersample attention makes SAINT more expensive than
        FT-Transformer at equal width).
    n_blocks : int, default=1
        Number of (feature attention, row attention) stages. Real use wants
        2-6.
    n_heads : int, default=2
        Attention heads per block.
    dropout : float, default=0.1
        Dropout rate in the attention weights and the feed-forward network.
    learning_rate : float, default=1e-3
        AdamW step size.
    weight_decay : float, default=1e-5
        AdamW decoupled weight decay.
    batch_size : int, default=64
        Mini-batch size. It matters more here than in other models: it is the
        population intersample attention gets to look at, at train **and**
        predict time.
    n_epochs : int, default=60
        Passes over the training set; real use wants several hundred.
    early_stopping : bool, default=False
        Hold out a validation split and restore the best weights.
    validation_fraction : float, default=0.15
        Fraction held out when ``early_stopping`` is enabled.
    patience : int, default=10
        Epochs without validation improvement before training stops.
    categorical_features : sequence of int, optional
        Column indices holding integer-coded categorical features.
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
        Feature standardisation statistics, fitted on train.
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

    **Predictions depend on the batch.** Intersample attention is not a
    per-row function, so a row's prediction depends on which rows share its
    batch. Predictions are deterministic for a given ``X`` and ``batch_size``
    because inference batches follow input order, but scoring the same row
    inside a different set of rows can move it. This is inherent to the
    architecture, not an implementation artifact.

    **Complexity.** Per epoch,
    :math:`O(n m^2 d + n b m d)` -- the second term is intersample attention,
    quadratic in the batch size :math:`b`. Memory is :math:`O(b^2 + b m^2)`.

    **When to use.** SAINT pays off when rows are informative about each other
    -- semi-supervised settings, or data with cluster structure the label
    respects. When rows are genuinely i.i.d. given the features,
    :class:`~tuiml.algorithms.tabular_foundation.FTTransformerClassifier` gives
    similar accuracy for less compute.

    References
    ----------
    .. [Somepalli2021] Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss,
       C. B., & Goldstein, T. (2021). SAINT: Improved Neural Networks for
       Tabular Data via Row Attention and Contrastive Pre-Training.
       :doi:`10.48550/arXiv.2106.01342`

    See Also
    --------
    :class:`~tuiml.algorithms.tabular_foundation.FTTransformerClassifier` : Feature attention only.
    :class:`~tuiml.algorithms.tabular_foundation.NODEClassifier` : Differentiable oblivious trees.
    :class:`~tuiml.algorithms.tabular_foundation.SAINTRegressor` : The regression counterpart.

    Examples
    --------
    Constructing and inspecting a model needs no torch:

    >>> from tuiml.algorithms.tabular_foundation import SAINTClassifier
    >>> model = SAINTClassifier(d_token=8, n_blocks=2, random_state=0)
    >>> model.n_blocks
    2
    >>> "batch_size" in SAINTClassifier.get_parameter_schema()
    True

    Fitting requires ``pip install 'tuiml[torch]'``; the example below is a
    no-op on an install without it:

    >>> import numpy as np
    >>> from tuiml.utils.torch_backend import has_torch
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 4))
    >>> y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    >>> if has_torch():
    ...     model = SAINTClassifier(n_epochs=150, random_state=0).fit(X, y)
    ...     print(float(model.score(X, y)) > 0.85)
    ... else:
    ...     print(True)
    True
    """

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "categorical", "binary_class", "multiclass",
                "probabilistic", "non_linear"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return ("Training: O(epochs * n * (m^2*d + b*m*d)), "
                "Prediction: O(n * (m^2*d + b*m*d)), b=batch_size, m=n_features")

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C.B. and "
            "Goldstein, T., 2021. SAINT: Improved neural networks for tabular "
            "data via row attention and contrastive pre-training. arXiv:2106.01342.",
        ]


@regressor(
    tags=["deep-learning", "transformer", "attention", "tabular", "torch"],
    version="1.0.0",
)
class SAINTRegressor(_SAINTCore, _DeepTabularRegressorMixin, Regressor):
    """SAINT for regression: feature attention plus intersample attention.

    The regression counterpart of
    :class:`~tuiml.algorithms.tabular_foundation.SAINTClassifier`. The stack is
    identical -- tokenizer, alternating feature and row attention, ``[CLS]``
    readout -- and only the objective changes: one output unit trained with
    mean squared error against a standardised target.

    Overview
    --------
    1. Standardise the target.
    2. Tokenize the features and prepend ``[CLS]``.
    3. Alternate feature attention (within a row) and intersample attention
       (across the batch) for ``n_blocks`` stages.
    4. Read ``[CLS]`` through a linear head, then undo the standardisation.

    Theory
    ------
    With :math:`Z \\in \\mathbb{R}^{b \\times td}` the flattened batch of
    tokenized rows, intersample attention makes the prediction for row
    :math:`i` a weighted combination of every row in the batch:

    .. math::
        \\hat{y}_i = h\\Big(\\sum_{j=1}^{b} \\alpha_{ij} v(Z_j)\\Big),
        \\quad
        \\alpha_{i\\cdot} = \\mathrm{softmax}\\!
        \\left(\\frac{q(Z_i) K^{\\top}}{\\sqrt{d_h}}\\right)

    which is a learned kernel regression over the batch, trained jointly with
    the representation it attends over. The loss is mean squared error on the
    standardised target :math:`(y - \\mu_y)/\\sigma_y`.

    Parameters
    ----------
    d_token : int, default=16
        Width of each feature token; must be divisible by ``n_heads``. Real
        problems want 32-64.
    n_blocks : int, default=1
        Number of (feature attention, row attention) stages; real use wants 2-6.
    n_heads : int, default=2
        Attention heads per block.
    dropout : float, default=0.1
        Dropout rate in the attention weights and the feed-forward network.
    learning_rate : float, default=1e-3
        AdamW step size.
    weight_decay : float, default=1e-5
        AdamW decoupled weight decay.
    batch_size : int, default=64
        Mini-batch size, and the population intersample attention sees.
    n_epochs : int, default=60
        Passes over the training set; real use wants several hundred.
    early_stopping : bool, default=False
        Hold out a validation split and restore the best weights.
    validation_fraction : float, default=0.15
        Fraction held out when ``early_stopping`` is enabled.
    patience : int, default=10
        Epochs without validation improvement before training stops.
    categorical_features : sequence of int, optional
        Column indices holding integer-coded categorical features.
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

    **Predictions depend on the batch**, because intersample attention is not
    a per-row function. See the classifier's notes.

    **Complexity.** :math:`O(n m^2 d + n b m d)` per epoch, with the second
    term quadratic in ``batch_size``.

    **When to use.** Choose SAINT over
    :class:`~tuiml.algorithms.tabular_foundation.FTTransformerRegressor` when
    neighbouring rows carry information about the target -- clustered or
    grouped data -- and accept the extra compute for it.

    References
    ----------
    .. [Somepalli2021] Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss,
       C. B., & Goldstein, T. (2021). SAINT: Improved Neural Networks for
       Tabular Data via Row Attention and Contrastive Pre-Training.
       :doi:`10.48550/arXiv.2106.01342`

    See Also
    --------
    :class:`~tuiml.algorithms.tabular_foundation.SAINTClassifier` : The classification counterpart.
    :class:`~tuiml.algorithms.tabular_foundation.FTTransformerRegressor` : Feature attention only.
    :class:`~tuiml.algorithms.tabular_foundation.NODERegressor` : Differentiable oblivious trees.

    Examples
    --------
    >>> from tuiml.algorithms.tabular_foundation import SAINTRegressor
    >>> model = SAINTRegressor(d_token=8, n_heads=2)
    >>> model.d_token
    8
    >>> "regression" in SAINTRegressor.get_capabilities()
    True

    Fitting requires ``pip install 'tuiml[torch]'``; the example below is a
    no-op on an install without it:

    >>> import numpy as np
    >>> from tuiml.utils.torch_backend import has_torch
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 4))
    >>> y = np.sin(X[:, 0]) * X[:, 1]
    >>> if has_torch():
    ...     model = SAINTRegressor(n_epochs=300, random_state=0).fit(X, y)
    ...     print(float(model.score(X, y)) > 0.8)
    ... else:
    ...     print(True)
    True
    """

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "categorical", "numeric_class", "regression",
                "non_linear"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return ("Training: O(epochs * n * (m^2*d + b*m*d)), "
                "Prediction: O(n * (m^2*d + b*m*d)), b=batch_size, m=n_features")

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C.B. and "
            "Goldstein, T., 2021. SAINT: Improved neural networks for tabular "
            "data via row attention and contrastive pre-training. arXiv:2106.01342.",
        ]


__all__ = ["SAINTClassifier", "SAINTRegressor"]
