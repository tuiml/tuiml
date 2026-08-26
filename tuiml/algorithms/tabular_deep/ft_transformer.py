"""FT-Transformer: the Transformer, adapted to tabular data.

Implements the architecture of Gorishniy et al. (NeurIPS 2021), *Revisiting
Deep Learning Models for Tabular Data*: a **feature tokenizer** turns each
column into its own embedding, a ``[CLS]`` token is prepended, and a stack of
pre-norm Transformer blocks attends over the resulting feature sequence.

PyTorch is an optional dependency -- ``pip install 'tuiml[torch]'``. Nothing in
this module imports torch until :meth:`fit` is called.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.algorithms.tabular_deep._base import (
    _DeepTabularClassifierMixin,
    _DeepTabularRegressorMixin,
    _build_head,
    _build_tokenizer,
    _build_transformer_block,
    _categorical_schema,
    _training_schema,
)


class _FTTransformerCore:
    """Constructor, schema and architecture shared by the two FT models.

    Internal mixin. It is not registered and defines no task behaviour; the
    classifier and regressor pair it with the matching task mixin from
    :mod:`tuiml.algorithms.tabular_deep._base`.
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
                "description": "Number of Transformer blocks",
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
        """Assemble tokenizer, Transformer blocks and the ``[CLS]`` head.

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
        tokenizer = _build_tokenizer(
            torch, nn, n_numeric, cardinalities, d_token, prepend_cls=True
        )
        blocks = nn.ModuleList([
            _build_transformer_block(
                torch, nn, d_token, int(self.n_heads), float(self.dropout)
            )
            for _ in range(int(self.n_blocks))
        ])
        head = _build_head(torch, nn, d_token, n_outputs)

        class _FTTransformerNet(nn.Module):
            """Tokenize, attend over features, read out the ``[CLS]`` token."""

            def __init__(self):
                super().__init__()
                self.tokenizer = tokenizer
                self.blocks = blocks
                self.head = head

            def forward(self, x_num, x_cat):
                """Map one batch of rows to output units."""
                tokens = self.tokenizer(x_num, x_cat)
                for block in self.blocks:
                    tokens = block(tokens)
                return self.head(tokens[:, 0])

        return _FTTransformerNet()


@classifier(
    tags=["deep-learning", "transformer", "attention", "tabular", "torch"],
    version="1.0.0",
)
class FTTransformerClassifier(_FTTransformerCore, _DeepTabularClassifierMixin, Classifier):
    """FT-Transformer: **per-feature tokens** attended by a Transformer.

    The obstacle to using a Transformer on tabular data is that a table has no
    sequence. FT-Transformer manufactures one: the **feature tokenizer** gives
    every column its own learned embedding, so a row of :math:`m` scalars
    becomes a sequence of :math:`m` tokens that self-attention can relate to
    one another. A ``[CLS]`` token rides along and collects the evidence; the
    prediction head reads only that token.

    Overview
    --------
    1. Tokenize: numerical feature :math:`x_j` becomes :math:`W_j x_j + b_j`,
       a ``d_token``-dimensional vector with its **own** weights; categorical
       features index an embedding table.
    2. Prepend a learned ``[CLS]`` token, giving ``m + 1`` tokens.
    3. Apply ``n_blocks`` pre-norm Transformer blocks: multi-head self-attention
       across features, then a feed-forward network, each residual.
    4. Predict from the final ``[CLS]`` token.

    Theory
    ------
    The tokenizer is an affine map applied per feature, not per row, which is
    what distinguishes it from a plain input layer:

    .. math::
        T_j = W_j x_j + b_j, \\quad W_j \\in \\mathbb{R}^{d},
        \\quad j = 1, \\dots, m

    Each block then applies pre-norm attention and a feed-forward network:

    .. math::
        T \\leftarrow T + \\mathrm{MHSA}(\\mathrm{LN}(T)),
        \\quad
        T \\leftarrow T + \\mathrm{FFN}(\\mathrm{LN}(T))

    with attention over the feature axis,

    .. math::
        \\mathrm{Attn}(Q, K, V) = \\mathrm{softmax}\\!
        \\left(\\frac{Q K^{\\top}}{\\sqrt{d_h}}\\right) V.

    Because attention scores are computed *between features*, the model learns
    multiplicative feature interactions directly, rather than approximating
    them with axis-aligned splits as a tree ensemble does.

    Parameters
    ----------
    d_token : int, default=16
        Width of each feature token. Must be divisible by ``n_heads``. The
        default is deliberately small so that a fit on a toy problem takes
        milliseconds; real tabular problems want 64-192.
    n_blocks : int, default=1
        Number of Transformer blocks. Real use wants 2-4.
    n_heads : int, default=2
        Attention heads per block.
    dropout : float, default=0.1
        Dropout rate in the attention weights and the feed-forward network.
    learning_rate : float, default=1e-3
        AdamW step size.
    weight_decay : float, default=1e-5
        AdamW decoupled weight decay.
    batch_size : int, default=64
        Mini-batch size for training and inference.
    n_epochs : int, default=60
        Passes over the training set. Real use wants several hundred, paired
        with ``early_stopping=True``.
    early_stopping : bool, default=False
        Hold out ``validation_fraction`` of the rows and restore the
        best-scoring weights when validation loss stops improving.
    validation_fraction : float, default=0.15
        Fraction held out when ``early_stopping`` is enabled.
    patience : int, default=10
        Epochs without validation improvement before training stops.
    categorical_features : sequence of int, optional
        Column indices holding integer-coded categorical features; they get an
        embedding table instead of the numerical tokenizer. ``None`` treats
        every column as numerical.
    device : {"cpu", "auto", "cuda", "mps"}, default="cpu"
        Compute device. The default is ``"cpu"`` because accelerators make
        results non-reproducible; ``"auto"`` picks CUDA, then MPS, then CPU
        and is the right choice when speed matters more than bit-exactness.
    random_state : int, optional, default=0
        Seed for parameter initialisation, batch shuffling and the validation
        split. With ``device="cpu"`` the same seed reproduces predictions
        exactly.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted class labels seen during :meth:`fit`.
    network_ : torch.nn.Module
        The fitted network. Rebuilt lazily after unpickling.
    feature_mean_, feature_scale_ : np.ndarray
        Standardisation statistics of the numerical columns, fitted on train.
    loss_curve_ : np.ndarray of shape (n_iter_,)
        Mean training loss per epoch.
    n_iter_ : int
        Epochs actually run, which is below ``n_epochs`` if early stopping
        triggered.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    **Requires PyTorch.** Install with ``pip install 'tuiml[torch]'``. The
    class can be constructed and inspected without torch; :meth:`fit` raises
    ``ImportError`` naming the install command.

    **Complexity.** One block costs :math:`O(n m^2 d + n m d^2)` per epoch,
    quadratic in the number of *features* rather than of samples, so wide
    tables cost more than tall ones. Memory is :math:`O(b m^2)` for the
    attention matrix of a batch.

    **When to use.** FT-Transformer is the strongest of the attention-based
    tabular baselines and the one to reach for when features interact in ways
    an additive model misses, and when there is enough data (roughly tens of
    thousands of rows) to train a Transformer. On small tables a gradient
    boosted ensemble is usually both faster and more accurate.

    References
    ----------
    .. [Gorishniy2021] Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko,
       A. (2021). Revisiting Deep Learning Models for Tabular Data. *Advances
       in Neural Information Processing Systems 34*, 18932-18943.
       :doi:`10.48550/arXiv.2106.11959`
    .. [Vaswani2017] Vaswani, A., et al. (2017). Attention Is All You Need.
       *Advances in Neural Information Processing Systems 30*.
       :doi:`10.48550/arXiv.1706.03762`

    See Also
    --------
    :class:`~tuiml.algorithms.tabular_deep.SAINTClassifier` : Adds attention across rows.
    :class:`~tuiml.algorithms.tabular_deep.NODEClassifier` : Differentiable oblivious trees instead of attention.
    :class:`~tuiml.algorithms.tabular_deep.FTTransformerRegressor` : The regression counterpart.

    Examples
    --------
    Constructing and inspecting a model needs no torch:

    >>> from tuiml.algorithms.tabular_deep import FTTransformerClassifier
    >>> model = FTTransformerClassifier(d_token=8, n_blocks=1, random_state=0)
    >>> model.d_token
    8
    >>> "n_epochs" in FTTransformerClassifier.get_parameter_schema()
    True

    Fitting requires ``pip install 'tuiml[torch]'``; the example below is a
    no-op on an install without it:

    >>> import numpy as np
    >>> from tuiml.utils.torch_backend import has_torch
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 4))
    >>> y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    >>> if has_torch():
    ...     model = FTTransformerClassifier(n_epochs=150, random_state=0).fit(X, y)
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
        return ("Training: O(epochs * n * (m^2*d + m*d^2)), "
                "Prediction: O(n * (m^2*d + m*d^2)), m=n_features, d=d_token")

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A., 2021. "
            "Revisiting deep learning models for tabular data. NeurIPS 34.",
        ]


@regressor(
    tags=["deep-learning", "transformer", "attention", "tabular", "torch"],
    version="1.0.0",
)
class FTTransformerRegressor(_FTTransformerCore, _DeepTabularRegressorMixin, Regressor):
    """FT-Transformer for regression: per-feature tokens, ``[CLS]`` readout.

    The regression counterpart of
    :class:`~tuiml.algorithms.tabular_deep.FTTransformerClassifier`. The
    architecture is identical -- feature tokenizer, ``[CLS]`` token, pre-norm
    Transformer blocks -- and only the output layer and objective change: a
    single unit trained with mean squared error against a standardised target,
    rescaled back on prediction.

    Overview
    --------
    1. Standardise the target, so the loss scale is independent of the units.
    2. Tokenize each feature into a ``d_token``-dimensional embedding.
    3. Attend across features through ``n_blocks`` pre-norm Transformer blocks.
    4. Read the ``[CLS]`` token through a linear head and undo the target
       standardisation.

    Theory
    ------
    Given tokens :math:`T \\in \\mathbb{R}^{(m+1) \\times d}` the network
    minimises

    .. math::
        \\mathcal{L} = \\frac{1}{n} \\sum_{i=1}^{n}
        \\left(\\hat{y}_i - \\tilde{y}_i\\right)^2,
        \\quad
        \\tilde{y}_i = \\frac{y_i - \\mu_y}{\\sigma_y}

    where :math:`\\hat{y}` is the head applied to the final ``[CLS]`` token.
    Standardising :math:`y` matters more than it does for trees: with a raw
    target of large magnitude the initial gradients dominate the attention
    weights and the model spends its early epochs learning the mean.

    Parameters
    ----------
    d_token : int, default=16
        Width of each feature token; must be divisible by ``n_heads``. Real
        problems want 64-192.
    n_blocks : int, default=1
        Number of Transformer blocks. Real use wants 2-4.
    n_heads : int, default=2
        Attention heads per block.
    dropout : float, default=0.1
        Dropout rate in the attention weights and the feed-forward network.
    learning_rate : float, default=1e-3
        AdamW step size.
    weight_decay : float, default=1e-5
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
        Target standardisation statistics, fitted on train.
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
    **Requires PyTorch.** Install with ``pip install 'tuiml[torch]'``. Building
    the object works without it; :meth:`fit` raises ``ImportError``.

    **Complexity.** Identical to the classifier:
    :math:`O(n m^2 d + n m d^2)` per epoch, quadratic in the feature count.

    **When to use.** Prefer it over
    :class:`~tuiml.algorithms.glassbox.ExplainableBoostingRegressor` when the
    target depends on feature *interactions* rather than on an additive sum of
    shape functions, and there is enough data to train a Transformer.

    References
    ----------
    .. [Gorishniy2021] Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko,
       A. (2021). Revisiting Deep Learning Models for Tabular Data. *Advances
       in Neural Information Processing Systems 34*, 18932-18943.
       :doi:`10.48550/arXiv.2106.11959`

    See Also
    --------
    :class:`~tuiml.algorithms.tabular_deep.FTTransformerClassifier` : The classification counterpart.
    :class:`~tuiml.algorithms.tabular_deep.SAINTRegressor` : Adds attention across rows.
    :class:`~tuiml.algorithms.tabular_deep.NODERegressor` : Differentiable oblivious trees.

    Examples
    --------
    >>> from tuiml.algorithms.tabular_deep import FTTransformerRegressor
    >>> model = FTTransformerRegressor(d_token=8, n_epochs=10)
    >>> model.n_epochs
    10
    >>> sorted(FTTransformerRegressor.get_capabilities())[:2]
    ['categorical', 'non_linear']

    Fitting requires ``pip install 'tuiml[torch]'``; the example below is a
    no-op on an install without it:

    >>> import numpy as np
    >>> from tuiml.utils.torch_backend import has_torch
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 4))
    >>> y = np.sin(X[:, 0]) * X[:, 1]
    >>> if has_torch():
    ...     model = FTTransformerRegressor(n_epochs=300, random_state=0).fit(X, y)
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
        return ("Training: O(epochs * n * (m^2*d + m*d^2)), "
                "Prediction: O(n * (m^2*d + m*d^2)), m=n_features, d=d_token")

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A., 2021. "
            "Revisiting deep learning models for tabular data. NeurIPS 34.",
        ]


__all__ = ["FTTransformerClassifier", "FTTransformerRegressor"]
