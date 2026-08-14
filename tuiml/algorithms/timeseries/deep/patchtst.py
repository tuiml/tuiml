"""PatchTST: a patch-based Transformer forecaster with channel independence."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from tuiml.algorithms.timeseries.deep._base import (
    DeepForecaster,
    instance_denormalize,
    instance_normalize,
)
from tuiml.base.algorithms import regressor

# Re-exported so the RevIN pair PatchTST relies on is importable from the
# model that documents it, while there is still exactly one implementation.
revin_normalize = instance_normalize
revin_denormalize = instance_denormalize


def num_patches(length: int, patch_len: int, stride: int) -> int:
    """Return how many patches a window of ``length`` is split into.

    The end of the window is padded by repeating its last value so the final
    patch lands exactly on the padded end. Padding the *end* rather than the
    start matters: the most recent points are the ones a forecast leans on,
    and truncating them to make the arithmetic divide would throw away the
    most informative part of the window.

    Parameters
    ----------
    length : int
        Window length.
    patch_len : int
        Points per patch. Clipped to ``length`` when the window is shorter.
    stride : int
        Step between consecutive patch starts. Equal to ``patch_len`` gives
        disjoint patches; smaller gives overlap.

    Returns
    -------
    n_patches : int
        Number of patches, at least 1.

    Examples
    --------
    >>> from tuiml.algorithms.timeseries.deep.patchtst import num_patches
    >>> num_patches(24, patch_len=8, stride=4)   # divides exactly
    5
    >>> num_patches(25, patch_len=8, stride=4)   # padded to 29
    6
    >>> num_patches(5, patch_len=8, stride=4)    # window shorter than a patch
    1
    """
    length = int(length)
    patch_len = max(1, min(int(patch_len), length))
    stride = max(1, int(stride))
    if length <= patch_len:
        return 1
    return int(math.ceil((length - patch_len) / stride)) + 1


def padded_length(length: int, patch_len: int, stride: int) -> int:
    """Return the window length after end padding.

    Parameters
    ----------
    length : int
        Window length.
    patch_len : int
        Points per patch.
    stride : int
        Step between patch starts.

    Returns
    -------
    padded : int
        ``patch_len + (n_patches - 1) * stride``, never less than ``length``.

    Examples
    --------
    >>> from tuiml.algorithms.timeseries.deep.patchtst import padded_length
    >>> padded_length(25, patch_len=8, stride=4)
    28
    """
    patch_len = max(1, min(int(patch_len), int(length)))
    stride = max(1, int(stride))
    n = num_patches(length, patch_len, stride)
    return max(int(length), patch_len + (n - 1) * stride)


def patchify(torch: Any, x: Any, patch_len: int, stride: int) -> Any:
    """Split each window into overlapping or disjoint patches.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module, passed in so this function never
        imports it.
    x : torch.Tensor of shape (batch, length)
        Input windows.
    patch_len : int
        Points per patch.
    stride : int
        Step between patch starts.

    Returns
    -------
    patches : torch.Tensor of shape (batch, n_patches, patch_len)
        The patch tokens, oldest first.
    """
    length = int(x.shape[-1])
    patch_len = max(1, min(int(patch_len), length))
    stride = max(1, int(stride))
    target = padded_length(length, patch_len, stride)
    if target > length:
        tail = x[:, -1:].expand(x.shape[0], target - length)
        x = torch.cat([x, tail], dim=1)
    return x.unfold(dimension=1, size=patch_len, step=stride)


def _build_patchtst_module(
    torch: Any,
    nn: Any,
    lookback: int,
    horizon: int,
    patch_len: int,
    stride: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> Any:
    """Build the PatchTST network.

    The ``nn.Module`` subclass is declared inside this function because ``nn``
    does not exist at module scope: importing this file must work on an
    install without PyTorch.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.
    nn : module
        ``torch.nn``.
    lookback : int
        Input window length.
    horizon : int
        Forecast length.
    patch_len : int
        Points per patch.
    stride : int
        Step between patch starts.
    d_model : int
        Token embedding width.
    n_heads : int
        Attention heads per encoder layer.
    n_layers : int
        Encoder layers.
    dim_feedforward : int
        Width of the position-wise feed-forward network.
    dropout : float
        Dropout probability inside the encoder and before the head.

    Returns
    -------
    module : torch.nn.Module
        Maps ``(batch, lookback)`` or ``(batch, channels, lookback)`` to
        ``(batch, horizon)`` or ``(batch, channels, horizon)``.
    """
    effective_patch = max(1, min(int(patch_len), int(lookback)))
    effective_stride = max(1, int(stride))
    n_patches = num_patches(lookback, effective_patch, effective_stride)

    class _PatchTST(nn.Module):
        """Patch embedding, Transformer encoder, flatten-and-linear head."""

        def __init__(self):
            """Build the embedding, positional table, encoder and head."""
            super().__init__()
            self.patch_len = effective_patch
            self.stride = effective_stride
            self.n_patches = n_patches

            self.embedding = nn.Linear(effective_patch, d_model)
            self.position = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
            self.embed_dropout = nn.Dropout(dropout)

            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # ``enable_nested_tensor`` is inert with ``norm_first=True`` and
            # only emits a warning; turn it off rather than warn on every fit.
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=int(n_layers), enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(d_model)
            self.head_dropout = nn.Dropout(dropout)
            self.head = nn.Linear(n_patches * d_model, horizon)

        def encode(self, x):
            """Run the shared backbone on a batch of single-channel windows.

            Parameters
            ----------
            x : torch.Tensor of shape (batch, lookback)
                Normalised single-channel windows.

            Returns
            -------
            forecast : torch.Tensor of shape (batch, horizon)
                The forecast for each window.
            """
            patches = patchify(torch, x, self.patch_len, self.stride)
            tokens = self.embedding(patches) + self.position
            tokens = self.embed_dropout(tokens)
            encoded = self.norm(self.encoder(tokens))
            flat = encoded.reshape(encoded.shape[0], -1)
            return self.head(self.head_dropout(flat))

        def forward(self, x):
            """Forecast, treating any channel dimension independently.

            Channel independence is implemented literally: a
            ``(batch, channels, lookback)`` input is folded into
            ``(batch * channels, lookback)`` so every series meets the *same*
            backbone weights on its own, and the result is unfolded back.

            Parameters
            ----------
            x : torch.Tensor of shape (batch, lookback) or (batch, channels, lookback)
                Normalised input windows.

            Returns
            -------
            forecast : torch.Tensor
                Shape ``(batch, horizon)`` for 2-D input, or
                ``(batch, channels, horizon)`` for 3-D input.
            """
            if x.dim() == 2:
                return self.encode(x)
            if x.dim() != 3:
                raise ValueError(
                    "PatchTST expects (batch, lookback) or "
                    f"(batch, channels, lookback), got {tuple(x.shape)}."
                )
            batch, channels, length = x.shape
            folded = x.reshape(batch * channels, length)
            out = self.encode(folded)
            return out.reshape(batch, channels, -1)

    return _PatchTST()


@regressor(
    tags=["timeseries", "forecasting", "deep-learning"],
    version="1.0.0",
)
class PatchTSTForecaster(DeepForecaster):
    r"""PatchTST: a Transformer that attends over **patches**, not time steps.

    Point-wise Transformers on time series attend over individual time steps,
    which are semantically thin — a single reading carries almost no meaning
    on its own — and give attention a sequence as long as the lookback.
    PatchTST instead cuts the lookback into **patches**, short subseries of
    ``patch_len`` points, and treats each patch as one token. A patch carries
    local shape, so attention finally has something meaningful to relate, and
    the sequence length drops by roughly ``stride``, making attention's
    quadratic cost about :math:`\\text{stride}^2` cheaper.

    The second idea is **channel independence**. A multivariate series is not
    mixed inside the model; every channel is pushed through the *same* shared
    backbone on its own. This helps for two reasons. It cuts parameters
    sharply — one backbone rather than one per channel pair, and a head sized
    by patches rather than by channels — and it removes the chance to overfit
    spurious cross-channel correlations, which are abundant and unstable in
    real series. Each channel also contributes its own training examples to
    the same weights, so the effective dataset grows with the channel count
    instead of the parameter count. The public API here is univariate, but the
    backbone implements this folding, so a channel dimension is handled by
    construction.

    Instance normalisation (RevIN) wraps the whole thing: each window is
    standardised on its own mean and standard deviation and the forecast is
    mapped back with the same statistics, which is what lets one set of
    weights serve windows at wildly different levels.

    Overview
    --------
    1. Slide a ``(lookback, horizon)`` window over the series.
    2. Normalise each window by its own statistics (RevIN).
    3. Cut the window into ``n_patches`` patches, padding the end by repeating
       the last value when the arithmetic does not divide.
    4. Embed each patch to ``d_model`` and add a learned positional encoding.
    5. Run a Transformer encoder over the patch tokens.
    6. Flatten the encoded tokens and project linearly to the horizon.
    7. Denormalise with the window's own statistics.

    Theory
    ------
    A window :math:`x \\in \\mathbb{R}^{L}` is normalised,

    .. math::
        \\tilde{x} = \\frac{x - \\mu}{\\sigma}, \\qquad
        \\mu = \\frac{1}{L}\\sum_t x_t, \\quad
        \\sigma = \\sqrt{\\frac{1}{L}\\sum_t (x_t - \\mu)^2},

    and cut into :math:`N = \\lceil (L - P)/S \\rceil + 1` patches of length
    :math:`P` at stride :math:`S`. Each patch is embedded and encoded,

    .. math::
        z_i = W_e p_i + e_i, \\qquad
        Z = \\text{TransformerEncoder}(z_1, \\dots, z_N),

    where attention costs :math:`O(N^2 d)` rather than the :math:`O(L^2 d)` of
    a point-wise model. The head flattens and projects,

    .. math::
        \\hat{y} = W_h\\, \\text{vec}(Z) \\in \\mathbb{R}^{H},

    and the forecast is returned to the original units by
    :math:`\\sigma \\hat{y} + \\mu`, which is an exact inverse of the
    normalisation.

    Parameters
    ----------
    lookback : int, default=24
        Length of the input window. Automatically shrunk when the series is
        too short to yield training windows.
    horizon : int, default=8
        Number of steps the network is trained to emit at once. Longer
        forecasts are produced by autoregressive rollout.
    patch_len : int, default=8
        Points per patch. Clipped to the resolved lookback.
    stride : int, default=4
        Step between patch starts; equal to ``patch_len`` gives disjoint
        patches, smaller gives overlap.
    d_model : int, default=32
        Token embedding width. Must be divisible by ``n_heads``.
    n_heads : int, default=4
        Attention heads per encoder layer.
    n_layers : int, default=1
        Transformer encoder layers. The paper uses 3; the default here keeps
        the generic algorithm contract, which fits every model on sixty
        points, fast. Real use wants 2-4.
    dim_feedforward : int, default=64
        Width of the position-wise feed-forward network.
    dropout : float, default=0.0
        Dropout inside the encoder and before the head. The default is 0 so a
        fit is exactly reproducible; 0.1-0.3 is usual for real training runs.
    revin : bool, default=True
        Apply reversible instance normalisation to each window. Turning this
        off is almost always a mistake on a trending series.
    n_epochs : int, default=100
        Maximum passes over the window dataset. Real use wants several
        hundred.
    batch_size : int, default=32
        Windows per gradient step.
    learning_rate : float, default=0.001
        Adam step size.
    patience : int, default=15
        Epochs without an improvement in training loss before stopping early.
    device : {"cpu", "auto", "cuda", "mps"}, default="cpu"
        Where to run. The default is ``"cpu"`` because it is the only setting
        that gives bit-identical forecasts across machines; use ``"auto"``
        when speed matters more than exact reproducibility.
    random_state : int, optional, default=None
        Seed for weight initialisation and batch shuffling.

    Attributes
    ----------
    module_ : torch.nn.Module
        The trained network.
    lookback_ : int
        Lookback actually used, after any shrinking.
    horizon_ : int
        Horizon actually used, after any shrinking.
    n_windows_ : int
        Number of training windows built from the series.
    offset_ : float
        Mean of the training series, removed before fitting.
    scale_ : float
        Standard deviation of the training series, divided out before fitting.
    series_ : np.ndarray of shape (n_samples,)
        The globally scaled training series, kept for the rollout.
    loss_curve_ : np.ndarray of shape (n_epochs_run_,)
        Mean training loss per epoch.
    n_epochs_run_ : int
        Epochs actually run before early stopping.
    device_ : str
        Resolved device string.

    Notes
    -----
    **Requires PyTorch:** ``pip install 'tuiml[torch]'``. The class imports,
    constructs, registers and reports its schema without torch; only
    :meth:`fit` needs it.

    **Complexity:**

    - Training: :math:`O(E \\cdot W \\cdot (N^2 d + N d^2))` for :math:`E`
      epochs, :math:`W` windows, :math:`N` patches and width :math:`d`. The
      patching is what turns :math:`L^2` into :math:`N^2`.
    - Prediction: :math:`O(\\lceil s / H \\rceil \\cdot (N^2 d + N d^2))`.

    **When to use PatchTSTForecaster:**

    - Long lookbacks, where a point-wise Transformer's attention is both
      expensive and unfocused.
    - Series with repeating local shapes that patches can capture as units.
    - Multivariate problems where channel mixing has been found to overfit.
    - Not for very short series: with a few dozen points there are too few
      patches for attention to say anything, and
      :class:`~tuiml.algorithms.timeseries.ExponentialSmoothing` will win.

    References
    ----------
    .. [Nie2023] Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J.
           (2023). **A time series is worth 64 words: Long-term forecasting
           with Transformers.** *International Conference on Learning
           Representations (ICLR)*.
           https://doi.org/10.48550/arXiv.2211.14730
    .. [Kim2022] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J.
           (2022). **Reversible instance normalization for accurate
           time-series forecasting against distribution shift.** *ICLR*.

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.deep.NBEATSForecaster` : Fully connected doubly-residual forecaster.
    :class:`~tuiml.algorithms.timeseries.deep.NHITSForecaster` : Multi-rate variant of N-BEATS.
    :class:`~tuiml.algorithms.timeseries.ARIMA` : Classical alternative for short series.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.deep import PatchTSTForecaster
    >>> from tuiml.utils.torch_backend import has_torch
    >>> model = PatchTSTForecaster(lookback=24, horizon=6, random_state=0)
    >>> from tuiml.algorithms.timeseries.deep.patchtst import num_patches
    >>> num_patches(24, model.patch_len, model.stride)
    5
    >>> if has_torch():
    ...     y = np.sin(np.arange(200) / 5.0)
    ...     _ = model.fit(y)
    ...     print(model.predict(steps=6).shape)
    ... else:
    ...     print("(6,)")
    (6,)
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 8,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        revin: bool = True,
        n_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 15,
        device: str = "cpu",
        random_state: Optional[int] = None,
    ):
        """Record the hyperparameters. Never imports torch.

        Parameters
        ----------
        lookback : int, default=24
            Input window length.
        horizon : int, default=8
            Trained forecast length.
        patch_len : int, default=8
            Points per patch.
        stride : int, default=4
            Step between patch starts.
        d_model : int, default=32
            Token embedding width; must be divisible by ``n_heads``.
        n_heads : int, default=4
            Attention heads.
        n_layers : int, default=1
            Encoder layers.
        dim_feedforward : int, default=64
            Feed-forward width.
        dropout : float, default=0.0
            Dropout probability.
        revin : bool, default=True
            Apply reversible instance normalisation.
        n_epochs : int, default=100
            Maximum training epochs.
        batch_size : int, default=32
            Windows per gradient step.
        learning_rate : float, default=0.001
            Adam step size.
        patience : int, default=15
            Early stopping patience in epochs.
        device : {"cpu", "auto", "cuda", "mps"}, default="cpu"
            Compute device.
        random_state : int, optional, default=None
            Seed for reproducible fits.
        """
        super().__init__()
        if int(d_model) % int(n_heads) != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}); "
                "each head takes an equal slice of the embedding."
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.lookback = lookback
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.revin = revin
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device
        self.random_state = random_state

        self._init_fitted_attributes()

    def _use_instance_norm(self) -> bool:
        """Report whether RevIN is enabled.

        Returns
        -------
        enabled : bool
            The ``revin`` constructor flag.
        """
        return bool(self.revin)

    def _build_module(self, torch: Any, nn: Any, lookback: int, horizon: int) -> Any:
        """Build the PatchTST network for a resolved window.

        Parameters
        ----------
        torch : module
            The imported ``torch`` module.
        nn : module
            ``torch.nn``.
        lookback : int
            Resolved input length.
        horizon : int
            Resolved output length.

        Returns
        -------
        module : torch.nn.Module
            The network.
        """
        return _build_patchtst_module(
            torch,
            nn,
            lookback,
            horizon,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "lookback": {
                "type": "integer",
                "default": 24,
                "minimum": 1,
                "description": "Length of the input window",
            },
            "horizon": {
                "type": "integer",
                "default": 8,
                "minimum": 1,
                "description": "Steps forecast per forward pass",
            },
            "patch_len": {
                "type": "integer",
                "default": 8,
                "minimum": 1,
                "description": "Points per patch token",
            },
            "stride": {
                "type": "integer",
                "default": 4,
                "minimum": 1,
                "description": "Step between consecutive patch starts",
            },
            "d_model": {
                "type": "integer",
                "default": 32,
                "minimum": 1,
                "description": "Token embedding width; divisible by n_heads",
            },
            "n_heads": {
                "type": "integer",
                "default": 4,
                "minimum": 1,
                "description": "Attention heads per encoder layer",
            },
            "n_layers": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Transformer encoder layers",
            },
            "dim_feedforward": {
                "type": "integer",
                "default": 64,
                "minimum": 1,
                "description": "Width of the feed-forward network",
            },
            "dropout": {
                "type": "number",
                "default": 0.0,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Dropout probability; 0 keeps fits reproducible",
            },
            "revin": {
                "type": "boolean",
                "default": True,
                "description": "Apply reversible instance normalisation",
            },
            "n_epochs": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Maximum training epochs",
            },
            "batch_size": {
                "type": "integer",
                "default": 32,
                "minimum": 1,
                "description": "Windows per gradient step",
            },
            "learning_rate": {
                "type": "number",
                "default": 0.001,
                "minimum": 0.0,
                "description": "Adam step size",
            },
            "patience": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "description": "Epochs without improvement before stopping",
            },
            "device": {
                "type": "string",
                "default": "cpu",
                "enum": ["cpu", "auto", "cuda", "mps"],
                "description": "Compute device; cpu is the reproducible choice",
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Seed for initialisation and batch shuffling",
            },
        }

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return (
            "Training: O(epochs * windows * (patches^2 * d_model + "
            "patches * d_model^2)); patching reduces the sequence length from "
            "the lookback to the patch count"
        )

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Nie, Y., Nguyen, N.H., Sinthong, P. and Kalagnanam, J., 2023. A "
            "time series is worth 64 words: Long-term forecasting with "
            "Transformers. ICLR. doi:10.48550/arXiv.2211.14730",
            "Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H. and Choo, J., "
            "2022. Reversible instance normalization for accurate time-series "
            "forecasting against distribution shift. ICLR.",
        ]


__all__ = [
    "PatchTSTForecaster",
    "num_patches",
    "padded_length",
    "patchify",
    "revin_denormalize",
    "revin_normalize",
]
