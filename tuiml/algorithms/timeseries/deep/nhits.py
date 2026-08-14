"""N-HiTS: neural hierarchical interpolation for time series forecasting."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tuiml.algorithms.timeseries.deep._base import (
    DeepForecaster,
    residual_stack_loss,
)
from tuiml.base.algorithms import regressor


def pooled_length(length: int, pooling_size: int) -> int:
    """Return the length of a max-pooled sequence.

    Pooling uses ``ceil_mode``, so a partial final window still yields an
    output point and no data is silently dropped from the end of the lookback
    — the end being the part that matters most for a forecast.

    Parameters
    ----------
    length : int
        Input sequence length.
    pooling_size : int
        Pooling kernel and stride.

    Returns
    -------
    length : int
        Output sequence length, at least 1.

    Examples
    --------
    >>> from tuiml.algorithms.timeseries.deep.nhits import pooled_length
    >>> pooled_length(24, 4), pooled_length(10, 4), pooled_length(3, 8)
    (6, 3, 1)
    """
    pooling_size = max(1, min(int(pooling_size), int(length)))
    return max(1, int(math.ceil(length / pooling_size)))


def interpolation_length(horizon: int, downsample: int) -> int:
    """Return how many knots a stack predicts before interpolating up.

    Parameters
    ----------
    horizon : int
        Full forecast length.
    downsample : int
        Expressiveness ratio for this stack. A large value means few knots and
        so a smooth, low-frequency contribution.

    Returns
    -------
    n_knots : int
        Number of predicted knots, at least 1 and at most ``horizon``.

    Examples
    --------
    >>> from tuiml.algorithms.timeseries.deep.nhits import interpolation_length
    >>> interpolation_length(24, 8), interpolation_length(24, 1)
    (3, 24)
    """
    downsample = max(1, int(downsample))
    return max(1, min(int(horizon), int(math.ceil(horizon / downsample))))


def _build_nhits_module(
    torch: Any,
    nn: Any,
    lookback: int,
    horizon: int,
    pooling_sizes: Sequence[int],
    n_freq_downsample: Sequence[int],
    n_blocks: int,
    n_layers: int,
    hidden_size: int,
    interpolation_mode: str,
) -> Any:
    """Build the N-HiTS network.

    The ``nn.Module`` subclasses are declared inside this function because
    ``nn`` does not exist at module scope: importing this file must work on an
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
    pooling_sizes : sequence of int
        Max-pooling rate per stack, coarsest first.
    n_freq_downsample : sequence of int
        Interpolation ratio per stack, coarsest first.
    n_blocks : int
        Blocks per stack.
    n_layers : int
        Fully connected layers per block trunk.
    hidden_size : int
        Trunk width.
    interpolation_mode : {"linear", "nearest"}
        How knots are expanded to the full resolution.

    Returns
    -------
    module : torch.nn.Module
        Maps ``(batch, lookback)`` to ``(batch, horizon)``.
    """
    functional = nn.functional

    def _expand(x, size):
        """Expand ``(batch, n_knots)`` to ``(batch, size)``.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, n_knots)
            Predicted knots.
        size : int
            Target length.

        Returns
        -------
        expanded : torch.Tensor of shape (batch, size)
            The interpolated signal.
        """
        if x.shape[1] == size:
            return x
        x = x.unsqueeze(1)
        if x.shape[2] == 1 or interpolation_mode == "nearest":
            out = functional.interpolate(x, size=size, mode="nearest")
        else:
            out = functional.interpolate(
                x, size=size, mode="linear", align_corners=True
            )
        return out.squeeze(1)

    class _NHiTSBlock(nn.Module):
        """One N-HiTS block: pool, encode, predict knots, interpolate up."""

        def __init__(self, pooling_size: int, downsample: int):
            """Build the pooling layer, the trunk and the two knot heads.

            Parameters
            ----------
            pooling_size : int
                Max-pooling kernel and stride for this stack.
            downsample : int
                Interpolation ratio for this stack.
            """
            super().__init__()
            self.pooling_size = max(1, min(int(pooling_size), lookback))
            self.pool = nn.MaxPool1d(
                kernel_size=self.pooling_size,
                stride=self.pooling_size,
                ceil_mode=True,
            )
            self.pooled_size = pooled_length(lookback, self.pooling_size)
            self.n_forecast_knots = interpolation_length(horizon, downsample)
            self.n_backcast_knots = self.pooled_size

            layers: List[Any] = []
            width = self.pooled_size
            for _ in range(int(n_layers)):
                layers.append(nn.Linear(width, hidden_size))
                layers.append(nn.ReLU())
                width = hidden_size
            self.trunk = nn.Sequential(*layers)
            self.theta_backcast = nn.Linear(width, self.n_backcast_knots)
            self.theta_forecast = nn.Linear(width, self.n_forecast_knots)

        def forward(self, x):
            """Return this block's backcast and interpolated forecast.

            Parameters
            ----------
            x : torch.Tensor of shape (batch, lookback)
                The residual reaching this block.

            Returns
            -------
            backcast : torch.Tensor of shape (batch, lookback)
                What the block explains of its input.
            forecast : torch.Tensor of shape (batch, horizon)
                The block's contribution to the forecast.
            """
            pooled = self.pool(x.unsqueeze(1)).squeeze(1)
            hidden = self.trunk(pooled)
            backcast = _expand(self.theta_backcast(hidden), lookback)
            forecast = _expand(self.theta_forecast(hidden), horizon)
            return backcast, forecast

    class _NHiTS(nn.Module):
        """Multi-rate doubly-residual stack."""

        def __init__(self):
            """Assemble one stack per (pooling size, downsample) pair."""
            super().__init__()
            blocks: List[Any] = []
            owner: List[int] = []
            for stack_index, (pool, down) in enumerate(
                zip(pooling_sizes, n_freq_downsample)
            ):
                for _ in range(int(n_blocks)):
                    blocks.append(_NHiTSBlock(pool, down))
                    owner.append(stack_index)
            self.blocks = nn.ModuleList(blocks)
            self.block_stack = owner
            self.n_stacks = len(list(pooling_sizes))

        def forward(self, x):
            """Return the summed forecast.

            Parameters
            ----------
            x : torch.Tensor of shape (batch, lookback)
                Normalised input windows.

            Returns
            -------
            forecast : torch.Tensor of shape (batch, horizon)
                The model's forecast.
            """
            return self.forward_detail(x)["forecast"]

        def forward_detail(self, x):
            """Return the forecast alongside every intermediate quantity.

            Parameters
            ----------
            x : torch.Tensor of shape (batch, lookback)
                Normalised input windows.

            Returns
            -------
            detail : dict of str to torch.Tensor
                ``"forecast"``, ``"block_forecasts"``, ``"stack_forecasts"``
                and ``"residuals"``.
            """
            residual = x
            residuals = [residual]
            block_forecasts = []
            stack_totals = [
                torch.zeros(x.shape[0], horizon, dtype=x.dtype, device=x.device)
                for _ in range(self.n_stacks)
            ]
            for block, stack_index in zip(self.blocks, self.block_stack):
                backcast, forecast = block(residual)
                residual = residual - backcast
                residuals.append(residual)
                block_forecasts.append(forecast)
                stack_totals[stack_index] = stack_totals[stack_index] + forecast

            stacked = torch.stack(block_forecasts, dim=1)
            return {
                "forecast": stacked.sum(dim=1),
                "block_forecasts": stacked,
                "stack_forecasts": torch.stack(stack_totals, dim=1),
                "residuals": torch.stack(residuals, dim=1),
            }

    return _NHiTS()


@regressor(
    tags=["timeseries", "forecasting", "deep-learning"],
    version="1.0.0",
)
class NHITSForecaster(DeepForecaster):
    r"""N-HiTS: N-BEATS with **multi-rate sampling** and **hierarchical interpolation**.

    N-HiTS keeps the doubly-residual skeleton of N-BEATS and adds the two
    ideas that make long horizons tractable. First, **multi-rate signal
    sampling**: each stack max-pools its input at a different rate before the
    fully connected trunk sees it, so a stack with a large pooling size
    literally cannot see high-frequency detail and is forced to model the slow
    component. Second, **hierarchical interpolation**: a stack predicts only a
    handful of knots and interpolates them up to the full horizon, so the
    number of parameters in the output layer no longer grows with the horizon.

    Together the two make the stacks specialise by frequency — coarse stacks
    supply the smooth backbone, fine stacks the detail — while cutting both
    compute and parameter count against N-BEATS at long horizons. Remove
    either one and what remains is N-BEATS with extra steps.

    Overview
    --------
    1. Slide a ``(lookback, horizon)`` window over the series and normalise
       each window by its own statistics.
    2. For stack :math:`s`, max-pool the incoming residual with kernel
       :math:`k_s` — large for the first stack, 1 for the last.
    3. Run the pooled signal through a fully connected trunk and predict
       :math:`\\lceil H / r_s \\rceil` forecast knots and a pooled backcast.
    4. Interpolate both back to full resolution: the backcast to the lookback,
       the forecast to the horizon.
    5. Subtract the backcast, pass the residual on, and sum the forecasts.

    Theory
    ------
    Stack :math:`s` first pools its input :math:`x_s` at rate :math:`k_s`,

    .. math::
        x^{p}_s = \\text{MaxPool}_{k_s}(x_s),

    which acts as an anti-alias filter: frequencies above :math:`1/(2k_s)` are
    removed before the trunk sees them. The trunk emits knots
    :math:`\\theta_s \\in \\mathbb{R}^{\\lceil H/r_s \\rceil}` that are expanded
    by a temporal interpolation operator :math:`g`,

    .. math::
        \\hat{y}_s[t] = g(\\theta_s)[t], \\qquad t = 1, \\dots, H,

    with :math:`g` linear interpolation over the knot grid. Choosing
    :math:`r_s` in step with :math:`k_s` gives each stack a matched input and
    output bandwidth; the forecast is again the sum,
    :math:`\\hat{y} = \\sum_s \\hat{y}_s`, over residuals
    :math:`x_{s+1} = x_s - \\hat{x}_s`.

    Parameters
    ----------
    lookback : int, default=24
        Length of the input window. Automatically shrunk when the series is
        too short to yield training windows.
    horizon : int, default=8
        Number of steps the network is trained to emit at once. Longer
        forecasts are produced by autoregressive rollout.
    pooling_sizes : tuple of int, default=(4, 2, 1)
        Max-pooling rate per stack, coarsest first. One entry per stack.
    n_freq_downsample : tuple of int, default=(4, 2, 1)
        Interpolation ratio per stack; a stack predicts
        ``ceil(horizon / ratio)`` knots. Must be the same length as
        ``pooling_sizes``.
    n_blocks : int, default=1
        Blocks per stack.
    n_layers : int, default=2
        Fully connected layers in each block's trunk.
    hidden_size : int, default=64
        Width of the trunk. The paper uses 512; the default here keeps the
        generic algorithm contract, which fits every model on sixty points,
        fast. Real use wants 256-512.
    interpolation_mode : {"linear", "nearest"}, default="linear"
        How knots are expanded to full resolution.
    backcast_loss_weight : float, default=0.5
        Weight on a penalty applied to the residual leaving the last block.
        The published objective supervises only the forecast, which leaves the
        backcast heads free to *grow* the residual; a small weight here is what
        makes the residual cascade real. Set to 0 for the paper's exact
        objective.
    n_epochs : int, default=100
        Maximum passes over the window dataset. Real use wants several
        hundred to a few thousand.
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

    - Training: :math:`O(E \\cdot W \\cdot S \\cdot B \\cdot h^2)` for
      :math:`E` epochs, :math:`W` windows, :math:`S` stacks and :math:`B`
      trunk layers of width :math:`h`. Unlike N-BEATS the output layer is
      :math:`O(h \\cdot H / r)` rather than :math:`O(h \\cdot H)`.
    - Prediction: :math:`O(\\lceil s / H \\rceil \\cdot S \\cdot B \\cdot h^2)`.

    **When to use NHITSForecaster:**

    - Long horizons, where N-BEATS output layers become the bottleneck.
    - Series with structure at clearly separated timescales.
    - When you want N-BEATS accuracy at a fraction of the parameters.
    - Not for very short series; a classical model will usually win.

    References
    ----------
    .. [Challu2023] Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F.,
           Mergenthaler-Canseco, M., & Dubrawski, A. (2023). **N-HiTS: Neural
           hierarchical interpolation for time series forecasting.**
           *Proceedings of the AAAI Conference on Artificial Intelligence*,
           37(6), 6989-6997. https://doi.org/10.1609/aaai.v37i6.25854

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.deep.NBEATSForecaster` : The architecture N-HiTS extends.
    :class:`~tuiml.algorithms.timeseries.deep.PatchTSTForecaster` : Patch-based transformer forecaster.
    :class:`~tuiml.algorithms.timeseries.STLDecomposition` : Classical multi-scale decomposition.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.deep import NHITSForecaster
    >>> from tuiml.utils.torch_backend import has_torch
    >>> model = NHITSForecaster(lookback=24, horizon=6, random_state=0)
    >>> model.pooling_sizes
    (4, 2, 1)
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
        pooling_sizes: Tuple[int, ...] = (4, 2, 1),
        n_freq_downsample: Tuple[int, ...] = (4, 2, 1),
        n_blocks: int = 1,
        n_layers: int = 2,
        hidden_size: int = 64,
        interpolation_mode: str = "linear",
        backcast_loss_weight: float = 0.5,
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
        pooling_sizes : tuple of int, default=(4, 2, 1)
            Max-pooling rate per stack, coarsest first.
        n_freq_downsample : tuple of int, default=(4, 2, 1)
            Interpolation ratio per stack.
        n_blocks : int, default=1
            Blocks per stack.
        n_layers : int, default=2
            Trunk depth.
        hidden_size : int, default=64
            Trunk width.
        interpolation_mode : {"linear", "nearest"}, default="linear"
            Knot expansion mode.
        backcast_loss_weight : float, default=0.5
            Weight on the final-residual penalty.
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
        pooling_sizes = tuple(int(v) for v in pooling_sizes)
        n_freq_downsample = tuple(int(v) for v in n_freq_downsample)
        if not pooling_sizes:
            raise ValueError("pooling_sizes must name at least one stack.")
        if len(pooling_sizes) != len(n_freq_downsample):
            raise ValueError(
                "pooling_sizes and n_freq_downsample must have one entry per "
                f"stack, got {len(pooling_sizes)} and {len(n_freq_downsample)}."
            )
        if any(v < 1 for v in pooling_sizes + n_freq_downsample):
            raise ValueError(
                "pooling_sizes and n_freq_downsample entries must be >= 1."
            )
        if interpolation_mode not in ("linear", "nearest"):
            raise ValueError(
                "interpolation_mode must be 'linear' or 'nearest', got "
                f"{interpolation_mode!r}."
            )

        self.lookback = lookback
        self.horizon = horizon
        self.pooling_sizes = pooling_sizes
        self.n_freq_downsample = n_freq_downsample
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.interpolation_mode = interpolation_mode
        self.backcast_loss_weight = backcast_loss_weight
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device
        self.random_state = random_state

        self._init_fitted_attributes()

    def _build_module(self, torch: Any, nn: Any, lookback: int, horizon: int) -> Any:
        """Build the N-HiTS network for a resolved window.

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
        return _build_nhits_module(
            torch,
            nn,
            lookback,
            horizon,
            pooling_sizes=self.pooling_sizes,
            n_freq_downsample=self.n_freq_downsample,
            n_blocks=self.n_blocks,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size,
            interpolation_mode=self.interpolation_mode,
        )

    def _loss(self, torch, module, xb, yb):
        """Return the forecast loss plus the backcast reconstruction term.

        Parameters
        ----------
        torch : module
            The imported ``torch`` module.
        module : torch.nn.Module
            The network being trained.
        xb : torch.Tensor of shape (batch, lookback)
            Normalised input windows.
        yb : torch.Tensor of shape (batch, horizon)
            Normalised target windows.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss.
        """
        return residual_stack_loss(
            torch, module, xb, yb, self.backcast_loss_weight
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
            "pooling_sizes": {
                "type": "array",
                "default": [4, 2, 1],
                "items": {"type": "integer", "minimum": 1},
                "description": "Max-pooling rate per stack, coarsest first",
            },
            "n_freq_downsample": {
                "type": "array",
                "default": [4, 2, 1],
                "items": {"type": "integer", "minimum": 1},
                "description": "Interpolation ratio per stack",
            },
            "n_blocks": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Blocks per stack",
            },
            "n_layers": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Fully connected layers per block trunk",
            },
            "hidden_size": {
                "type": "integer",
                "default": 64,
                "minimum": 1,
                "description": "Width of each block trunk",
            },
            "interpolation_mode": {
                "type": "string",
                "default": "linear",
                "enum": ["linear", "nearest"],
                "description": "How knots are expanded to full resolution",
            },
            "backcast_loss_weight": {
                "type": "number",
                "default": 0.5,
                "minimum": 0.0,
                "description": "Weight on the final-residual reconstruction term",
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
            "Training: O(epochs * windows * stacks * layers * hidden^2); "
            "output layer O(hidden * horizon / ratio) rather than "
            "O(hidden * horizon)"
        )

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Challu, C., Olivares, K.G., Oreshkin, B.N., Garza, F., "
            "Mergenthaler-Canseco, M. and Dubrawski, A., 2023. N-HiTS: Neural "
            "hierarchical interpolation for time series forecasting. AAAI, "
            "37(6), pp.6989-6997. doi:10.1609/aaai.v37i6.25854",
        ]


__all__ = ["NHITSForecaster", "interpolation_length", "pooled_length"]
