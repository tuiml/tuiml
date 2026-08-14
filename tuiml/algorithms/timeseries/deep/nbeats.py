"""N-BEATS: neural basis expansion analysis for time series forecasting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml.algorithms.timeseries.deep._base import (
    DeepForecaster,
    residual_stack_loss,
)
from tuiml.base.algorithms import regressor


def polynomial_basis(degree: int, length: int) -> np.ndarray:
    """Build the polynomial basis N-BEATS uses for its trend stack.

    Row :math:`p` holds :math:`t^p` on a time grid normalised to
    :math:`[0, 1)`, so a linear combination of the rows is a polynomial of
    order ``degree`` — a deliberately low-order, slowly varying function.

    Parameters
    ----------
    degree : int
        Highest power in the basis. The basis has ``degree + 1`` rows.
    length : int
        Number of time steps to evaluate the basis on.

    Returns
    -------
    basis : np.ndarray of shape (degree + 1, length)
        The basis matrix.

    Examples
    --------
    >>> from tuiml.algorithms.timeseries.deep.nbeats import polynomial_basis
    >>> polynomial_basis(1, 4).round(2)
    array([[1.  , 1.  , 1.  , 1.  ],
           [0.  , 0.25, 0.5 , 0.75]])
    """
    grid = np.arange(length, dtype=np.float64) / float(max(1, length))
    return np.stack([grid ** power for power in range(int(degree) + 1)])


def fourier_basis(n_harmonics: int, length: int) -> np.ndarray:
    """Build the Fourier basis N-BEATS uses for its seasonality stack.

    Harmonics run from 1 to ``n_harmonics`` over a fundamental period of
    exactly ``length``. The zeroth (constant) harmonic is deliberately
    omitted: it would let the seasonality stack absorb the level, blurring the
    split against the trend stack. Because it is omitted, every row — and so
    every linear combination of them — has exactly zero mean over one period
    and repeats with period ``length``.

    Parameters
    ----------
    n_harmonics : int
        Number of harmonics. The basis has ``2 * n_harmonics`` rows, a cosine
        and a sine per harmonic.
    length : int
        Number of time steps to evaluate the basis on. Passing a multiple of
        the fundamental period yields the periodic extension.

    Returns
    -------
    basis : np.ndarray of shape (2 * n_harmonics, length)
        The basis matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.deep.nbeats import fourier_basis
    >>> basis = fourier_basis(2, 8)
    >>> basis.shape
    (4, 8)
    >>> bool(np.allclose(basis.mean(axis=1), 0.0))
    True
    """
    grid = np.arange(length, dtype=np.float64) / float(max(1, length))
    rows: List[np.ndarray] = []
    for harmonic in range(1, int(n_harmonics) + 1):
        angle = 2.0 * np.pi * harmonic * grid
        rows.append(np.cos(angle))
        rows.append(np.sin(angle))
    return np.stack(rows)


def _build_nbeats_module(
    torch: Any,
    nn: Any,
    lookback: int,
    horizon: int,
    stack_type: str,
    n_stacks: int,
    n_blocks: int,
    n_layers: int,
    hidden_size: int,
    trend_polynomial_degree: int,
    n_harmonics: int,
) -> Any:
    """Build the N-BEATS network.

    Every ``nn.Module`` subclass is declared inside this function because
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
    stack_type : {"generic", "interpretable"}
        Basis configuration.
    n_stacks : int
        Number of stacks; forced to 2 for the interpretable configuration.
    n_blocks : int
        Blocks per stack.
    n_layers : int
        Fully connected layers in each block's trunk.
    hidden_size : int
        Width of the trunk.
    trend_polynomial_degree : int
        Polynomial order of the trend stack.
    n_harmonics : int
        Harmonics in the seasonality stack.

    Returns
    -------
    module : torch.nn.Module
        Maps ``(batch, lookback)`` to ``(batch, horizon)``.
    """

    class _Block(nn.Module):
        """One N-BEATS block: a trunk, then a backcast and a forecast head."""

        def __init__(self, backcast_basis, forecast_basis):
            """Build the trunk and the two expansion-coefficient heads.

            Parameters
            ----------
            backcast_basis : np.ndarray or None
                Basis matrix of shape ``(theta_b, lookback)``; ``None`` for the
                generic block, whose basis is learned directly.
            forecast_basis : np.ndarray or None
                Basis matrix of shape ``(theta_f, horizon)``; ``None`` for the
                generic block.
            """
            super().__init__()
            layers: List[Any] = []
            width = lookback
            for _ in range(n_layers):
                layers.append(nn.Linear(width, hidden_size))
                layers.append(nn.ReLU())
                width = hidden_size
            self.trunk = nn.Sequential(*layers)

            theta_b = lookback if backcast_basis is None else backcast_basis.shape[0]
            theta_f = horizon if forecast_basis is None else forecast_basis.shape[0]
            self.theta_backcast = nn.Linear(width, theta_b)
            self.theta_forecast = nn.Linear(width, theta_f)

            self.has_basis = backcast_basis is not None
            if self.has_basis:
                self.register_buffer(
                    "backcast_basis",
                    torch.as_tensor(backcast_basis, dtype=torch.float32),
                )
                self.register_buffer(
                    "forecast_basis",
                    torch.as_tensor(forecast_basis, dtype=torch.float32),
                )

        def forward(self, x):
            """Return this block's backcast and forecast.

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
            hidden = self.trunk(x)
            theta_b = self.theta_backcast(hidden)
            theta_f = self.theta_forecast(hidden)
            if not self.has_basis:
                return theta_b, theta_f
            return theta_b @ self.backcast_basis, theta_f @ self.forecast_basis

    class _NBeats(nn.Module):
        """Doubly-residual stack of blocks."""

        def __init__(self):
            """Assemble the stacks and record which stack each block is in."""
            super().__init__()
            blocks: List[Any] = []
            owner: List[int] = []

            if stack_type == "interpretable":
                trend_b = polynomial_basis(trend_polynomial_degree, lookback)
                trend_f = polynomial_basis(trend_polynomial_degree, horizon)
                season_b = fourier_basis(n_harmonics, lookback)
                season_f = fourier_basis(n_harmonics, horizon)
                specs = [(trend_b, trend_f), (season_b, season_f)]
            else:
                specs = [(None, None)] * int(n_stacks)

            for stack_index, (basis_b, basis_f) in enumerate(specs):
                for _ in range(int(n_blocks)):
                    blocks.append(_Block(basis_b, basis_f))
                    owner.append(stack_index)

            self.blocks = nn.ModuleList(blocks)
            self.block_stack = owner
            self.n_stacks = len(specs)

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
                ``"forecast"`` of shape ``(batch, horizon)``;
                ``"block_forecasts"`` of shape ``(batch, n_blocks, horizon)``;
                ``"stack_forecasts"`` of shape ``(batch, n_stacks, horizon)``;
                ``"residuals"`` of shape ``(batch, n_blocks + 1, lookback)``,
                starting with the input itself.
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

    return _NBeats()


@regressor(
    tags=["timeseries", "forecasting", "deep-learning"],
    version="1.0.0",
)
class NBEATSForecaster(DeepForecaster):
    r"""N-BEATS: **doubly-residual** stacks of fully connected blocks.

    N-BEATS forecasts a series from a lookback window using nothing but dense
    layers — no recurrence, no convolution, no attention — and still beat the
    winner of the M4 competition. Its one structural idea is the **double
    residual**: every block emits both a **backcast** (its reconstruction of
    the input it saw) and a **forecast**. The backcast is *subtracted* from the
    block's input before the next block runs, so each block only ever works on
    the part of the signal its predecessors could not explain, and the final
    forecast is the plain sum of the block forecasts.

    Two basis configurations are available. The **generic** one learns the
    backcast and forecast vectors directly and is the stronger forecaster. The
    **interpretable** one constrains the first stack to a low-order polynomial
    (trend) and the second to a Fourier series (seasonality), which costs a
    little accuracy and buys a decomposition you can plot — see
    :meth:`decompose`.

    Overview
    --------
    1. Slide a ``(lookback, horizon)`` window over the series to build a
       supervised dataset.
    2. Normalise each window by its own mean and standard deviation.
    3. Pass the window through the first block: a stack of ReLU layers
       producing expansion coefficients :math:`\\theta`, projected onto a
       backcast basis and a forecast basis.
    4. Subtract the backcast from the block input and hand the residual to the
       next block; accumulate the forecasts.
    5. Denormalise the summed forecast with the window's own statistics.

    Theory
    ------
    Block :math:`\\ell` receives residual :math:`x_\\ell` and produces

    .. math::
        \\hat{x}_\\ell = V^b \\theta^b_\\ell, \\qquad
        \\hat{y}_\\ell = V^f \\theta^f_\\ell,

    where :math:`\\theta_\\ell = g_\\ell(x_\\ell)` is the output of the fully
    connected trunk and :math:`V^b, V^f` are the backcast and forecast bases.
    The residual recursion and the forecast aggregation are

    .. math::
        x_{\\ell+1} = x_\\ell - \\hat{x}_\\ell, \\qquad
        \\hat{y} = \\sum_{\\ell=1}^{L} \\hat{y}_\\ell .

    For the generic configuration :math:`V^b` and :math:`V^f` are identity —
    the trunk emits the vectors themselves. For the interpretable
    configuration the trend stack uses a polynomial basis of order :math:`p`,

    .. math::
        \\hat{y}^{\\text{tr}}_\\ell = \\sum_{i=0}^{p} \\theta_{\\ell,i}\\, t^i,
        \\qquad t = 0, \\tfrac{1}{H}, \\dots, \\tfrac{H-1}{H},

    and the seasonality stack a Fourier basis with fundamental period
    :math:`H`,

    .. math::
        \\hat{y}^{\\text{se}}_\\ell = \\sum_{k=1}^{K}
        \\left[ \\alpha_{\\ell,k} \\cos(2\\pi k t)
              + \\beta_{\\ell,k} \\sin(2\\pi k t) \\right].

    The constant harmonic :math:`k = 0` is omitted so the seasonal component
    has exactly zero mean and cannot absorb the level.

    Parameters
    ----------
    lookback : int, default=24
        Length of the input window. Automatically shrunk when the series is
        too short to yield training windows.
    horizon : int, default=8
        Number of steps the network is trained to emit at once. Forecasts
        longer than this are produced by autoregressive rollout.
    stack_type : {"generic", "interpretable"}, default="generic"
        ``"generic"`` learns the bases; ``"interpretable"`` uses a trend stack
        and a seasonality stack and enables :meth:`decompose`.
    n_stacks : int, default=2
        Number of stacks in the generic configuration. Ignored when
        ``stack_type="interpretable"``, which always has exactly two.
    n_blocks : int, default=2
        Blocks per stack.
    n_layers : int, default=2
        Fully connected layers in each block's trunk.
    hidden_size : int, default=64
        Width of the trunk. The paper uses 512; the default here is sized so
        the generic algorithm contract, which fits every model on sixty
        points, stays fast. Real use wants 256-512.
    trend_polynomial_degree : int, default=2
        Polynomial order of the interpretable trend stack.
    n_harmonics : int, default=4
        Harmonics in the interpretable seasonality stack.
    backcast_loss_weight : float, default=0.5
        Weight on a penalty applied to the residual leaving the last block.
        The paper's objective supervises only the forecast, which leaves the
        backcast heads free to *grow* the residual; a small weight here is
        what makes the doubly-residual cascade real. Set to 0 for the paper's
        exact objective.
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
        Seed for weight initialisation and batch shuffling. With a seed and
        ``device="cpu"`` the forecast is reproducible.

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

    - Training: :math:`O(E \\cdot W \\cdot L \\cdot B \\cdot h)` for :math:`E`
      epochs, :math:`W` windows, :math:`L` blocks, :math:`B` trunk layers of
      width :math:`h`.
    - Prediction: :math:`O(\\lceil s / H \\rceil \\cdot L \\cdot B \\cdot h)`
      for :math:`s` steps.

    **When to use NBEATSForecaster:**

    - Univariate series with a few hundred points or more, where a classical
      model underfits the shape.
    - When you want a strong forecaster with no feature engineering.
    - When an explicit trend/seasonality split is wanted, via
      ``stack_type="interpretable"``.
    - Not for very short series: with fewer than roughly a hundred points,
      :class:`~tuiml.algorithms.timeseries.ExponentialSmoothing` or
      :class:`~tuiml.algorithms.timeseries.ARIMA` will usually win.

    References
    ----------
    .. [Oreshkin2020] Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y.
           (2020). **N-BEATS: Neural basis expansion analysis for
           interpretable time series forecasting.** *International Conference
           on Learning Representations (ICLR)*.
           https://doi.org/10.48550/arXiv.1905.10437

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.deep.NHITSForecaster` : N-BEATS plus multi-rate pooling and hierarchical interpolation.
    :class:`~tuiml.algorithms.timeseries.deep.PatchTSTForecaster` : Patch-based transformer forecaster.
    :class:`~tuiml.algorithms.timeseries.ARIMA` : Classical alternative for short series.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.deep import NBEATSForecaster
    >>> from tuiml.utils.torch_backend import has_torch
    >>> model = NBEATSForecaster(lookback=24, horizon=6, random_state=0)
    >>> sorted(model.get_parameter_schema())[:3]
    ['backcast_loss_weight', 'batch_size', 'device']
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
        stack_type: str = "generic",
        n_stacks: int = 2,
        n_blocks: int = 2,
        n_layers: int = 2,
        hidden_size: int = 64,
        trend_polynomial_degree: int = 2,
        n_harmonics: int = 4,
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
        stack_type : {"generic", "interpretable"}, default="generic"
            Basis configuration.
        n_stacks : int, default=2
            Stacks in the generic configuration.
        n_blocks : int, default=2
            Blocks per stack.
        n_layers : int, default=2
            Trunk depth.
        hidden_size : int, default=64
            Trunk width.
        trend_polynomial_degree : int, default=2
            Polynomial order of the trend stack.
        n_harmonics : int, default=4
            Harmonics in the seasonality stack.
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
        if stack_type not in ("generic", "interpretable"):
            raise ValueError(
                f"stack_type must be 'generic' or 'interpretable', got "
                f"{stack_type!r}."
            )
        self.lookback = lookback
        self.horizon = horizon
        self.stack_type = stack_type
        self.n_stacks = n_stacks
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.trend_polynomial_degree = trend_polynomial_degree
        self.n_harmonics = n_harmonics
        self.backcast_loss_weight = backcast_loss_weight
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device
        self.random_state = random_state

        self._init_fitted_attributes()

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    def _build_module(self, torch: Any, nn: Any, lookback: int, horizon: int) -> Any:
        """Build the N-BEATS network for a resolved window.

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
        return _build_nbeats_module(
            torch,
            nn,
            lookback,
            horizon,
            stack_type=self.stack_type,
            n_stacks=self.n_stacks,
            n_blocks=self.n_blocks,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size,
            trend_polynomial_degree=self.trend_polynomial_degree,
            n_harmonics=self.n_harmonics,
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

    def _forward_parts(self, torch: Any, xn: Any) -> Dict[str, Any]:
        """Run the network, exposing the stack split when interpretable.

        Parameters
        ----------
        torch : module
            The imported ``torch`` module.
        xn : torch.Tensor of shape (1, lookback)
            One normalised input window.

        Returns
        -------
        outputs : dict of str to torch.Tensor
            ``"forecast"`` always; ``"trend"`` and ``"seasonality"`` when
            ``stack_type="interpretable"``.
        """
        detail = self.module_.forward_detail(xn)
        outputs = {"forecast": detail["forecast"]}
        if self.stack_type == "interpretable":
            stacks = detail["stack_forecasts"]
            outputs["trend"] = stacks[:, 0, :]
            outputs["seasonality"] = stacks[:, 1, :]
        return outputs

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    def decompose(self, steps: int = 1) -> Dict[str, np.ndarray]:
        """Split the forecast into its trend and seasonality components.

        Only available when the model was built with
        ``stack_type="interpretable"``. The two components sum exactly to
        :meth:`predict`; the series level is attributed to the trend, which is
        where it belongs, since the Fourier basis has zero mean by
        construction.

        Parameters
        ----------
        steps : int, default=1
            Number of future points to decompose.

        Returns
        -------
        components : dict of str to np.ndarray
            ``"trend"``, ``"seasonality"`` and ``"forecast"``, each of shape
            ``(steps,)`` and in the units of the training series.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If the model is not the interpretable configuration.
        """
        self._check_is_fitted()
        if self.stack_type != "interpretable":
            raise ValueError(
                "decompose() needs the interpretable configuration; this model "
                f"was built with stack_type={self.stack_type!r}. Construct it "
                "as NBEATSForecaster(stack_type='interpretable')."
            )
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}.")

        forecast, parts = self._rollout(steps)
        return {
            "trend": parts["trend"] * self.scale_ + self.offset_,
            "seasonality": parts["seasonality"] * self.scale_,
            "forecast": forecast * self.scale_ + self.offset_,
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
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
            "stack_type": {
                "type": "string",
                "default": "generic",
                "enum": ["generic", "interpretable"],
                "description": "Learned bases, or trend plus seasonality bases",
            },
            "n_stacks": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Stacks in the generic configuration",
            },
            "n_blocks": {
                "type": "integer",
                "default": 2,
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
            "trend_polynomial_degree": {
                "type": "integer",
                "default": 2,
                "minimum": 0,
                "description": "Polynomial order of the trend stack",
            },
            "n_harmonics": {
                "type": "integer",
                "default": 4,
                "minimum": 1,
                "description": "Harmonics in the seasonality stack",
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
            "Training: O(epochs * windows * blocks * layers * hidden^2); "
            "Prediction: O(ceil(steps/horizon) * blocks * layers * hidden^2)"
        )

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Oreshkin, B.N., Carpov, D., Chapados, N. and Bengio, Y., 2020. "
            "N-BEATS: Neural basis expansion analysis for interpretable time "
            "series forecasting. ICLR. doi:10.48550/arXiv.1905.10437",
        ]


__all__ = ["NBEATSForecaster", "fourier_basis", "polynomial_basis"]
