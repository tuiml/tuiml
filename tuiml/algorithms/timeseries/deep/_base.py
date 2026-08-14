"""Shared machinery for the deep, torch-backed forecasters.

The three models in this subpackage — N-BEATS, N-HiTS and PatchTST — differ
only in the neural block that maps a lookback window to a horizon. Everything
around that block is identical: turning one series into a supervised
window dataset, scaling, seeding, device placement, the training loop with
early stopping, the autoregressive rollout when the requested number of steps
exceeds the trained horizon, and the not-fitted guard.

All of that lives here so the model files stay about their architecture.

Notes
-----
**No torch at module scope.** Nothing in this module imports torch on import.
The helpers below are pure NumPy; :class:`DeepForecaster` imports torch inside
``fit`` via :func:`~tuiml.utils.torch_backend.require_torch`. This keeps the
classes constructible, registrable and inspectable on an install without
PyTorch.

Examples
--------
>>> import numpy as np
>>> from tuiml.algorithms.timeseries.deep._base import resolve_window
>>> resolve_window(60, lookback=24, horizon=8)
(24, 8)
"""

from __future__ import annotations

import copy
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tuiml.base.algorithms import Regressor
from tuiml.utils.torch_backend import require_torch, resolve_device

#: Libraries that bundle their own ``libomp.dylib`` on macOS. Importing
#: :mod:`tuiml.algorithms` pulls all three in, so by the time a lazy
#: ``import torch`` runs there is already a second OpenMP runtime in the
#: process.
_OPENMP_CONFLICTING_MODULES = ("xgboost", "lightgbm", "catboost")

#: The guard changes a process-wide setting, so it is applied at most once.
_openmp_guard_checked = False

#: Shortest lookback the models are allowed to shrink to. Below four points a
#: window carries no usable shape and the block degenerates to a bias term.
MIN_LOOKBACK = 4

#: Fewest training windows a fit will accept. One window is not a dataset;
#: four is the smallest number that still lets a batch mean mean anything.
MIN_WINDOWS = 4


def guard_duplicate_openmp(torch: Any) -> bool:
    """Force torch to one thread when a second OpenMP runtime is loaded.

    On macOS, xgboost, LightGBM and CatBoost each load a bundled
    ``libomp.dylib``, and :mod:`tuiml.algorithms` imports all three. Torch is
    then imported *afterwards* — which is precisely what the optional
    dependency contract demands — leaving two OpenMP runtimes in one process.
    The first torch operation to open a parallel region segfaults the
    interpreter; in practice that is the fused optimiser step, so the crash
    lands in the middle of training with no Python traceback.

    Running torch single-threaded avoids parallel regions entirely. These
    networks are small, so the cost is minor, and a slower fit beats a dead
    interpreter. The guard does nothing when no conflicting library is loaded,
    so a torch-only process keeps all its threads.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.

    Returns
    -------
    applied : bool
        Whether the thread count was actually clamped.
    """
    global _openmp_guard_checked
    if _openmp_guard_checked:
        return False
    _openmp_guard_checked = True

    if sys.platform != "darwin":
        return False
    if not any(name in sys.modules for name in _OPENMP_CONFLICTING_MODULES):
        return False
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        return True
    return False


def check_series(y: Any) -> np.ndarray:
    """Validate and coerce a forecasting target into a 1-D float array.

    Parameters
    ----------
    y : array-like
        The series to forecast. A column vector of shape ``(n, 1)`` is
        accepted and flattened; anything genuinely multivariate is rejected.

    Returns
    -------
    series : np.ndarray of shape (n_samples,)
        Contiguous ``float64`` copy of the series.

    Raises
    ------
    ValueError
        If the input is not one-dimensional, is empty, or holds non-finite
        values.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.deep._base import check_series
    >>> check_series([1.0, 2.0, 3.0])
    array([1., 2., 3.])
    """
    series = np.asarray(y, dtype=np.float64)
    if series.ndim == 2 and series.shape[1] == 1:
        series = series.ravel()
    if series.ndim != 1:
        raise ValueError(
            "Deep forecasters take a single series: y must be 1-D (or a "
            f"column vector), got an array of shape {series.shape}. These "
            "models forecast a series from its own history, so y is the "
            "series itself, not a design matrix."
        )
    if series.size == 0:
        raise ValueError("y is empty: nothing to forecast from.")
    if not np.all(np.isfinite(series)):
        raise ValueError(
            "y contains NaN or infinite values. Deep forecasters do not "
            "support missing values; impute or drop them first."
        )
    return np.ascontiguousarray(series)


def resolve_window(
    n_samples: int,
    lookback: int,
    horizon: int,
    min_windows: int = MIN_WINDOWS,
    min_lookback: int = MIN_LOOKBACK,
) -> Tuple[int, int]:
    """Shrink the requested window so it fits the series, or explain why not.

    A window model needs ``lookback + horizon`` points for its first training
    example and one more per additional example. Rather than failing on a
    short series — which is common, and which the generic algorithm contract
    exercises with only 60 points — the window is shrunk: first the horizon to
    at most a quarter of the series, then the lookback, and only if the series
    is genuinely too short does this raise.

    Parameters
    ----------
    n_samples : int
        Length of the training series.
    lookback : int
        Requested lookback length.
    horizon : int
        Requested forecast horizon used for training.
    min_windows : int, default=4
        Fewest training windows to leave after shrinking.
    min_lookback : int, default=4
        Shortest acceptable lookback.

    Returns
    -------
    lookback : int
        Lookback that fits.
    horizon : int
        Horizon that fits.

    Raises
    ------
    ValueError
        If even the smallest window leaves fewer than ``min_windows``
        examples, with the exact number of points required.

    Examples
    --------
    >>> from tuiml.algorithms.timeseries.deep._base import resolve_window
    >>> resolve_window(500, lookback=48, horizon=12)
    (48, 12)
    >>> resolve_window(60, lookback=96, horizon=24)
    (42, 15)
    >>> resolve_window(20, lookback=48, horizon=12)
    (12, 5)
    """
    lookback = int(lookback)
    horizon = int(horizon)
    if lookback < 1 or horizon < 1:
        raise ValueError(
            f"lookback and horizon must be >= 1, got lookback={lookback}, "
            f"horizon={horizon}."
        )

    needed = min_lookback + 1 + min_windows - 1
    if n_samples < needed:
        raise ValueError(
            f"Series has {n_samples} points, which is too short to train a "
            f"window model. At least {needed} points are required (a lookback "
            f"of {min_lookback}, a horizon of 1, and {min_windows} training "
            "windows). Use a classical forecaster such as "
            "ExponentialSmoothing or ARIMA on series this short."
        )

    horizon = max(1, min(horizon, n_samples // 4))
    lookback = min(lookback, n_samples - horizon - min_windows + 1)
    if lookback < min_lookback:
        lookback = min_lookback
        horizon = max(1, n_samples - lookback - min_windows + 1)
    return int(lookback), int(horizon)


def make_windows(
    series: np.ndarray, lookback: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the sliding ``(lookback, horizon)`` supervised dataset.

    Parameters
    ----------
    series : np.ndarray of shape (n_samples,)
        The training series.
    lookback : int
        Number of past points in each input window.
    horizon : int
        Number of future points each window must predict.

    Returns
    -------
    X : np.ndarray of shape (n_windows, lookback)
        Input windows, oldest point first.
    Y : np.ndarray of shape (n_windows, horizon)
        The points immediately following each input window.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.deep._base import make_windows
    >>> X, Y = make_windows(np.arange(6.0), lookback=3, horizon=2)
    >>> X
    array([[0., 1., 2.],
           [1., 2., 3.]])
    >>> Y
    array([[3., 4.],
           [4., 5.]])
    """
    n_windows = len(series) - lookback - horizon + 1
    if n_windows < 1:
        raise ValueError(
            f"Cannot build any window: series of length {len(series)} is "
            f"shorter than lookback ({lookback}) + horizon ({horizon})."
        )
    idx = np.arange(n_windows)[:, None]
    X = series[idx + np.arange(lookback)[None, :]]
    Y = series[idx + lookback + np.arange(horizon)[None, :]]
    return np.ascontiguousarray(X), np.ascontiguousarray(Y)


def instance_normalize(torch: Any, x: Any) -> Tuple[Any, Any, Any]:
    """Normalise each window by its own mean and standard deviation (RevIN).

    This is reversible instance normalisation without the learnable affine
    term. Each window is standardised on its own statistics rather than the
    dataset's, which removes the level and scale drift that a trending or
    regime-switching series presents to the network. The statistics are
    returned so the forecast can be mapped back;
    :func:`instance_denormalize` inverts this exactly.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module, passed in so this function never
        imports it.
    x : torch.Tensor of shape (batch, length)
        Input windows.

    Returns
    -------
    x_norm : torch.Tensor of shape (batch, length)
        Normalised windows.
    loc : torch.Tensor of shape (batch, 1)
        Per-window mean.
    scale : torch.Tensor of shape (batch, 1)
        Per-window standard deviation, floored away from zero so a constant
        window does not divide by zero.
    """
    loc = x.mean(dim=-1, keepdim=True)
    scale = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (x - loc) / scale, loc, scale


def instance_denormalize(torch: Any, y: Any, loc: Any, scale: Any) -> Any:
    """Undo :func:`instance_normalize` on a forecast.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.
    y : torch.Tensor of shape (batch, length)
        Normalised values.
    loc : torch.Tensor of shape (batch, 1)
        Location returned by :func:`instance_normalize`.
    scale : torch.Tensor of shape (batch, 1)
        Scale returned by :func:`instance_normalize`.

    Returns
    -------
    y_orig : torch.Tensor of shape (batch, length)
        Values in the original units.
    """
    return y * scale + loc


def residual_stack_loss(
    torch: Any, module: Any, xb: Any, yb: Any, weight: float
) -> Any:
    """Return the forecast loss plus a backcast reconstruction term.

    The doubly-residual claim of N-BEATS and N-HiTS is that each block removes
    the part of the input it can explain, so the residual entering the last
    block is close to noise. Nothing in a forecast-only objective actually
    asks for that: the backcast heads are unsupervised, and left to themselves
    they happily *grow* the residual while the forecast still fits. Adding a
    small penalty on the final residual supervises the backcasts, which is
    what makes the residual cascade real rather than decorative — and it acts
    as a mild regulariser on top, since the blocks must agree on a
    decomposition of the input rather than each fitting the target alone.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.
    module : torch.nn.Module
        A network exposing ``forward_detail``.
    xb : torch.Tensor of shape (batch, lookback)
        Normalised input windows.
    yb : torch.Tensor of shape (batch, horizon)
        Normalised target windows.
    weight : float
        Weight on the backcast term. Zero recovers the paper's objective
        exactly.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss.
    """
    detail = module.forward_detail(xb)
    loss = torch.mean((detail["forecast"] - yb) ** 2)
    weight = float(weight)
    if weight > 0.0:
        loss = loss + weight * torch.mean(detail["residuals"][:, -1, :] ** 2)
    return loss


class DeepForecaster(Regressor):
    """Base class for the torch-backed window forecasters.

    Subclasses supply exactly one thing — :meth:`_build_module`, a factory
    that imports torch, builds an ``nn.Module`` mapping ``(batch, lookback)``
    to ``(batch, horizon)`` and returns it. Everything else (windowing,
    scaling, seeding, training, rollout) is handled here.

    Notes
    -----
    **Instance normalisation.** Every window is normalised by its own mean and
    standard deviation before it reaches the network, and the forecast is
    denormalised with the same statistics. This is RevIN without the learnable
    affine term. It matters far more than it sounds: without it a trending
    series presents the network with inputs whose scale drifts between the
    start and the end of training, and the extrapolation degrades badly.
    Subclasses may opt out by overriding :meth:`_use_instance_norm`.

    These models require ``pip install 'tuiml[torch]'``.
    """

    # --- hyperparameters every subclass shares; set in the subclass __init__
    lookback: int
    horizon: int
    n_epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    device: str
    random_state: Optional[int]

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _use_instance_norm(self) -> bool:
        """Report whether each window is normalised by its own statistics.

        Returns
        -------
        enabled : bool
            ``True`` by default; PatchTST ties this to its ``revin`` flag.
        """
        return True

    def _build_module(self, torch: Any, nn: Any, lookback: int, horizon: int) -> Any:
        """Build the network. Implemented by each model.

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
            Maps ``(batch, lookback)`` to ``(batch, horizon)``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _init_fitted_attributes(self) -> None:
        """Reset every fitted attribute to ``None``.

        Called from each subclass ``__init__`` so an unfitted model has the
        same attribute set as a fitted one.
        """
        self.module_ = None
        self.lookback_ = None
        self.horizon_ = None
        self.n_windows_ = None
        self.offset_ = None
        self.scale_ = None
        self.loss_curve_ = None
        self.n_epochs_run_ = None
        self.series_ = None
        self.device_ = None

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> "DeepForecaster":
        """Train the network on windows drawn from a single series.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            The series to learn. This is the series itself, not a design
            matrix.
        X : np.ndarray, optional, default=None
            Exogenous regressors. Accepted for interface compatibility with
            the other forecasters and ignored: none of these three models
            takes covariates in this implementation.

        Returns
        -------
        self : DeepForecaster
            The fitted forecaster.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        ValueError
            If the series is too short, empty or non-finite.
        """
        torch, nn = require_torch(type(self).__name__)
        guard_duplicate_openmp(torch)

        series = check_series(y)
        lookback, horizon = resolve_window(len(series), self.lookback, self.horizon)

        # Centre and scale the whole series once. The per-window instance
        # normalisation on top of this handles level drift; this outer step
        # only keeps the raw magnitudes near unity so a shared learning rate
        # behaves the same on a series measured in units or in millions.
        self.offset_ = float(series.mean())
        spread = float(series.std())
        self.scale_ = spread if spread > 1e-12 else 1.0
        scaled = (series - self.offset_) / self.scale_

        Xw, Yw = make_windows(scaled, lookback, horizon)

        self.lookback_ = lookback
        self.horizon_ = horizon
        self.n_windows_ = int(Xw.shape[0])
        self.series_ = scaled

        device = resolve_device(self.device, torch)
        self.device_ = str(device)

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        module = self._build_module(torch, nn, lookback, horizon).to(device)

        self._train_module(torch, module, Xw, Yw, device)
        self.module_ = module
        self._is_fitted = True
        return self

    def _train_module(
        self,
        torch: Any,
        module: Any,
        Xw: np.ndarray,
        Yw: np.ndarray,
        device: Any,
    ) -> None:
        """Run Adam over the windows with early stopping on the training loss.

        Early stopping watches the *training* loss rather than a validation
        split on purpose. These fits are routinely asked to work on sixty
        points, where carving out a validation tail costs more information
        than the stopping rule recovers; the epoch budget and the small
        networks are what keep overfitting in check.

        Parameters
        ----------
        torch : module
            The imported ``torch`` module.
        module : torch.nn.Module
            The network to train, already on ``device``.
        Xw : np.ndarray of shape (n_windows, lookback)
            Input windows.
        Yw : np.ndarray of shape (n_windows, horizon)
            Target windows.
        device : torch.device
            Where to place the tensors.

        Returns
        -------
        None
        """
        Xt = torch.as_tensor(Xw, dtype=torch.float32, device=device)
        Yt = torch.as_tensor(Yw, dtype=torch.float32, device=device)
        n = int(Xt.shape[0])

        optimiser = torch.optim.Adam(module.parameters(), lr=float(self.learning_rate))
        seed = 0 if self.random_state is None else int(self.random_state)
        rng = np.random.default_rng(seed)

        batch_size = max(1, min(int(self.batch_size), n))
        best_loss = float("inf")
        best_state: Optional[Dict[str, Any]] = None
        waited = 0
        curve: List[float] = []

        for epoch in range(int(self.n_epochs)):
            module.train()
            order = rng.permutation(n)
            running = 0.0
            for start in range(0, n, batch_size):
                sel = torch.as_tensor(
                    order[start:start + batch_size].copy(),
                    dtype=torch.long,
                    device=device,
                )
                xb, yb = Xt[sel], Yt[sel]
                xb, loc, spread = self._normalise(torch, xb)
                yb = (yb - loc) / spread
                loss = self._loss(torch, module, xb, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                running += float(loss.detach()) * int(sel.shape[0])

            epoch_loss = running / n
            curve.append(epoch_loss)

            if epoch_loss < best_loss - 1e-7:
                best_loss = epoch_loss
                best_state = copy.deepcopy(module.state_dict())
                waited = 0
            else:
                waited += 1
                if waited >= int(self.patience):
                    break

        if best_state is not None:
            module.load_state_dict(best_state)
        module.eval()
        self.loss_curve_ = np.asarray(curve, dtype=np.float64)
        self.n_epochs_run_ = len(curve)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _loss(self, torch: Any, module: Any, xb: Any, yb: Any) -> Any:
        """Return the training loss for one batch.

        Plain forecast mean squared error. The doubly-residual models override
        this to add a backcast term.

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
        return torch.mean((module(xb) - yb) ** 2)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    def _normalise(self, torch: Any, x: Any) -> Tuple[Any, Any, Any]:
        """Normalise each window by its own mean and standard deviation.

        Parameters
        ----------
        torch : module
            The imported ``torch`` module.
        x : torch.Tensor of shape (batch, lookback)
            Input windows.

        Returns
        -------
        x_norm : torch.Tensor of shape (batch, lookback)
            Normalised windows.
        loc : torch.Tensor of shape (batch, 1)
            Per-window location, for inverting the transform.
        spread : torch.Tensor of shape (batch, 1)
            Per-window scale, for inverting the transform.
        """
        if not self._use_instance_norm():
            zero = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
            one = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
            return x, zero, one
        return instance_normalize(torch, x)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, steps: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast the next ``steps`` values of the training series.

        Requests longer than the trained horizon are served by an
        autoregressive rollout: the horizon is forecast, appended to the
        history, and the window slides forward until enough points exist.

        Parameters
        ----------
        steps : int, default=1
            Number of future points to forecast.
        X : np.ndarray, optional, default=None
            Exogenous regressors; ignored.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecast in the units of the training series.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        self._check_is_fitted()
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}.")
        scaled = self._rollout(steps)[0]
        return scaled * self.scale_ + self.offset_

    def _rollout(self, steps: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forecast ``steps`` scaled points, collecting any per-part outputs.

        Parameters
        ----------
        steps : int
            Number of scaled points to produce.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecast in the globally scaled space.
        parts : dict of str to np.ndarray
            Per-component forecasts, in the same scaled space, for models that
            expose a decomposition. Empty for models that do not.
        """
        import torch

        history = self.series_.copy()
        chunks: List[np.ndarray] = []
        part_chunks: Dict[str, List[np.ndarray]] = {}

        self.module_.eval()
        device = torch.device(self.device_)
        produced = 0
        while produced < steps:
            window = history[-self.lookback_:]
            xb = torch.as_tensor(
                window[None, :], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                xn, loc, spread = self._normalise(torch, xb)
                out = self._forward_parts(torch, xn)
                forecast = out.pop("forecast")
                block = (forecast * spread + loc).cpu().numpy().ravel()
                for name, tensor in out.items():
                    # A component is denormalised without the location term,
                    # except for the one that carries the level (handled by the
                    # caller); this keeps the components summing to the total.
                    part_chunks.setdefault(name, []).append(
                        (tensor * spread).cpu().numpy().ravel()
                    )
                if "trend" in part_chunks:
                    part_chunks["trend"][-1] = part_chunks["trend"][-1] + \
                        float(loc.item())
            chunks.append(block)
            history = np.concatenate([history, block])
            produced += len(block)

        forecast = np.concatenate(chunks)[:steps]
        parts = {
            name: np.concatenate(values)[:steps]
            for name, values in part_chunks.items()
        }
        return forecast, parts

    def _forward_parts(self, torch: Any, xn: Any) -> Dict[str, Any]:
        """Run the network on one normalised window.

        Parameters
        ----------
        torch : module
            The imported ``torch`` module.
        xn : torch.Tensor of shape (1, lookback)
            One normalised input window.

        Returns
        -------
        outputs : dict of str to torch.Tensor
            Always contains ``"forecast"``; models with an interpretable
            decomposition add one entry per component.
        """
        return {"forecast": self.module_(xn)}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def __getstate__(self) -> Dict[str, Any]:
        """Return picklable state, with the network reduced to NumPy weights.

        The ``nn.Module`` subclasses are defined inside their factory
        functions — they have to be, since ``nn`` does not exist at module
        scope on a torch-free install — and a locally defined class cannot be
        pickled by reference. The weights are therefore stored as plain arrays
        and the network is rebuilt on unpickling, which also makes the pickle
        readable on a machine with a different torch build.

        Returns
        -------
        state : dict
            The instance dictionary with ``module_`` replaced by
            ``_module_state``, a mapping of parameter name to ``np.ndarray``.
        """
        state = dict(self.__dict__)
        module = state.pop("module_", None)
        if module is not None:
            state["_module_state"] = {
                key: value.detach().cpu().numpy()
                for key, value in module.state_dict().items()
            }
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore the instance, rebuilding the network from saved weights.

        Parameters
        ----------
        state : dict
            The dictionary produced by :meth:`__getstate__`.

        Returns
        -------
        None
        """
        weights = state.pop("_module_state", None)
        self.__dict__.update(state)
        self.module_ = None
        if weights is None:
            return
        torch, nn = require_torch(type(self).__name__)
        guard_duplicate_openmp(torch)
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        module = self._build_module(torch, nn, self.lookback_, self.horizon_)
        module.load_state_dict(
            {key: torch.as_tensor(value) for key, value in weights.items()}
        )
        module.to(torch.device(self.device_ or "cpu"))
        module.eval()
        self.module_ = module

    def fit_predict(self, y: np.ndarray, steps: int = 1) -> np.ndarray:
        """Fit on a series and immediately forecast it forward.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            The series to learn.
        steps : int, default=1
            Number of future points to forecast.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecast in the units of the training series.
        """
        self.fit(y)
        return self.predict(steps)

    # ------------------------------------------------------------------
    # Metadata shared by all three models
    # ------------------------------------------------------------------
    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric",
            "timeseries",
            "forecasting",
            "univariate",
            "trend",
            "seasonality",
            "non_linear",
        ]


__all__ = [
    "DeepForecaster",
    "MIN_LOOKBACK",
    "MIN_WINDOWS",
    "check_series",
    "guard_duplicate_openmp",
    "instance_denormalize",
    "instance_normalize",
    "make_windows",
    "residual_stack_loss",
    "resolve_window",
]
