"""TimesFM — Google's pretrained decoder-only forecaster, used zero-shot.

Wraps the upstream ``timesfm`` package. TuiML ships no weights; ``timesfm``
fetches its own checkpoint from the Hugging Face Hub on first use. The package
and the ``google/timesfm-2.5-200m-pytorch`` checkpoint are both Apache-2.0,
which is compatible with TuiML's BSD-3-Clause license, so there is no usage
restriction to accept.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tuiml.foundation._base import fd_regressor, require_package
from tuiml.foundation._forecast import DEFAULT_QUANTILES, _ZeroShotForecaster

#: TimesFM's context and horizon are compiled in multiples of these.
_CONTEXT_MULTIPLE = 32
_HORIZON_MULTIPLE = 128


def _round_up(n: int, multiple: int) -> int:
    """Round ``n`` up to the nearest multiple of ``multiple``.

    Parameters
    ----------
    n : int
        Value to round.
    multiple : int
        Granularity.

    Returns
    -------
    rounded : int
        Smallest multiple of ``multiple`` that is ``>= n``.
    """
    return int(np.ceil(n / multiple) * multiple)


@fd_regressor(tags=["foundation-model", "pretrained", "zero-shot", "transformer",
                    "timeseries", "forecasting"], version="1.0.0")
class TimesFMForecaster(_ZeroShotForecaster):
    r"""Zero-shot forecasting with Google's **pretrained TimesFM** checkpoint.

    TimesFM is a **time series foundation model**: a decoder-only transformer
    pretrained on a corpus of roughly 100 billion real-world time points, so
    that it forecasts any univariate series it is shown without being trained
    on it. Your series is the *context*; the model emits a point forecast and
    a full set of **decile quantiles** for every future step in one forward
    pass. Nothing is estimated on your data -- ``fit`` only stores the
    context and loads the checkpoint, and all the cost lands in ``predict``.

    This wrapper targets **TimesFM 2.5** (200M parameters), the PyTorch
    release with the continuous quantile head and a context window of up to
    16k points.

    Overview
    --------
    1. The last ``context_length`` points are per-series normalised, so the
       model never sees the raw units.
    2. The context is cut into non-overlapping input *patches* of 32 points;
       each is embedded by a residual MLP.
    3. A stack of causal self-attention layers runs over the patch sequence --
       a decoder-only design, so every patch attends only to those before it.
    4. The final patch's representation is decoded by a second residual MLP
       into an *output* patch of 128 points, longer than the input patch, so
       most horizons need a single step.
    5. A continuous quantile head produces the deciles; :meth:`predict`
       returns the median and :meth:`predict_quantiles` the whole set.

    Theory
    ------
    TimesFM treats forecasting as **next-patch prediction**, the time series
    analogue of next-token prediction in a language model. With input patches
    :math:`\\tilde{y}_1, \\dots, \\tilde{y}_N` of length :math:`p` and output
    patches of length :math:`h > p`, the pretraining objective is the mean
    squared error of each output patch given all preceding input patches:

    .. math::
        \\mathcal{L} = \\frac{1}{N} \\sum_{j=1}^{N}
            \\big\\| \\hat{y}_{pj+1:pj+h} - y_{pj+1:pj+h} \\big\\|^2

    Making the output patch longer than the input patch is the design choice
    that matters: it cuts the number of autoregressive steps a long horizon
    needs, which both speeds up inference and limits error accumulation.
    Version 2.5 adds a quantile head trained with the pinball loss, giving
    calibrated deciles rather than a point alone.

    Parameters
    ----------
    checkpoint : str, default="google/timesfm-2.5-200m-pytorch"
        Hugging Face model id, or a local directory holding one.
    context_length : int, default=1024
        Trailing points of the series handed to the model. Rounded up to a
        multiple of 32 when the model is compiled; the model supports up to
        16384.
    max_horizon : int, default=128
        Longest horizon the compiled model serves without recompiling.
        Rounded up to a multiple of 128. A ``predict`` beyond it recompiles
        the model for the larger horizon, which costs a second or two once.
    infer_is_positive : bool, default=True
        Clip forecasts at zero for series whose context is entirely
        non-negative, as counts and prices are.
    batch_size : int, default=32
        Series per forward pass inside the model. Only matters for memory.

    Attributes
    ----------
    model_ : object
        The underlying ``timesfm.TimesFM_2p5_200M_torch``.
    series_ : np.ndarray of shape (n_context,)
        The context window stored at ``fit``.
    n_obs_ : int
        Length of the series seen during ``fit``.
    quantile_levels_ : np.ndarray of shape (9,)
        The deciles 0.1 … 0.9, in the column order of
        :meth:`predict_quantiles`.
    compiled_horizon_ : int
        Horizon the model is currently compiled for.

    Notes
    -----
    **Requires** ``pip install 'tuiml[foundation]'``, which pulls in
    ``timesfm`` and PyTorch. Importing TuiML and constructing this class work
    without either; only :meth:`fit` needs them.

    **Weights are not shipped by TuiML.** On first use ``timesfm`` downloads
    the checkpoint (about 800 MB) from the Hugging Face Hub into
    ``~/.cache/huggingface``. Code and weights are Apache-2.0, compatible with
    TuiML's BSD-3-Clause, so there is no commercial-use restriction to
    accept.

    **Device:** the upstream model picks CUDA when it is available and the
    CPU otherwise; there is no device knob to expose.

    **Saved models stay small.** Pickling a fitted wrapper stores the context
    and the hyperparameters, never the checkpoint or its compiled state; the
    model is reloaded and recompiled from the Hugging Face cache on the first
    forecast after loading.

    **Complexity:** fitting costs one checkpoint load plus a compile.
    Prediction is one decoder forward pass over
    :math:`n_{\\text{context}} / 32` patches per 128-step chunk of horizon.
    A GPU is worth roughly an order of magnitude.

    **When to use TimesFMForecaster:**

    - A strong forecast with zero tuning, on a series you have not modelled
      before -- the natural first baseline alongside
      :class:`~tuiml.foundation.ChronosForecaster`.
    - Long contexts: the 16k-point window is far longer than Chronos-Bolt's.
    - When you need calibrated deciles, not just a point.

    Prefer :class:`~tuiml.algorithms.timeseries.ThetaForecaster` or
    :class:`~tuiml.algorithms.timeseries.SARIMAXForecaster` when you need a
    model you can inspect, exogenous regressors, or a CPU-only deployment
    without PyTorch.

    References
    ----------
    .. [Das2024] Das, A., Kong, W., Sen, R. and Zhou, Y. (2024). **A
           decoder-only foundation model for time-series forecasting.**
           *Proceedings of the 41st International Conference on Machine
           Learning (ICML)*.
           arXiv: `2310.10688 <https://arxiv.org/abs/2310.10688>`_

    See Also
    --------
    :class:`~tuiml.foundation.ChronosForecaster` : Amazon's pretrained forecaster, same interface.
    :class:`~tuiml.algorithms.timeseries.PatchTSTForecaster` : Native patch transformer trained on your data.
    :class:`~tuiml.algorithms.timeseries.ThetaForecaster` : The classical zero-tuning baseline.

    Examples
    --------
    Constructing and inspecting the model needs no optional dependency:

    >>> from tuiml.foundation import TimesFMForecaster
    >>> model = TimesFMForecaster(context_length=512)
    >>> model.context_length
    512
    >>> "max_horizon" in model.get_parameter_schema()
    True

    Fitting requires ``pip install 'tuiml[foundation]'`` and downloads the
    checkpoint on first use:

    >>> import numpy as np  # doctest: +SKIP
    >>> t = np.arange(200)  # doctest: +SKIP
    >>> y = 10 + 0.05 * t + np.sin(2 * np.pi * t / 12)  # doctest: +SKIP
    >>> model.fit(y).predict(steps=3).round(1)  # doctest: +SKIP
    array([20.5, 21. , 21.4])
    >>> model.predict_quantiles(steps=3).shape  # doctest: +SKIP
    (3, 9)
    """

    _module = "timesfm"
    _transient_state = ("model_", "compiled_horizon_")

    def __init__(
        self,
        checkpoint: str = "google/timesfm-2.5-200m-pytorch",
        context_length: int = 1024,
        max_horizon: int = 128,
        infer_is_positive: bool = True,
        batch_size: int = 32,
    ) -> None:
        """Record hyperparameters. Imports nothing — see the class Notes."""
        super().__init__()
        self.checkpoint = checkpoint
        self.context_length = context_length
        self.max_horizon = max_horizon
        self.infer_is_positive = infer_is_positive
        self.batch_size = batch_size
        self.compiled_horizon_ = None

    def _quantile_levels(self) -> Sequence[float]:
        """Return the deciles: the only levels TimesFM's quantile head emits."""
        return DEFAULT_QUANTILES

    def _compile(self, horizon: int) -> None:
        """Compile ``model_`` for the stored context and the given horizon.

        Parameters
        ----------
        horizon : int
            Longest horizon the compiled model must serve.
        """
        timesfm = require_package(self._module, type(self).__name__)
        self.compiled_horizon_ = _round_up(max(int(horizon), 1), _HORIZON_MULTIPLE)
        self.model_.compile(timesfm.ForecastConfig(
            max_context=_round_up(int(self.context_length), _CONTEXT_MULTIPLE),
            max_horizon=self.compiled_horizon_,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=bool(self.infer_is_positive),
            fix_quantile_crossing=True,
            per_core_batch_size=int(self.batch_size),
        ))

    def _load_model(self) -> Any:
        """Import ``timesfm``, load the checkpoint and compile it.

        Returns
        -------
        model : timesfm.TimesFM_2p5_200M_torch
            The frozen model, compiled for ``max_horizon``.
        """
        if int(self.context_length) < 1:
            raise ValueError("context_length must be >= 1")
        if int(self.max_horizon) < 1:
            raise ValueError("max_horizon must be >= 1")
        timesfm = require_package(self._module, type(self).__name__)
        self.model_ = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)
        self._compile(self.max_horizon)
        return self.model_

    def _forecast_quantiles(self, context: np.ndarray, steps: int) -> np.ndarray:
        """Run one forward pass and return the decile forecast.

        Parameters
        ----------
        context : np.ndarray of shape (n_context,)
            The float32 history.
        steps : int
            Forecast horizon. Beyond ``compiled_horizon_`` the model is
            recompiled first.

        Returns
        -------
        quantiles : np.ndarray of shape (steps, 9)
            Deciles 0.1 … 0.9.
        """
        import torch

        if steps > self.compiled_horizon_:
            self._compile(steps)
        with torch.inference_mode():
            _point, quant = self.model_.forecast(horizon=steps, inputs=[context])
        # quant has shape (batch, horizon, 10): the mean, then deciles 0.1..0.9.
        return np.asarray(quant)[0, :steps, 1:10]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "checkpoint": {
                "type": "string",
                "default": "google/timesfm-2.5-200m-pytorch",
                "description": "Hugging Face model id or local directory.",
            },
            "context_length": {
                "type": "integer",
                "default": 1024,
                "minimum": 1,
                "maximum": 16384,
                "description": "Trailing points handed to the model; rounded "
                               "up to a multiple of 32.",
            },
            "max_horizon": {
                "type": "integer",
                "default": 128,
                "minimum": 1,
                "description": "Longest horizon served without recompiling; "
                               "rounded up to a multiple of 128.",
            },
            "infer_is_positive": {
                "type": "boolean",
                "default": True,
                "description": "Clip forecasts at zero when the context is "
                               "entirely non-negative.",
            },
            "batch_size": {
                "type": "integer",
                "default": 32,
                "minimum": 1,
                "description": "Series per forward pass inside the model.",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the data and task properties this model supports."""
        return _ZeroShotForecaster.get_capabilities()


__all__ = ["TimesFMForecaster"]
