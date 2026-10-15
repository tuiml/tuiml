"""Chronos — Amazon's pretrained time series forecasters, used zero-shot.

Wraps the upstream ``chronos-forecasting`` package. TuiML ships no weights;
``chronos`` fetches its own checkpoint from the Hugging Face Hub on first use.
The package and every checkpoint this wrapper names (``chronos-bolt-small``,
``chronos-bolt-base``, ``chronos-2``) are Apache-2.0, which is compatible with
TuiML's BSD-3-Clause license, so there is no usage restriction to accept.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tuiml.foundation._base import fd_regressor, require_package
from tuiml.foundation._forecast import DEFAULT_QUANTILES, _ZeroShotForecaster

#: Checkpoints this wrapper has been exercised against. Any other Hugging Face
#: id that ``chronos.BaseChronosPipeline`` can load also works.
KNOWN_CHECKPOINTS = (
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-bolt-mini",
    "amazon/chronos-bolt-small",
    "amazon/chronos-bolt-base",
    "amazon/chronos-2",
)


@fd_regressor(tags=["foundation-model", "pretrained", "zero-shot", "transformer",
                    "timeseries", "forecasting"], version="1.0.0")
class ChronosForecaster(_ZeroShotForecaster):
    r"""Zero-shot forecasting with a **pretrained Chronos** checkpoint.

    Chronos is a family of **time series foundation models** from Amazon: a
    transformer trained once, on a large corpus of real and synthetic series,
    to forecast any univariate series it is shown. Your series is the
    *context*; the model emits a full **quantile forecast** for every future
    step in a single forward pass. Nothing is estimated on your data -- there
    is no gradient step, no hyperparameter search and no fitted coefficient --
    which is why ``fit`` is instant and all the cost lands in ``predict``.

    One class covers the whole family. ``checkpoint`` selects the model:

    - ``amazon/chronos-bolt-{tiny,mini,small,base}`` -- **Chronos-Bolt**, a
      T5 encoder-decoder over *patches* of the series that decodes all
      horizons at once. Up to 250x faster than the original Chronos and the
      right default for most series.
    - ``amazon/chronos-2`` -- **Chronos-2**, the successor that extends the
      same recipe to multivariate series and covariates via in-context group
      attention. Used univariately here.

    Overview
    --------
    1. The last ``context_length`` points of the series are scaled by their
       mean absolute value, so the model never sees the raw units.
    2. The scaled context is cut into fixed-length *patches* and each patch is
       embedded by a small residual network.
    3. A transformer encoder attends over the patch sequence; the decoder emits
       every horizon step at once, as a set of quantiles (direct multi-step
       forecasting, no autoregression inside the horizon).
    4. The quantiles are unscaled back to the series' units.
    5. :meth:`predict` returns the median; :meth:`predict_quantiles` returns
       the whole set.

    Theory
    ------
    Chronos-Bolt is trained with the **quantile (pinball) loss** summed over a
    fixed set of levels :math:`Q`, which makes each output head a consistent
    estimator of the corresponding conditional quantile:

    .. math::
        \\mathcal{L} = \\sum_{q \\in Q} \\sum_{h=1}^{H}
            \\rho_q\\!\\left(y_{t+h} - \\hat{y}^{(q)}_{t+h}\\right),
        \\qquad
        \\rho_q(u) = u \\,\\big(q - \\mathbb{1}[u < 0]\\big)

    The original Chronos instead *tokenised* values into a vocabulary and
    trained a language model over them; Bolt replaces that with patching and
    the direct quantile head, which is where the speed-up comes from. Both
    rest on the same bet: a model that has seen enough series learns the
    patterns -- trend, seasonality, regime change -- as general knowledge, so
    at your series the whole "learning algorithm" is one forward pass.

    Parameters
    ----------
    checkpoint : str, default="amazon/chronos-bolt-small"
        Hugging Face model id, or a local directory holding one. See
        ``KNOWN_CHECKPOINTS``.
    context_length : int or None, default=None
        Trailing points of the series handed to the model. ``None`` passes
        the whole series; the model itself caps what it attends to (2048 for
        Chronos-Bolt, 8192 for Chronos-2) and silently drops older points.
    quantile_levels : sequence of float, default=(0.1, …, 0.9)
        Levels returned by :meth:`predict_quantiles`. Must include a value at
        or near 0.5, which :meth:`predict` reports.
    device : str or None, default=None
        Torch device map: ``'cpu'``, ``'cuda'``, ``'mps'``, or ``None`` to let
        the upstream pipeline choose.

    Attributes
    ----------
    model_ : object
        The underlying ``chronos.BaseChronosPipeline``.
    series_ : np.ndarray of shape (n_context,)
        The context window stored at ``fit``.
    n_obs_ : int
        Length of the series seen during ``fit``.
    quantile_levels_ : np.ndarray of shape (n_quantiles,)
        The levels, in the column order of :meth:`predict_quantiles`.

    Notes
    -----
    **Requires** ``pip install 'tuiml[foundation]'``, which pulls in
    ``chronos-forecasting`` and PyTorch. Importing TuiML and constructing this
    class work without either; only :meth:`fit` needs them.

    **Weights are not shipped by TuiML.** On first use ``chronos`` downloads
    the checkpoint from the Hugging Face Hub into ``~/.cache/huggingface``
    (about 190 MB for Bolt-small, 800 MB for Bolt-base and Chronos-2). Code
    and weights are Apache-2.0, compatible with TuiML's BSD-3-Clause, so there
    is no commercial-use restriction to accept. That is *not* true of every
    forecasting foundation model: Moirai's weights, for instance, are
    non-commercial, which is why it is not integrated here.

    **Saved models stay small.** Pickling a fitted wrapper stores the context
    and the hyperparameters, never the checkpoint; the model is reloaded
    from the Hugging Face cache on the first forecast after loading.

    **Complexity:** fitting costs one checkpoint load. Prediction is one
    transformer forward pass over :math:`n_{\\text{context}} / P` patches
    (patch length :math:`P = 16`), independent of the horizon up to the
    model's native horizon of 64 steps; longer horizons are produced by
    feeding predictions back as context. A GPU is worth roughly an order of
    magnitude.

    **When to use ChronosForecaster:**

    - A strong forecast with zero tuning, on a series you have not modelled
      before -- the natural first baseline.
    - Many short series, where fitting a statistical model to each is the
      bottleneck: one checkpoint load serves them all.
    - When you need calibrated quantiles, not just a point.

    Prefer :class:`~tuiml.algorithms.timeseries.ThetaForecaster` or
    :class:`~tuiml.algorithms.timeseries.SARIMAXForecaster` when you need a
    model you can inspect, exogenous regressors, or a CPU-only deployment
    without PyTorch.

    References
    ----------
    .. [Ansari2024] Ansari, A. F., Stella, L., Turkmen, C., Zhang, X.,
           Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P.,
           Kapoor, S., Zschiegner, J., Maddix, D. C., Wang, H., Mahoney,
           M. W., Torkkola, K., Wilson, A. G., Bohlke-Schneider, M. and Wang,
           Y. (2024). **Chronos: Learning the Language of Time Series.**
           *Transactions on Machine Learning Research*.
           arXiv: `2403.07815 <https://arxiv.org/abs/2403.07815>`_
    .. [Ansari2025] Ansari, A. F., Shchur, O., Küken, J., Auer, A., Han, B.,
           Mercado, P., Rangapuram, S. S., Shen, H., Stella, L., Sun, X.,
           Turkmen, C., Williams, A. C., Wang, Y., Bohlke-Schneider, M.,
           Hochreiter, S., Wilson, A. G. and others (2025). **Chronos-2: From
           Univariate to Universal Forecasting.**
           arXiv: `2510.15821 <https://arxiv.org/abs/2510.15821>`_

    See Also
    --------
    :class:`~tuiml.foundation.TimesFMForecaster` : Google's pretrained forecaster, same interface.
    :class:`~tuiml.algorithms.timeseries.PatchTSTForecaster` : Native patch transformer trained on your data.
    :class:`~tuiml.algorithms.timeseries.ThetaForecaster` : The classical zero-tuning baseline.

    Examples
    --------
    Constructing and inspecting the model needs no optional dependency:

    >>> from tuiml.foundation import ChronosForecaster
    >>> model = ChronosForecaster(checkpoint="amazon/chronos-bolt-base")
    >>> model.checkpoint
    'amazon/chronos-bolt-base'
    >>> "context_length" in model.get_parameter_schema()
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

    _module = "chronos"

    def __init__(
        self,
        checkpoint: str = "amazon/chronos-bolt-small",
        context_length: Optional[int] = None,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        device: Optional[str] = None,
    ) -> None:
        """Record hyperparameters. Imports nothing — see the class Notes."""
        super().__init__()
        self.checkpoint = checkpoint
        self.context_length = context_length
        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        self.device = device

    def _quantile_levels(self) -> Sequence[float]:
        """Return the configured quantile levels, validated."""
        levels = self.quantile_levels
        if not levels or any(not (0.0 < q < 1.0) for q in levels):
            raise ValueError("quantile_levels must be non-empty and lie in (0, 1)")
        return levels

    def _load_model(self) -> Any:
        """Import ``chronos`` and load the pipeline for ``checkpoint``.

        Returns
        -------
        pipeline : chronos.BaseChronosPipeline
            The frozen model, on the requested device.
        """
        chronos = require_package(self._module, type(self).__name__)
        import torch

        kwargs: Dict[str, Any] = {"torch_dtype": torch.float32}
        if self.device is not None:
            kwargs["device_map"] = self.device
        return chronos.BaseChronosPipeline.from_pretrained(self.checkpoint, **kwargs)

    def _forecast_quantiles(self, context: np.ndarray, steps: int) -> np.ndarray:
        """Run one forward pass and return the quantile forecast.

        Parameters
        ----------
        context : np.ndarray of shape (n_context,)
            The float32 history.
        steps : int
            Forecast horizon.

        Returns
        -------
        quantiles : np.ndarray of shape (steps, n_quantiles)
            One column per configured level.
        """
        import torch

        with torch.inference_mode():
            q, _mean = self.model_.predict_quantiles(
                [torch.tensor(context)],
                prediction_length=steps,
                quantile_levels=list(self.quantile_levels_),
            )
        # Chronos-Bolt returns a tensor (batch, horizon, n_quantiles).
        # Chronos-2 returns a list of (n_variates, horizon, n_quantiles) tensors.
        if isinstance(q, (list, tuple)):
            out = q[0].float().cpu().numpy().reshape(-1, len(self.quantile_levels_))
        else:
            out = q[0].float().cpu().numpy()
        return out[:steps]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "checkpoint": {
                "type": "string",
                "default": "amazon/chronos-bolt-small",
                "description": "Hugging Face model id or local directory. "
                               "Bolt {tiny,mini,small,base} or chronos-2.",
            },
            "context_length": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 1,
                "description": "Trailing points handed to the model; null "
                               "passes the whole series.",
            },
            "quantile_levels": {
                "type": "array",
                "items": {"type": "number", "exclusiveMinimum": 0.0,
                          "exclusiveMaximum": 1.0},
                "default": DEFAULT_QUANTILES,
                "description": "Quantile levels returned by predict_quantiles; "
                               "predict reports the one nearest 0.5.",
            },
            "device": {
                "type": ["string", "null"],
                "default": None,
                "description": "Torch device: 'cpu', 'cuda', 'mps', or null "
                               "to let chronos choose.",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the data and task properties this model supports."""
        return _ZeroShotForecaster.get_capabilities()


__all__ = ["ChronosForecaster", "KNOWN_CHECKPOINTS"]
