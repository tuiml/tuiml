"""Shared base for the zero-shot time series forecasters in :mod:`tuiml.foundation`.

A pretrained forecaster has the same fit/predict shape as the native
forecasters in :mod:`tuiml.algorithms.timeseries` -- ``fit(y)`` then
``predict(steps)`` -- but ``fit`` does no estimation. It stores the series as
the model's *context* and loads the frozen checkpoint; every forecast is one
forward pass over that context at predict time.

The subclasses implement two hooks: :meth:`_load_model`, which imports the
upstream package and returns the ready-to-run model, and
:meth:`_forecast_quantiles`, which maps the stored context and a horizon onto
an array of quantile forecasts. Everything else -- argument checks, context
truncation, the median point forecast, ``fit_predict`` -- lives here once.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tuiml.base.algorithms import Regressor

#: The quantile levels every wrapper reports by default. They are the deciles
#: because TimesFM's quantile head emits exactly these and nothing else, so the
#: two wrappers agree on the shape of :meth:`predict_quantiles` out of the box.
DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class _ZeroShotForecaster(Regressor):
    """Delegation shared by the pretrained forecaster wrappers.

    Holds fitting (context storage plus checkpoint loading), prediction (median
    of the quantile forecast) and the quantile API. A subclass supplies the
    model loader and the quantile hook.
    """

    #: Importable name of the upstream package, for the install hint.
    _module: str = ""

    #: Fitted attributes that hold the loaded upstream model or anything
    #: derived from it. They are dropped on pickling and rebuilt on demand, so
    #: a saved TuiML model stays a few kilobytes and never embeds a checkpoint.
    _transient_state: Sequence[str] = ("model_",)

    def __init__(self) -> None:
        """Initialise the fitted-state slots. Imports nothing."""
        super().__init__()
        self.model_ = None
        self.series_ = None
        self.n_obs_ = None
        self.quantile_levels_ = None

    # ------------------------------------------------------------------
    # Hooks for the subclasses
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Import the upstream package and return the loaded, frozen model."""
        raise NotImplementedError

    def _forecast_quantiles(self, context: np.ndarray, steps: int) -> np.ndarray:
        """Forecast ``steps`` ahead from ``context``.

        Parameters
        ----------
        context : np.ndarray of shape (n_context,)
            The float32 history handed to the model.
        steps : int
            Forecast horizon.

        Returns
        -------
        quantiles : np.ndarray of shape (steps, n_quantiles)
            One column per level in ``quantile_levels_``, in that order.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared machinery
    # ------------------------------------------------------------------

    def _context_window(self, y: np.ndarray, context_length: Optional[int]) -> np.ndarray:
        """Return the tail of ``y`` the model will see as context.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            The full series.
        context_length : int or None
            Maximum number of trailing points to keep; ``None`` keeps all.

        Returns
        -------
        context : np.ndarray of shape (min(n_samples, context_length),)
            Trailing window as float32.
        """
        if context_length is not None:
            if int(context_length) < 1:
                raise ValueError("context_length must be >= 1 or None")
            y = y[-int(context_length):]
        return y.astype(np.float32)

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> "_ZeroShotForecaster":
        """Store the series as context and load the pretrained checkpoint.

        No parameter is estimated. The series is validated and its trailing
        window kept; the upstream model is imported and loaded here so that a
        missing dependency or a failed download surfaces at ``fit`` rather
        than at the first ``predict``.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        self : object
            The fitted wrapper.
        """
        y = np.asarray(y, dtype=float).ravel()
        if y.size < 2:
            raise ValueError(f"{type(self).__name__} requires at least 2 observations")
        if not np.isfinite(y).all():
            raise ValueError("y must not contain NaN or infinite values")
        self.model_ = self._load_model()
        self.series_ = self._context_window(y, getattr(self, "context_length", None))
        self.n_obs_ = int(y.size)
        self.quantile_levels_ = np.asarray(self._quantile_levels(), dtype=float)
        self._is_fitted = True
        return self

    def _quantile_levels(self) -> Sequence[float]:
        """Return the quantile levels the model will emit, in column order."""
        return DEFAULT_QUANTILES

    def _ensure_model(self) -> Any:
        """Return the loaded upstream model, loading it if it was dropped.

        After unpickling, ``model_`` is ``None`` and the checkpoint is fetched
        again from the local Hugging Face cache on the first forecast.

        Returns
        -------
        model : object
            The loaded upstream model.
        """
        if self.model_ is None:
            self.model_ = self._load_model()
        return self.model_

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Return picklable state without the loaded upstream model.

        The checkpoint is someone else's artifact, hundreds of megabytes, and
        in TimesFM's case holds compiled closures that cannot be pickled at
        all. Only the context and the hyperparameters are saved; the model is
        reloaded from the Hugging Face cache on the next forecast.

        Returns
        -------
        state : dict
            The instance dictionary with every ``_transient_state`` entry
            reset to ``None``.
        """
        state = dict(self.__dict__)
        for key in self._transient_state:
            if key in state:
                state[key] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore the instance; the upstream model is reloaded lazily.

        Parameters
        ----------
        state : dict
            The dictionary produced by :meth:`__getstate__`.
        """
        self.__dict__.update(state)
        for key in self._transient_state:
            setattr(self, key, None)

    def predict_quantiles(self, steps: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast a set of quantiles for each future step.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        quantiles : np.ndarray of shape (steps, n_quantiles)
            Column ``j`` is the forecast at level ``quantile_levels_[j]``.
            Columns are sorted per row so quantiles never cross.
        """
        self._check_is_fitted()
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self._ensure_model()
        q = np.asarray(self._forecast_quantiles(self.series_, steps), dtype=float)
        q = q.reshape(steps, -1)
        return np.sort(q, axis=1)

    def predict(self, steps: int = 1, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast future values of the series.

        The point forecast is the **median** of the model's quantile forecast,
        which is the convention both upstream packages use for their headline
        number and the one that stays well defined when the predictive
        distribution is skewed.

        Parameters
        ----------
        steps : int, default=1
            Number of future time steps to forecast.
        X : np.ndarray, optional, default=None
            Ignored. Present for API consistency with regressors.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecasted values.
        """
        q = self.predict_quantiles(steps)
        levels = self.quantile_levels_
        j = int(np.argmin(np.abs(levels - 0.5)))
        return q[:, j]

    def fit_predict(self, y: np.ndarray, steps: int = 1) -> np.ndarray:
        """Fit the model and forecast in one call.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Time series values.
        steps : int, default=1
            Number of future time steps to forecast.

        Returns
        -------
        forecast : np.ndarray of shape (steps,)
            Forecasted values.
        """
        self.fit(y)
        return self.predict(steps)

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the data and task properties this model supports."""
        return [
            "numeric", "timeseries", "forecasting", "univariate",
            "probabilistic", "trend", "seasonality", "non_linear",
        ]


__all__ = ["DEFAULT_QUANTILES", "_ZeroShotForecaster"]
