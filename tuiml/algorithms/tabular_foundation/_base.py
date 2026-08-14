"""Shared machinery for the torch-backed deep tabular models.

Everything in this module is internal. It holds the parts that
:mod:`~tuiml.algorithms.tabular_foundation.ft_transformer`,
:mod:`~tuiml.algorithms.tabular_foundation.saint` and
:mod:`~tuiml.algorithms.tabular_foundation.node` would otherwise each repeat: label
encoding, feature standardisation, the mini-batch training loop, early
stopping, seeding, device placement, prediction, and the pickling protocol that
lets a fitted network round-trip without its layer classes being importable.

**No torch at import time.** Not one symbol here imports torch at module scope.
Every helper that needs tensors either takes the ``torch``/``nn`` modules as
arguments or imports them inside its own body, which is what allows the model
classes to be defined, registered and introspected on an install that has never
seen PyTorch. See :mod:`tuiml.utils.torch_backend` for the full contract.

The layer builders (:func:`_build_tokenizer`, :func:`_build_attention`,
:func:`_build_transformer_block`) are *factories*: they define their
``nn.Module`` subclass inside the call and return an instance. A class
statement cannot appear at module scope because ``nn`` does not exist there.
"""

from __future__ import annotations

import copy
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from tuiml.utils.torch_backend import require_torch, resolve_device


# ---------------------------------------------------------------------------
# Duplicate-OpenMP guard
# ---------------------------------------------------------------------------

#: Libraries that ship their own ``libomp.dylib`` on macOS. Loading one of
#: them *before* torch leaves two OpenMP runtimes in the process, and the
#: second one to open a parallel region crashes the interpreter.
_OPENMP_CONFLICTING_MODULES = ("xgboost", "lightgbm", "catboost")

#: The guard is a process-wide setting, so it is applied at most once.
_openmp_guard_checked = False


def _guard_duplicate_openmp(torch) -> bool:
    """Force torch to one thread when a second OpenMP runtime is loaded.

    On macOS, importing xgboost/LightGBM/CatBoost loads their bundled
    ``libomp.dylib``. If torch is imported afterwards -- which is exactly what
    lazy loading does -- the process holds two OpenMP runtimes, and the first
    torch op that opens a parallel region (``LayerNorm`` is usually the one)
    segfaults. Running torch single-threaded avoids the parallel region
    entirely. The models here are small enough that the cost is minor, and a
    slow fit beats a dead interpreter.

    The guard is skipped when no conflicting library is loaded, so a
    torch-only process keeps all its threads.

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


# ---------------------------------------------------------------------------
# Parameter schema fragments
# ---------------------------------------------------------------------------

def _training_schema() -> Dict[str, Dict[str, Any]]:
    """Return the JSON Schema entries every deep tabular model shares.

    Returns
    -------
    schema : dict
        Schema fragments for the optimiser and training-loop parameters. Each
        model merges these with its own architecture parameters so that
        ``get_parameter_schema`` covers every constructor argument.
    """
    return {
        "learning_rate": {
            "type": "number",
            "default": 1e-3,
            "minimum": 1e-6,
            "maximum": 1.0,
            "description": "AdamW step size",
        },
        "weight_decay": {
            "type": "number",
            "default": 1e-5,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "AdamW decoupled weight decay",
        },
        "batch_size": {
            "type": "integer",
            "default": 64,
            "minimum": 1,
            "maximum": 8192,
            "description": "Mini-batch size for training and inference",
        },
        "n_epochs": {
            "type": "integer",
            "default": 60,
            "minimum": 1,
            "maximum": 10000,
            "description": "Number of passes over the training set",
        },
        "early_stopping": {
            "type": "boolean",
            "default": False,
            "description": "Hold out a validation split and stop when it stops improving",
        },
        "validation_fraction": {
            "type": "number",
            "default": 0.15,
            "minimum": 0.01,
            "maximum": 0.5,
            "description": "Fraction held out when early_stopping is enabled",
        },
        "patience": {
            "type": "integer",
            "default": 10,
            "minimum": 1,
            "maximum": 1000,
            "description": "Epochs without validation improvement before stopping",
        },
        "device": {
            "type": "string",
            "default": "cpu",
            "enum": ["cpu", "auto", "cuda", "mps"],
            "description": "Compute device; 'auto' picks CUDA, then MPS, then CPU",
        },
        "random_state": {
            "type": "integer",
            "default": 0,
            "description": "Seed for parameter init, shuffling and the validation split",
        },
    }


def _categorical_schema() -> Dict[str, Dict[str, Any]]:
    """Return the schema entry for the ``categorical_features`` parameter.

    Returns
    -------
    schema : dict
        Single-entry schema fragment, shared by the tokenizer-based models.
    """
    return {
        "categorical_features": {
            "type": "array",
            "default": None,
            "items": {"type": "integer"},
            "description": (
                "Column indices holding integer-coded categorical features; "
                "None means every column is numerical"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Layer factories (torch is passed in; never imported at module scope)
# ---------------------------------------------------------------------------

def _build_tokenizer(torch, nn, n_numeric: int, cardinalities: Sequence[int],
                     d_token: int, prepend_cls: bool = True):
    """Build the feature tokenizer shared by FT-Transformer and SAINT.

    Every numerical feature owns a learned weight vector and bias, so scalar
    :math:`x_j` becomes :math:`W_j x_j + b_j`, a ``d_token``-dimensional
    embedding. Categorical features index an embedding table instead. A
    learned ``[CLS]`` token is optionally prepended, giving the sequence the
    prediction head reads from.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.
    nn : module
        ``torch.nn``.
    n_numeric : int
        Number of numerical columns.
    cardinalities : sequence of int
        Category count per categorical column; empty when there are none.
    d_token : int
        Embedding width.
    prepend_cls : bool, default=True
        Whether to prepend the learned ``[CLS]`` token.

    Returns
    -------
    tokenizer : torch.nn.Module
        Module mapping ``(x_num, x_cat)`` to ``(batch, n_tokens, d_token)``.
    """

    class _Tokenizer(nn.Module):
        """Per-feature embedding of a tabular row into a token sequence."""

        def __init__(self):
            super().__init__()
            self.n_numeric = n_numeric
            self.prepend_cls = prepend_cls
            if n_numeric > 0:
                self.num_weight = nn.Parameter(torch.empty(n_numeric, d_token))
                self.num_bias = nn.Parameter(torch.empty(n_numeric, d_token))
                nn.init.normal_(self.num_weight, std=d_token ** -0.5)
                nn.init.normal_(self.num_bias, std=d_token ** -0.5)
            self.cat_embeddings = nn.ModuleList(
                [nn.Embedding(int(c), d_token) for c in cardinalities]
            )
            for emb in self.cat_embeddings:
                nn.init.normal_(emb.weight, std=d_token ** -0.5)
            if prepend_cls:
                self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
                nn.init.normal_(self.cls_token, std=d_token ** -0.5)

        def forward(self, x_num, x_cat):
            """Embed one batch of rows into a token sequence."""
            tokens = []
            if self.n_numeric > 0:
                # (batch, n_numeric, 1) * (n_numeric, d) -> (batch, n_numeric, d)
                tokens.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
            for j, emb in enumerate(self.cat_embeddings):
                tokens.append(emb(x_cat[:, j]).unsqueeze(1))
            out = torch.cat(tokens, dim=1)
            if self.prepend_cls:
                cls = self.cls_token.expand(out.shape[0], -1, -1)
                out = torch.cat([cls, out], dim=1)
            return out

    return _Tokenizer()


def _build_attention(torch, nn, d_model: int, n_heads: int, dropout: float):
    """Build a multi-head self-attention block over the sequence axis.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.
    nn : module
        ``torch.nn``.
    d_model : int
        Token width. Must be divisible by ``n_heads``.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout applied to the attention weights.

    Returns
    -------
    attention : torch.nn.Module
        Module mapping ``(batch, n_tokens, d_model)`` to the same shape.
    """
    if d_model % n_heads != 0:
        raise ValueError(
            f"d_token={d_model} must be divisible by n_heads={n_heads}"
        )

    class _MultiHeadAttention(nn.Module):
        """Scaled dot-product self-attention with ``n_heads`` heads."""

        def __init__(self):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.q = nn.Linear(d_model, d_model)
            self.k = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, d_model)
            self.dropout = float(dropout)

        def _split(self, t):
            """Reshape ``(b, n, d)`` to ``(b, heads, n, head_dim)``."""
            b, n, _ = t.shape
            return t.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        def forward(self, x):
            """Attend every token to every other token."""
            b, n, _ = x.shape
            q, k, v = self._split(self.q(x)), self._split(self.k(x)), self._split(self.v(x))
            drop = self.dropout if self.training else 0.0
            attended = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=drop
            )
            attended = attended.transpose(1, 2).reshape(b, n, -1)
            return self.out(attended)

    return _MultiHeadAttention()


def _build_transformer_block(torch, nn, d_model: int, n_heads: int,
                             dropout: float, d_ffn_factor: float = 2.0):
    """Build one pre-norm Transformer block (attention + FFN, both residual).

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.
    nn : module
        ``torch.nn``.
    d_model : int
        Token width.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate used in the attention weights and the FFN.
    d_ffn_factor : float, default=2.0
        Hidden width of the feed-forward network, as a multiple of ``d_model``.

    Returns
    -------
    block : torch.nn.Module
        Module mapping ``(batch, n_tokens, d_model)`` to the same shape.
    """
    attention = _build_attention(torch, nn, d_model, n_heads, dropout)
    d_hidden = max(1, int(d_model * d_ffn_factor))

    class _Block(nn.Module):
        """Pre-norm residual attention block followed by a residual FFN."""

        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attention = attention
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_model),
            )
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            """Apply attention then the FFN, each with a residual connection."""
            x = x + self.drop(self.attention(self.norm1(x)))
            x = x + self.drop(self.ffn(self.norm2(x)))
            return x

    return _Block()


def _build_head(torch, nn, d_model: int, n_outputs: int):
    """Build the prediction head that reads the ``[CLS]`` token.

    Parameters
    ----------
    torch : module
        The imported ``torch`` module.
    nn : module
        ``torch.nn``.
    d_model : int
        Token width.
    n_outputs : int
        Number of outputs (classes, or 1 for regression).

    Returns
    -------
    head : torch.nn.Module
        Module mapping ``(batch, d_model)`` to ``(batch, n_outputs)``.
    """
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.ReLU(),
        nn.Linear(d_model, n_outputs),
    )


# ---------------------------------------------------------------------------
# Shared estimator behaviour
# ---------------------------------------------------------------------------

class _BaseDeepTabular:
    """Preprocessing, training loop and prediction for the deep tabular models.

    An internal mixin: it is never registered and never instantiated directly.
    Subclasses supply the architecture through :meth:`_build_network` and
    inherit everything else -- standardisation, the AdamW loop with cosine
    decay and optional early stopping, seeding, device placement, batched
    inference, and the ``__getstate__``/``__setstate__`` pair that pickles a
    fitted network as plain NumPy arrays.

    Notes
    -----
    Fitted networks are pickled as a NumPy state dict rather than as live
    ``nn.Module`` objects, because those classes are defined inside factory
    functions and so have no importable qualified name. The network is rebuilt
    from the stored architecture on first use after unpickling.
    """

    #: ``"classification"`` or ``"regression"``; set by the task mixins.
    _task = "classification"

    # -- architecture hook --------------------------------------------------

    def _build_network(self, torch, nn, n_numeric: int,
                       cardinalities: Sequence[int], n_outputs: int):
        """Return the ``nn.Module`` implementing this model's architecture.

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
        raise NotImplementedError

    # -- preprocessing ------------------------------------------------------

    def _fit_preprocessing(self, X: np.ndarray) -> None:
        """Learn the column split, category codes and standardisation stats.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Raw training features.

        Returns
        -------
        None
        """
        n_features = X.shape[1]
        requested = getattr(self, "categorical_features", None) or []
        cat_idx = sorted(int(j) % n_features for j in requested)
        num_idx = [j for j in range(n_features) if j not in set(cat_idx)]

        self.n_features_in_ = n_features
        self.categorical_indices_ = np.asarray(cat_idx, dtype=int)
        self.numeric_indices_ = np.asarray(num_idx, dtype=int)
        self.categories_ = [np.unique(X[:, j]) for j in cat_idx]
        self.cardinalities_ = [max(1, len(c)) for c in self.categories_]

        if num_idx:
            sub = X[:, num_idx]
            mean = sub.mean(axis=0)
            scale = sub.std(axis=0)
            scale[scale < 1e-8] = 1.0
        else:
            mean = np.zeros(0)
            scale = np.ones(0)
        self.feature_mean_ = mean
        self.feature_scale_ = scale

    def _transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Standardise numerical columns and code categorical ones.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Raw features.

        Returns
        -------
        x_num : np.ndarray of shape (n_samples, n_numeric), dtype float32
            Standardised numerical block.
        x_cat : np.ndarray of shape (n_samples, n_categorical), dtype int64
            Category codes, clipped into the training vocabulary.
        """
        num = X[:, self.numeric_indices_] if self.numeric_indices_.size else X[:, :0]
        x_num = ((num - self.feature_mean_) / self.feature_scale_).astype(np.float32)

        if self.categorical_indices_.size:
            codes = np.empty((X.shape[0], self.categorical_indices_.size), dtype=np.int64)
            for k, j in enumerate(self.categorical_indices_):
                cats = self.categories_[k]
                pos = np.searchsorted(cats, X[:, j])
                codes[:, k] = np.clip(pos, 0, len(cats) - 1)
        else:
            codes = np.zeros((X.shape[0], 0), dtype=np.int64)
        return x_num, codes

    @staticmethod
    def _check_X(X) -> np.ndarray:
        """Validate and coerce ``X`` to a finite 2-D float array.

        Parameters
        ----------
        X : array-like
            Feature data.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Two-dimensional float copy of the input.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinity; the deep tabular models do not "
                "support missing values. Impute first."
            )
        return X

    # -- training -----------------------------------------------------------

    def _fit_network(self, X: np.ndarray, target: np.ndarray, n_outputs: int):
        """Standardise, build the network and run the training loop.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Raw training features.
        target : np.ndarray of shape (n_samples,)
            Integer class indices (classification) or standardised targets
            (regression).
        n_outputs : int
            Number of output units.

        Returns
        -------
        None
        """
        torch, nn = require_torch(type(self).__name__)
        _guard_duplicate_openmp(torch)

        seed = 0 if self.random_state is None else int(self.random_state)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)

        self._fit_preprocessing(X)
        x_num, x_cat = self._transform(X)
        self.n_outputs_ = int(n_outputs)

        device = resolve_device(self.device, torch)
        self._device_ = device
        network = self._build_network(
            torch, nn, x_num.shape[1], self.cardinalities_, n_outputs
        ).to(device)

        num_t = torch.as_tensor(x_num, device=device)
        cat_t = torch.as_tensor(x_cat, device=device)
        if self._task == "classification":
            y_t = torch.as_tensor(np.asarray(target, dtype=np.int64), device=device)
            loss_fn = nn.CrossEntropyLoss()
        else:
            y_t = torch.as_tensor(
                np.asarray(target, dtype=np.float32).reshape(-1, 1), device=device
            )
            loss_fn = nn.MSELoss()

        n = num_t.shape[0]
        train_idx = np.arange(n)
        valid_idx = np.empty(0, dtype=int)
        if self.early_stopping and n >= 8:
            perm = rng.permutation(n)
            n_valid = max(1, int(round(self.validation_fraction * n)))
            valid_idx, train_idx = perm[:n_valid], perm[n_valid:]

        optimizer = torch.optim.AdamW(
            network.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(self.n_epochs))
        )

        batch_size = max(1, int(self.batch_size))
        best_loss, best_state, bad_epochs = float("inf"), None, 0
        history: List[float] = []

        network.train()
        for _ in range(int(self.n_epochs)):
            order = rng.permutation(train_idx)
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, len(order), batch_size):
                batch = torch.as_tensor(
                    np.ascontiguousarray(order[start:start + batch_size]),
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                out = network(num_t[batch], cat_t[batch])
                loss = loss_fn(out, y_t[batch])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
            scheduler.step()
            history.append(epoch_loss / max(1, n_batches))

            if valid_idx.size:
                network.eval()
                with torch.no_grad():
                    vidx = torch.as_tensor(valid_idx, device=device)
                    vloss = float(loss_fn(network(num_t[vidx], cat_t[vidx]), y_t[vidx]))
                network.train()
                if vloss < best_loss - 1e-6:
                    best_loss, bad_epochs = vloss, 0
                    best_state = copy.deepcopy(network.state_dict())
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(self.patience):
                        break

        if best_state is not None:
            network.load_state_dict(best_state)
        network.eval()

        self.network_ = network
        self.loss_curve_ = np.asarray(history, dtype=float)
        self.n_iter_ = len(history)

    # -- inference ----------------------------------------------------------

    def _ensure_network(self):
        """Rebuild the network after unpickling, if it is not live.

        Returns
        -------
        network : torch.nn.Module
            The fitted network, on the resolved device.
        """
        torch, nn = require_torch(type(self).__name__)
        _guard_duplicate_openmp(torch)
        if getattr(self, "network_", None) is None:
            device = resolve_device(self.device, torch)
            network = self._build_network(
                torch, nn, int(self.numeric_indices_.size),
                self.cardinalities_, self.n_outputs_,
            ).to(device)
            state = {
                key: torch.as_tensor(value, device=device)
                for key, value in self._state_arrays_.items()
            }
            network.load_state_dict(state)
            network.eval()
            self.network_ = network
            self._device_ = device
        return self.network_

    def _forward(self, X) -> np.ndarray:
        """Run the fitted network over ``X`` in evaluation mode.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature data.

        Returns
        -------
        outputs : np.ndarray of shape (n_samples, n_outputs)
            Raw network outputs (logits for classification).
        """
        torch, _nn = require_torch(type(self).__name__)
        network = self._ensure_network()
        X = self._check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{type(self).__name__} was fitted on {self.n_features_in_} "
                f"features but got {X.shape[1]}"
            )
        x_num, x_cat = self._transform(X)
        device = self._device_
        num_t = torch.as_tensor(x_num, device=device)
        cat_t = torch.as_tensor(x_cat, device=device)

        batch_size = max(1, int(self.batch_size))
        chunks = []
        with torch.no_grad():
            for start in range(0, num_t.shape[0], batch_size):
                stop = start + batch_size
                chunks.append(
                    network(num_t[start:stop], cat_t[start:stop]).cpu().numpy()
                )
        return np.concatenate(chunks, axis=0)

    # -- persistence --------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Return picklable state, with the network flattened to NumPy arrays.

        Returns
        -------
        state : dict
            Instance dictionary in which ``network_`` has been replaced by
            ``_state_arrays_``.
        """
        state = dict(self.__dict__)
        network = state.pop("network_", None)
        state.pop("_device_", None)
        if network is not None:
            state["_state_arrays_"] = {
                key: value.detach().cpu().numpy()
                for key, value in network.state_dict().items()
            }
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state, deferring network reconstruction to first use.

        Parameters
        ----------
        state : dict
            State produced by :meth:`__getstate__`.

        Returns
        -------
        None
        """
        self.__dict__.update(state)
        self.network_ = None


class _DeepTabularClassifierMixin(_BaseDeepTabular):
    """Classification behaviour: label encoding, cross-entropy, softmax."""

    _task = "classification"

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the network on labelled data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Class labels of any hashable dtype.

        Returns
        -------
        self : object
            The fitted estimator.
        """
        require_torch(type(self).__name__)
        X = self._check_X(X)
        y = np.asarray(y).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]} labels"
            )
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError(
                f"{type(self).__name__} requires at least two classes, got "
                f"{self.classes_.size}"
            )
        self._fit_network(X, encoded, int(self.classes_.size))
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Predicted labels, in the dtype of the training labels.
        """
        self._check_is_fitted()
        return self.classes_[np.argmax(self._forward(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via a softmax over the logits.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Row-stochastic class probabilities.
        """
        self._check_is_fitted()
        logits = self._forward(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


class _DeepTabularRegressorMixin(_BaseDeepTabular):
    """Regression behaviour: target standardisation and mean-squared error."""

    _task = "regression"

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the network on a continuous target.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Continuous targets.

        Returns
        -------
        self : object
            The fitted estimator.
        """
        require_torch(type(self).__name__)
        X = self._check_X(X)
        y = np.asarray(y, dtype=float).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]} targets"
            )
        self.target_mean_ = float(y.mean())
        scale = float(y.std())
        self.target_scale_ = scale if scale > 1e-8 else 1.0
        standardised = (y - self.target_mean_) / self.target_scale_
        self._fit_network(X, standardised, 1)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous targets on the original scale.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        self._check_is_fitted()
        out = self._forward(X).ravel()
        return out * self.target_scale_ + self.target_mean_


__all__: List[str] = []
