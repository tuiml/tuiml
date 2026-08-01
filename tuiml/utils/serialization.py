"""Persist trained models to disk, and export them to other runtimes.

Reach for this module whenever a fitted TuiML model (or any picklable
Python object) needs to survive past the current process: hand it
:func:`save_model` / :func:`load_model` for a quick round trip,
:class:`ModelSerializer` when you want to reuse one format/protocol
configuration across many saves, :class:`ModelCheckpoint` to snapshot a
model periodically during a training loop, or :func:`export_to_onnx` to
hand the model to a non-Python runtime.

Overview
--------
1. **Formats**: every format ultimately stores a ``pickle``-serialized
   dict of ``{model, metadata, saved_at, model_class, model_module,
   params}``. ``"pickle"`` writes it directly; ``"compressed"`` gzips the
   pickle stream; ``"joblib"`` defers to the optional ``joblib`` package,
   which is more efficient for models holding large NumPy arrays.
2. **Metadata tracking**: :meth:`ModelSerializer.save` (and
   :func:`save_model`) automatically capture the save timestamp, the
   model's class/module, and its constructor parameters (via
   ``get_params()`` if available) alongside any caller-supplied
   ``metadata`` dict. :func:`load_model_info` gives convenient access to
   this record; note it still unpickles the whole saved bundle (model
   included) internally, it just returns a lighter dict.
3. **Checkpointing**: :class:`ModelCheckpoint` wraps :func:`save_model` to
   snapshot a model during long-running training, optionally keeping only
   the best-scoring checkpoint and pruning old ones.

Security
--------
Every format here is ``pickle``-based (``joblib`` also pickles under the
hood). Unpickling executes arbitrary code embedded in the file. **Only
load files you created yourself or otherwise trust** — never call
:func:`load_model`, :meth:`ModelSerializer.load`, or
:meth:`ModelCheckpoint.load_latest`/:meth:`~ModelCheckpoint.load_best` on
a file from an untrusted source.
"""

import pickle
import json
import gzip
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

class ModelSerializer:
    """Unified interface for machine learning model serialization.

    Provides a consistent API for persistent storage and retrieval of trained 
    models using several backend formats. It handles automatic metadata 
    tracking (e.g., save time, model class, hyperparameters).

    Overview
    --------
    The serializer supports the following backend formats:

    - **pickle**: Python's native serialization format. Best for general use.
    - **joblib**: Optimized for models containing large NumPy arrays.
    - **compressed**: Gzip-compressed pickle for significant disk space savings.

    Parameters
    ----------
    format : {"pickle", "joblib", "compressed"}, default="pickle"
        The serialization strategy to employ.

    protocol : int, default=4
        The pickle protocol version to use (typically 0-5). Higher versions 
        are more efficient but less compatible with older Python versions.

    Attributes
    ----------
    format : str
        The active serialization format.

    protocol : int
        The active pickle protocol version.

    Raises
    ------
    ValueError
        If ``format`` is not one of :attr:`SUPPORTED_FORMATS`.

    Notes
    -----
    Every format is pickle-based under the hood (``joblib`` also pickles).
    Only call :meth:`load` on files you trust — see the module-level
    "Security" note.

    Examples
    --------
    Save and reload a fitted model together with custom metadata:

    >>> import numpy as np
    >>> from tuiml.algorithms.bayesian import NaiveBayesClassifier
    >>> from tuiml.utils.serialization import ModelSerializer
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> my_model = NaiveBayesClassifier().fit(X, y)
    >>> serializer = ModelSerializer(format="compressed")
    >>> meta = {"accuracy": 0.98, "dataset": "iris_v1"}
    >>> serializer.save(my_model, "model.pkl.gz", metadata=meta)  # doctest: +SKIP
    'model.pkl.gz'
    >>> loaded_model = serializer.load("model.pkl.gz")  # doctest: +SKIP
    """

    SUPPORTED_FORMATS = ['pickle', 'joblib', 'compressed']

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "format": {
                "type": "string",
                "default": "pickle",
                "enum": ["pickle", "joblib", "compressed"],
                "description": "Serialization format: pickle (native), joblib (numpy-optimized), compressed (gzip)"
            },
            "protocol": {
                "type": "integer",
                "default": 4,
                "minimum": 0,
                "maximum": 5,
                "description": "Pickle protocol version (0-5). Higher versions are more efficient"
            }
        }

    def __init__(
        self,
        format: str = 'pickle',
        protocol: int = 4
    ):
        """Validate ``format`` and store the serialization settings."""
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
        self.format = format
        self.protocol = protocol

    def save(
        self,
        model: Any,
        filepath: Union[str, Path],
        metadata: Optional[Dict] = None
    ) -> str:
        """Save a trained model to disk in this instance's ``format``.

        Bundles ``model`` with automatically-collected metadata (save
        timestamp, model class/module, and ``model.get_params()`` if
        available) plus any caller-supplied ``metadata``, then writes the
        bundle with :mod:`pickle` (or ``joblib``, which also pickles).

        Parameters
        ----------
        model : Any
            Trained model object to save. Must be picklable.
        filepath : str or Path
            Path to save the model to. For ``format="compressed"``, a
            ``.gz`` suffix is appended to the file actually written on disk
            if ``filepath`` doesn't already end in ``.gz``.
        metadata : dict, optional
            Additional metadata to store alongside the model.

        Returns
        -------
        filepath : str
            The ``filepath`` argument, unchanged, as a string.

        Raises
        ------
        ImportError
            If ``format="joblib"`` and the optional ``joblib`` package is
            not installed.

        Notes
        -----
        For ``format="compressed"``, this returns the original
        ``filepath`` argument even when the file actually written to disk
        has a ``.gz`` suffix appended (i.e. when ``filepath`` did not
        already end in ``.gz``). Pass a path already ending in ``.gz`` to
        avoid the mismatch.
        """
        filepath = Path(filepath)

        # Prepare data with metadata
        data = {
            'model': model,
            'metadata': metadata or {},
            'saved_at': datetime.now().isoformat(),
            'format': self.format,
            'model_class': model.__class__.__name__,
            'model_module': model.__class__.__module__,
        }

        # Add model parameters if available
        if hasattr(model, 'get_params'):
            data['params'] = model.get_params()
        elif hasattr(model, '__dict__'):
            # Filter out private attributes and numpy arrays
            data['params'] = {
                k: v for k, v in model.__dict__.items()
                if not k.startswith('_') and not hasattr(v, 'shape')
            }

        if self.format == 'pickle':
            self._save_pickle(data, filepath)
        elif self.format == 'joblib':
            self._save_joblib(data, filepath)
        elif self.format == 'compressed':
            # _save_compressed appends .gz when the caller did not, so take the
            # path it actually wrote; returning the original would name a file
            # that does not exist.
            filepath = self._save_compressed(data, filepath)

        return str(filepath)

    def load(self, filepath: Union[str, Path]) -> Any:
        """Load a model from disk, ignoring any stored metadata.

        The format is auto-detected from the file extension: ``.gz`` loads
        as gzip-compressed pickle, ``.joblib``/``.jbl`` loads via
        ``joblib``, anything else loads as a plain pickle.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved model.

        Returns
        -------
        model : Any
            Loaded model object.

        Raises
        ------
        FileNotFoundError
            If ``filepath`` does not exist.
        ImportError
            If the file requires ``joblib`` and it is not installed.

        Notes
        -----
        Unpickling runs arbitrary code embedded in the file. Only load
        files you trust (see the module-level "Security" note).
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Detect format from file
        if filepath.suffix == '.gz':
            data = self._load_compressed(filepath)
        elif filepath.suffix in ['.joblib', '.jbl']:
            data = self._load_joblib(filepath)
        else:
            data = self._load_pickle(filepath)

        return data['model']

    def load_with_metadata(
        self,
        filepath: Union[str, Path]
    ) -> Dict[str, Any]:
        """Load a model together with everything :meth:`save` recorded about it.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved model.

        Returns
        -------
        data : dict
            Dictionary with keys ``"model"``, ``"metadata"``, ``"params"``,
            ``"saved_at"``, ``"format"``, ``"model_class"``, and
            ``"model_module"``.

        Raises
        ------
        FileNotFoundError
            If ``filepath`` does not exist (raised by the underlying file
            open, unlike :meth:`load` this method does not pre-check
            existence).
        ImportError
            If the file requires ``joblib`` and it is not installed.

        Notes
        -----
        Unpickling runs arbitrary code embedded in the file. Only load
        files you trust (see the module-level "Security" note).
        """
        filepath = Path(filepath)

        if filepath.suffix == '.gz':
            return self._load_compressed(filepath)
        elif filepath.suffix in ['.joblib', '.jbl']:
            return self._load_joblib(filepath)
        else:
            return self._load_pickle(filepath)

    def _save_pickle(self, data: Dict, filepath: Path):
        """Save using pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=self.protocol)

    def _load_pickle(self, filepath: Path) -> Dict:
        """Load using pickle."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def _save_compressed(self, data: Dict, filepath: Path) -> Path:
        """Save using gzip-compressed pickle.

        Parameters
        ----------
        data : dict
            Bundle of model and metadata to write.
        filepath : Path
            Target path. ``.gz`` is appended when it is not already present.

        Returns
        -------
        filepath : Path
            The path actually written, including any appended ``.gz``.
        """
        if not str(filepath).endswith('.gz'):
            filepath = Path(str(filepath) + '.gz')
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=self.protocol)
        return filepath

    def _load_compressed(self, filepath: Path) -> Dict:
        """Load gzip-compressed pickle."""
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)

    def _save_joblib(self, data: Dict, filepath: Path):
        """Save using joblib.

        Raises
        ------
        ImportError
            If the optional ``joblib`` package is not installed.
        """
        try:
            import joblib
            joblib.dump(data, filepath)
        except ImportError:
            raise ImportError(
                "joblib is required for joblib format. "
                "Install it with: pip install joblib"
            )

    def _load_joblib(self, filepath: Path) -> Dict:
        """Load using joblib.

        Raises
        ------
        ImportError
            If the optional ``joblib`` package is not installed.
        """
        try:
            import joblib
            return joblib.load(filepath)
        except ImportError:
            raise ImportError(
                "joblib is required for joblib format. "
                "Install it with: pip install joblib"
            )

# Convenience functions

def save_model(
    model: Any,
    filepath: Union[str, Path],
    format: str = 'pickle',
    metadata: Optional[Dict] = None,
    compress: bool = False
) -> str:
    """Save a trained model to disk with automatic metadata tracking.

    Thin functional wrapper around :class:`ModelSerializer`: builds one
    with the resolved format and delegates to :meth:`ModelSerializer.save`.

    Parameters
    ----------
    model : Any
        The trained model instance (or any picklable object) to save.

    filepath : str or Path
        The destination file path.

    format : {"pickle", "joblib", "compressed"}, default="pickle"
        The underlying serialization format. Ignored if ``compress=True``
        (which forces ``"compressed"``).

    metadata : dict, optional
        A dictionary of custom metadata (e.g., metrics, data version)
        to store alongside the model.

    compress : bool, default=False
        If ``True``, applies gzip compression to the output file
        (shorthand for ``format="compressed"``).

    Returns
    -------
    str
        The ``filepath`` argument, unchanged, as a string. Note: if
        ``compress=True`` (or ``format="compressed"``) and ``filepath``
        does not already end in ``.gz``, the file written to disk has
        ``.gz`` appended, but this return value does not reflect that —
        pass a path already ending in ``.gz`` to avoid the mismatch.

    Raises
    ------
    ImportError
        If ``format="joblib"`` and the optional ``joblib`` package is not
        installed.

    See Also
    --------
    :func:`~tuiml.utils.serialization.load_model` : The inverse operation.

    Examples
    --------
    Save a fitted model with gzip compression:

    >>> import numpy as np
    >>> from tuiml.algorithms.bayesian import NaiveBayesClassifier
    >>> from tuiml.utils.serialization import save_model
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> my_clf = NaiveBayesClassifier().fit(X, y)
    >>> save_model(my_clf, "classifier.pkl.gz", compress=True)  # doctest: +SKIP
    'classifier.pkl.gz'
    """
    actual_format = 'compressed' if compress else format
    serializer = ModelSerializer(format=actual_format)
    return serializer.save(model, filepath, metadata)

def load_model(filepath: Union[str, Path]) -> Any:
    """Load a model from disk, discarding any stored metadata.

    Detects the format from ``filepath``'s extension: ``.gz`` is loaded as
    gzip-compressed pickle, ``.joblib``/``.jbl`` via ``joblib``, and
    anything else as a plain pickle.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved model file.

    Returns
    -------
    model : Any
        The reconstructed model object.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    ImportError
        If the file requires ``joblib`` and it is not installed.

    Notes
    -----
    Unpickling runs arbitrary code embedded in the file. Only load files
    you trust (see the module-level "Security" note).

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.bayesian import NaiveBayesClassifier
    >>> from tuiml.utils.serialization import save_model, load_model
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> save_model(NaiveBayesClassifier().fit(X, y), "classifier.pkl")  # doctest: +SKIP
    >>> model = load_model("classifier.pkl")  # doctest: +SKIP
    >>> model.predict(X[:1])  # doctest: +SKIP
    array([0])
    """
    serializer = ModelSerializer()
    return serializer.load(filepath)

def load_model_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load a saved model's metadata and hyper-parameters, without the model itself.

    Useful for inspecting model properties, comparing performance, or
    retrieving training timestamps when only the recorded metadata is
    needed, so the caller's code doesn't have to unpack :meth:`load_with_metadata`'s
    dict by hand.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved model file.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``"model_class"``: Name of the Python class.
        - ``"model_module"``: Fully-qualified module the class lives in.
        - ``"params"``: Configuration parameters/hyper-parameters (from
          ``model.get_params()`` if available, else public ``__dict__``
          attributes).
        - ``"metadata"``: The custom metadata dict provided during save.
        - ``"saved_at"``: ISO timestamp of the save operation.
        - ``"format"``: The serialization format used (``"pickle"``,
          ``"joblib"``, or ``"compressed"``).

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    ImportError
        If the file requires ``joblib`` and it is not installed.

    Notes
    -----
    This still unpickles the whole saved bundle (model included) — it
    only *returns* a lighter dict. It offers no memory savings over
    :func:`load_model`, but is convenient when metadata and the model
    object are both wanted, or when only the metadata is of interest and
    reading it is simpler this way.

    Examples
    --------
    Check the accuracy of a saved model:

    >>> import numpy as np
    >>> from tuiml.algorithms.bayesian import NaiveBayesClassifier
    >>> from tuiml.utils.serialization import save_model, load_model_info
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> save_model(NaiveBayesClassifier().fit(X, y), "model.pkl",
    ...            metadata={"accuracy": 0.98})  # doctest: +SKIP
    >>> info = load_model_info("model.pkl")  # doctest: +SKIP
    >>> print(f"Model: {info['model_class']} | Acc: {info['metadata']['accuracy']}")  # doctest: +SKIP
    Model: NaiveBayesClassifier | Acc: 0.98
    """
    serializer = ModelSerializer()
    data = serializer.load_with_metadata(filepath)
    return {
        'model_class': data.get('model_class'),
        'model_module': data.get('model_module'),
        'params': data.get('params', {}),
        'metadata': data.get('metadata', {}),
        'saved_at': data.get('saved_at'),
        'format': data.get('format'),
    }

class ModelCheckpoint:
    """Manage model persistence during iterative training processes.

    Automatically handles saving model snapshots at regular intervals, 
    tracking the 'best' model based on a chosen metric, and cleaning up 
    disk space by keeping only a limited number of recent checkpoints.

    Parameters
    ----------
    directory : str or Path
        The directory where checkpoint files will be written.

    prefix : str, default="checkpoint"
        The filename prefix for all generated snapshots.

    max_to_keep : int, default=5
        The rolling limit of recent checkpoints to retain.

    save_best_only : bool, default=False
        If ``True``, a new checkpoint is only committed if the monitored 
        metric improves.

    monitor : str, default="loss"
        The key name of the metric to track (passed to :meth:`save`).

    mode : {"min", "max"}, default="min"
        Whether to look for the minimum (e.g., loss) or maximum (e.g., accuracy).

    Attributes
    ----------
    best_value : float
        The best recorded value of the monitored metric so far
        (``inf`` / ``-inf`` until the first checkpoint when
        ``save_best_only=True``).

    checkpoints : list of Path
        A chronologically ordered list of active checkpoint file paths
        (only files written by this instance; not populated from disk).

    Notes
    -----
    Checkpoints are always written with :func:`save_model`'s default
    ``format="pickle"``. Unpickling runs arbitrary code embedded in the
    file — only call :meth:`load_latest`/:meth:`load_best` on a
    ``directory`` you trust.

    Examples
    --------
    Keep only the 3 best checkpoints by F1-score:

    >>> import tempfile
    >>> import numpy as np
    >>> from tuiml.algorithms.bayesian import NaiveBayesClassifier
    >>> from tuiml.utils.serialization import ModelCheckpoint
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> my_model = NaiveBayesClassifier().fit(X, y)
    >>> checkpoint = ModelCheckpoint(
    ...     directory=tempfile.mkdtemp(),
    ...     save_best_only=True,
    ...     monitor="f1",
    ...     mode="max",
    ...     max_to_keep=3,
    ... )  # doctest: +SKIP
    >>> for i, score in enumerate([0.5, 0.6, 0.55, 0.7, 0.65]):
    ...     checkpoint.save(my_model, epoch=i, f1=score)  # doctest: +SKIP
    >>> round(checkpoint.best_value, 2)  # doctest: +SKIP
    0.7
    >>> len(checkpoint.checkpoints)  # doctest: +SKIP
    3
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "directory": {
                "type": "string",
                "description": "Directory path where checkpoints will be saved"
            },
            "prefix": {
                "type": "string",
                "default": "checkpoint",
                "description": "Prefix for checkpoint filenames"
            },
            "max_to_keep": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Maximum number of checkpoint files to retain"
            },
            "save_best_only": {
                "type": "boolean",
                "default": False,
                "description": "If true, only save when monitored metric improves"
            },
            "monitor": {
                "type": "string",
                "default": "loss",
                "description": "Metric name to monitor when save_best_only is true"
            },
            "mode": {
                "type": "string",
                "default": "min",
                "enum": ["min", "max"],
                "description": "Whether lower (min) or higher (max) metric value is better"
            }
        }

    def __init__(
        self,
        directory: Union[str, Path],
        prefix: str = 'checkpoint',
        max_to_keep: int = 5,
        save_best_only: bool = False,
        monitor: str = 'loss',
        mode: str = 'min'
    ):
        """Store the checkpoint settings and create ``directory`` if needed."""
        self.directory = Path(directory)
        self.prefix = prefix
        self.max_to_keep = max_to_keep
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints = []

        # Create directory if needed
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: Any,
        epoch: int = None,
        **metrics
    ) -> Optional[str]:
        """Save a checkpoint, unless ``save_best_only`` says to skip it.

        Writes ``model`` via :func:`save_model` (pickle format) under
        ``directory`` as ``"{prefix}_epoch{epoch}_{timestamp}.pkl"``, then
        prunes old checkpoints beyond ``max_to_keep``.

        Parameters
        ----------
        model : Any
            Model to checkpoint.
        epoch : int, optional
            Current epoch number, included in the checkpoint filename.
        **metrics
            Metric values (e.g., ``accuracy=0.95, loss=0.1``), stored in
            the checkpoint's metadata. Must include the key named by
            ``self.monitor`` when ``save_best_only=True``.

        Returns
        -------
        filepath : str or None
            Path to the saved checkpoint, or ``None`` if
            ``save_best_only=True`` and ``self.monitor``'s value did not
            improve on :attr:`best_value` (so nothing was written).

        Raises
        ------
        ValueError
            If ``save_best_only=True`` and ``metrics`` does not contain
            the key named by ``self.monitor``.
        """
        # Check if we should save
        if self.save_best_only:
            current_value = metrics.get(self.monitor)
            if current_value is None:
                raise ValueError(
                    f"Monitor metric '{self.monitor}' not found in metrics"
                )

            is_better = (
                (self.mode == 'min' and current_value < self.best_value) or
                (self.mode == 'max' and current_value > self.best_value)
            )

            if not is_better:
                return None

            self.best_value = current_value

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_str = f'_epoch{epoch}' if epoch is not None else ''
        filename = f'{self.prefix}{epoch_str}_{timestamp}.pkl'
        filepath = self.directory / filename

        # Save
        metadata = {'epoch': epoch, 'metrics': metrics}
        save_model(model, filepath, metadata=metadata)
        self.checkpoints.append(filepath)

        # Clean up old checkpoints
        self._cleanup()

        return str(filepath)

    def _cleanup(self):
        """Remove old checkpoints beyond max_to_keep."""
        while len(self.checkpoints) > self.max_to_keep:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

    def load_latest(self) -> Any:
        """Load the most recently saved checkpoint.

        Uses :attr:`checkpoints` if this instance has saved any; otherwise
        falls back to globbing ``directory`` for ``"{prefix}*.pkl"`` files
        (e.g. after re-creating the ``ModelCheckpoint`` in a new process)
        and picks the lexicographically-last match, which is also the most
        recent since filenames are timestamp-suffixed.

        Returns
        -------
        Any
            The loaded model object.

        Raises
        ------
        FileNotFoundError
            If :attr:`checkpoints` is empty and no matching files are
            found in ``directory``.

        Notes
        -----
        Unpickling runs arbitrary code embedded in the file. Only call
        this on a ``directory`` you trust.
        """
        if not self.checkpoints:
            # Find checkpoints in directory
            pattern = f'{self.prefix}*.pkl'
            found = sorted(self.directory.glob(pattern))
            if not found:
                raise FileNotFoundError(
                    f"No checkpoints found in {self.directory}"
                )
            return load_model(found[-1])

        return load_model(self.checkpoints[-1])

    def load_best(self) -> Any:
        """Load the best checkpoint saved so far.

        Only meaningful with ``save_best_only=True``, in which case every
        saved checkpoint improved on the last, so the most recent one
        (delegated to :meth:`load_latest`) is also the best.

        Returns
        -------
        Any
            The loaded model object.

        Raises
        ------
        ValueError
            If ``save_best_only=False``.
        FileNotFoundError
            If no checkpoint has been saved yet.
        """
        if not self.save_best_only:
            raise ValueError(
                "load_best() only works when save_best_only=True"
            )
        return self.load_latest()

def export_to_onnx(
    model: Any,
    filepath: Union[str, Path],
    input_shape: tuple,
    input_names: list = None,
    output_names: list = None
) -> str:
    """Export a trained model to ONNX format for cross-platform interoperability.

    Overview
    --------
    ONNX (Open Neural Network Exchange) allows models trained in Python to 
    be executed in high-performance runtimes like ONNX Runtime, C++, or 
    mobile devices.

    Parameters
    ----------
    model : Any
        A Scikit-learn compatible model or pipeline to be converted.

    filepath : str or Path
        The output path for the ``.onnx`` file.

    input_shape : tuple
        The expected input dimensions (e.g., ``(None, 20)`` for a batch size
        of 20 features). Only the last element (the feature count) is
        currently used.

    input_names : list of str, optional
        Custom names for the input nodes. **Not currently applied** — the
        input node is always named ``"input"`` regardless of this
        argument; see Notes.

    output_names : list of str, optional
        Custom names for the output nodes. **Not currently applied** —
        this argument is accepted but not passed through to the
        underlying converter; see Notes.

    Returns
    -------
    str
        The path to the successfully exported ONNX model, with a
        ``.onnx`` suffix appended if ``filepath`` didn't already have one.

    Raises
    ------
    ImportError
        If the optional ``skl2onnx`` package is not installed.

    Notes
    -----
    This utility requires the ``skl2onnx`` package.
    Install via: ``pip install skl2onnx``

    ``input_names`` renames the single input node of the exported graph.
    ``output_names`` is not supported yet and raises :exc:`NotImplementedError`
    rather than being silently ignored, so a caller never ends up with a graph
    whose node names differ from what it asked for.
    """
    if output_names is not None:
        raise NotImplementedError(
            "output_names is not supported yet: skl2onnx names the output "
            "nodes and this exporter does not rename them afterwards. Omit "
            "the argument, or rename the nodes on the returned .onnx graph."
        )
    if input_names is not None and len(input_names) != 1:
        raise ValueError(
            f"input_names must name exactly one input node, got "
            f"{len(input_names)}: {input_names}"
        )

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        raise ImportError(
            "skl2onnx is required for ONNX export. "
            "Install it with: pip install skl2onnx"
        )

    filepath = Path(filepath)
    if not str(filepath).endswith('.onnx'):
        filepath = Path(str(filepath) + '.onnx')

    # Define input type
    n_features = input_shape[-1] if input_shape else None
    input_name = input_names[0] if input_names else 'input'
    initial_type = [
        (input_name, FloatTensorType([None, n_features]))
    ]

    # Convert
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12
    )

    # Save
    with open(filepath, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    return str(filepath)
