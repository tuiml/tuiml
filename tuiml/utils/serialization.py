"""Model Serialization - Save, load, and export trained models."""

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

    Examples
    --------
    Save and reload a model with metadata:

    >>> from tuiml.utils.serialization import ModelSerializer
    >>> serializer = ModelSerializer(format="compressed")
    >>> meta = {"accuracy": 0.98, "dataset": "iris_v1"}
    >>> serializer.save(my_model, "model.pkl.gz", metadata=meta)
    >>>
    >>> # Reload the model later
    >>> loaded_model = serializer.load("model.pkl.gz")
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
        """
        Save a trained model to disk.

        Parameters
        ----------
        model : Any
            Trained model object to save.
        filepath : str or Path
            Path to save the model.
        metadata : dict, optional
            Additional metadata to store with the model.

        Returns
        -------
        filepath : str
            Actual path where model was saved.
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
            self._save_compressed(data, filepath)

        return str(filepath)

    def load(self, filepath: Union[str, Path]) -> Any:
        """
        Load a model from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved model.

        Returns
        -------
        model : Any
            Loaded model object.
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
        """
        Load a model with its metadata.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved model.

        Returns
        -------
        data : dict
            Dictionary with 'model', 'metadata', 'params', etc.
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

    def _save_compressed(self, data: Dict, filepath: Path):
        """Save using gzip-compressed pickle."""
        if not str(filepath).endswith('.gz'):
            filepath = Path(str(filepath) + '.gz')
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=self.protocol)

    def _load_compressed(self, filepath: Path) -> Dict:
        """Load gzip-compressed pickle."""
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)

    def _save_joblib(self, data: Dict, filepath: Path):
        """Save using joblib."""
        try:
            import joblib
            joblib.dump(data, filepath)
        except ImportError:
            raise ImportError(
                "joblib is required for joblib format. "
                "Install it with: pip install joblib"
            )

    def _load_joblib(self, filepath: Path) -> Dict:
        """Load using joblib."""
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

    Parameters
    ----------
    model : Any
        The trained model instance (or any picklable object) to save.

    filepath : str or Path
        The destination file path.

    format : {"pickle", "joblib"}, default="pickle"
        The underlying serialization format.

    metadata : dict, optional
        A dictionary of custom metadata (e.g., metrics, data version) 
        to store alongside the model.

    compress : bool, default=False
        If ``True``, applies gzip compression to the output file.

    Returns
    -------
    str
        The absolute path where the model was saved.

    See Also
    --------
    :func:`~tuiml.utils.serialization.load_model` : The inverse operation.

    Examples
    --------
    Save a model with high compression:

    >>> from tuiml.utils.serialization import save_model
    >>> save_model(my_clf, "classifier.pkl.gz", compress=True)
    '/path/to/classifier.pkl.gz'
    """
    actual_format = 'compressed' if compress else format
    serializer = ModelSerializer(format=actual_format)
    return serializer.save(model, filepath, metadata)

def load_model(filepath: Union[str, Path]) -> Any:
    """Load a model from disk.

    Automagically detects the format (pickle, joblib, or compressed) based 
    on the file signature and extension.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved model file.

    Returns
    -------
    Any
        The reconstructed model object.

    Examples
    --------
    >>> from tuiml.utils.serialization import load_model
    >>> model = load_model("classifier.pkl")
    >>> result = model.predict(X_test)
    """
    serializer = ModelSerializer()
    return serializer.load(filepath)

def load_model_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load model metadata and hyper-parameters without loading the heavy model object.

    Useful for inspecting model properties, comparing performance, or 
    retrieving training timestamps without the memory overhead of loading 
    large model arrays.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved model file.

    Returns
    -------
    dict
        A dictionary containing tracked info:
        - ``"model_class"``: Name of the Python class.
        - ``"params"``: Configuration parameters/hyper-parameters.
        - ``"saved_at"``: ISO timestamp of the save operation.
        - ``"metadata"``: The custom metadata provided during save.

    Examples
    --------
    Check the accuracy of a saved model:

    >>> from tuiml.utils.serialization import load_model_info
    >>> info = load_model_info("model.pkl")
    >>> print(f"Model: {info['model_class']} | Acc: {info['metadata']['accuracy']}")
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
        The best recorded value of the monitored metric so far.

    checkpoints : list of Path
        A chronologically ordered list of active checkpoint file paths.

    Examples
    --------
    Save the top 3 models by F1-score:

    >>> from tuiml.utils.serialization import ModelCheckpoint
    >>> checkpoint = ModelCheckpoint(
    ...     directory="./models",
    ...     save_best_only=True,
    ...     monitor="f1",
    ...     mode="max",
    ...     max_to_keep=3
    ... )
    >>> for i in range(10):
    ...     score = train_epoch()
    ...     checkpoint.save(my_model, epoch=i, f1=score)
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
        """
        Save a checkpoint.

        Parameters
        ----------
        model : Any
            Model to checkpoint.
        epoch : int, optional
            Current epoch number.
        **metrics
            Metric values (e.g., accuracy=0.95, loss=0.1).

        Returns
        -------
        filepath : str or None
            Path to saved checkpoint, or None if not saved.
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
        """Load the most recent checkpoint."""
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
        """Load the best checkpoint (only valid with save_best_only=True)."""
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
        of 20 features).

    input_names : list of str, optional
        Custom names for the input nodes.

    output_names : list of str, optional
        Custom names for the output nodes.

    Returns
    -------
    str
        The path to the successfully exported ONNX model.

    Notes
    -----
    This utility requires the ``skl2onnx`` package. 
    Install via: ``pip install skl2onnx``
    """
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
    initial_type = [
        ('input', FloatTensorType([None, n_features]))
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
