"""Preprocessing pipelines and stages."""

import os
import uuid
from typing import Any, Dict

from .._spec import ToolSpec
from .._shared import _load_data


def execute_preprocess(**kwargs) -> Dict[str, Any]:
    """Apply preprocessing steps or a specific atomic stage to a dataset.

    Backs the ``tuiml_preprocess`` tool. Two modes: ``steps`` runs an
    ordered pipeline of named preprocessors; ``stage`` runs one atomic
    operation (``split``, ``impute``, ``balance``, ``scale``, ``encode``,
    ``discretize``) with a default class per stage. Output is written to
    CSV so downstream tools can consume it by path.

    Parameters
    ----------
    data : str
        Dataset to preprocess: dataset_id, file path, or built-in name
        (arrives via ``**kwargs``, like all parameters below).
    steps : list, default=None
        Pipeline mode: preprocessor steps, each a class name or
        ``{"name", **params}`` dict. One of ``steps`` / ``stage`` is
        required.
    stage : str, default=None
        Atomic stage mode: ``'split'``, ``'impute'``, ``'balance'``,
        ``'scale'``, ``'encode'``, or ``'discretize'``.
    stage_kwargs : dict, default=None
        Stage options, e.g. ``method`` to pick a specific class, or for
        ``split``: ``n_splits``/``kfold``, ``test_size``, ``train_size``,
        ``shuffle``, ``stratify``, ``random_seed``.
    output : str, default=None
        Output file or directory (alias: ``save_as``); defaults to a
        temp location.
    save_as : str, default=None
        Alias for ``output``.
    target : str, default='target'
        Column name for the label in the output CSV.
    random_seed : int, default=None
        Random seed forwarded to seed-aware stages.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``original_shape``, and
        either ``file_path``, ``new_shape`` and ``steps_applied``, or --
        for ``stage='split'`` -- ``split_type`` (``'kfold'`` /
        ``'holdout'``) and ``files`` (train/test CSV paths, per fold for
        k-fold). On failure: ``status`` (``'error'``), ``error`` and
        optionally ``error_type`` / ``suggestion``.
    """
    import numpy as np
    import pandas as pd

    try:
        dataset = _load_data(kwargs['data'])
        X = np.asarray(dataset.X, dtype=float)
        y = dataset.y
        original_shape = list(X.shape)
        feature_names = list(dataset.feature_names) if hasattr(dataset, 'feature_names') and dataset.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]

        stage = kwargs.get('stage')
        steps = kwargs.get('steps')

        if stage is None and steps is None:
            return {
                'status': 'error',
                'error': "Either 'steps' or 'stage' must be specified for preprocessing."
            }

        if stage is not None:
            # Atomic stage execution
            stage = stage.strip().lower()
            if stage == 'split':
                from pathlib import Path
                # Get splitting arguments from stage_kwargs
                stage_kwargs = kwargs.get('stage_kwargs') or {}
                # Support common aliases
                n_splits = stage_kwargs.get('kfold') or stage_kwargs.get('n_splits')
                test_size = stage_kwargs.get('test_size')
                train_size = stage_kwargs.get('train_size')
                shuffle = stage_kwargs.get('shuffle', True)
                stratify_flag = stage_kwargs.get('stratify', False)
                # Use random_seed or random_state
                seed = stage_kwargs.get('random_seed') or stage_kwargs.get('random_state') or kwargs.get('random_seed')

                # Determine output location
                output = kwargs.get('output') or kwargs.get('save_as')
                if output:
                    output_path = Path(output)
                    if output_path.suffix == '' or output_path.is_dir():
                        os.makedirs(output_path, exist_ok=True)
                        prefix = ""
                        out_dir = output_path
                    else:
                        os.makedirs(output_path.parent, exist_ok=True)
                        prefix = output_path.name.rsplit('.', 1)[0] + "_"
                        out_dir = output_path.parent
                else:
                    import tempfile
                    out_dir = Path(tempfile.mkdtemp(prefix='tuiml_split_'))
                    prefix = ""

                out_cols = feature_names[:X.shape[1]] if len(feature_names) >= X.shape[1] else [f'feature_{i}' for i in range(X.shape[1])]

                # If n_splits is provided, we perform K-Fold / StratifiedKFold split
                if n_splits is not None:
                    n_splits = int(n_splits)
                    from tuiml.evaluation.splitting import KFold, StratifiedKFold
                    if stratify_flag and y is not None:
                        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
                    else:
                        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

                    split_files = {}
                    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
                        X_train, X_test = X[train_idx], X[test_idx]

                        df_train = pd.DataFrame(X_train, columns=out_cols)
                        df_test = pd.DataFrame(X_test, columns=out_cols)

                        if y is not None:
                            target_name = kwargs.get('target', 'target')
                            df_train[target_name] = y[train_idx]
                            df_test[target_name] = y[test_idx]

                        train_path = out_dir / f"{prefix}train_{i}.csv"
                        test_path = out_dir / f"{prefix}test_{i}.csv"

                        df_train.to_csv(train_path, index=False)
                        df_test.to_csv(test_path, index=False)

                        split_files[f"fold_{i}"] = {
                            "train": str(train_path),
                            "test": str(test_path)
                        }

                    return {
                        'status': 'success',
                        'stage': 'split',
                        'split_type': 'kfold',
                        'n_splits': n_splits,
                        'files': split_files,
                        'original_shape': original_shape
                    }
                else:
                    # Simple holdout split (train_test_split)
                    from tuiml.evaluation.splitting import train_test_split

                    # Prepare arguments for train_test_split
                    split_args = [X]
                    if y is not None:
                        split_args.append(y)

                    stratify_arr = y if (stratify_flag and y is not None) else None

                    splits = train_test_split(
                        *split_args,
                        test_size=test_size,
                        train_size=train_size,
                        shuffle=shuffle,
                        stratify=stratify_arr,
                        random_state=seed
                    )

                    if y is not None:
                        X_train, X_test, y_train, y_test = splits
                    else:
                        X_train, X_test = splits
                        y_train, y_test = None, None

                    df_train = pd.DataFrame(X_train, columns=out_cols)
                    df_test = pd.DataFrame(X_test, columns=out_cols)

                    if y is not None:
                        target_name = kwargs.get('target', 'target')
                        df_train[target_name] = y_train
                        df_test[target_name] = y_test

                    train_path = out_dir / f"{prefix}train.csv"
                    test_path = out_dir / f"{prefix}test.csv"

                    df_train.to_csv(train_path, index=False)
                    df_test.to_csv(test_path, index=False)

                    return {
                        'status': 'success',
                        'stage': 'split',
                        'split_type': 'holdout',
                        'files': {
                            'train': str(train_path),
                            'test': str(test_path)
                        },
                        'original_shape': original_shape,
                        'train_shape': list(X_train.shape),
                        'test_shape': list(X_test.shape)
                    }

            else:
                # Other atomic preprocessing stages
                import tuiml.preprocessing as pp_module

                stage_kwargs = kwargs.get('stage_kwargs') or {}
                method = stage_kwargs.get('method')

                # Filter out params that shouldn't be passed directly to initialization
                estimator_params = {k: v for k, v in stage_kwargs.items() if k != 'method'}

                # Check for random_seed / random_state / seed and inject if appropriate
                seed = stage_kwargs.get('random_seed') or stage_kwargs.get('random_state') or kwargs.get('random_seed')

                # Map stage to default class and suffix
                if stage == 'impute':
                    class_name = 'SimpleImputer'
                    if method:
                        if method.lower() in ('knn', 'knnimputer'):
                            class_name = 'KNNImputer'
                        elif method.lower() in ('simple', 'simpleimputer'):
                            class_name = 'SimpleImputer'
                        else:
                            class_name = method
                elif stage == 'balance':
                    class_name = 'SMOTESampler'
                    if method:
                        class_name = method
                elif stage == 'scale':
                    class_name = 'StandardScaler'
                    if method:
                        class_name = method
                elif stage == 'encode':
                    class_name = 'OneHotEncoder'
                    if method:
                        class_name = method
                elif stage == 'discretize':
                    class_name = 'EqualWidthDiscretizer'
                    if method:
                        class_name = method
                else:
                    return {
                        'status': 'error',
                        'error': f"Unknown preprocessing stage: '{stage}'"
                    }

                # Helper to perform case-insensitive attribute lookup
                def resolve_class_name(name):
                    """Resolve a preprocessor class name case-insensitively in tuiml.preprocessing.

                    Parameters
                    ----------
                    name : str
                        Candidate class name to look up.

                    Returns
                    -------
                    resolved : str or None
                        The exact attribute name in ``tuiml.preprocessing``,
                        or None if no match exists.
                    """
                    if hasattr(pp_module, name):
                        return name
                    for attr in dir(pp_module):
                        if attr.lower() == name.lower():
                            return attr
                    return None

                resolved_name = resolve_class_name(class_name)
                if not resolved_name:
                    # Try matching with suffix based on stage
                    suffix = ""
                    if stage == 'balance': suffix = 'sampler'
                    elif stage == 'scale': suffix = 'scaler'
                    elif stage == 'encode': suffix = 'encoder'
                    elif stage == 'discretize': suffix = 'discretizer'
                    elif stage == 'impute': suffix = 'imputer'

                    if suffix and not class_name.lower().endswith(suffix):
                        resolved_name = resolve_class_name(class_name + suffix)

                if resolved_name:
                    preprocessor_cls = getattr(pp_module, resolved_name)
                    class_name = resolved_name
                else:
                    # Fallback to registry lookup
                    from tuiml.registry import registry
                    try:
                        preprocessor_cls = registry.get(class_name)
                    except Exception:
                        preprocessor_cls = None

                if preprocessor_cls is None:
                    return {
                        'status': 'error',
                        'error': f"Preprocessor class '{class_name}' for stage '{stage}' not found."
                    }

                # If random_seed is supported, inject it
                import inspect
                init_sig = inspect.signature(preprocessor_cls.__init__)
                if seed is not None:
                    if 'random_state' in init_sig.parameters and 'random_state' not in estimator_params:
                        estimator_params['random_state'] = seed
                    elif 'random_seed' in init_sig.parameters and 'random_seed' not in estimator_params:
                        estimator_params['random_seed'] = seed

                preprocessor = preprocessor_cls(**estimator_params)

                if hasattr(preprocessor, 'fit_resample') and y is not None:
                    X, y = preprocessor.fit_resample(X, y)
                else:
                    from tuiml.base.preprocessing import InstanceTransformer
                    if isinstance(preprocessor, InstanceTransformer):
                        result = preprocessor.fit_transform(X, y)
                        X, y = result[0], result[1]
                    else:
                        from tuiml.base.preprocessing import SupervisedTransformer
                        if isinstance(preprocessor, SupervisedTransformer) and y is not None:
                            X = preprocessor.fit_transform(X, y)
                        else:
                            X = preprocessor.fit_transform(X)

                # Save result
                output = kwargs.get('output') or kwargs.get('save_as')
                if output:
                    from pathlib import Path
                    output_path = Path(output)
                    if output_path.is_dir() or output.endswith('/') or output.endswith('\\'):
                        os.makedirs(output_path, exist_ok=True)
                        file_path = str(output_path / f"preprocessed_{uuid.uuid4().hex[:8]}.csv")
                    else:
                        os.makedirs(output_path.parent, exist_ok=True)
                        file_path = str(output_path)
                else:
                    import tempfile
                    upload_dir = os.path.join(tempfile.gettempdir(), 'tuiml_preprocessed')
                    os.makedirs(upload_dir, exist_ok=True)
                    file_path = os.path.join(upload_dir, f'preprocessed_{uuid.uuid4().hex[:8]}.csv')

                out_cols = feature_names[:X.shape[1]] if len(feature_names) >= X.shape[1] else [f'feature_{i}' for i in range(X.shape[1])]
                df = pd.DataFrame(X, columns=out_cols)
                if y is not None:
                    target_name = kwargs.get('target', 'target')
                    df[target_name] = y
                df.to_csv(file_path, index=False)

                return {
                    'status': 'success',
                    'stage': stage,
                    'file_path': file_path,
                    'original_shape': original_shape,
                    'new_shape': list(X.shape),
                    'steps_applied': [class_name],
                }

        else:
            # Standard step-by-step pipeline execution
            steps_applied = []
            from tuiml.registry import registry
            import tuiml.preprocessing  # noqa: F401 - trigger registration

            for step in steps:
                if isinstance(step, str):
                    name, params = step, {}
                elif isinstance(step, dict):
                    name = step.get('name')
                    params = {k: v for k, v in step.items() if k != 'name'}
                else:
                    continue

                # Resolve preprocessor class
                preprocessor_cls = None
                try:
                    preprocessor_cls = registry.get(name)
                except (KeyError, Exception):
                    pass

                if preprocessor_cls is None:
                    # Fallback: try direct import
                    try:
                        from tuiml import preprocessing as pp_module
                        preprocessor_cls = getattr(pp_module, name, None)
                    except ImportError:
                        pass

                if preprocessor_cls is None:
                    return {
                        'status': 'error',
                        'error': f"Preprocessor '{name}' not found.",
                        'suggestion': "Use tuiml_list with category='preprocessing' to see available preprocessors."
                    }

                preprocessor = preprocessor_cls(**params)
                if hasattr(preprocessor, 'fit_resample') and y is not None:
                    X, y = preprocessor.fit_resample(X, y)
                else:
                    from tuiml.base.preprocessing import InstanceTransformer
                    if isinstance(preprocessor, InstanceTransformer):
                        result = preprocessor.fit_transform(X, y)
                        X, y = result[0], result[1]
                    else:
                        X = preprocessor.fit_transform(X)

                steps_applied.append(name)

            # Save result to CSV
            save_as = kwargs.get('save_as') or kwargs.get('output')
            if save_as:
                file_path = save_as
                os.makedirs(os.path.dirname(os.path.abspath(save_as)) or '.', exist_ok=True)
            else:
                import tempfile
                upload_dir = os.path.join(tempfile.gettempdir(), 'tuiml_preprocessed')
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, f'preprocessed_{uuid.uuid4().hex[:8]}.csv')

            # Build output DataFrame
            out_cols = feature_names[:X.shape[1]] if len(feature_names) >= X.shape[1] else [f'feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=out_cols)
            if y is not None:
                target_name = kwargs.get('target', 'target')
                df[target_name] = y
            df.to_csv(file_path, index=False)

            return {
                'status': 'success',
                'file_path': file_path,
                'original_shape': original_shape,
                'new_shape': list(X.shape),
                'steps_applied': steps_applied,
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


SPEC = ToolSpec(
    name='tuiml_preprocess',
    description="Apply preprocessing steps to a dataset and return the result as a new file. "
        "Supports running standard pipelines or single atomic stages like split, impute, "
        "balance, scale, encode, and discretize.",
    input_schema={
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (excluded from preprocessing, re-appended to output)"
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                                "additionalProperties": True
                            }
                        ]
                    },
                    "description": (
                        "Preprocessing steps as names or objects with params.\n"
                        "Examples: ['StandardScaler', 'SimpleImputer'] or "
                        "[{'name': 'SimpleImputer', 'strategy': 'median'}, 'MinMaxScaler']"
                    )
                },
                "stage": {
                    "type": "string",
                    "description": "Atomic preprocessing stage to execute: 'split', 'impute', 'balance', 'scale', 'encode', 'discretize'"
                },
                "stage_kwargs": {
                    "type": "object",
                    "description": "Arbitrary keyword arguments for the selected stage (e.g. kfold, test_size, strategy, method)"
                },
                "output": {
                    "type": "string",
                    "description": "Output path to save the generated file(s)"
                },
                "save_as": {
                    "type": "string",
                    "description": "Custom output file path (optional, alias for output)"
                }
            },
            "required": ["data"]
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "file_path": {"type": "string"},
                "files": {
                    "type": "object",
                    "description": "Mapping of split names/folds to file paths"
                },
                "stage": {"type": "string"},
                "split_type": {"type": "string"},
                "n_splits": {"type": "integer"},
                "original_shape": {"type": "array", "items": {"type": "integer"}},
                "new_shape": {"type": "array", "items": {"type": "integer"}},
                "steps_applied": {"type": "array", "items": {"type": "string"}},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_preprocess,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=False, open_world=False,
)
