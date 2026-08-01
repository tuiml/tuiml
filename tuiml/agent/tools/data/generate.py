"""Synthetic data generation."""

import os
import tempfile
import uuid
from typing import Any, Dict

from .._spec import ToolSpec


def execute_generate_data(**kwargs) -> Dict[str, Any]:
    """Generate synthetic data using a generator class.

    Backs the ``tuiml_generate_data`` tool. The generated dataset is
    written to a temporary CSV whose path can be passed to other tools.

    Parameters
    ----------
    generator : str
        Generator class name, one of: RandomRBF, Agrawal, LED,
        Hyperplane, Friedman, MexicanHat, Sine, Blobs, Moons, Circles,
        SwissRoll (arrives via ``**kwargs``, like all parameters below).
    n_samples : int, default=None
        Number of samples to generate.
    n_features : int, default=None
        Number of features (generator dependent).
    n_classes : int, default=None
        Number of classes (classification generators).
    n_clusters : int, default=None
        Number of clusters (clustering generators).
    noise : float, default=None
        Noise level (generator dependent).
    random_seed : int, default=None
        Random seed; mapped to the generator's ``random_state``.
    generator_params : dict, default=None
        Additional constructor parameters merged on top.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``generator``,
        ``file_path`` (CSV on disk), ``shape``, ``feature_names``,
        ``preview`` (first 5 rows of up to 6 columns) and optionally
        ``target_names``. On failure: ``status`` (``'error'``),
        ``error`` and optionally ``suggestion`` / ``error_type``.
    """
    import numpy as np

    try:
        generator_name = kwargs['generator']

        from tuiml.datasets.generators import (
            RandomRBF, Agrawal, LED, Hyperplane,
            Friedman, MexicanHat, Sine,
            Blobs, Moons, Circles, SwissRoll,
        )

        generators = {
            'RandomRBF': RandomRBF, 'Agrawal': Agrawal, 'LED': LED, 'Hyperplane': Hyperplane,
            'Friedman': Friedman, 'MexicanHat': MexicanHat, 'Sine': Sine,
            'Blobs': Blobs, 'Moons': Moons, 'Circles': Circles, 'SwissRoll': SwissRoll,
        }

        gen_cls = generators.get(generator_name)
        if gen_cls is None:
            return {
                'status': 'error',
                'error': f"Generator '{generator_name}' not found.",
                'suggestion': f"Available generators: {list(generators.keys())}"
            }

        # Build constructor params
        params = {}
        extra_params = kwargs.get('generator_params', {})
        if 'random_seed' in kwargs:
            kwargs['random_state'] = kwargs.pop('random_seed')

        for key in ('n_samples', 'n_features', 'n_classes', 'n_clusters', 'noise', 'random_state'):
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        params.update(extra_params)

        gen = gen_cls(**params)
        data = gen.generate()

        # Save to CSV temp file
        upload_dir = os.path.join(tempfile.gettempdir(), 'tuiml_generated')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f'{generator_name.lower()}_{uuid.uuid4().hex[:8]}.csv')

        import pandas as pd
        feature_names = list(data.feature_names) if data.feature_names else [f'x{i}' for i in range(data.X.shape[1])]
        df = pd.DataFrame(data.X, columns=feature_names)
        if data.y is not None:
            df['target'] = data.y
        df.to_csv(file_path, index=False)

        # Preview: first 5 rows
        preview = {col: df[col].head(5).tolist() for col in df.columns[:6]}

        result = {
            'status': 'success',
            'generator': generator_name,
            'file_path': file_path,
            'shape': [int(data.X.shape[0]), int(data.X.shape[1])],
            'feature_names': feature_names,
            'preview': preview,
        }
        if data.target_names:
            result['target_names'] = list(data.target_names)

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


SPEC = ToolSpec(
    name='tuiml_generate_data',
    description="Generate synthetic datasets for testing and demos. Supports classification "
        "(RandomRBF, Agrawal, LED, Hyperplane), regression (Friedman, MexicanHat, Sine), "
        "and clustering (Blobs, Moons, Circles, SwissRoll) generators.",
    input_schema={
            "type": "object",
            "properties": {
                "generator": {
                    "type": "string",
                    "enum": [
                        "RandomRBF", "Agrawal", "LED", "Hyperplane",
                        "Friedman", "MexicanHat", "Sine",
                        "Blobs", "Moons", "Circles", "SwissRoll"
                    ],
                    "description": "Generator class name"
                },
                "n_samples": {
                    "type": "integer",
                    "default": 100,
                    "description": "Number of samples to generate"
                },
                "n_features": {
                    "type": "integer",
                    "description": "Number of features (not all generators support this)"
                },
                "n_classes": {
                    "type": "integer",
                    "description": "Number of classes (classification generators only)"
                },
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of clusters (clustering generators only)"
                },
                "noise": {
                    "type": "number",
                    "description": "Noise level (regression generators only)"
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                },
                "generator_params": {
                    "type": "object",
                    "description": "Additional generator-specific parameters"
                }
            },
            "required": ["generator"]
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "generator": {"type": "string"},
                "file_path": {"type": "string"},
                "shape": {"type": "array", "items": {"type": "integer"}},
                "feature_names": {"type": "array", "items": {"type": "string"}},
                "target_names": {"type": "array", "items": {"type": "string"}},
                "preview": {"type": "object"},
                "random_seed": {"type": "integer"},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_generate_data,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=False, open_world=False,
    seeded=True,
)
