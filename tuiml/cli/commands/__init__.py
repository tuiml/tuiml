"""
CLI commands for TuiML.
"""

from . import train
from . import predict
from . import evaluate
from . import experiment
from . import list_cmd
from . import upload
from . import serve

__all__ = [
    'train',
    'predict',
    'evaluate',
    'experiment',
    'list_cmd',
    'upload',
    'serve',
]
