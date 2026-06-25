"""CapyMOA (streaming / online-learning) bridge for TuiML.

Importing this package registers a curated set of CapyMOA incremental learners
into the TuiML hub under ``capymoa.<ClassName>`` keys. These are genuinely
streaming learners (concept-drift aware, instance-incremental) and are distinct
from TuiML's native streaming algorithms in :mod:`tuiml.algorithms.streaming`.

CapyMOA is an optional dependency (it pulls in a JVM-backed MOA runtime). If it
is not installed, importing this package still succeeds, but instantiating a
wrapper raises a clear ``ImportError`` pointing at ``pip install tuiml[capymoa]``.
"""

from tuiml.capymoa import streaming  # noqa: F401  triggers registration
from tuiml.capymoa.streaming import *  # noqa: F401,F403

__all__ = [*streaming.__all__]
