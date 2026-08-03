"""Registry- and API-driven contract checks, one module per component family.

Laid out the way scikit-learn splits ``estimator_checks`` from the sweep that
runs it: the check functions live here, and the ``tests/common/`` modules
parametrise them over whatever the registry currently holds, one sweep module
per family mirroring the module names below. Adding a component subscribes it
to its family's whole battery; adding a check here applies it to every
component in that family at once.

============================ ==============================================
Module                       Covers
============================ ==============================================
``algorithms``               classifiers, regressors, clusterers, forecasters
``transformers``             preprocessing, feature transformers, samplers
``splitters``                cross-validation splitters
``associators``              association-rule miners
``_data``                    shared fixture router
============================ ==============================================
"""

from . import algorithms, associators, splitters, transformers

__all__ = ["algorithms", "transformers", "splitters", "associators"]
