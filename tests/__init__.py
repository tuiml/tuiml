"""TuiML Test Suite.

Laid out to mirror the package: one subpackage per top-level namespace, and
within it one module per component family.

===================== =====================================================
Path                  Covers
===================== =====================================================
``algorithms/``       learners, kernels, distances, tree internals
``preprocessing/``    scaling, encoding, discretization, sampling, text, ...
``features/``         selection, extraction, generation
``datasets/``         loaders, builtin datasets, synthetic generators
``serving/``          model manager, schemas, HTTP server
``common/``           registry-wide sweeps over every registered component
``contract/``         the check batteries ``common/`` parametrises
===================== =====================================================

Modules at the top level (``test_workflow``, ``test_base``, ``test_agent``,
``test_integration``) cover behaviour that spans namespaces.

Run all tests:
    cd tuiml && uv run pytest

Run with coverage:
    cd tuiml && uv run pytest --cov=tuiml --cov-report=html

Run one family:
    cd tuiml && uv run pytest tests/algorithms/test_trees.py
"""
