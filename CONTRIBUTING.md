# Contributing to TuiML

The full guide — design philosophy, base classes, how to structure and register an algorithm, the docstring convention, and how external libraries are wrapped — lives at **[tuiml.ai/contributing](https://tuiml.ai/contributing.html)**, built from [`website/templates/pages/contributing.html`](website/templates/pages/contributing.html).

This file is the short version, plus the things you only need once you are actually opening a pull request.

## Getting set up

TuiML has a compiled C++ core, so a source install needs a C++17 compiler and CMake ≥ 3.15.

```bash
git clone https://github.com/tuiml/tuiml
cd tuiml
uv sync --extra dev              # library + pytest, black, flake8, mypy
uv run pytest                    # ~3 minutes
```

Always run Python through `uv run`. Never invoke `python3` directly — it will pick up the wrong environment.

Confirm the C++ extension actually built, because a source install that silently lacks it makes two dozen test modules skip and the suite still passes:

```bash
uv run python -c "import tuiml._cpp; print('ok')"
```

Optional extras change what runs. `uv sync --all-extras` installs everything, including torch. Without them the affected algorithms are skipped rather than failed — `uv run pytest -rs` lists exactly what was skipped and why, which is worth checking before you open a PR if you touched `algorithms/tabular_deep/` or `algorithms/timeseries/deep/`.

## What CI checks

`.github/workflows/test.yml` runs on every push and pull request:

- pytest on ubuntu, macOS and Windows across Python 3.10–3.13
- an assertion that `tuiml._cpp` imported
- one wheel build per platform, imported after building

`black --check` and `mypy` also run but do **not** block: the tree currently has 426 files black would reformat and 2,111 mypy errors, so enforcing either would fail every commit. Don't add to those numbers; format new files with `black` before committing.

## Things that are easy to get wrong

**Use TuiML's own implementations.** `tuiml.evaluation.splitting` for cross-validation, `tuiml.evaluation.metrics` for scoring, `tuiml.base.algorithms` for base classes. Never `from sklearn.model_selection import ...` in core code. Wrappers around external libraries belong in `tuiml/sklearn/`, `tuiml/capymoa/` or `tuiml/weka/`, never in `algorithms/`, `preprocessing/` or `features/`.

**Docstrings are NumPy style, not Google.** Section headers use dash underlines (`Parameters` over `-----`). The documentation generator does not parse `Args:`. Every `Examples` block must open with the import that defines the names it uses — a reader copies it straight out of the rendered docs, and a block starting with a bare constructor call fails with `NameError`.

**Escape LaTeX backslashes in docstrings.** Python eats `\t`, `\a`, `\b`, `\f`, `\n`, `\r` and `\v`, so `\text`, `\alpha`, `\beta`, `\frac`, `\begin` and `\nabla` must be written `\\text`, `\\alpha` and so on.

**Registering an algorithm subscribes it to the contract suite** — eleven invariants including seeded reproducibility, no mutation of input, and a pickle round-trip. If it fails one of those, fix the algorithm rather than adding an entry to `XFAIL_CHECKS` in `tests/common/test_algorithms.py`. That table is a list of known bugs and the goal is for it to be empty.

**Suggest C++ where it would pay.** If your algorithm has a hot loop — splitting, distance computation, an iterative solver — say so in the PR. C++ lives in `tuiml/_cpp/` and is shared across algorithms. Do not write a Python fallback for something implemented in C++; the compiler is a hard requirement, not an optional one.

## Opening a pull request

Branch from `main`, keep the change focused, and make sure `uv run pytest` passes. In the description, say what changed and why — a reviewer should not have to reconstruct the reasoning from the diff.

Add a `CHANGELOG.md` entry under `[Unreleased]` for anything user-visible. Say what was wrong or missing, not just what you did.

If you changed a docstring that appears in the API reference, the site regenerates it on deploy; you do not need to run `scripts/generate_docs.py` yourself.

## Reporting things

- **Bugs and features** — [GitHub issues](https://github.com/tuiml/tuiml/issues). A minimal reproduction is worth more than a description.
- **Security vulnerabilities** — do not open an issue. See [SECURITY.md](SECURITY.md).
- **Looking for something to work on** — the [Build Board](https://tuiml.ai/projects.html) lists algorithms, integrations and good first issues.

## Licence

TuiML is BSD-3-Clause. Contributions are accepted under the same licence.
