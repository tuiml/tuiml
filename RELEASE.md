# Release Process

How to ship a new tuiml version to PyPI and update the bundled website.

Everything now lives in this single repo: the library at the root, the
landing/docs site in `website/`, and tooling in `scripts/`.

## Quick reference

```bash
# From the repo root
uv run scripts/bump_version.py patch     # 1.2.3 → 1.2.4
                                         # or: minor / major / X.Y.Z

# Fill in the new section
$EDITOR CHANGELOG.md

# Regenerate the API docs so the site reflects the new code
uv run scripts/generate_docs.py

# Commit + tag + release (this ships to PyPI)
git add -A && git commit -m "Bump version to X.Y.Z"
git tag vX.Y.Z
git push origin main --tags
gh release create vX.Y.Z --generate-notes
```

## What gets bumped

`scripts/bump_version.py` updates the version string in every file that
carries one. Current tracked files:

**Library (repo root):**
- `pyproject.toml`
- `tuiml/__init__.py`
- `tuiml/_cpp/module.cpp` (`m.attr("__version__")`)
- `tuiml/agent/prompts/SKILL.md` (frontmatter `version:`)
- `tutorials/mcp_server.ipynb` (a stored output cell prints
  the version, and the tutorial is published on the site)
- `CHANGELOG.md` (adds an empty section)

**Bundled website (`website/`):**
- `website/pyproject.toml`
- `website/build.py` (`APP_VERSION` — the site's single source of truth)
- All `website/templates/pages/**/*.html` and `_generated/**/*.html`
  (raw string replace of the old version)

The site's changelog page is built from the root `CHANGELOG.md` directly —
there is no website copy to keep in sync.

If you add a new file with a hardcoded version, add it to the
`VERSION_FILES` list in `scripts/bump_version.py` so future bumps catch
it. Avoid hardcoding the release version in tests: `test_schemas.py` used
to, and every bump broke it, because the script rewrote the constructor
argument and left the assertion on the old value.

To audit, run (substituting the version you just moved *off*):

```bash
grep -rn "0\.1\.6" --include='*.py' --include='*.toml' \
    --include='*.md' --include='*.cpp' --include='*.h' \
    --include='*.html' --include='*.yml' --include='*.json' \
    | grep -v '.venv\|__pycache__\|benchmark_results\|uv.lock\|.git/\|_site/'
```

## What triggers PyPI publish

`.github/workflows/publish.yml` fires on:
- `release: published` — the `gh release create` step above
- `workflow_dispatch` — manual trigger from the GitHub UI

It builds wheels for Linux x86_64/aarch64, macOS, Windows across Python
3.10–3.13 (`cibuildwheel`), then publishes to PyPI. The `website/` folder
is excluded from the package (`wheel.packages = ["tuiml"]` and
`sdist.exclude = ["website"]`), so it never ships to PyPI.

## Website deploy

The site is static and served by **GitHub Pages** at https://tuiml.ai.
`website/build.py` is the whole toolchain: it renders the Jinja templates in
`website/templates/`, converts the root `tutorials/` notebooks, and parses the
root `CHANGELOG.md` into static HTML:

```bash
cd website
uv run python build.py      # → website/_site/  (git-ignored build output)
```

Pushing to `main` triggers the Pages workflow (`.github/workflows/pages.yml`)
to rebuild and deploy automatically — no manual step. Regenerate the API docs
(`scripts/generate_docs.py`) before a release so the published site matches
the new version.

## Common gotchas

- **Tag already exists from a failed run:** delete it locally and on the
  remote before re-tagging:
  ```bash
  git tag -d vX.Y.Z
  git push origin :refs/tags/vX.Y.Z
  ```
- **PyPI shows old version after release:** the GitHub Actions run uses
  `skip-existing: true`, so a re-run without bumping the version is a
  no-op. Bump first.
- **`bump_version.py` aborts with "VERSION_FILES is out of date":** a
  registered file has moved, been deleted, or no longer matches its regex.
  The check runs before anything is written, so the tree is untouched — fix
  the entry in `scripts/bump_version.py` (or drop it if the file no longer
  carries a version) and re-run. This used to be a `SKIP`/`WARN` that let
  the bump continue, which is how `SKILL.md` shipped a release still on the
  previous version after it moved to `tuiml/agent/prompts/`.
