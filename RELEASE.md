# Release Process

How to ship a new tuiml version to PyPI and update the bundled website.

Everything now lives in this single repo: the library at the root, the
landing/docs site in `website/`, and tooling in `scripts/`.

## Quick reference

```bash
# From the repo root
uv run scripts/bump_version.py patch     # 0.1.6 → 0.1.7
                                         # or: minor / major / X.Y.Z

# Fill in the new section
$EDITOR CHANGELOG.md

# Regenerate the API docs so the site reflects the new code
uv run scripts/generate_docs.py

# Commit + tag + release (this ships to PyPI)
git add -A && git commit -m "Bump version to 0.1.7"
git tag v0.1.7
git push origin main --tags
gh release create v0.1.7 --generate-notes
```

## What gets bumped

`scripts/bump_version.py` updates the version string in every file that
carries one. Current tracked files:

**Library (repo root):**
- `pyproject.toml`
- `tuiml/__init__.py`
- `tuiml/_cpp/module.cpp` (`m.attr("__version__")`)
- `tuiml/agent/SKILL.md` (frontmatter `version:`)
- `tests/test_serving/test_schemas.py`
- `CHANGELOG.md` (adds an empty section)

**Bundled website (`website/`):**
- `website/pyproject.toml`
- `website/build.py` (`APP_VERSION` — the site's single source of truth)
- All `website/templates/docs/**/*.html` and `docs_api/**/*.html`
  (raw string replace of the old version)

The site's changelog page is built from the root `CHANGELOG.md` directly —
there is no website copy to keep in sync.

If you add a new file with a hardcoded version, add it to the
`VERSION_FILES` list in `scripts/bump_version.py` so future bumps catch
it. To audit, run:

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
  git tag -d v0.1.7
  git push origin :refs/tags/v0.1.7
  ```
- **PyPI shows old version after release:** the GitHub Actions run uses
  `skip-existing: true`, so a re-run without bumping the version is a
  no-op. Bump first.
- **`bump_version.py` reports `SKIP` for a file:** the path in
  `VERSION_FILES` no longer exists. Either remove or update it.
- **`bump_version.py` reports `WARN (no match)`:** the regex didn't match.
  The version is likely missing from that file, or the file format changed.
