#!/usr/bin/env python3
"""Bump version across all tuiml project files.

Usage:
    python scripts/bump_version.py 0.2.0
    python scripts/bump_version.py patch      # 0.1.1 -> 0.1.2
    python scripts/bump_version.py minor      # 0.1.1 -> 0.2.0
    python scripts/bump_version.py major      # 0.1.1 -> 1.0.0
"""

import re
import sys
from datetime import date
from pathlib import Path

# scripts/ lives at the repo root, so parent.parent IS the tuiml repo root. The
# library files sit directly under it; the bundled website lives in website/.
ROOT = Path(__file__).resolve().parent.parent
TUIML = ROOT
HUB = ROOT / "website"

VERSION_FILES = [
    # (file_path, regex_pattern, replacement_template)
    # --- tuiml core ---
    (
        TUIML / "pyproject.toml",
        r'(^version\s*=\s*")[^"]+(")',
        r"\g<1>{version}\2",
    ),
    (
        TUIML / "tuiml" / "__init__.py",
        r'(__version__\s*=\s*")[^"]+(")',
        r"\g<1>{version}\2",
    ),
    (
        TUIML / "tuiml" / "_cpp" / "module.cpp",
        r'(m\.attr\("__version__"\)\s*=\s*")[^"]+(")',
        r"\g<1>{version}\2",
    ),
    (
        TUIML / "tests" / "test_serving" / "test_schemas.py",
        r'(version=")[^"]+(")',
        r"\g<1>{version}\2",
    ),
    (
        TUIML / "tuiml" / "agent" / "SKILL.md",
        r"(^version:\s*)[^\s]+",
        r"\g<1>{version}",
    ),
    # --- bundled website (website/) ---
    (
        HUB / "pyproject.toml",
        r'(^version\s*=\s*")[^"]+(")',
        r"\g<1>{version}\2",
    ),
    # build.py's APP_VERSION is the site's single source of truth (injected into
    # every template as `settings`). A broad `version="..."` rule here once
    # matched the sitemap's `<?xml version="...">` declaration — keep this
    # anchored to APP_VERSION.
    (
        HUB / "build.py",
        r'(APP_VERSION\s*=\s*")[^"]+(")',
        r"\g<1>{version}\2",
    ),
]


def get_current_version() -> str:
    """Read current version from tuiml/pyproject.toml."""
    text = (TUIML / "pyproject.toml").read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', text)
    if not match:
        print("ERROR: Could not find version in tuiml/pyproject.toml")
        sys.exit(1)
    return match.group(1)


def compute_next_version(current: str, bump_type: str) -> str:
    """Compute next version from bump type (patch/minor/major)."""
    parts = current.split(".")
    if len(parts) != 3:
        print(f"ERROR: Current version '{current}' is not semver")
        sys.exit(1)
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if bump_type == "patch":
        patch += 1
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        print(f"ERROR: Unknown bump type '{bump_type}'")
        sys.exit(1)

    return f"{major}.{minor}.{patch}"


def update_version_files(old_version: str, new_version: str) -> None:
    """Replace version strings in all tracked files."""
    for filepath, pattern, template in VERSION_FILES:
        if not filepath.exists():
            print(f"  SKIP {filepath.relative_to(ROOT)} (not found)")
            continue

        text = filepath.read_text()
        replacement = template.format(version=new_version)
        new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)

        if count == 0:
            print(f"  WARN {filepath.relative_to(ROOT)} (no match for pattern)")
        else:
            filepath.write_text(new_text)
            print(f"  OK   {filepath.relative_to(ROOT)} ({count} replacement(s))")


def update_hub_templates(old_version: str, new_version: str) -> None:
    """Replace version strings in bundled website HTML templates and docs."""
    template_dirs = [
        HUB / "templates" / "docs",
        HUB / "templates" / "docs_api",
    ]
    count = 0
    for tdir in template_dirs:
        if not tdir.exists():
            continue
        for html_file in tdir.rglob("*.html"):
            text = html_file.read_text()
            if old_version in text:
                html_file.write_text(text.replace(old_version, new_version))
                count += 1
    if count:
        print(f"  OK   website/templates/ ({count} HTML file(s) updated)")
    else:
        print(f"  SKIP website/templates/ (no version references found)")


def update_changelog(old_version: str, new_version: str) -> None:
    """Add a new version section to CHANGELOG.md, preserving history."""
    changelog = TUIML / "CHANGELOG.md"
    if not changelog.exists():
        print("  SKIP CHANGELOG.md (not found)")
        return

    text = changelog.read_text()
    today = date.today().isoformat()

    # Don't add duplicate section if version already exists in changelog
    if f"## [{new_version}]" in text:
        print(f"  SKIP tuiml/CHANGELOG.md (section for {new_version} already exists)")
        return

    new_section = (
        f"## [{new_version}] - {today}\n"
        f"\n"
        f"### Added\n"
        f"- \n"
        f"\n"
        f"### Changed\n"
        f"- \n"
        f"\n"
        f"### Fixed\n"
        f"- \n"
        f"\n"
    )

    # Insert new section before the first existing ## [x.x.x] entry
    pattern = r"(## \[\d+\.\d+\.\d+\])"
    match = re.search(pattern, text)
    if match:
        insert_pos = match.start()
        text = text[:insert_pos] + new_section + text[insert_pos:]
    else:
        text += "\n" + new_section

    # Update or add release link at bottom
    old_link = f"[{old_version}]: https://github.com/tuiml/tuiml/releases/tag/v{old_version}"
    new_link = f"[{new_version}]: https://github.com/tuiml/tuiml/releases/tag/v{new_version}"

    if old_link in text:
        text = text.replace(old_link, new_link + "\n" + old_link)
    else:
        text = text.rstrip() + "\n" + new_link + "\n"

    changelog.write_text(text)
    print(f"  OK   tuiml/CHANGELOG.md (new section for {new_version})")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]
    current = get_current_version()

    # Determine new version
    if arg in ("patch", "minor", "major"):
        new_version = compute_next_version(current, arg)
    elif re.match(r"^\d+\.\d+\.\d+$", arg):
        new_version = arg
    else:
        print(f"ERROR: '{arg}' is not a valid version or bump type (patch/minor/major)")
        sys.exit(1)

    if new_version == current:
        print(f"Version is already {current}, nothing to do.")
        sys.exit(0)

    print(f"\nBumping version: {current} -> {new_version}\n")

    update_version_files(current, new_version)
    update_hub_templates(current, new_version)
    update_changelog(current, new_version)

    print(f"\nDone! Version bumped to {new_version}")
    print(f"Next steps:")
    print(f"  1. Edit tuiml/CHANGELOG.md to fill in the changes")
    print(f"  2. git add -A && git commit -m 'Bump version to {new_version}'")
    print(f"  3. git tag v{new_version}")
    print(f"  4. git push origin main --tags")


if __name__ == "__main__":
    main()
