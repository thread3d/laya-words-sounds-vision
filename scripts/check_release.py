#!/usr/bin/env python3
"""Refuse a release that is not a release, and refuse a savepoint that looks like one.

A tag does two things here: `v<version>` starts `release.yml`, which publishes to PyPI and opens a
GitHub Release, and anything else is a marker on a commit. The rules that keep those two apart were
three inline shell snippets in three workflows, each checking one property, and none of them checked
that the changelog says what the version says.

What this enforces:

* **The version is one number.** `pyproject.toml`, `laya.__version__` and `nix/package.nix` agree.
* **The changelog documents it.** The newest `## [x.y.z] - date` section is the current version and
  has a body. A heading with nothing under it reads as a release and is not one; a version bumped
  without a section is a release nobody can read. Both have happened in this repository while a
  heading was edited by hand.
* **A release tag matches the version.** `v4.0.7` requires version 4.0.7.
* **A savepoint tag cannot be mistaken for a release.** Any tag beginning with `v` has to be a full
  `vX.Y.Z`, because `v*` is what starts the publish; a marker belongs in a name like
  `ci-<date>-<topic>`.

Use it before tagging (`--tag`), and let CI run the repo-state half on every push:

    python scripts/check_release.py                    # the tree is releasable in principle
    python scripts/check_release.py --tag v4.0.7       # ...and this tag releases it
"""
import argparse
import os
import re
import sys

DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VERSION_SOURCES = {
    "pyproject.toml": re.compile(r'^\s*version\s*=\s*"([^"]+)"', re.M),
    "laya/__init__.py": re.compile(r'^__version__\s*=\s*"([^"]+)"', re.M),
    "nix/package.nix": re.compile(r'^\s*version\s*=\s*"([^"]+)"', re.M),
}
SECTION = re.compile(r"(?m)^## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})\s*$")
RELEASE_TAG = re.compile(r"^v(\d+\.\d+\.\d+)$")
SAVEPOINT_EXAMPLE = "ci-20261106-something"


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def version_of(root, name, pattern):
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        return None, "%s is missing" % name
    match = pattern.search(read(path))
    if match is None:
        return None, "%s declares no version" % name
    return match.group(1), None


def changelog_sections(root):
    """[(version, date, body)] in file order, newest first as the file is written."""
    path = os.path.join(root, "CHANGELOG.md")
    if not os.path.isfile(path):
        return None, "CHANGELOG.md is missing"
    text = read(path)
    matches = list(SECTION.finditer(text))
    if not matches:
        return None, "CHANGELOG.md has no `## [x.y.z] - YYYY-MM-DD` section"
    sections = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end]
        body_lines = [line for line in body.splitlines() if line.strip()]
        sections.append((match.group(1), match.group(2), body_lines))
    return sections, None


def check_tree(root):
    """Problems with the tree as it stands, independent of any tag."""
    problems = []
    versions = {}
    for name, pattern in VERSION_SOURCES.items():
        version, error = version_of(root, name, pattern)
        if error:
            problems.append(error)
        else:
            versions[name] = version
    distinct = sorted(set(versions.values()))
    if len(distinct) > 1:
        problems.append(
            "the version is declared as %s across %s"
            % (" and ".join(distinct), ", ".join(sorted(versions)))
        )
    current = distinct[0] if len(distinct) == 1 else (versions.get("pyproject.toml") or "")

    sections, error = changelog_sections(root)
    if error:
        problems.append(error)
        return problems, current, None

    newest_version, newest_date, newest_body = sections[0]
    if current and newest_version != current:
        if not any(version == current for version, _, _ in sections):
            # The shape of a hand edit that replaced a heading instead of adding above it: the new
            # section's prose is now attached to the version below.
            problems.append(
                "no changelog section exists for %s at all, and the newest is %s. If a heading was "
                "edited, check that it was added ABOVE the previous one rather than replacing it"
                % (current, newest_version)
            )
        else:
            problems.append(
                "the newest changelog section is %s but the version is %s; a release nobody can read "
                "is not a release" % (newest_version, current)
            )
    if not newest_body:
        problems.append(
            "the changelog section for %s has no body; a heading alone reads as a release and says "
            "nothing" % newest_version
        )
    return problems, current, (newest_version, newest_date, newest_body)


def check_tag(tag, current):
    """Problems with the tag itself."""
    match = RELEASE_TAG.match(tag)
    if match:
        if current and match.group(1) != current:
            return [
                "tag %s releases version %s but the tree is %s; bump the version (three files) or "
                "move the tag" % (tag, match.group(1), current)
            ]
        return []
    if tag.startswith("v"):
        return [
            "tag %s starts with `v`, which is the pattern `release.yml` publishes on, but it is not a "
            "`vX.Y.Z` release. A savepoint must not look like a release: name it "
            "`%s`" % (tag, SAVEPOINT_EXAMPLE)
        ]
    return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT, help="repository root (default: this checkout)")
    ap.add_argument("--tag", help="the tag being created, checked against the version")
    ap.add_argument("--quiet", action="store_true", help="say nothing on success")
    args = ap.parse_args(argv)

    problems, current, newest = check_tree(args.root)
    if args.tag:
        problems.extend(check_tag(args.tag, current))
    if problems:
        print("check_release: this is not releasable", file=sys.stderr)
        for problem in problems:
            print("  * " + problem, file=sys.stderr)
        return 1
    if not args.quiet:
        detail = " (%s, %s)" % (newest[0], newest[1]) if newest else ""
        where = "tag %s" % args.tag if args.tag else "no tag"
        print("check_release: version %s, changelog section%s, %s: OK" % (current, detail, where))
    return 0


if __name__ == "__main__":
    sys.exit(main())
