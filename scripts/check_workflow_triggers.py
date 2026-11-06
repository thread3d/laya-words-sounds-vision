#!/usr/bin/env python3
"""Refuse a workflow trigger that makes one commit pay for two runs.

GitHub Actions reads a `push` trigger with no `branches:`/`tags:` filter as "any push", and a tag
push is a push. Five lanes here (`cpp`, `dotnet`, `go`, `java`, `rust`) declared exactly that, so
tagging a commit re-ran those lanes against a tree that had just been built: 36 jobs on the tag
push of `v4.0.0`, none of which could report anything the branch run had not already reported.
A private repository bills those minutes, and the bill is what stopped every workflow in this
repository -- each job failing with "The job was not started because recent account payments have
failed or your spending limit needs to be increased", before a single step ran.

What this enforces:

* `push:` is either branch-scoped (`branches:` naming `main`) or tag-scoped (`tags:`). Never bare.
* Only `release*.yml` fires on tags. Publishing is the only thing a tag means here; a commit
  reaches `main` through a branch push, and that push is the run that verifies it.
* A commit pushed to `main` and then tagged costs at most `--max-main-jobs` jobs for the branch
  push and `--max-tag-jobs` for the tag push. The budgets are the ratchet: a workflow that adds a
  matrix leg has to say so here.

Job counts are matrix-expanded exactly as Actions expands them (the product of the dimensions,
plus every `include:` entry that names a combination the matrix does not already produce), so the
number printed is the number of runners the commit asks for, not the number of `jobs:` keys.

Run: `python scripts/check_workflow_triggers.py`
"""
import argparse
import fnmatch
import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW_DIR = os.path.join(ROOT, ".github", "workflows")

# A release-shaped tag: `release.yml` fires on it because its pattern is `v*`. `laya-java-v*` and
# `laya-ts-v*` deliberately do not match it, so the SDK releases are not counted here.
TAG = "v9.9.9"

MAIN_PUSH_JOB_BUDGET = 70
TAG_PUSH_JOB_BUDGET = 5

ABSENT = object()


def load(path):
    import yaml  # imported here so `--help` works without the dependency installed

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def events(doc):
    """The `on:` block as {event: config}, where config is None for a filterless event."""
    raw = doc.get("on", doc.get(True))  # a YAML 1.1 loader turns the key `on` into True
    if raw is None:
        return {}
    if isinstance(raw, str):
        return {raw: None}
    if isinstance(raw, list):
        return {name: None for name in raw}
    return dict(raw)


def push_filters(doc):
    """None when there is no push trigger; otherwise the filter dict (possibly empty)."""
    config = events(doc).get("push", ABSENT)
    if config is ABSENT:
        return None
    if not isinstance(config, dict):
        return {}
    filters = {}
    for key in ("branches", "tags", "branches-ignore", "tags-ignore"):
        value = config.get(key)
        if value is not None:
            filters[key] = [value] if isinstance(value, str) else list(value)
    return filters


def fires_on_main(doc):
    filters = push_filters(doc)
    if filters is None:
        return False
    if "branches-ignore" in filters:
        return not any(fnmatch.fnmatchcase("main", p) for p in filters["branches-ignore"])
    if "branches" in filters:
        return any(fnmatch.fnmatchcase("main", p) for p in filters["branches"])
    return not ("tags" in filters or "tags-ignore" in filters)


def fires_on_tag(doc, tag=TAG):
    filters = push_filters(doc)
    if filters is None:
        return False
    if "tags-ignore" in filters:
        return not any(fnmatch.fnmatchcase(tag, p) for p in filters["tags-ignore"])
    if "tags" in filters:
        return any(fnmatch.fnmatchcase(tag, p) for p in filters["tags"])
    return not ("branches" in filters or "branches-ignore" in filters)


def expand_matrix(job):
    """How many runners one job entry asks for once `strategy.matrix` is expanded."""
    matrix = (job.get("strategy") or {}).get("matrix")
    if not matrix:
        return 1
    if isinstance(matrix, str):  # fromJSON(...) or similar: not countable statically
        return 1
    include = matrix.get("include") or []
    dims = {
        key: value
        for key, value in matrix.items()
        if key not in ("include", "exclude") and isinstance(value, list)
    }
    if not dims:
        return max(len(include), 1)
    combos = [dict(zip(dims, values)) for values in itertools.product(*dims.values())]
    for entry in include:
        if not isinstance(entry, dict):
            continue
        shared = [key for key in entry if key in dims]
        # An `include:` that only adds variables to a combination that already exists is not an
        # extra job; one that overrides a dimension value into a new combination is.
        if shared and not any(all(c.get(key) == entry[key] for key in shared) for c in combos):
            combos.append(entry)
    return len(combos)


def job_count(doc):
    return sum(expand_matrix(job) for job in (doc.get("jobs") or {}).values())


def collect():
    rows = []
    for name in sorted(os.listdir(WORKFLOW_DIR)):
        if not name.endswith((".yml", ".yaml")):
            continue
        path = os.path.join(WORKFLOW_DIR, name)
        doc = load(path)
        rows.append(
            {
                "file": name,
                "doc": doc,
                "main": fires_on_main(doc),
                "tag": fires_on_tag(doc),
                "jobs": job_count(doc),
            }
        )
    return rows


def violations(rows, max_main, max_tag):
    problems = []
    for row in rows:
        filters = push_filters(row["doc"])
        if filters is None:
            continue
        where = f".github/workflows/{row['file']}"
        release = row["file"].startswith("release")
        if not filters:
            problems.append(
                f"{where}: `push:` has no `branches:` or `tags:` filter, so a tag push re-runs this "
                f"lane ({row['jobs']} jobs) for a tree the branch push already built. Give it "
                f"`branches: [main]`, or `tags:` if it belongs to a release."
            )
        if "tags" in filters and not release:
            problems.append(
                f"{where}: fires on tag pushes but is not a release workflow; release*.yml is the "
                f"only place that belongs."
            )
        # Only branch pushes supersede each other. A release run must NOT: two tags pointing at
        # different commits are two things to publish, so cancelling one would lose a release.
        concurrency = row["doc"].get("concurrency") or {}
        if row["main"] and (
            not isinstance(concurrency, dict)
            or "group" not in concurrency
            or "cancel-in-progress" not in concurrency
        ):
            problems.append(
                f"{where}: runs on a push without a `concurrency:` group and `cancel-in-progress:`, "
                f"so a burst of pushes queues every commit's full matrix instead of superseding it."
            )
    main_total = sum(row["jobs"] for row in rows if row["main"])
    tag_total = sum(row["jobs"] for row in rows if row["tag"])
    if main_total > max_main:
        problems.append(f"a push to main asks for {main_total} jobs, budget is {max_main}")
    if tag_total > max_tag:
        problems.append(f"tagging a commit asks for {tag_total} more jobs, budget is {max_tag}")
    return problems, main_total, tag_total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-main-jobs", type=int, default=MAIN_PUSH_JOB_BUDGET)
    ap.add_argument("--max-tag-jobs", type=int, default=TAG_PUSH_JOB_BUDGET)
    args = ap.parse_args()

    rows = collect()
    width = max(len(row["file"]) for row in rows)
    print(f"{'workflow'.ljust(width)}  push:main  push:tag  jobs")
    for row in rows:
        print(
            f"{row['file'].ljust(width)}  {'yes' if row['main'] else '-':^9}  "
            f"{'yes' if row['tag'] else '-':^8}  {row['jobs']:>4}"
        )
    problems, main_total, tag_total = violations(rows, args.max_main_jobs, args.max_tag_jobs)
    print(f"\none commit, pushed to main then tagged: {main_total} + {tag_total} jobs")

    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  * {problem}")
        return 1
    print(f"OK: triggers within budget (main {main_total}<={args.max_main_jobs}, tag {tag_total}<={args.max_tag_jobs})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
