#!/usr/bin/env python3
"""Refuse a workflow trigger that makes one commit pay for a run it does not need.

GitHub Actions reads a `push` trigger with no `branches:`/`tags:` filter as "any push", and a tag
push is a push. Five lanes here declared exactly that, so tagging a commit re-ran them against a
tree that had just been built: 36 jobs on the tag push of `v4.0.0`, none of which could report
anything the branch run had not already reported. A private repository bills those minutes, and the
bill is what stopped every workflow in this repository -- each job failing with "The job was not
started because recent account payments have failed or your spending limit needs to be increased",
before a single step ran.

So the question this file answers is not "is the YAML valid" (actionlint does that) but "what does
one event cost", and it fails when the answer is bigger than the strategy allows.

What this enforces:

* `push:` is either branch-scoped (`branches:` naming `main`) or tag-scoped (`tags:`). Never bare.
* Only `release*.yml` fires on tags. Publishing is the only thing a tag means here; a commit
  reaches `main` through a branch push, and that push is the run that verifies it.
* A push-triggered lane supersedes the previous run on the same ref instead of queueing behind it:
  `concurrency.group` keyed on `github.ref` (plus `github.event_name` where a cron shares the lane)
  and `cancel-in-progress: true`. Grouping by `github.sha`, which is what these lanes used to do,
  gives every commit its own group, so five quick pushes bill five full matrices.
* The five port lanes are frozen off the push and pull-request path: they report weekly and on
  demand. Every job in them is advisory, so no verdict they produce can block a pull request or a
  release, while each matrix cell re-records its goldens from the Python package. They were 33 of
  the 66 jobs a push to `main` asked for.
* Each scenario -- a push to `main`, a tag on the same commit, a pull request, the weekly crons --
  stays inside its own budget, printed on every run.
* A job names a hosted runner, a matrix dimension, or the opt-in selector
  `fromJSON(vars.LAYA_RUNNER || '["ubuntu-latest"]')` -- never a hardcoded `self-hosted`. A lane with
  no runner queues until it times out and then fails, which is a failure report nobody asked for;
  docs/self-hosted-runner.md has the contract.

## How breadth is expressed, and how it is counted

A lane runs its full matrix on a pull request and on a schedule and one cell on a push, written as

    python: ${{ github.event_name == 'push' && fromJSON('["3.11"]') || fromJSON('["3.10", ...]') }}

and a whole job that belongs off the push path carries `if: github.event_name != 'push'`. This file
models both forms per event, and *fails* on a `strategy.matrix` or job `if:` it cannot model rather
than counting it as one job -- an unreadable expression that silently undercounts is worse than no
check at all.

Job counts are matrix-expanded exactly as Actions expands them (the product of the dimensions, plus
every `include:` entry that names a combination the matrix does not already produce).

Run: `python scripts/check_workflow_triggers.py`
"""
import argparse
import fnmatch
import itertools
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW_DIR = os.path.join(ROOT, ".github", "workflows")

# The four scenarios a workflow can be asked about.
PUSH = "push"  # a branch push to main
TAG = "tag"  # a v* tag on the same commit
PR = "pull_request"
WEEKLY = "schedule"
SCENARIOS = (PUSH, TAG, PR, WEEKLY)

# A release-shaped tag: `release.yml` fires on it because its pattern is `v*`. `laya-java-v*` and
# `laya-ts-v*` deliberately do not match it, so the SDK releases are not counted here.
TAG_NAME = "v9.9.9"

# Runners a job may name directly. Anything else has to arrive through the opt-in selector, so that
# an unset variable means "hosted" and a machine being down cannot park a lane forever.
HOSTED_RUNNERS = {"ubuntu-latest", "ubuntu-24.04-arm", "windows-latest", "macos-latest"}
RUNNER_SELECTOR = "vars.LAYA_RUNNER"

BUDGETS = {
    PUSH: 14,
    TAG: 5,
    PR: 36,
    WEEKLY: 70,
}

# A push run in these lanes does something that must not stop halfway. `docs.yml`'s `deploy` job
# publishes the Pages site, and a half-cancelled deploy is a broken site rather than a duplicated
# verdict, so that lane keeps `cancel-in-progress` off for pushes and serialises instead.
KEEP_RUNNING_ON_PUSH = {"docs.yml"}

# The parity lanes for the ports. Each is advisory end to end or nearly so (a port that goes red is a
# follow-up port, not a revert), and each re-records its goldens from the Python package in every
# matrix cell, which is the most expensive thing this repository runs. They report weekly and on
# demand; docs/ci.md says what that means for a maintainer.
FROZEN_PORTS = {"cpp.yml", "dotnet.yml", "go.yml", "java.yml", "rust.yml"}

ABSENT = object()
# `${{ github.event_name == 'push' && fromJSON('[...]') || fromJSON('[...]') }}`, push branch first.
_PUSH_JSON = re.compile(r"github\.event_name\s*==\s*'push'")
_FROM_JSON = re.compile(r"fromJSON\('(\[[^']*\])'\)")


class Uncountable(Exception):
    """A matrix or job condition this file cannot model. Reported, never silently dropped."""


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


def fires_on_tag(doc, tag=TAG_NAME):
    filters = push_filters(doc)
    if filters is None:
        return False
    if "tags-ignore" in filters:
        return not any(fnmatch.fnmatchcase(tag, p) for p in filters["tags-ignore"])
    if "tags" in filters:
        return any(fnmatch.fnmatchcase(tag, p) for p in filters["tags"])
    return not ("branches" in filters or "branches-ignore" in filters)


def fires_on(doc, scenario):
    names = events(doc)
    if scenario == PUSH:
        return fires_on_main(doc)
    if scenario == TAG:
        return fires_on_tag(doc)
    return scenario in names


def job_condition(job):
    raw = job.get("if")
    if raw is None:
        return ""
    return raw if isinstance(raw, str) else json.dumps(raw)


def job_runs(job, scenario):
    """Whether one job runs for a scenario. Anything unrecognised runs everywhere, except an `if:`
    that names `github.event_name` in a shape this file does not model: that is reported."""
    condition = job_condition(job)
    if not condition:
        return True
    if "github.event_name" in condition:
        if re.search(r"github\.event_name\s*!=\s*'push'", condition):
            return scenario in (PR, WEEKLY)
        if re.search(r"github\.event_name\s*==\s*'push'", condition):
            return scenario in (PUSH, TAG)
        # `docs.yml`'s deploy: "not a pull request, and only on a repository that has Pages".
        if re.search(r"github\.event_name\s*!=\s*'pull_request'", condition):
            return scenario in (PUSH, TAG, WEEKLY)
        raise Uncountable(f"job `if:` names github.event_name in a form this check cannot model: {condition!r}")
    if "startsWith(github.ref" in condition and "refs/tags/" in condition:
        return scenario == TAG
    return True


def dimension_values(value, scenario):
    """The values a matrix dimension takes for a scenario: a literal list, or the modelled form."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        arrays = _FROM_JSON.findall(value)
        if len(arrays) == 2 and _PUSH_JSON.search(value):
            push_values, other_values = json.loads(arrays[0]), json.loads(arrays[1])
            if not push_values or not other_values:
                raise Uncountable(f"a branch of this matrix is empty: {value!r}")
            # The event test exists to make a push cheaper. If the push branch is the larger one the
            # test is inverted, which costs more than it did before and would otherwise look fine.
            if len(push_values) > len(other_values):
                raise Uncountable(
                    f"the push branch of this matrix is larger ({len(push_values)}) than the "
                    f"non-push branch ({len(other_values)}); the event test is probably inverted: "
                    f"{value!r}"
                )
            return push_values if scenario in (PUSH, TAG) else other_values
    raise Uncountable(f"matrix dimension not modelable statically: {value!r}")


def expand_matrix(job, scenario):
    """How many runners one job entry asks for once `strategy.matrix` is expanded."""
    matrix = (job.get("strategy") or {}).get("matrix")
    if not matrix:
        return 1
    include = matrix.get("include") or []
    dims = {
        key: dimension_values(value, scenario)
        for key, value in matrix.items()
        if key not in ("include", "exclude")
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


def job_count(doc, scenario):
    total = 0
    for name, job in (doc.get("jobs") or {}).items():
        try:
            if job_runs(job, scenario):
                total += expand_matrix(job, scenario)
        except Uncountable as exc:
            raise Uncountable(f"job `{name}`: {exc}") from None
    return total


def collect():
    rows = []
    for name in sorted(os.listdir(WORKFLOW_DIR)):
        if not name.endswith((".yml", ".yaml")):
            continue
        doc = load(os.path.join(WORKFLOW_DIR, name))
        names = events(doc)
        row = {
            "file": name,
            "doc": doc,
            "fires": {scenario: fires_on(doc, scenario) for scenario in SCENARIOS},
            "pr": PR in names,
            "jobs": {},
        }
        for scenario in SCENARIOS:
            row["jobs"][scenario] = job_count(doc, scenario) if row["fires"][scenario] else 0
        rows.append(row)
    return rows


def totals(rows):
    return {scenario: sum(row["jobs"][scenario] for row in rows) for scenario in SCENARIOS}


def violations(rows, budgets):
    problems = []
    for row in rows:
        where = f".github/workflows/{row['file']}"
        if row["file"] in FROZEN_PORTS and (row["fires"][PUSH] or row["pr"]):
            on = "+".join(name for name, hit in (("push", row["fires"][PUSH]), ("pull_request", row["pr"])) if hit)
            problems.append(
                f"{where}: a frozen port lane is back on the {on} path ({row['jobs'][PUSH]} jobs on "
                f"a push). Its verdicts are advisory, so they cannot block anything; it belongs on "
                f"`schedule` + `workflow_dispatch` only (see docs/ci.md)."
            )
        filters = push_filters(row["doc"])
        if filters is None:
            continue
        release = row["file"].startswith("release")
        if not filters:
            problems.append(
                f"{where}: `push:` has no `branches:` or `tags:` filter, so a tag push re-runs this "
                f"lane ({row['jobs'][PUSH]} jobs) for a tree the branch push already built. Give it "
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
        group = concurrency.get("group", "") if isinstance(concurrency, dict) else ""
        group = group if isinstance(group, str) else ""
        cancel = concurrency.get("cancel-in-progress") if isinstance(concurrency, dict) else None
        if row["fires"][PUSH]:
            if "github.ref" not in group or "github.sha" in group:
                problems.append(
                    f"{where}: the `concurrency:` group is not keyed on `github.ref`, so each commit "
                    f"gets its own group and a burst of pushes queues every commit's "
                    f"{row['jobs'][PUSH]} jobs instead of superseding the previous one."
                )
            if row["file"] not in KEEP_RUNNING_ON_PUSH and cancel is not True:
                problems.append(
                    f"{where}: `cancel-in-progress` is not `true` for pushes, so the previous run on "
                    f"the same ref keeps billing after a newer commit arrives."
                )
            if row["fires"][WEEKLY] and "github.event_name" not in group:
                problems.append(
                    f"{where}: runs on both a schedule and a push but its group omits "
                    f"`github.event_name`, so the cron run and the push run collide and one cancels "
                    f"the other."
                )
    problems.extend(runner_problems(rows))
    for scenario, budget in budgets.items():
        total = sum(row["jobs"][scenario] for row in rows)
        if total > budget:
            problems.append(f"{scenario}: {total} jobs, budget is {budget}")
    return problems


def runner_label_problem(where, value):
    """None when a `runs-on` is acceptable, else why it is not."""
    self_hosted = (
        f"{where}: names `self-hosted` directly. Reach it through `{RUNNER_SELECTOR}` instead, so "
        f"an unset variable still means a hosted runner (docs/self-hosted-runner.md)."
    )
    if isinstance(value, list):
        if any(str(item).lower() == "self-hosted" for item in value):
            return self_hosted
        return None
    text = str(value)
    if "self-hosted" in text.lower() and RUNNER_SELECTOR not in text:
        return self_hosted
    if RUNNER_SELECTOR in text:
        if not any(runner in text for runner in HOSTED_RUNNERS):
            return (
                f"{where}: the `{RUNNER_SELECTOR}` selector has no hosted fallback, so the job "
                f"would ask for an empty runner set when the variable is unset."
            )
        return None
    if text.startswith("${{"):
        # An expression, such as the event-conditional matrix used for the Docker smoke cells. Any
        # literal label inside it has to be a hosted runner; a `matrix.` reference is checked where
        # the matrix is declared.
        labels = set(re.findall(r'"([A-Za-z0-9._-]+)"', text))
        unknown = sorted(labels - HOSTED_RUNNERS)
        if unknown:
            return f"{where}: names {unknown} inside a `runs-on` expression; only hosted labels belong there."
        return None
    if text in HOSTED_RUNNERS:
        return None
    return f"{where}: unknown runner {text!r}; name a hosted runner or the `{RUNNER_SELECTOR}` selector."


def runner_problems(rows):
    problems = []
    for row in rows:
        for job_name, job in (row["doc"].get("jobs") or {}).items():
            where = f".github/workflows/{row['file']}:{job_name}"
            problem = runner_label_problem(where, job.get("runs-on"))
            if problem:
                problems.append(problem)
            # A matrix can carry the label out of sight: a `runner:` or `os:` dimension, or an
            # `include:` entry that sets one.
            matrix = (job.get("strategy") or {}).get("matrix") or {}
            for key, value in matrix.items():
                entries = value if key in ("include", "exclude") else [{key: value}]
                for entry in entries or []:
                    if not isinstance(entry, dict):
                        continue
                    for dim, dim_value in entry.items():
                        if not (dim.lower().endswith("runner") or dim == "os"):
                            continue
                        for candidate in dim_value if isinstance(dim_value, list) else [dim_value]:
                            problem = runner_label_problem(f"{where}/{dim}", candidate)
                            if problem:
                                problems.append(problem)
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for scenario in SCENARIOS:
        ap.add_argument(f"--max-{scenario}-jobs", type=int, default=BUDGETS[scenario])
    args = ap.parse_args()
    budgets = {scenario: getattr(args, f"max_{scenario}_jobs") for scenario in SCENARIOS}

    try:
        rows = collect()
    except Uncountable as exc:
        print(f"FAIL: {exc}")
        return 1

    width = max(len(row["file"]) for row in rows)
    print(f"{'workflow'.ljust(width)}  " + "  ".join(f"{s[:9]:>9}" for s in SCENARIOS))
    for row in rows:
        cells = "  ".join(
            f"{(str(row['jobs'][s]) if row['fires'][s] else '-'):>9}" for s in SCENARIOS
        )
        print(f"{row['file'].ljust(width)}  {cells}")
    totals_by_scenario = totals(rows)
    print("\nper event: " + ", ".join(f"{scenario} {totals_by_scenario[scenario]}" for scenario in SCENARIOS))
    print(
        "one commit, pushed to main then tagged: "
        f"{totals_by_scenario[PUSH]} + {totals_by_scenario[TAG]} jobs"
    )

    problems = violations(rows, budgets)
    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  * {problem}")
        return 1
    print("OK: " + ", ".join(f"{scenario} {totals_by_scenario[scenario]}<={budgets[scenario]}" for scenario in SCENARIOS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
