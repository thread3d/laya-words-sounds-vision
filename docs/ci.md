# Continuous integration

Which lanes must be green, which report on a schedule, and what each one costs.

**The constraint that shapes everything here:** this repository is private, so every Actions minute
is billed against a monthly allowance. That allowance ran out once, and the failure did not look
like a budget problem — it looked like this, on every job of every workflow, with no step ever
running:

```
The job was not started because recent account payments have failed or your spending limit
needs to be increased. Please check the 'Billing & plans' section in your settings
```

Two things follow. Verification has to be *cheap* enough to run on every commit, and the expensive
breadth has to run somewhere other than the commit path.

## The tiers

| tier | fires on | lanes |
|---|---|---|
| **blocking** | every push to `main`, every pull request | `ci.yml` (the Python suites via `scripts/test_suites.py`, lint, byte-compile, packaging, the workflow-trigger audit), `docker.yml` (smoke, HTTP), `security.yml` (secret scan, dependency CVEs, CodeQL) |
| **blocking (release)** | a `v*` tag | `release.yml` — the same suite list, then build, `twine check`, PyPI, GitHub Release |
| **weekly report** | Monday 03:00 UTC (`evals.yml`), 04:00 UTC (`cpp`, `dotnet`, `go`, `java`, `rust`), 06:17 UTC (`security.yml`) | the five port lanes, the CVE re-scan, the eval baseline |
| **on demand** | `workflow_dispatch` | every lane above, plus `docs.yml`'s deploy |

`docs.yml` runs its strict build on pushes to `main` and on pull requests that touch docs, `laya/`,
or the site config; the deploy job publishes only from `main`.

## Why the port lanes are weekly

`laya-cpp`, `laya-dotnet`, `laya-go`, `laya-java` and `laya-rust` each re-record their goldens from
the Python package at the commit under test — that is the point of them, and it is also the most
expensive thing this repository runs. And every job in them that can fail is advisory
(`continue-on-error`): a port that goes red is a follow-up port, not a revert of the Python change,
so no verdict they produce can block a merge or a release.

Per-commit breadth plus advisory verdicts is the worst of both: the lanes were **33 of the 66 jobs**
a single push to `main` asked for, and none of those 33 could change any decision. They now report
weekly over the whole tree, and a maintainer can still run one against any branch:

```bash
gh workflow run "Java" --ref my-branch
gh run list --workflow "Java" --limit 3
```

A change to `laya/` that moves a golden therefore surfaces in the weekly report rather than on the
commit that caused it. That is the deliberate trade: the blocking lanes still catch a broken Python
package immediately (`ci.yml`), and the ports answer "has the port drifted?" on a schedule that
costs nothing on a normal day.

## What one commit costs

```bash
python scripts/check_workflow_triggers.py
```

The script parses every workflow, matrix-expands it the way Actions does (the product of the
dimensions, plus each `include:` entry that names a combination the matrix does not already
produce), and prints the cost of a push, a tag and the weekly crons. It fails if:

- a `push:` trigger has neither a `branches:` nor a `tags:` filter, so a tag push re-runs a lane;
- a non-release workflow fires on tags;
- a push-triggered lane does not supersede the previous run on its ref
  (`concurrency.group` keyed on `github.ref` and `github.event_name`, `cancel-in-progress: true`);
- a frozen port lane is back on the push or pull-request path;
- a push, a tag, or the weekly crons exceed their budget.

Budgets today: **33 jobs** for a push to `main`, **3** more if that commit is tagged, **41** for the
weekly crons. The script runs in the CI lint job, so a lane cannot quietly rejoin the push path;
raise a budget in the same change that earns it.

## When the allowance runs out

The bill is account-level, so no workflow edit lifts the block. Either raise the spending limit
(`Settings → Billing`) or stop the jobs entirely until it is sorted:

```bash
gh api -X PUT repos/thread3d/laya-words-sounds-vision/actions/permissions -F enabled=false
gh api repos/thread3d/laya-words-sounds-vision/actions/permissions    # {"enabled":false}
```

Pushes, tags and releases keep working while this is set; only workflow execution stops. The
practical cost of leaving it off is that nothing verifies the code, so it is a stopgap, not a
policy.

## Release tags

`release.yml` fires on any `v*` tag and requires `pyproject.toml`'s version to equal the tag before
it publishes. A savepoint that is not a release therefore must not use the `v*` shape — this
repository uses `ci-<date>-<topic>` for those (see `git tag -l 'ci-*'`), so a commit can be marked
without starting a publish.
