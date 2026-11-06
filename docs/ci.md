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
| **blocking (push)** | every push to `main` | `ci.yml` one cell per matrix — `lint` (ruff, `actionlint`, the workflow-trigger audit, the release check, byte-compile), `test` (3.11), `build`, `typescript` (node 22), `python-sdk` (ubuntu, 3.11), `feishu-benchmark`; `docker.yml` x86 smoke + HTTP; `security.yml` secret scan, unsafe-deserialization check, CodeQL (python); `docs.yml` strict build (deploy only where `vars.LAYA_PAGES` is set) |
| **breadth (pull request, weekly, dispatch)** | `pull_request`, the Monday crons, `workflow_dispatch` | the whole `ci.yml` matrix (4 Pythons, both Node versions, Windows, the SDK integration and TypeScript-SDK lanes, ONNX export, evals), both Docker runners, all four CodeQL languages, dependency CVEs |
| **blocking (release)** | a `v*` tag | `release.yml` — the same suite list, then build, `twine check`, PyPI, GitHub Release |
| **weekly report** | Monday 05:00 UTC (`ci.yml`), 05:30 (`docker.yml`), 03:00 (`evals.yml`), 04:00 (`cpp`, `dotnet`, `go`, `java`, `rust`), 06:17 (`security.yml`) | the full gate on a fresh tree, the five port lanes, the CVE re-scan, the eval baseline |
| **on demand** | `workflow_dispatch` | every lane above, plus `docs.yml`'s deploy |

A push therefore runs the fast gate (13 jobs); a pull request, a dispatch or the Monday cron runs the
full one (32–65). `docs.yml` runs its strict build on pushes to `main` and on pull requests that
touch docs, `laya/`, or the site config. Its deploy job publishes only from `main` **and only where
Pages is configured**, which is a repository variable rather than a condition on the repository name:

```bash
gh variable set LAYA_PAGES --body true     # Settings -> Pages -> Source: GitHub Actions
```

That switch replaced a list naming two repositories, neither of which was this one — so the deploy had
silently not run since the rename while the workflow still read as if it did. A fork that has not
enabled Pages leaves the variable unset and skips, which is what the list was reaching for.

Two consequences worth knowing:

- **A breadth job reports `skipped` on a push.** If branch protection "requires" one of them (say
  `tests (windows)`), require it on pull requests rather than on `main`, or a push will wait on a
  check that never runs.
- **A push is not the full matrix.** A commit that breaks Python 3.12 only is caught by the next
  pull request, dispatch, or Monday 05:00 run — not by the push that broke it. That is the trade the
  allowance forces; `ci.yml`'s fast gate still runs every suite on 3.11.

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

## Record once, fan out

A port lane's cells compare against goldens recorded from the Python package at the commit under
test. Recording is the expensive half — a checkpoint fetch plus a generator run — and its result
depends only on the commit and the pinned revision, so a matrix of N cells used to produce N
byte-identical copies of one answer, each paying for its own torch install and its own multi-gigabyte
checkpoint fetch.

Two lanes now have a `goldens` job that records once and publishes artifacts for the run:

| lane | the recorder | what the cells take |
|---|---|---|
| `cpp.yml` | `cpp-goldens` (the nine families) and `cpp-checkpoint-parts` (`<checkpoint>/tokenizer/` and `rl_agent_config.json` for english and multilingual — the only files a cell opens, not the weights) | both, and install nothing: no torch, no hub, no generator |
| `dotnet.yml` | `dotnet-goldens` (routing, round4, and the three checkpoints, recorded by one `regen_golden.py --checkpoint all`) | the artifact; the ONNX exports they still need come from the cache the recorder warms |
| `rust.yml` | `rust-goldens` (routing plus the three checkpoints — nine recording passes for four recordings before this, because every cell also recorded `routing` and the router cell recorded english and multilingual again) | the artifact; the ONNX exports come from the cache the recorder warms |

The five `build` cells of `cpp.yml` and the model-free `build` job of `dotnet.yml` run no Python
generator at all, so they no longer install torch or reach the hub. They assert the shape of what
arrived (`The recorded inputs arrived`), which turns a wrong artifact depth into an immediate
failure instead of a missing-checkpoint skip after a full compile — and in `dotnet.yml` a skip is
not a failure to `dotnet test`, which is exactly the hole the arrival check closes.

Two cells keep a generator, and only under a condition: the ones that record a *graph-backed*
family. `cpp.yml`'s `onnx` cells record `predict.json` from the graph they exported in that cell, so
the recording cannot be shared; the rule is that every generator call they make stays `--only`-scoped
to those families.

Retention is one day on purpose: an artifact here is an input to this run's cells, not an archive,
and artifact storage is billed the way minutes are.

`java.yml` and `go.yml` need no recorder, because neither ever paid twice for a recording: java's
three parity cells record **disjoint** `--only` sets (the tokenizer's six families, `predict`'s two,
`typed-decisions`' one), and go has a single recording job. That is a second legitimate shape for the
same rule, not an exception to it.

`tests/test_ci_record_once.py` is the guard, and it enforces the rule rather than a job layout: **one
write per fixture family per workflow**. It reads that out of the YAML — the families each job or
matrix cell writes, from `--only` and `--checkpoint`, with a cell that passes a shell variable
resolved through its own `checkpoints` value — and fails on a family recorded by two jobs, or twice
inside one matrix unless that matrix is a declared graph-backed one. On top of that, where a recorder
exists: a consumer that downloads an artifact without `needs: goldens`, a consumer that starts
recording the shared set again, a cache entry the recorder warms under a key no cell restores, or a
lane that half-migrates (consumers rewired, recorder missing).

Run against the trees before each migration, it names the waste it was written for: `rust.yml`'s
`routing` recorded four times with english and multilingual twice each, `dotnet.yml`'s english three
times, and `cpp.yml`'s eight families recorded by both `build` and `onnx`.

## What one commit costs

```bash
python scripts/check_workflow_triggers.py
```

The script parses every workflow, matrix-expands it the way Actions does (the product of the
dimensions, plus each `include:` entry that names a combination the matrix does not already
produce), and prints the cost of each scenario: a push, a tag on the same commit, a pull request, and
the weekly crons. It fails if:

- a `push:` trigger has neither a `branches:` nor a `tags:` filter, so a tag push re-runs a lane;
- a non-release workflow fires on tags;
- a push-triggered lane does not supersede the previous run on its ref
  (`concurrency.group` keyed on `github.ref` and `github.event_name`, `cancel-in-progress: true`);
- a frozen port lane is back on the push or pull-request path;
- a scenario exceeds its budget;
- it cannot model a `strategy.matrix` or a job `if:` — an unreadable expression that silently counts
  as one job is worse than no check, so it fails instead. The modelled forms are
  `${{ github.event_name == 'push' && fromJSON('[…]') || fromJSON('[…]') }}` (push branch first) and
  `if: github.event_name != 'push'`. A reversed pair is reported too, since that makes a push cost
  *more* than it did before.

Budgets today: **13 jobs** for a push to `main` (**3** more if that commit is tagged), **32** for a
pull request, **66** for the weekly crons. The script runs in the CI lint job, so a lane cannot
quietly rejoin the push path; raise a budget in the same change that earns it.

## Turning it back on

Actions has been off long enough that this tree has never run: every lane here was edited while it was
disabled, and every gate in this repository is therefore a *local* gate. The first push after
re-enabling is the first machine evidence, so it is worth knowing what working looks like.

| workflow | on the first push |
|---|---|
| `CI` | 6 jobs — `lint`, `tests (py3.11)`, `package`, `TypeScript (node22)`, `Python SDK (ubuntu-latest, py3.11)`, `Chinese benchmark audit`. The ONNX export, evals, Windows and both SDK breadth lanes report **skipped**: they are pull-request, dispatch and cron lanes |
| `Docker` | 2 jobs — x86 smoke and HTTP; the arm runner and the Spark build are skipped |
| `Security` | 3 jobs — secret scan, unsafe-deserialization, `CodeQL (python)` |
| `Docs` | the strict build; the deploy reports **skipped** unless `vars.LAYA_PAGES` is set |
| `C++`, `.NET`, `Go`, `Java`, `Rust` | absent — they are a weekly report |

Then prove the paths a push cannot reach:

```bash
gh workflow run "Rust"  --ref main   # a recorder lane end to end: the goldens job, the artifact, the cells
gh workflow run "Bench" --ref main   # the benchmark lane and its artifact
```

Both are cold on the first run: the .NET and Rust recorders restore no ONNX cache, so they export the
graphs themselves, which is the case their timeouts are sized for and much slower than the steady state.

Four shapes that are **not** code failures, in the order they are likely to appear:

- `The job was not started because recent account payments have failed or your spending limit needs to
  be increased` — the bill, again. No workflow edit changes it; [the allowance section](#when-the-allowance-runs-out)
  has the two ways out.
- `skipped` on a breadth job — correct. It is not a push lane.
- `cancelled` with no failing step — a newer push superseded it on the same ref. That is 4.0.1's
  concurrency change doing its job.
- a truncated Hugging Face download inside a recorder — it fails at the fetch and names the file. Re-run;
  if it repeats, `HF_TOKEN` is missing or the anonymous rate limit is being hit.

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

The two durable answers to the bill are [the self-hosted runner](self-hosted-runner.md) — the port
lanes read one repository variable, and a machine you already own spends no minutes — and making the
repository public, which grants free hosted minutes and is a visibility decision rather than a CI
one.

## Releasing

A release is three facts that have to agree:

| fact | where |
|---|---|
| one version | `pyproject.toml`, `laya.__version__` and `nix/package.nix` declare the same `x.y.z` |
| a readable section | `CHANGELOG.md`'s newest `## [x.y.z] - YYYY-MM-DD` is that version and has a body |
| a matching tag | `vX.Y.Z` names that version, because `release.yml` publishes on `v*` |

`scripts/check_release.py` is the single check for all three. CI runs the tree half on every push
(the `lint` job) and the tag half on a tag (the `package` job), and `release.yml` runs it before it
installs anything — a tag naming the wrong version fails in seconds rather than after the suite run
and a build.

```bash
# bump the three files and add the changelog section, then:
python scripts/check_release.py --tag v4.0.8
git tag -a v4.0.8 -m "4.0.8: ..." && git push origin main v4.0.8
```

The tree half exists because of a hand-edit that happened twice here: replacing a `## [x.y.z]`
heading instead of adding above it, which leaves the new prose attached to the release beneath it and
nothing noticing. The checker names that shape when it sees it.

**The workflows are linted too.** `lint` downloads `actionlint` at a pinned version, verifies its
SHA-256 before extracting it, and runs it over all fourteen workflows — a bad expression or a
shellcheck finding in a `run:` block is a failure like any other. Like the runner bootstrap, the pin
is a version plus the checksum published with that release.

**Savepoint tags are not releases.** A tag that marks a commit without publishing anything must not
begin with `v`, because `v*` is the pattern that starts the publish; this repository uses
`ci-<date>-<topic>` for those (`git tag -l 'ci-*'`). The checker refuses a tag like `v4.0` for exactly
that reason: it looks like a release to `release.yml` and is not one.

After the tag, `release.yml` runs the shared suite list, then `python -m build` and `twine check`,
publishes to PyPI with a trusted publisher (no token lives in the repository), and writes the GitHub
Release.
