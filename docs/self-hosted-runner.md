# The self-hosted runner

**Status: wired, not installed.** The port lanes ask for a runner through a repository variable; with
the variable unset they run on GitHub's hosted runners exactly as before. Nothing here changes until
someone sets it.

## What it buys

This repository is private, so every hosted minute is billed ([CI](ci.md)) — and the port lanes are
the expensive part: they fetch multi-gigabyte checkpoints and run model inference. Two things follow.

- **A self-hosted runner spends no minutes.** The hardware is already paid for.
- **The machine that measured the benchmarks is the machine the lanes want.** The numbers in
  [the changelog](https://github.com/thread3d/laya-words-sounds-vision/blob/main/CHANGELOG.md) were
  taken on three gfx1030s with all three checkpoints staged locally; a lane running there measures the
  same hardware instead of a hosted runner's shared vCPUs, and it does not fetch from the hub at all.

## The contract

The five port workflows — `cpp`, `dotnet`, `go`, `java`, `rust` — name their runner through one
expression:

```yaml
runs-on: ${{ fromJSON(vars.LAYA_RUNNER || '["ubuntu-latest"]') }}
```

`LAYA_RUNNER` is a repository variable holding a JSON array. Unset, the fallback is a hosted runner.
Set, every job in those five workflows moves:

```bash
gh variable set LAYA_RUNNER --body '["self-hosted", "laya-gpu"]'
gh variable delete LAYA_RUNNER      # back to hosted
```

**The blocking gate does not read it.** `ci.yml`, `security.yml`, `docs.yml`, `docker.yml` and the
`release*.yml` lanes stay on hosted runners on purpose: a machine that is off, rebooting, or busy must
not be able to stop a push from being verified or a release from publishing. The lanes that move are
the ones that are already a weekly report.

`scripts/check_workflow_triggers.py` enforces the contract, because the failure mode is quiet and
expensive: a job asking for a label no runner has waits in the queue and is eventually refused, which
is a failure report that has nothing to do with the code.

- a `runs-on` may name a hosted label, a `matrix.` dimension, or the selector above;
- naming `self-hosted` directly fails, including inside a matrix dimension or an `include:` entry;
- a selector whose fallback is not a hosted runner fails.

## What the machine needs

| requirement | why |
|---|---|
| Linux x86_64 | `scripts/setup_github_runner.sh` pins the linux-x64 runner archive |
| the three checkpoints | `models/` or `~/laya_models`, per [Local setup](https://github.com/thread3d/laya-words-sounds-vision/blob/main/LOCAL_SETUP.md) |
| ~30 GB free | checkpoints, the fused and split ONNX exports, and the cargo target directory |
| ROCm for the GPU paths | the benchmark arms in the changelog need it; the CPU-only test paths do not |
| outbound HTTPS | to reach GitHub, and the hub on a cache miss |

If a GPU run of a sound checkpoint segfaults inside MIOpen's convolution, that is the known ROCm
defect described in the 4.0.0 notes, not the runner: see
[Upgrading](https://github.com/thread3d/laya-words-sounds-vision/blob/main/docs/upgrading.md).

## Install

```bash
scripts/setup_github_runner.sh --dry-run          # what it would do, touching nothing
scripts/setup_github_runner.sh --token <token>    # token: Settings -> Actions -> Runners -> New runner
```

It downloads the pinned archive (`v2.338.2`, linux-x64), **verifies its SHA-256**, configures it with
the `laya-gpu` label, and then prints the two commands it will not run for you
(`sudo ./svc.sh install && sudo ./svc.sh start`).
`--verify <tarball>` re-checks any copy against the pin; `--tarball` installs from a local file.

Then watch a lane pick it up:

```bash
gh variable set LAYA_RUNNER --body '["self-hosted", "laya-gpu"]'
gh api repos/thread3d/laya-words-sounds-vision/actions/runners \
  --jq '.runners[] | "\(.name) \(.status) \(.labels|map(.name)|join(","))"'
gh workflow run "Rust" --ref main
```

`--disableupdate` is deliberate: the runner version is pinned the same way the toolchains are, and an
upgrade should be a reviewed change that moves the checksum beside it.

## Security, plainly

A runner executes repository code with direct access to the machine it sits on. That is fine for a
private repository with no outside contributors and unacceptable for a public one with fork pull
requests — a fork PR can reach the machine. Keep the repository private while a runner is attached,
and stop the service (`sudo ./svc.sh stop`) rather than leaving it idle on a laptop.

The realistic risk is the one hosted runners exist to remove: whoever can push here can run code on
that box.

## The alternative

A **public** repository gets free hosted minutes for standard runners, which removes the billing
constraint with no hardware at all. That is a visibility decision rather than a CI one, and it is
written down in [CI](ci.md) next to the budget numbers.
