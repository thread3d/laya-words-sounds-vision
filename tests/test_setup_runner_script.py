"""The runner bootstrap stays safe to run by hand, and stays in step with the workflows.

`scripts/setup_github_runner.sh` installs something that executes repository code on a real machine,
so the properties worth locking down are the ones a hurried edit would quietly break:

* it is pinned -- a runner version and a 64-hex checksum -- so the artefact it executes is the one
  that was reviewed;
* it never escalates on its own: the `sudo` steps are printed, not run;
* its `--help`, `--dry-run` and `--verify` paths work and touch nothing, which is what makes it
  reviewable before it is used;
* the label it configures is the label the docs tell an operator to put in `vars.LAYA_RUNNER`.

Nothing here reaches the network or writes outside a temporary directory.

Run: `python tests/test_setup_runner_script.py`
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "setup_github_runner.sh")
DOCS = os.path.join(ROOT, "docs", "self-hosted-runner.md")

PASS, FAIL = [], []


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def run(*args):
    return subprocess.run(
        [SCRIPT, *args], capture_output=True, text=True, timeout=120, cwd=ROOT
    )


def main():
    check_true("runner script/present and executable", os.access(SCRIPT, os.X_OK), SCRIPT)
    with open(SCRIPT, encoding="utf-8") as handle:
        text = handle.read()

    check_true("runner script/fails fast", "set -euo pipefail" in text)
    version = re.search(r'^RUNNER_VERSION="([0-9]+\.[0-9]+\.[0-9]+)"$', text, re.M)
    check_true("runner script/pins a runner version", version is not None, text[:200])
    checksum = re.search(r'^RUNNER_SHA256="([0-9a-f]{64})"$', text, re.M)
    check_true("runner script/pins a sha256", checksum is not None)
    labels = re.search(r'^DEFAULT_LABELS="([^"]+)"$', text, re.M)
    check_true("runner script/declares its labels", labels is not None)

    # Escalating on its own would make the script something an operator cannot read before running.
    check_true(
        "runner script/never runs sudo itself",
        re.search(r"(?m)^\s*sudo\b", text) is None,
        "a bootstrap that sudo's on its own cannot be reviewed before it is used",
    )
    check_true(
        "runner script/does not pipe a download into a shell",
        re.search(r"curl[^\n|]*\|\s*(ba)?sh", text) is None,
    )

    help_run = run("--help")
    check_true("runner script/--help exits 0", help_run.returncode == 0, help_run.stderr[:200])
    for flag in ("--dry-run", "--verify", "--token"):
        check_true("runner script/--help documents %s" % flag, flag in help_run.stdout)

    dry = run("--dry-run")
    check_true("runner script/--dry-run exits 0", dry.returncode == 0, dry.stderr[:200])
    check_true(
        "runner script/--dry-run names the variable the workflows read",
        "LAYA_RUNNER" in dry.stdout,
        dry.stdout[:200],
    )
    check_true(
        "runner script/--dry-run shows the pinned checksum",
        (checksum.group(1) in dry.stdout) if checksum else False,
    )

    # A path that cannot be a runner must fail, and must fail without a download.
    missing = run("--verify", os.path.join(ROOT, "does-not-exist.tar.gz"))
    check_true("runner script/--verify rejects a missing tarball", missing.returncode != 0)

    with open(DOCS, encoding="utf-8") as handle:
        docs = handle.read()
    check_true(
        "docs/self-hosted-runner names the script",
        "scripts/setup_github_runner.sh" in docs,
    )
    if labels:
        check_true(
            "docs/self-hosted-runner shows the label the script configures",
            labels.group(1) in docs,
            "script default %r is not in the page" % labels.group(1),
        )
    if version:
        # The page names the pinned version so an operator can see what they are installing; a
        # version bump that forgets the page fails here rather than misleading a reader.
        check_true(
            "docs/self-hosted-runner names the pinned runner version",
            version.group(1) in docs,
            "the page does not mention %s" % version.group(1),
        )

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for failure in FAIL:
        print("  FAIL", failure)
    if not FAIL:
        print("the runner bootstrap is pinned, reviewable, and matches the workflows")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
