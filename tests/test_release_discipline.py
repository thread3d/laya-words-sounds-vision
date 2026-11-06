"""A release is a version, a changelog section and a matching tag -- and nothing else looks like one.

`scripts/check_release.py` is the rule; this file is what proves the rule bites. Every case below is
a small synthetic tree, because the interesting cases are the broken ones and a repository cannot be
in all of them at once:

* the truncated heading that this file exists for -- while editing CHANGELOG.md by hand the heading
  `## [4.0.2] - ...` was replaced instead of preceded, twice, leaving the section's prose attached to
  the version above it;
* a version bumped in one of the three files but not the others;
* a changelog whose newest section is an older version than the tree;
* a savepoint tag that begins with `v` and would start the publish when its author meant "mark this
  commit";
* a release tag that names a different version than the tree.

Runs the shipped script against a temporary tree (`--root`), so this tests what CI runs.

Run: `python tests/test_release_discipline.py`
"""
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "check_release.py")

PASS, FAIL = [], []


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def load_module():
    spec = importlib.util.spec_from_file_location("check_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GOOD_SECTIONS = """# Changelog

Intro prose.

## [4.0.2] - 2026-11-06

The newest section, with something under it.

## [4.0.1] - 2026-11-06

An older release.

"""


def make_tree(tmp, version="4.0.2", init_version=None, nix_version=None, changelog=GOOD_SECTIONS):
    with open(os.path.join(tmp, "pyproject.toml"), "w", encoding="utf-8") as handle:
        handle.write('[project]\nname = "laya"\nversion = "%s"\n' % version)
    os.makedirs(os.path.join(tmp, "laya"), exist_ok=True)
    with open(os.path.join(tmp, "laya", "__init__.py"), "w", encoding="utf-8") as handle:
        handle.write('__version__ = "%s"\n' % (init_version or version))
    os.makedirs(os.path.join(tmp, "nix"), exist_ok=True)
    with open(os.path.join(tmp, "nix", "package.nix"), "w", encoding="utf-8") as handle:
        handle.write('{\n    version = "%s";\n}\n' % (nix_version or version))
    if changelog is not None:
        with open(os.path.join(tmp, "CHANGELOG.md"), "w", encoding="utf-8") as handle:
            handle.write(changelog)


def main():
    module = load_module()

    def run(root, *args):
        return module.main(["--root", root, *args])

    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp)
        check_true("release/a coherent tree passes", run(tmp) == 0)

    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, init_version="4.0.1")
        check_true("release/a version out of step between files fails", run(tmp) == 1)

    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, nix_version="3.9.0")
        check_true("release/nix drifting from pyproject fails", run(tmp) == 1)

    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, version="4.0.3")
        check_true(
            "release/a bump with no changelog section fails",
            run(tmp) == 1,
            "the newest section is 4.0.2 and the version is 4.0.3",
        )

    truncated = GOOD_SECTIONS.replace("## [4.0.2] - 2026-11-06\n", "")
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, changelog=truncated)
        check_true(
            "release/the truncated-heading case fails",
            run(tmp) == 1,
            "prose left attached to the section above is the bug this exists for",
        )

    empty = GOOD_SECTIONS.replace(
        "The newest section, with something under it.\n", ""
    )
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, changelog=empty)
        check_true("release/a heading with no body fails", run(tmp) == 1)

    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp)
        check_true("release/a matching release tag passes", run(tmp, "--tag", "v4.0.2") == 0)
        check_true("release/a mismatched release tag fails", run(tmp, "--tag", "v4.0.1") == 1)
        check_true("release/a malformed v-tag fails", run(tmp, "--tag", "v4.0") == 1)
        check_true("release/a savepoint tag passes", run(tmp, "--tag", "ci-20261106-topic") == 0)
        check_true("release/no tag passes on a coherent tree", run(tmp) == 0)

    with tempfile.TemporaryDirectory() as tmp:
        make_tree(tmp, changelog=None)
        check_true("release/a tree with no CHANGELOG fails", run(tmp) == 1)

    # The shipped script's exit codes, as CI invokes it (no --root): this is the real checkout.
    real = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True, cwd=ROOT)
    check_true("release/the checkout is releasable", real.returncode == 0, real.stderr[:300])
    version = re.search(r"version (\S+?),", real.stdout or "")
    check_true("release/the checkout reports its version", version is not None, real.stdout[:120])

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for failure in FAIL:
        print("  FAIL", failure)
    if not FAIL:
        print("a release is a version, a changelog section and a matching tag -- and nothing else")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
