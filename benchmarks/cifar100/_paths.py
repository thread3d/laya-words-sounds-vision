"""Where the CIFAR-100 benchmark's inputs and outputs live.

The harness began life outside the repository, with one developer's checkout written into every
script (`cd /home/threaded/projects/Laya`, `--base /home/threaded/projects/Laya/laya/models/laya`).
When the repository moved, those paths still resolved -- to *different code* -- so a reproduction
run silently measured a tree that did not contain the vision branch. That is the failure this
module exists to prevent: a path that cannot be wrong.

Everything is derived from this file, plus two environment variables:

    LAYA_BENCH_DATA   root holding the parquet, the PNGs, the JSONL splits, the cached SigLIP2
                      weights and the trained checkpoints. Default: ``<repo>/.vision-cifar100``,
                      falling back to the sibling scratch ``.benchprobe/`` when that is where the
                      data already lives (the layout this benchmark was first measured in).
    LAYA_MODELS       the base checkpoints ``laya-train`` starts from. Default: ``<repo>/models``.

Neither directory is committed: ``.vision-cifar100/`` and ``models/`` are both in ``.gitignore``.

Nothing here imports torch, so a data-prep script can import it on a machine with no GPU.
"""
import os

# .../<repo>/benchmarks/cifar100/_paths.py -> <repo>
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Candidate roots for the benchmark's data, in priority order.
_BENCH_CANDIDATES = (
    os.path.join(REPO, ".vision-cifar100"),
    os.path.join(os.path.dirname(REPO), ".benchprobe"),
)

#: The benchmark's data and reports live directly under this root, so the default is
#: ``<repo>/.vision-cifar100`` and ``LAYA_BENCH_DATA`` overrides it.
BENCH = _BENCH_CANDIDATES[0]

#: JSONL splits, cached embeddings, controls and eval reports.
CIFAR = os.path.join(BENCH, "cifar100")

#: The HF parquet the splits are cut from.
PARQUET = os.path.join(BENCH, "data")

#: A local SigLIP2 checkpoint, for the controls (the tower the vision head projects from).
SIGLIP = os.path.join(BENCH, "siglip2")

#: Base checkpoints for `laya-train --base`.
MODELS = os.path.join(REPO, "models")

#: The checkpoint every run in the published tables starts from.
BASE = os.path.join(MODELS, "laya")


def _resolve_bench():
    """Pick the data root: the environment variable, then whichever candidate is populated."""
    override = os.environ.get("LAYA_BENCH_DATA")
    if override:
        return os.path.abspath(override)
    for candidate in _BENCH_CANDIDATES:
        if os.path.isdir(os.path.join(candidate, "cifar100")):
            return candidate
    return _BENCH_CANDIDATES[0]


BENCH = _resolve_bench()
CIFAR = os.path.join(BENCH, "cifar100")
PARQUET = os.path.join(BENCH, "data")
SIGLIP = os.path.join(BENCH, "siglip2")

if os.environ.get("LAYA_MODELS"):
    MODELS = os.path.abspath(os.environ["LAYA_MODELS"])
BASE = os.path.join(MODELS, "laya")


def add_repo_to_path():
    """Make `import laya` work from a source checkout, without installing the package."""
    import sys

    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    return REPO


def add_harness_to_path():
    """Make `import tower_patch` work: it sits beside this module."""
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    return here


def describe():
    """One line per resolved path, for the head of a run log."""
    return (
        "harness: repo=%s" % REPO,
        "harness: bench=%s" % BENCH,
        "harness: base checkpoint=%s" % BASE,
    )
