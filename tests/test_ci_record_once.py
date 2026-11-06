"""A port lane records its goldens once, and every cell consumes that recording.

The pattern, and why it exists: a parity cell used to fetch the checkpoints from the hub, install
torch, and run the generator itself. The recorded output depends only on the commit and the pinned
revision, so a matrix of N cells produced N byte-identical copies of one answer -- and on a private
repository every copy is billed. `docs/ci.md` ("Record once, fan out") states the rule; this file
is what stops a lane from quietly drifting back, or from half-migrating: a job that consumes an
artifact without `needs: goldens`, or a cell that starts recording again, fails here.

Text and YAML only. Nothing is imported from `laya`, so this runs wherever the suite list runs, and
a missing PyYAML is a SKIP line rather than a false failure (the CI lint job installs it, and so does
the `test` job).

Run: `python tests/test_ci_record_once.py`
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")

PASS, FAIL, SKIPPED = [], [], []

# The generator each lane records with. A consumer that still names one is recording again.
GENERATORS = ("gen_fixtures.py", "gen_go_fixtures.py", "regen_golden.py", "dump_golden.py")
# Lanes that still record per cell. Reported as SKIP, not enforced -- `docs/ci.md` says which lanes
# are next, and a lane that gains a `goldens` job is held to the full rule below automatically.
PER_CELL_LANES = ("java.yml", "rust.yml", "go.yml")

# Cells that must record something themselves, because the family is recorded from a graph the cell
# exports in that cell -- it cannot be shared, and the cell has to carry the exporter with it. The
# exception is narrow on purpose: the generator call must stay `--only`-scoped to those families.
GRAPH_BACKED_CONSUMERS = {
    "cpp.yml": {"onnx"},   # predict.json / predict-english.json, traced from that cell's graph
}


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def skip(name, why):
    SKIPPED.append("%s: %s" % (name, why))


def load(name):
    import yaml

    with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def steps(job):
    return job.get("steps") or []


def run_text(job):
    return "\n".join(str(step.get("run") or "") for step in steps(job))


def uses(step, action):
    return str(step.get("uses") or "").startswith(action)


def artifact_names(doc, kind):
    found = []
    for job in (doc.get("jobs") or {}).values():
        for step in steps(job):
            if uses(step, "actions/%s-artifact" % kind):
                name = (step.get("with") or {}).get("name")
                if name:
                    found.append(name)
    return found


def needs(job):
    raw = job.get("needs")
    if raw is None:
        return []
    return [raw] if isinstance(raw, str) else list(raw)


def cache_keys(job):
    return [
        (step.get("with") or {}).get("key")
        for step in steps(job)
        if uses(step, "actions/cache")
    ]


def cache_paths(job):
    out = []
    for step in steps(job):
        if uses(step, "actions/cache"):
            path = (step.get("with") or {}).get("path") or ""
            out.append(frozenset(line.strip() for line in str(path).splitlines() if line.strip()))
    return out


def audit_lane(name, doc):
    """Every rule a migrated lane has to satisfy."""
    jobs = doc.get("jobs") or {}
    recorder = jobs.get("goldens")
    if recorder is None:
        return False
    where = name
    check("%s/one recorder job" % where, True)
    check(
        "%s/the recorder runs a generator" % where,
        any(gen in run_text(recorder) for gen in GENERATORS),
        "no generator in the `goldens` job",
    )
    published = artifact_names(doc, "upload")
    check("%s/the recorder publishes an artifact" % where, bool(published), "no upload-artifact")

    # Billing hygiene: an artifact that exists only for this run's cells is not an archive.
    for step in steps(recorder):
        if uses(step, "actions/upload-artifact"):
            with_ = step.get("with") or {}
            check(
                "%s/%s is short-lived" % (where, with_.get("name")),
                str(with_.get("retention-days", "")) not in ("", "None"),
                "an intra-run artifact must set retention-days",
            )

    downloaded = artifact_names(doc, "download")
    for artifact in downloaded:
        check(
            "%s/%s is published by this workflow" % (where, artifact),
            artifact in published,
            "downloaded but never uploaded (typo?)",
        )

    for job_name, job in jobs.items():
        if job_name == "goldens":
            continue
        graph_backed = job_name in GRAPH_BACKED_CONSUMERS.get(name, ())
        if artifact_names({"jobs": {job_name: job}}, "download"):
            check(
                "%s/%s waits for the recorder" % (where, job_name),
                "goldens" in needs(job),
                "downloads an artifact without `needs: goldens`, so it can start before it exists",
            )
        runs = run_text(job)
        recording = [gen for gen in GENERATORS if gen in runs]
        if graph_backed:
            # Allowed to record, but only the families it traces itself: every generator call has to
            # be scoped, or the cell is recording the shared set again.
            check(
                "%s/%s records only what it traces" % (where, job_name),
                all(
                    "--only" in line or not any(gen in line for gen in GENERATORS)
                    for line in runs.splitlines()
                ),
                "a graph-backed cell must scope every generator call with --only",
            )
        else:
            check(
                "%s/%s does not record its own goldens" % (where, job_name),
                not recording,
                "still runs %s" % ", ".join(recording),
            )
            check(
                "%s/%s does not install torch" % (where, job_name),
                "torch==" not in runs,
                "a consumer that records nothing does not need torch",
            )

    # A cache entry is only restored by a job that keys it the same way. Group the caches by the
    # directories they hold and require one key per group across the whole workflow: that is what
    # makes the recorder's warm cache reach the cells (dotnet's ONNX exports), and it is also what
    # catches two cells that quietly forked an entry.
    groups = {}
    for job_name, job in jobs.items():
        for path_set, key in zip(cache_paths(job), cache_keys(job)):
            groups.setdefault(path_set, set()).add(key)
    for path_set, keys in groups.items():
        check(
            "%s/caches %s under one key" % (where, "/".join(sorted(path_set))[:60]),
            len(keys) == 1,
            "found %d different keys: %s" % (len(keys), sorted(k for k in keys if k)[:2]),
        )
    return True


def main():
    try:
        import yaml  # noqa: F401
    except ImportError:
        skip("record-once/audit", "PyYAML is not installed, so the workflow YAML cannot be read")
        return _report()

    files = sorted(f for f in os.listdir(WORKFLOWS) if f.endswith((".yml", ".yaml")))
    check("record-once/workflows found", len(files) >= 10, files[:3])

    migrated = []
    for name in files:
        doc = load(name)
        jobs = doc.get("jobs") or {}
        has_needs = any("goldens" in needs(job) for job in jobs.values())
        if "goldens" in jobs:
            migrated.append(name)
            audit_lane(name, doc)
        elif has_needs:
            # The half-migration this file exists to catch: consumers were rewired, the recorder
            # never landed.
            check("record-once/%s has a recorder" % name, False,
                  "a job declares `needs: goldens` but the workflow has no `goldens` job")

    check("record-once/at least two lanes migrated", len(migrated) >= 2, migrated)
    for name in PER_CELL_LANES:
        if name in migrated:
            continue
        skip("record-once/%s" % name, "still records per cell (docs/ci.md names it as next)")

    return _report()


def _report():
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for failure in FAIL:
        print("  FAIL", failure)
    for skipped in SKIPPED:
        print("  SKIP", skipped)
    if not FAIL:
        print("every port lane records once and every cell consumes it")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
