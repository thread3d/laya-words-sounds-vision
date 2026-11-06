"""Every port lane records each golden once, and every cell consumes that recording.

The rule this file enforces is about *recordings*, not about job shapes: a fixture family must be
written once per workflow. Two shapes satisfy it.

* **A recorder and its cells.** `cpp.yml`, `dotnet.yml` and `rust.yml` have a `goldens` job that
  records everything and publishes it for the run; their cells download the artifact and install no
  torch. Those lanes are held to the harder rules below -- a consumer must declare
  `needs: goldens`, must not run a generator (except a graph-backed cell, under a narrow and
  explicit exception), and a cache the recorder warms must be keyed the way the cells restore it.
* **Disjoint cells.** `java.yml`'s three parity cells record disjoint `--only` sets and `go.yml` has
  a single recording job, so neither has anything to deduplicate and neither needs a recorder. That
  is a legitimate way to satisfy the rule, and it is checked the same way.

What went wrong without it, measured: `cpp.yml` recorded the same eight families in eight cells,
`dotnet.yml` made four recordings eight times (its layout cells recorded byte-identical goldens),
and `rust.yml` made four recordings nine times because every cell also recorded `routing`. On a
private repository each of those copies is billed.

Text and YAML only: nothing is imported from `laya`, so this runs wherever the suite list runs, and a
missing PyYAML is a SKIP line rather than a false failure (both the CI lint job and the `test` job
install it).

Run: `python tests/test_ci_record_once.py`
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")

PASS, FAIL, SKIPPED = [], [], []

# The generators a lane records with, and the flag that names the families each writes. A call with
# neither flag writes the whole set.
FAMILY_FLAGS = {
    "gen_fixtures.py": "--only",
    "gen_go_fixtures.py": "--only",
    "dump_golden.py": "--checkpoint",
    "regen_golden.py": "--checkpoint",
}
ALL_FAMILIES = "<all>"
# Names a `--checkpoint` may resolve to when the call names a shell variable instead.
CHECKPOINT_WORDS = {"english", "multilingual", "typed-decisions", "routing", "repr"}

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
        FAIL.append("%s%s" % (name, ": " + str(detail) if detail else ""))


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


def recording_units(job):
    """[(generator, family)] for every family this job writes, matrix cells included.

    A step that also runs the same generator under `--check` is a drift report, not a recording:
    `java.yml`'s build job re-runs the generator only to show what a failed `--check` found.
    """
    run = run_text(job)
    units = []
    for generator, flag in FAMILY_FLAGS.items():
        if generator not in run:
            continue
        if re.search(r"%s[^\n]*--check" % re.escape(generator), run):
            continue  # drift check, not a recording
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        cells = [cell for cell in (matrix.get("include") or []) if isinstance(cell, dict)]
        cell_texts = [" ".join(str(value) for value in cell.values()) for cell in cells] or [""]
        for cell_text in cell_texts:
            raw = re.findall(r"%s\s+(\S+)" % re.escape(flag), run + " " + cell_text)
            families = {token.strip("\"'();,.") for token in raw if "$" not in token}
            families.discard("")
            if any("$" in token for token in raw):
                # `args+=(--checkpoint "$c")` names no family: the cell's own `checkpoints` values
                # do. Intersecting with the known vocabulary keeps `--workspace` or a timeout from
                # posing as one, and the literal families beside it (rust's `routing`) are kept.
                words = {name.strip("\"'();,.") for name in re.split(r"\s+", cell_text)}
                families |= words & CHECKPOINT_WORDS
            families.discard("")
            if families:
                units.extend((generator, family) for family in sorted(families))
            else:
                units.append((generator, ALL_FAMILIES))
    return units


def check_recorded_once(name, doc):
    """The rule every lane is held to: one write per family per workflow."""
    jobs = doc.get("jobs") or {}
    first_owner = {}
    duplicates = []
    for job_name, job in jobs.items():
        per_job = {}
        for generator, family in recording_units(job):
            per_job[(generator, family)] = per_job.get((generator, family), 0) + 1
            owner = first_owner.setdefault((generator, family), job_name)
            if owner != job_name:
                duplicates.append(
                    "%s/%s is recorded by both `%s` and `%s`" % (generator, family, owner, job_name)
                )
        for (generator, family), count in per_job.items():
            if count > 1 and job_name not in GRAPH_BACKED_CONSUMERS.get(name, ()):
                duplicates.append(
                    "%s/%s is recorded %d times inside `%s`" % (generator, family, count, job_name)
                )
    check(
        "%s/records each family once" % name,
        not duplicates,
        "; ".join(sorted(set(duplicates))[:4]),
    )


def audit_lane(name, doc):
    """The extra rules a lane with a `goldens` recorder has to satisfy."""
    jobs = doc.get("jobs") or {}
    recorder = jobs["goldens"]
    where = name
    check(
        "%s/the recorder runs a generator" % where,
        any(gen in run_text(recorder) for gen in FAMILY_FLAGS),
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
        if graph_backed:
            # Allowed to record, but only the families it traces itself: every generator call has to
            # be scoped, or the cell is recording the shared set again.
            check(
                "%s/%s records only what it traces" % (where, job_name),
                all(
                    "--only" in line or not any(gen in line for gen in FAMILY_FLAGS)
                    for line in runs.splitlines()
                ),
                "a graph-backed cell must scope every generator call with --only",
            )
        else:
            recording = [gen for gen in FAMILY_FLAGS if gen in runs]
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
    # makes the recorder's warm cache reach the cells, and it also catches two cells that quietly
    # forked an entry.
    groups = {}
    for job in jobs.values():
        for path_set, key in zip(cache_paths(job), cache_keys(job)):
            groups.setdefault(path_set, set()).add(key)
    for path_set, keys in groups.items():
        check(
            "%s/caches %s under one key" % (where, "/".join(sorted(path_set))[:60]),
            len(keys) == 1,
            "found %d different keys: %s" % (len(keys), sorted(k for k in keys if k)[:2]),
        )


def main():
    try:
        import yaml  # noqa: F401
    except ImportError:
        skip("record-once/audit", "PyYAML is not installed, so the workflow YAML cannot be read")
        return _report()

    files = sorted(f for f in os.listdir(WORKFLOWS) if f.endswith((".yml", ".yaml")))
    check("record-once/workflows found", len(files) >= 10, files[:3])

    recorders = []
    for name in files:
        doc = load(name)
        jobs = doc.get("jobs") or {}
        check_recorded_once(name, doc)
        if "goldens" in jobs:
            recorders.append(name)
            audit_lane(name, doc)
        elif any("goldens" in needs(job) for job in jobs.values()):
            # The half-migration this file exists to catch: consumers were rewired, the recorder
            # never landed.
            check(
                "record-once/%s has a recorder" % name,
                False,
                "a job declares `needs: goldens` but the workflow has no `goldens` job",
            )

    check("record-once/at least three lanes use a recorder", len(recorders) >= 3, recorders)
    return _report()


def _report():
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for failure in FAIL:
        print("  FAIL", failure)
    for skipped in SKIPPED:
        print("  SKIP", skipped)
    if not FAIL:
        print("every port lane records each golden once and every cell consumes that recording")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
