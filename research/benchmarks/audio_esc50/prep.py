"""ESC-50 and ESC-10 as Laya decision rows: one `choice` question per recording.

ESC-50 is the standard public benchmark for environmental sound classification: 2 000 five-second
clips, 50 classes, 40 per class, pre-arranged into 5 folds so that clips from one source recording
never straddle a fold. ESC-10 is its curated 10-class subset (400 clips), inside the same files.

Both become what Laya answers: a `choice` question whose options are the class names and whose state
carries the recording and nothing else that could leak the answer. That is the vision benchmark's
shape (`research/benchmarks/vision_cifar100/`), applied to audio.

    python3 research/benchmarks/audio_esc50/prep.py            # unzip if needed, write rows
    python3 research/benchmarks/audio_esc50/prep.py --check    # report counts only

Data lives in `$AUDIO_BENCH_DATA` (default `<repo>/.audio-bench`, gitignored): the upstream zip is
`data/esc50.zip`, unpacked to `data/ESC-50-master/`, and the rows are written to `esc50/`.
"""
import argparse
import csv
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATA_ROOT = os.environ.get("AUDIO_BENCH_DATA", os.path.join(REPO, ".audio-bench"))
ZIP_NAME = "esc50.zip"
ZIP_URL = "https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master"
UNPACKED = "ESC-50-master"
ALL_FOLDS = (1, 2, 3, 4, 5)

# What the state says about the recording: nothing that names or hints at the class. A decision
# engine that could read the answer off the state text would score the text, not the sound -- the
# mistake the vision benchmark's gate 1 exists to catch.
STATE_TEXT = "a recording of an environmental sound"
QUESTION_ID = "sound"
INSTRUCTIONS = "Which sound is this recording?"


def _data_paths():
    data = os.path.join(DATA_ROOT, "data")
    return {
        "zip": os.path.join(data, ZIP_NAME),
        "root": os.path.join(data, UNPACKED),
        "out": os.path.join(DATA_ROOT, "esc50"),
    }


def ensure_data():
    """Unpack the dataset, downloading it first if this is a fresh checkout."""
    paths = _data_paths()
    if os.path.isdir(os.path.join(paths["root"], "audio")):
        return paths
    os.makedirs(os.path.dirname(paths["zip"]), exist_ok=True)
    if not os.path.exists(paths["zip"]):
        import urllib.request
        print("downloading %s (~616 MB) ..." % ZIP_URL, flush=True)
        urllib.request.urlretrieve(ZIP_URL, paths["zip"])
    print("unpacking %s ..." % paths["zip"], flush=True)
    with zipfile.ZipFile(paths["zip"]) as archive:
        archive.extractall(os.path.dirname(paths["zip"]))
    return paths


def read_meta(paths):
    """The metadata CSV: filename, fold, target, category, esc10 flag."""
    meta_path = os.path.join(paths["root"], "meta", "esc50.csv")
    with open(meta_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["fold"] = int(row["fold"])
        row["target"] = int(row["target"])
        row["is_esc10"] = row["esc10"].strip().lower() == "true"
    return rows


def classes_of(rows):
    """Class names in target order -- the option order every row shares."""
    names = {}
    for row in rows:
        names.setdefault(row["target"], row["category"].replace("_", " "))
    return [names[i] for i in sorted(names)]


def rows_for(rows, classes, paths, folds, subset):
    """Laya rows for the given folds of one subset.

    The published protocol is 5-fold cross-validation: one fold is scored, the other four are
    trained on, and every clip is scored exactly once across the five runs.
    """
    labels = set(classes)
    wanted = set(folds)
    out = []
    for row in rows:
        if row["fold"] not in wanted:
            continue
        if subset == "esc10" and not row["is_esc10"]:
            continue
        label = row["category"].replace("_", " ")
        if label not in labels:                              # pragma: no cover - ESC-10 is a subset
            continue
        out.append({
            "state": {"audio_path": os.path.join(paths["root"], "audio", row["filename"]),
                      "text": STATE_TEXT},
            "questions": {QUESTION_ID: {"type": "choice", "instructions": INSTRUCTIONS,
                                        "criteria": classes}},
            "expected": {QUESTION_ID: label},
            "tags": ["fold%d" % row["fold"], subset, row["category"]],
        })
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report counts without writing rows")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="folds to write (default: all five, the published protocol)")
    parser.add_argument("--subsets", nargs="+", default=["esc50", "esc10"],
                        choices=["esc50", "esc10"])
    args = parser.parse_args(argv)

    paths = ensure_data()
    rows = read_meta(paths)
    classes50 = classes_of(rows)
    esc10_labels = {row["category"].replace("_", " ") for row in rows if row["is_esc10"]}
    classes10 = [name for name in classes50 if name in esc10_labels]

    print("ESC-50: %d clips, %d classes, folds %s"
          % (len(rows), len(classes50), sorted({r["fold"] for r in rows})))
    print("ESC-10: %d clips, %d classes (a subset of the same files)"
          % (sum(1 for r in rows if r["is_esc10"]), len(classes10)))
    if args.check:
        for fold in args.folds:
            held = [r for r in rows if r["fold"] == fold]
            print("  fold %d: %d held out, %d train" % (fold, len(held), len(rows) - len(held)))
        return 0

    os.makedirs(paths["out"], exist_ok=True)
    meta = {"classes50": classes50, "classes10": classes10, "folds": args.folds,
            "n_clips": len(rows), "state_text": STATE_TEXT, "instructions": INSTRUCTIONS,
            "question_id": QUESTION_ID, "source": os.path.join(paths["root"], "audio")}
    for subset in args.subsets:
        classes = classes50 if subset == "esc50" else classes10
        for fold in args.folds:
            # The held-out fold is always scored against the *other four* of the published
            # protocol, whatever `--folds` was asked to write: a single-fold run must not train
            # on nothing.
            train_folds = [f for f in ALL_FOLDS if f != fold]
            train = rows_for(rows, classes, paths, train_folds, subset)
            held = rows_for(rows, classes, paths, [fold], subset)
            write_jsonl(os.path.join(paths["out"], "%s_fold%d_train.jsonl" % (subset, fold)), train)
            write_jsonl(os.path.join(paths["out"], "%s_fold%d_eval.jsonl" % (subset, fold)), held)
            print("  %s fold %d: %d train, %d eval, %d options"
                  % (subset, fold, len(train), len(held), len(classes)))
    with open(os.path.join(paths["out"], "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    # Scored on one fold, trained on the other four: 1 600 / 400 clips (ESC-50) and 320 / 80
    # (ESC-10), which is the published cross-validation protocol.
    return 0


if __name__ == "__main__":
    sys.exit(main())
