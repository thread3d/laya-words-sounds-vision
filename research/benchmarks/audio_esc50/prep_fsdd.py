"""The Free Spoken Digit Dataset as Laya rows: 3 000 recordings, 10 classes, 6 speakers.

The fast tier. ESC-50 is the benchmark with published numbers to compare against, but it is 616 MB
and five-second clips, so a full engine arm on it costs tens of minutes of CPU. FSDD is 16 MB of
half-second 8 kHz recordings of the digits zero to nine, spoken by six people, and it asks the
question a decision engine is actually for: *which digit is this?* over a recording alone.

Its second job is to make the harness testable end to end in a couple of minutes -- prep, tower
pass, probe and an engine arm all fit inside one interactive session, which is what lets a bug like
"every eval row looks like a training row" be found by a human rather than by a benchmark run that
silently reports nothing held out.

    python3 research/benchmarks/audio_esc50/prep_fsdd.py

The split is **by speaker**, not by recording: the same voice must not appear on both sides, or the
number measures speaker recognition rather than digit recognition. Two speakers are held out
(the published "6 speakers, 4 train / 2 test" style split), which leaves 1 000 recordings on each
side and six ways to choose the pair.
"""
import argparse
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATA_ROOT = os.environ.get("AUDIO_BENCH_DATA", os.path.join(REPO, ".audio-bench"))
ZIP_URL = "https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/refs/heads/master"
UNPACKED = "free-spoken-digit-dataset-master"

STATE_TEXT = "a recording of someone saying a digit"
QUESTION_ID = "digit"
INSTRUCTIONS = "Which digit is being said?"


def paths():
    data = os.path.join(DATA_ROOT, "data")
    return {"zip": os.path.join(data, "fsdd.zip"),
            "root": os.path.join(data, UNPACKED),
            "out": os.path.join(DATA_ROOT, "fsdd")}


def ensure_data():
    out = paths()
    if os.path.isdir(os.path.join(out["root"], "recordings")):
        return out
    os.makedirs(os.path.dirname(out["zip"]), exist_ok=True)
    if not os.path.exists(out["zip"]):
        import urllib.request
        print("downloading %s (~16 MB) ..." % ZIP_URL, flush=True)
        urllib.request.urlretrieve(ZIP_URL, out["zip"])
    with zipfile.ZipFile(out["zip"]) as archive:
        archive.extractall(os.path.dirname(out["zip"]))
    return out


def recordings(out):
    """`(path, digit, speaker)` for every recording, parsed from `7_jackson_43.wav`."""
    folder = os.path.join(out["root"], "recordings")
    rows = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".wav"):
            continue
        stem = name[:-4].split("_")
        if len(stem) < 3:
            continue
        rows.append({"path": os.path.join(folder, name), "digit": int(stem[0]),
                     "speaker": stem[1], "index": stem[2]})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speakers", nargs="+", default=None,
                        help="held-out speaker pair (default: jackson and theo)")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    out = ensure_data()
    rows = recordings(out)
    speakers = sorted({row["speaker"] for row in rows})
    held = args.speakers or ["jackson", "theo"]
    unknown = [name for name in held if name not in speakers]
    if unknown:
        print("unknown speaker(s) %s; this dataset has %s" % (unknown, speakers))
        return 2

    classes = [str(digit) for digit in sorted({row["digit"] for row in rows})]
    print("FSDD: %d recordings, %d classes, speakers %s" % (len(rows), len(classes), speakers))
    print("held out: %s" % ", ".join(held))
    if args.check:
        return 0

    os.makedirs(out["out"], exist_ok=True)
    splits = {"train": [], "eval": []}
    for row in rows:
        split = "eval" if row["speaker"] in held else "train"
        splits[split].append({
            "state": {"audio_path": row["path"], "text": STATE_TEXT},
            "questions": {QUESTION_ID: {"type": "choice", "instructions": INSTRUCTIONS,
                                        "criteria": classes}},
            "expected": {QUESTION_ID: str(row["digit"])},
            "tags": [split, "speaker:" + row["speaker"]],
        })
    for split, split_rows in splits.items():
        path = os.path.join(out["out"], "fsdd_%s.jsonl" % split)
        with open(path, "w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("  %s: %d rows -> %s" % (split, len(split_rows), path))
    meta = {"classes": classes, "speakers": speakers, "held_out": held,
            "state_text": STATE_TEXT, "instructions": INSTRUCTIONS,
            "question_id": QUESTION_ID, "source": os.path.join(out["root"], "recordings")}
    with open(os.path.join(out["out"], "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
