"""The control that separates "cannot learn" from "did not train": a trivial two-class tone task.

When the first ESC-10 engine arm landed at chance, there were two possible explanations and they
lead to opposite conclusions: either the audio block does not carry learnable signal into the head
(a wiring bug, and the whole path is broken), or the small-data recipe simply did not train the head
far enough (an optimization finding, and the path is fine). This script answers that in two minutes.

It writes 64 one-second recordings -- 440 Hz against 880 Hz, the same state text for both -- and
trains the real path (frozen wav2vec2-base, projector, block, ModernBERT-large head) on them. A
working path reaches 100 %; a broken one stays at 50 %.

`--classes N` raises the cardinality. At 100 options the option head crosses
`summary_for_options_over`, so the block carries its summary row -- the high-cardinality path the
vision program once shipped broken (the plan reserved a row the projector never emitted). That case
is worth a control of its own: only ten of the 100 labels occur in the training rows, so the task is
"find the ten tones among 100 options", and a working summary row reaches well above the 1 % chance.

    python3 research/benchmarks/audio_esc50/synth.py                      # writes rows only
    python3 research/benchmarks/audio_esc50/synth.py --run                # two classes, trains
    python3 research/benchmarks/audio_esc50/synth.py --classes 100 --run  # summary-row path

Measured on this machine:

| task | setup | result |
|---|---|---|
| two tones | 64 rows, 4 epochs, 32 updates | loss 0.023 -> 0.000; held-out 0.5000 -> **1.0000** |
| 100 options, ten tones present | 120 rows, 6 epochs, 84 updates | held-out 0.1000 -> **0.3000** (chance 0.01) |
"""
import argparse
import json
import math
import os
import subprocess
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATA_ROOT = os.environ.get("AUDIO_BENCH_DATA", os.path.join(REPO, ".audio-bench"))
OUT = os.path.join(DATA_ROOT, "synth")


def _wav(path, freq, seconds=1.0, rate=16000):
    frames = bytearray()
    for i in range(int(seconds * rate)):
        frames += int(0.5 * math.sin(2 * math.pi * freq * i / rate) * 32767).to_bytes(
            2, "little", signed=True)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))


def build(n_train=64, n_eval=32, classes=2):
    """Write the rows. `classes > 10` exercises the summary-row path (option_tokens > 96)."""
    os.makedirs(OUT, exist_ok=True)
    labels = ["low", "high"] if classes == 2 else ["tone%02d" % i for i in range(classes)]
    # Only the first ten labels occur in training: with 100 options the task is finding ten tones
    # among a hundred, which is what makes chance 0.01 rather than 1/10.
    present = labels if classes == 2 else labels[:10]
    for name, n in (("train", n_train), ("eval", n_eval)):
        rows = []
        for i in range(n):
            label = present[i % len(present)]
            path = os.path.join(OUT, "%s_%s_%d.wav" % (name, label, i))
            _wav(path, 440.0 if classes == 2 and label == "low" else
                 440.0 + 110.0 * present.index(label))
            rows.append({"state": {"audio_path": path, "text": "a tone"},
                         "questions": {"sound": {"type": "choice",
                                                 "instructions": "Which tone is this?",
                                                 "criteria": labels}},
                         "expected": {"sound": label}})
        with open(os.path.join(OUT, "%s.jsonl" % name), "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print("wrote %s (%d train / %d eval, %d options)" % (OUT, n_train, n_eval, len(labels)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="train and score it")
    parser.add_argument("--base", default=os.path.join(REPO, "models", "laya"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--classes", type=int, default=2, choices=[2, 20, 50, 100])
    args = parser.parse_args(argv)
    build(classes=args.classes)
    if not args.run:
        return 0
    return subprocess.call([
        sys.executable, "-m", "laya.train_cli",
        "--data", os.path.join(OUT, "train.jsonl"),
        "--base", args.base,
        "--out", os.path.join(DATA_ROOT, "checkpoints", "synth"),
        "--audio", '{"n_audio_tokens": 32}',
        "--option-layout", "sequential", "--freeze-encoder",
        "--epochs", str(args.epochs), "--grad-accum", "1",
        # The two-class task needs a bigger step; the 100-option one diverges at 1e-3, so the
        # default rate is kept there (see the results page's learning-rate note).
        "--head-lr", "1e-3" if args.classes == 2 else "1e-4",
        "--micro-batch", "8", "--device", "cpu",
        "--max-len", "512", "--head-max-len", "400" if args.classes > 20 else "192",
        "--eval", os.path.join(OUT, "eval.jsonl"),
    ])


if __name__ == "__main__":
    sys.exit(main())
