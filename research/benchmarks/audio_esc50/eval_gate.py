"""Score one trained checkpoint: accuracy, both media gates, and the latency of a decision.

The engine arm's number is only half the result. The vision program learned the hard way that an
accuracy table can be true and meaningless -- a head that never looks at the input scores well and
means nothing -- so this runs the same two gates `laya-evals` runs, on the same rows, and reports
them beside the accuracy:

  * **gate 1** (`audio_ablation`): every row answered twice, once with the recording removed.
    The accuracy drop is the evidence that the answer depends on the sound at all.
  * **gate 2b** (`option_order_flip_rate`): every choice question answered twice, once with the
    options permuted. A distribution that moves when the caller reorders a list is not one a caller
    can act on.

    python3 research/benchmarks/audio_esc50/eval_gate.py --checkpoint .audio-bench/checkpoints/esc10-sc_fold1 \
        --rows esc10_fold1_eval.jsonl --limit 8            # smoke test
    python3 research/benchmarks/audio_esc50/eval_gate.py --checkpoint ... --rows esc10_fold1_eval.jsonl

Writes `results_<checkpoint>.json` beside the rows (default) or at `--out`.
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

DATA_ROOT = os.environ.get("AUDIO_BENCH_DATA", os.path.join(REPO, ".audio-bench"))
ROWS = os.path.join(DATA_ROOT, "esc50")


class _Timed:
    """A runner that records what each decision cost, and forwards everything else."""

    def __init__(self, agent):
        self.agent = agent
        self.latencies = []

    def predict(self, state, questions, model=None):
        started = time.time()
        try:
            return self.agent.predict(state, questions)
        finally:
            self.latencies.append(time.time() - started)

    def __getattr__(self, name):
        return getattr(self.agent, name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="a directory `laya.load` reads")
    parser.add_argument("--rows", required=True,
                        help="eval JSONL, relative to $AUDIO_BENCH_DATA/esc50 unless absolute")
    parser.add_argument("--limit", type=int, default=None, help="score at most N rows (a smoke test)")
    parser.add_argument("--skip-flip", action="store_true", help="skip gate 2b")
    parser.add_argument("--min-gain", type=float, default=None, help="ask gate 1 for a verdict")
    parser.add_argument("--max-flip-rate", type=float, default=None, help="ask gate 2b for a verdict")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="rows per forward pass in both gate passes (default 1); a CPU cannot "
                             "afford one pass per row at 50 options")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    import laya
    from laya.evals import Dataset
    from laya.evals_audio import audio_ablation, option_order_flip_rate

    path = args.rows if os.path.isabs(args.rows) else os.path.join(ROWS, args.rows)
    dataset = Dataset.from_jsonl(path)
    if args.limit:
        dataset = Dataset(dataset.examples[:args.limit])

    started = time.time()
    agent = laya.load(args.checkpoint, device="cpu")
    load_seconds = time.time() - started
    runner = _Timed(agent)

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "rows": os.path.abspath(path),
        "n_rows": len(dataset),
        "audio": getattr(agent.audio, "raw", None),
        "option_layout": agent.cfg.get("option_layout"),
        "load_seconds": round(load_seconds, 1),
        "batch_size": args.batch_size,
    }

    started = time.time()
    gate1 = audio_ablation(runner, dataset, min_gain=args.min_gain,
                           batch_size=args.batch_size)
    gate1_seconds = time.time() - started
    report["gate1"] = gate1

    gate2 = None
    if not args.skip_flip:
        started = time.time()
        gate2 = option_order_flip_rate(runner, dataset, max_flip_rate=args.max_flip_rate,
                                      batch_size=args.batch_size)
        report["gate2b"] = gate2
        report["gate2b_seconds"] = round(time.time() - started, 1)

    latencies = runner.latencies
    report["latency"] = {
        "decisions": len(latencies),
        "mean_ms": round(1000 * statistics.fmean(latencies), 1),
        "median_ms": round(1000 * statistics.median(latencies), 1),
        "p95_ms": round(1000 * sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
        "gate1_seconds": round(gate1_seconds, 1),
    }

    headline = {
        "accuracy": gate1["per_tag"]["all"]["with_audio"],
        "accuracy_without_audio": gate1["per_tag"]["all"]["without_audio"],
        "gate1_gain": gate1["gain"],
        "gate2b_flip_rate": None if gate2 is None else gate2["flip_rate"],
        "passed_gate1": gate1["passed"],
        "passed_gate2b": None if gate2 is None else gate2["passed"],
    }
    report["headline"] = headline

    out = args.out or os.path.join(
        ROWS, "results_%s.json" % os.path.basename(os.path.abspath(args.checkpoint)))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("rows %d  layout %s  load %.1fs"
          % (report["n_rows"], report["option_layout"], load_seconds), flush=True)
    print("accuracy %.4f  without-audio %.4f  gate1 gain %s  gate2b flips %s"
          % (headline["accuracy"] or float("nan"),
             headline["accuracy_without_audio"] or float("nan"),
             headline["gate1_gain"], headline["gate2b_flip_rate"]), flush=True)
    print("latency mean %(mean_ms).1f ms, median %(median_ms).1f ms, p95 %(p95_ms).1f ms"
          % report["latency"], flush=True)
    print("wrote %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
