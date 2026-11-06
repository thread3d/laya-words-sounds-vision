"""ESC-10, five-fold cross-validation, in the layout the product ships: the protocol-correct number.

Every other engine arm in this harness is one fold or one speaker split, because a fold costs minutes
to an hour. On one GPU a fold costs minutes, so the published protocol -- train on four folds, score
the fifth, five times, report mean ± std -- fits. ESC-10 is the subset small enough to do it here
(400 clips, 320 training per fold) and the one whose linear probe the tower table reports.

Each fold is trained with `laya.train.finetune` (the same function `laya-train` calls), then scored
with the two media gates on its own held-out fold. `state_conditioned` costs ten rows per clip, which
is what makes this the *shipping* layout's number rather than the joint layout's.

    source research/benchmarks/audio_esc50/gpuenv.sh
    python3 research/benchmarks/audio_esc50/cv_esc10.py [--epochs 15] [--folds 1 2 3 4 5]

Writes `$AUDIO_BENCH_DATA/esc50/results_esc10_cv.json`.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

DATA_ROOT = os.environ.get("AUDIO_BENCH_DATA", os.path.join(REPO, ".audio-bench"))
ROWS = os.path.join(DATA_ROOT, "esc50")
BASE = os.environ.get("AUDIO_BENCH_BASE", os.path.join(REPO, "models", "laya"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if str(args.device).startswith("cuda"):
        import gpu_patch
        gpu_patch.install()

    import laya
    from laya.evals import Dataset
    from laya.evals_audio import audio_ablation, option_order_flip_rate
    from laya.train import TrainConfig, finetune

    per_fold = []
    for fold in args.folds:
        started = time.time()
        out = os.path.join(DATA_ROOT, "checkpoints", "esc10-sc_cv_fold%d" % fold)
        config = TrainConfig(
            epochs=args.epochs, micro_batch=16, grad_accum=1, freeze_encoder=True,
            loss="soft-ce", encode_workers=1, checkpoint_every=0,
            audio={"n_audio_tokens": 32}, option_layout="state_conditioned",
            eval_data=os.path.join(ROWS, "esc10_fold%d_eval.jsonl" % fold),
        )
        finetune(data=os.path.join(ROWS, "esc10_fold%d_train.jsonl" % fold),
                 model_dir=BASE, output_dir=out, config=config, device=args.device)
        trained = time.time() - started

        agent = laya.load(out, device=args.device)
        dataset = Dataset.from_jsonl(os.path.join(ROWS, "esc10_fold%d_eval.jsonl" % fold))
        gate1 = audio_ablation(agent, dataset, batch_size=16)
        gate2 = option_order_flip_rate(agent, dataset, batch_size=16)
        entry = {
            "fold": fold,
            "accuracy": gate1["per_tag"]["all"]["with_audio"],
            "accuracy_without_audio": gate1["per_tag"]["all"]["without_audio"],
            "gate1_gain": gate1["gain"],
            "gate2b_flip_rate": gate2["flip_rate"],
            "train_seconds": round(trained, 1),
        }
        per_fold.append(entry)
        print("fold %d: accuracy %.4f  gate1 %+.3f  gate2b %s  (%.0fs)"
              % (fold, entry["accuracy"] or float("nan"), entry["gate1_gain"],
                 entry["gate2b_flip_rate"], trained), flush=True)

    accuracies = [f["accuracy"] for f in per_fold if f["accuracy"] is not None]
    gains = [f["gate1_gain"] for f in per_fold if f["gate1_gain"] is not None]
    flips = [f["gate2b_flip_rate"] for f in per_fold if f["gate2b_flip_rate"] is not None]
    mean = lambda xs: sum(xs) / len(xs) if xs else None                     # noqa: E731
    std = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else None  # noqa: E731
    report = {
        "dataset": "ESC-10", "protocol": "5-fold cross-validation (train 4, score 1)",
        "layout": "state_conditioned", "epochs": args.epochs, "per_fold": per_fold,
        "accuracy_mean": mean(accuracies), "accuracy_std": std(accuracies),
        "gate1_gain_mean": mean(gains), "gate2b_flip_rate_mean": mean(flips),
        "chance": 0.1,
    }
    out = args.out or os.path.join(ROWS, "results_esc10_cv.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("\nESC-10 %d-fold, state_conditioned: accuracy %.4f +/- %.4f, gate 1 %+.3f, gate 2b %.3f"
          % (len(per_fold), report["accuracy_mean"], report["accuracy_std"],
             report["gate1_gain_mean"], report["gate2b_flip_rate_mean"]))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
