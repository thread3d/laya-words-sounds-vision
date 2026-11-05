"""The control ladder for the ESC-50 run: what the frozen tower gets with a *linear* head.

Three rungs, all on the exact rows `laya-train` will be scored on, so the engine's number later is
comparable to something:

  * ``chance`` -- 1 / classes, the floor every arm has to beat;
  * ``centroid`` -- nearest class mean of the frozen embedding, a head with no learning at all;
  * ``ridge`` -- a linear probe on the frozen embedding, the standard "trained head on a frozen
    tower" measurement, with its ridge penalty chosen by an inner cross-validation on the training
    fold only.

The features are the ones Laya's audio path produces: each clip is decoded by `laya.audio`
(44.1 kHz stereo to a mono 16 kHz waveform, the product's own resampling), run through the frozen
wav2vec2-base tower, and mean-pooled over time -- 768 numbers per clip. Frames are cached on disk, so
a second run of the probe costs seconds and the *engine* arms can reuse them.

    python3 research/benchmarks/audio_esc50/probe.py --limit 8        # smoke test
    python3 research/benchmarks/audio_esc50/probe.py --folds 1 2 3 4 5

Writes `$AUDIO_BENCH_DATA/esc50/frames/<stem>.npz` (fp16) and `results_probe.json`.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

DATA_ROOT = os.environ.get("AUDIO_BENCH_DATA", os.path.join(REPO, ".audio-bench"))
ROWS = os.path.join(DATA_ROOT, "esc50")
FRAMES = os.path.join(ROWS, "frames")

TOWER = os.environ.get("AUDIO_BENCH_TOWER", "facebook/wav2vec2-base")
SAMPLE_RATE = 16000


def _encoder():
    """The product's own encoder object, with a throwaway projector width.

    `encode_frames` is what this probe needs, and building it through `laya.audio` is what makes the
    feature this probe scores the feature the engine will score.
    """
    from laya.audio import build_audio_encoder, parse_audio
    config = parse_audio({"audio": {"tower": TOWER, "sample_rate": SAMPLE_RATE,
                                    "n_audio_tokens": 32}})
    return build_audio_encoder(config, hidden_size=8)


_ENC = None


def _worker_init(threads):
    global _ENC
    import torch
    torch.set_num_threads(threads)
    _ENC = _encoder()
    _ENC.eval()


def _extract(path):
    """Mean/std/max of the tower's frames for one clip, cached beside the rows."""
    import torch
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(FRAMES, stem + ".npz")
    if os.path.exists(out):
        return out
    from laya.audio import load_waveform
    waveform = load_waveform(path, SAMPLE_RATE)
    with torch.no_grad():
        frames = _ENC.encode_frames(waveform.unsqueeze(0))[0].numpy().astype("float16")
    tmp = out + ".tmp.npz"
    np.savez(tmp, frames=frames)
    os.replace(tmp, out)
    return out


def extract_all(clips, workers, threads, limit=None):
    os.makedirs(FRAMES, exist_ok=True)
    todo = [c for c in clips if not os.path.exists(
        os.path.join(FRAMES, os.path.splitext(os.path.basename(c))[0] + ".npz"))]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print("frames: all %d clips already cached" % len(clips), flush=True)
        return
    print("frames: %d of %d clips to encode with %d workers x %d threads"
          % (len(todo), len(clips), workers, threads), flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(threads,)) as pool:
        for done, _ in enumerate(pool.map(_extract, todo, chunksize=8), 1):
            if done % 100 == 0 or done == len(todo):
                rate = done / max(1e-9, time.time() - started)
                print("  %d/%d  %.2f clips/s  eta %.1f min"
                      % (done, len(todo), rate, (len(todo) - done) / max(1e-9, rate) / 60),
                      flush=True)


# -- the arms ---------------------------------------------------------------------------------

def features(stem, mode):
    """Per-clip feature vector: mean-pooled frames, optionally with their standard deviation."""
    path = os.path.join(FRAMES, stem + ".npz")
    frames = np.load(path)["frames"].astype(np.float32)
    pooled = [frames.mean(axis=0)]
    if mode == "mean+std":
        pooled.append(frames.std(axis=0))
    return np.concatenate(pooled)


def _standardise(train, test):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mu) / sd, (test - mu) / sd


def fit_ridge_predict(x_train, y_train, x_test, alpha):
    """Closed-form ridge regression to one-hot targets; the argmax is the prediction."""
    n, d = x_train.shape
    k = y_train.shape[1]
    xb = np.hstack([x_train, np.ones((n, 1))])
    reg = np.eye(d + 1) * alpha
    reg[-1, -1] = 0.0                                  # never penalise the intercept
    w = np.linalg.solve(xb.T @ xb + reg, xb.T @ y_train)
    scores = np.hstack([x_test, np.ones((x_test.shape[0], 1))]) @ w
    return scores.argmax(axis=1), (n, d, k)


def centroid_predict(x_train, y_train, x_test):
    """Nearest class mean: a head with no learning beyond the class averages."""
    labels = y_train.argmax(axis=1)
    means = np.stack([x_train[labels == c].mean(axis=0) for c in range(y_train.shape[1])])
    d = ((x_test[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def choose_alpha(x_train, y_train, alphas, inner_folds=4, seed=0):
    """Pick the ridge penalty on the training fold alone, by inner cross-validation."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_train))
    best = (None, -1.0)
    for alpha in alphas:
        correct = 0
        for part in range(inner_folds):
            held = order[part::inner_folds]
            rest = np.setdiff1d(order, held)
            xa, xb = _standardise(x_train[rest], x_train[held])
            pred, _ = fit_ridge_predict(xa, y_train[rest], xb, alpha)
            correct += int((pred == y_train[held].argmax(axis=1)).sum())
        accuracy = correct / len(x_train)
        if accuracy > best[1]:
            best = (alpha, accuracy)
    return best


def _cached(row):
    """Whether this row's frames are on disk -- `--limit` scores whatever has been encoded."""
    stem = os.path.splitext(os.path.basename(row["state"]["audio_path"]))[0]
    return os.path.exists(os.path.join(FRAMES, stem + ".npz"))


def _row_path(subset, fold, split, dataset="esc50"):
    """Where the rows for one subset/fold/split live.

    ESC-50 is cross-validated (five folds per subset); FSDD is split by speaker and has one split
    pair, which the fold number is ignored for.
    """
    if dataset == "fsdd":
        return os.path.join(ROWS, "..", "fsdd", "fsdd_%s.jsonl" % split)
    return os.path.join(ROWS, "%s_fold%d_%s.jsonl" % (subset, fold, split))


def evaluate_fold(rows_dir, fold, subset, mode, alphas, dataset="esc50"):
    train_rows = [r for r in _read(_row_path(subset, fold, "train", dataset)) if _cached(r)]
    eval_rows = [r for r in _read(_row_path(subset, fold, "eval", dataset)) if _cached(r)]
    if not train_rows or not eval_rows:
        return None
    question = next(iter(train_rows[0]["questions"]))
    classes = train_rows[0]["questions"][question]["criteria"]
    index = {name: i for i, name in enumerate(classes)}

    def xy(rows):
        x = np.stack([features(os.path.splitext(os.path.basename(r["state"]["audio_path"]))[0], mode)
                      for r in rows])
        y = np.zeros((len(rows), len(classes)), dtype=np.float32)
        for i, row in enumerate(rows):
            y[i, index[row["expected"][question]]] = 1.0
        return x, y

    x_train_raw, y_train = xy(train_rows)
    x_eval_raw, y_eval = xy(eval_rows)
    alpha, inner = choose_alpha(x_train_raw, y_train, alphas)
    x_train, x_eval = _standardise(x_train_raw, x_eval_raw)

    ridge_pred, shape = fit_ridge_predict(x_train, y_train, x_eval, alpha)
    centroid_pred = centroid_predict(x_train, y_train, x_eval)
    truth = y_eval.argmax(axis=1)
    return {
        "fold": fold, "subset": subset, "features": mode,
        "train_rows": len(train_rows), "eval_rows": len(eval_rows), "classes": len(classes),
        "chance": 1.0 / len(classes),
        "centroid": float((centroid_pred == truth).mean()),
        "ridge": float((ridge_pred == truth).mean()),
        "ridge_alpha": alpha, "ridge_inner_accuracy": inner, "ridge_shape": shape,
    }


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarise(folds):
    out = {}
    for subset in sorted({f["subset"] for f in folds}):
        for mode in sorted({f["features"] for f in folds}):
            arm = [f for f in folds if f["subset"] == subset and f["features"] == mode]
            if not arm:
                continue
            for key in ("centroid", "ridge"):
                values = [f[key] for f in arm]
                out["%s/%s/%s" % (subset, mode, key)] = {
                    "mean": float(np.mean(values)), "std": float(np.std(values)),
                    "per_fold": values,
                }
            out["%s/%s/chance" % (subset, mode)] = {"mean": arm[0]["chance"], "std": 0.0}
    return out


def _clip_list(subsets, folds, per_split, dataset="esc50"):
    """Every clip to encode, taken split by split so `--limit` gives a *scoreable* subset.

    A limit applied to a sorted file list would encode one fold's clips only -- ESC-50 filenames
    start with their fold -- and a probe needs both sides of a split to score anything.
    """
    seen, out = set(), []
    for subset in subsets:
        for fold in folds:
            for name in ("train", "eval"):
                rows = _read(_row_path(subset, fold, name, dataset))
                if per_split is not None:
                    # Evenly spaced, not the first N: both row files are ordered by class, so a
                    # prefix is one or two classes and a smoke test on it scores nothing real.
                    stride = max(1, len(rows) // per_split)
                    rows = rows[::stride][:per_split]
                for row in rows:
                    path = row["state"]["audio_path"]
                    if path not in seen:
                        seen.add(path)
                        out.append(path)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="esc50", choices=["esc50", "fsdd"],
                        help="which rows to score: the 5-fold ESC-50 set or the speaker-split FSDD one")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--subsets", nargs="+", default=["esc50", "esc10"],
                        choices=["esc50", "esc10"])
    parser.add_argument("--features", nargs="+", default=["mean", "mean+std"],
                        choices=["mean", "mean+std"])
    parser.add_argument("--limit", type=int, default=None,
                        help="encode at most this many new clips (a smoke test)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) // 8))
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    args = parser.parse_args(argv)

    if args.dataset == "fsdd":
        args.subsets, args.folds = ["fsdd"], [0]
    clips = _clip_list(args.subsets, args.folds, args.limit, args.dataset)
    extract_all(clips, args.workers, args.threads)

    results = []
    for subset in args.subsets:
        for mode in args.features:
            for fold in args.folds:
                row = evaluate_fold(ROWS, fold, subset, mode, args.alphas, args.dataset)
                if row is None:
                    print("%s %-8s fold %d: no cached frames for that fold" % (subset, mode, fold))
                    continue
                results.append(row)
                print("%s %-8s fold %d (%d train / %d eval): chance %.3f  centroid %.3f  "
                      "ridge %.3f  (alpha %g)"
                      % (subset, mode, fold, row["train_rows"], row["eval_rows"], row["chance"],
                         row["centroid"], row["ridge"], row["ridge_alpha"]), flush=True)
    report = {"arms": results, "summary": summarise(results), "tower": TOWER,
              "sample_rate": SAMPLE_RATE}
    out = os.path.join(ROWS, "results_probe%s.json"
                       % ("" if args.dataset == "esc50" else "_%s" % args.dataset))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("\nwrote %s" % out)
    for name, stats in sorted(report["summary"].items()):
        print("  %-28s %.4f ± %.4f" % (name, stats["mean"], stats["std"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
