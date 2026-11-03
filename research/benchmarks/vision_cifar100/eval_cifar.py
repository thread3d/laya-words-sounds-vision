"""Score a trained CIFAR-100 vision checkpoint, and both gates, from one pass over the data.

Everything here is measured on the saved artifact through `Agent.predict`, not read off the
trainer's console report -- that report is known to disagree with the saved artifact (it scored
0.033 for a checkpoint that measures 0.873), so the checkpoint is the only thing worth measuring.

Reports, per checkpoint:
  * accuracy on the held-out split, against the controls in `controls.json`;
  * calibration: ECE (top-1, 100 classes), mean confidence, Brier, and the risk-coverage curve
    (accuracy of the answers that survive at each coverage);
  * gate 1, image ablation: the same rows with the image removed;
  * gate 2b, option-order invariance: the same rows with the 100 options permuted.

    python3 eval_cifar.py ckpt_cifar_frozen [--limit N]
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATA_ROOT = os.environ.get("VISION_CIFAR_DATA", os.path.join(REPO, ".vision-cifar100"))


def _jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _probs(answer, criteria):
    """Per-option probability vector in `criteria` order, whatever shape the answer came back in."""
    raw = answer.get("probs") or answer.get("probabilities") or {}
    if isinstance(raw, dict):
        return [float(raw.get(name, 0.0)) for name in criteria]
    return [float(x) for x in raw]


def _ece(probs, correct, bins=15):
    """Expected calibration error on the top-1 confidence, equal-width bins."""
    conf = [p for p in probs]
    total = len(conf)
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        avg = sum(conf[i] for i in idx) / len(idx)
        err += (len(idx) / total) * abs(acc - avg)
    return err


def _risk_coverage(probs, correct, points=(1.0, 0.9, 0.8, 0.7, 0.5, 0.3)):
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    out = {}
    for cov in points:
        k = max(1, int(round(cov * len(order))))
        kept = order[:k]
        out["%.0f%%" % (cov * 100)] = sum(correct[i] for i in kept) / len(kept)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="eval.jsonl", help="eval split under cifar100")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gate-limit", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16,
                    help="states per forward pass; 1 disables batching entirely")
    args = ap.parse_args()

    import torch  # noqa: F401  (torch first, as everything in this checkout requires)
    sys.path.insert(0, HERE)
    sys.path.insert(0, REPO)
    import tower_patch  # noqa: E402
    tower_patch.install()

    from laya import Agent  # noqa: E402
    from laya.evals_vision import strip_image  # noqa: E402

    rows = _jsonl(os.path.join(HERE, "cifar100", args.data))
    if args.limit:
        rows = rows[:args.limit]

    agent = Agent(model_id_or_path=args.checkpoint, device=args.device)
    qid = next(iter(rows[0]["questions"]))
    criteria = rows[0]["questions"][qid]["criteria"]

    def answers_for(states, questions, batch=None):
        batch = args.batch if batch is None else batch
        """Batched: `Agent.predict_batch` packs states into shared forward passes."""
        if batch <= 1:
            # one state per forward pass: `predict_batch` on image states measured ~16x slower per
            # item here (1.25 s/item against 75 ms), which is its own finding but not this run's job
            out = [agent.predict(st, questions) for st in states]
        else:
            out = agent.predict_batch(list(states), questions, batch_size=batch)
        if isinstance(out, dict) and "answers" in out:
            return [out["answers"]] * len(states)
        # one dict per state, each shaped {"model": ..., "answers": {qid: answer}}
        return [row.get("answers", row) for row in out]

    # ---- accuracy + calibration, as given ---------------------------------------------------
    probs, correct, conf = [], [], []
    questions = rows[0]["questions"]
    for start in range(0, len(rows), 32):
        chunk = rows[start:start + 32]
        got = answers_for([r["state"] for r in chunk], questions)
        for row, ans in zip(chunk, got):
            p = _probs(ans[qid], criteria)
            top = max(range(len(p)), key=lambda k: p[k])
            probs.append(p[top])
            correct.append(int(criteria[top] == row["expected"][qid]))
            conf.append(p[top])
        if start % 500 == 0:
            print("  scored %d/%d" % (start + len(chunk), len(rows)), flush=True)
    acc = sum(correct) / len(correct)
    result = {
        "checkpoint": args.checkpoint,
        "items": len(rows),
        "accuracy": acc,
        "mean_confidence": sum(conf) / len(conf),
        "ece": _ece(conf, correct),
        "risk_coverage": _risk_coverage(conf, correct),
        "chance": 1.0 / len(criteria),
    }
    print("accuracy %.4f | mean conf %.4f | ECE %.4f" % (acc, result["mean_confidence"], result["ece"]))

    # ---- gate 1: image ablation -------------------------------------------------------------
    gate_rows = rows[:args.gate_limit]
    both = []
    for row in gate_rows:
        both.append(row["state"])
        both.append(strip_image(row["state"]))
    got = answers_for(both, questions, batch=1)
    with_img = without_img = 0
    for i, row in enumerate(gate_rows):
        pa = max(range(len(criteria)), key=lambda k: _probs(got[2 * i][qid], criteria)[k])
        pb = max(range(len(criteria)), key=lambda k: _probs(got[2 * i + 1][qid], criteria)[k])
        gold = row["expected"][qid]
        with_img += int(criteria[pa] == gold)
        without_img += int(criteria[pb] == gold)
    n = len(gate_rows)
    result["gate1"] = {"with_image": with_img / n, "without_image": without_img / n,
                       "gain": (with_img - without_img) / n, "items": n}

    # ---- gate 2b: option-order invariance ---------------------------------------------------
    rng = random.Random(0)
    flips = 0
    base_q = rows[0]["questions"][qid]
    for start in range(0, len(gate_rows), 16):
        chunk = gate_rows[start:start + 16]
        states, orders = [], []
        for row in chunk:
            order = list(range(len(criteria)))
            rng.shuffle(order)
            orders.append(order)
            states.append(row["state"]); states.append(row["state"])
        got = answers_for(states, {qid: base_q}, batch=1)
        # score the permuted rows one pass at a time: the option order travels per call
        for i, row in enumerate(chunk):
            va = got[2 * i][qid].get("choice")
            perm_q = {qid: dict(base_q, option_order=orders[i])}
            vb = answers_for([states[2 * i + 1]], perm_q, batch=1)[0][qid].get("choice")
            flips += int(va != vb)
    result["gate2b"] = {"flip_rate": flips / n, "flips": flips, "items": n}

    print("gate 1  with %.4f / without %.4f  gain %+.4f"
          % (result["gate1"]["with_image"], result["gate1"]["without_image"], result["gate1"]["gain"]))
    print("gate 2b flip rate %.4f (%d/%d)" % (result["gate2b"]["flip_rate"], flips, n))

    out = os.path.join(HERE, "cifar100", "eval_%s%s.json" % (os.path.basename(args.checkpoint), args.tag))
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
