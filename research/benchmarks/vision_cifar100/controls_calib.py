"""The controls on the same axes Laya is judged on: calibration, risk-coverage, order invariance.

Reuses the cached embeddings from `controls.py`, so this costs seconds and runs on an idle card.

The point of the last column: a CLIP/SigLIP-style classifier scores each label independently and
takes an argmax, so permuting the label list cannot change its answer -- an order-flip rate of
exactly 0 by construction. That is the floor Laya's gate 2b is measured against.

    source gpuenv.sh && HIP_VISIBLE_DEVICES=2 python3 controls_calib.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATA_ROOT = os.environ.get("VISION_CIFAR_DATA", os.path.join(REPO, ".vision-cifar100"))
CIFAR = os.path.join(DATA_ROOT, "cifar100")
MODEL = os.path.join(DATA_ROOT, "siglip2")
TEMPLATES = ["a photo of {}.", "a photo of the {}.", "a close-up photo of a {}.",
             "a photo of a {} in a photograph.", "a blurry photo of a {}."]


def ece(conf, correct, bins=15):
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        err += (len(idx) / len(conf)) * abs(sum(correct[i] for i in idx) / len(idx)
                                            - sum(conf[i] for i in idx) / len(idx))
    return err


def risk_coverage(conf, correct, points=(1.0, 0.9, 0.8, 0.7, 0.5, 0.3)):
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    return {"%.0f%%" % (p * 100): sum(correct[i] for i in order[:max(1, int(round(p * len(order))))])
            / max(1, int(round(p * len(order)))) for p in points}


def summarise(name, probs, gold):
    top = probs.argmax(1)
    conf = probs[np.arange(len(top)), top]
    correct = [int(t == g) for t, g in zip(top, gold)]
    out = {"accuracy": float(np.mean(correct)), "mean_confidence": float(conf.mean()),
           "ece": ece(list(conf), correct), "risk_coverage": risk_coverage(list(conf), correct)}
    print("%-16s acc %.4f | conf %.4f | ECE %.4f | coverage-70%% acc %.4f"
          % (name, out["accuracy"], out["mean_confidence"], out["ece"], out["risk_coverage"]["70%"]))
    return out


def zero_shot_probs(x_eval, labels):
    """SigLIP 2's own zero-shot probabilities, with its learned logit scale.

    `controls.py` saved the softmax of the *raw* cosine similarities, which is fine for an argmax
    and wrong for a calibration number: SigLIP multiplies by `logit_scale` (~100) before the sigmoid,
    so the unscaled version is nearly uniform and its ECE says nothing. This recomputes the arm
    properly, on the CPU, from the cached embeddings.
    """
    sys.path.insert(0, HERE)
    from transformers import SiglipModel, SiglipProcessor
    model = SiglipModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    proc = SiglipProcessor.from_pretrained(MODEL)
    text = []
    with torch.no_grad():
        for label in labels:
            inputs = proc(text=[t.format(label) for t in TEMPLATES], return_tensors="pt",
                          padding="max_length", truncation=True)
            emb = model.get_text_features(**inputs).pooler_output.float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(0)
            text.append(mean / mean.norm())
    text = torch.stack(text)
    scale = float(model.logit_scale.exp())
    logits = (x_eval @ text.T) * scale
    print("zero-shot logit scale %.1f (raw cosine spread %.4f)"
          % (scale, float((x_eval @ text.T).std())))
    return torch.softmax(logits, dim=1).numpy()


def main():
    rows = [json.loads(l) for l in open(os.path.join(CIFAR, "eval.jsonl")) if l.strip()]
    train = [json.loads(l) for l in open(os.path.join(CIFAR, "train.jsonl")) if l.strip()]
    labels = sorted({r["expected"]["class"] for r in train})
    gold = np.array([labels.index(r["expected"]["class"]) for r in rows])

    x_train = torch.from_numpy(np.load(os.path.join(CIFAR, "x_train.npy")))
    x_eval = torch.from_numpy(np.load(os.path.join(CIFAR, "x_eval.npy")))
    zs_probs = zero_shot_probs(x_eval, labels)
    result = {"zero_shot": summarise("zero-shot", zs_probs, gold)}

    # Same frozen-linear-probe recipe as controls.py, seeded so the calibration numbers are stable.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    y_train = torch.tensor([labels.index(r["expected"]["class"]) for r in train])
    head = torch.nn.Linear(x_train.shape[1], len(labels)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=0.05, weight_decay=1e-4)
    xt, yt = x_train.to(device), y_train.to(device)
    for _ in range(400):
        perm = torch.randperm(xt.shape[0], device=device)
        for start in range(0, xt.shape[0], 512):
            idx = perm[start:start + 512]
            loss = torch.nn.functional.cross_entropy(head(xt[idx]) * 10.0, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits = head(x_eval.to(device)) * 10.0
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    result["linear_probe"] = summarise("linear probe", probs, gold)

    # Order invariance, measured the way gate 2b measures it: permute the label list, re-argmax.
    rng = np.random.default_rng(0)
    flips_zs = flips_lp = 0
    for i in range(min(1000, len(gold))):
        order = rng.permutation(len(labels))
        base_zs = int(zs_probs[i].argmax())
        flips_zs += int(int(order[zs_probs[i][order].argmax()]) != base_zs)
        base_lp = int(probs[i].argmax())
        flips_lp += int(int(order[probs[i][order].argmax()]) != base_lp)
    result["flip_rate"] = {"zero_shot": flips_zs / min(1000, len(gold)),
                           "linear_probe": flips_lp / min(1000, len(gold)),
                           "items": min(1000, len(gold))}
    print("flip rate: zero-shot %.4f | linear probe %.4f (both invariant by construction)"
          % (result["flip_rate"]["zero_shot"], result["flip_rate"]["linear_probe"]))

    def _plain(v):
        if isinstance(v, dict):
            return {k: _plain(x) for k, x in v.items()}
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        return v
    with open(os.path.join(CIFAR, "controls_calib.json"), "w") as handle:
        json.dump(_plain(result), handle, indent=2)
    print("wrote", os.path.join(CIFAR, "controls_calib.json"))


if __name__ == "__main__":
    main()
