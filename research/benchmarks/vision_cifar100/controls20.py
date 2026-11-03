"""Controls for the 20-way (coarse CIFAR-100) task, reusing the cached SigLIP2 embeddings.

The images and the cached embeddings are the same as the 100-way run's -- only the label mapping
changes -- so this costs a text-embedding pass and a small head, not 15k image encodes.

    source gpuenv.sh && HIP_VISIBLE_DEVICES=<idle card> python3 controls20.py
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
sys.path.insert(0, HERE)
import tower_patch  # noqa: E402
from transformers import SiglipModel, SiglipProcessor  # noqa: E402

TEMPLATES = ["a photo of {}.", "a photo of the {}.", "a close-up photo of a {}.",
             "a photo of a {} in a photograph.", "a blurry photo of a {}."]


def rows(name):
    return [json.loads(l) for l in open(os.path.join(CIFAR, name)) if l.strip()]


def ece(conf, correct, bins=15):
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if idx:
            err += (len(idx) / len(conf)) * abs(sum(correct[i] for i in idx) / len(idx)
                                                - sum(conf[i] for i in idx) / len(idx))
    return err


def report(name, probs, gold):
    top = probs.argmax(1)
    conf = probs[np.arange(len(top)), top]
    correct = [int(t == g) for t, g in zip(top, gold)]
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    rc = {p: sum(correct[i] for i in order[:max(1, int(round(p * len(order))))])
          / max(1, int(round(p * len(order)))) for p in (1.0, 0.8, 0.5)}
    out = {"accuracy": float(np.mean(correct)), "mean_confidence": float(conf.mean()),
           "ece": ece(list(conf), correct), "coverage_80": rc[0.8], "coverage_50": rc[0.5]}
    print("%-14s acc %.4f | conf %.4f | ECE %.4f | cov80 %.4f | cov50 %.4f"
          % (name, out["accuracy"], out["mean_confidence"], out["ece"], rc[0.8], rc[0.5]))
    return out


def _floats(d):
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in d.items()}


def main():
    train = rows("train20.jsonl")
    evaluation = rows("eval20.jsonl")
    base_train = rows("train.jsonl")
    base_eval = rows("eval.jsonl")
    labels = sorted({r["expected"]["class"] for r in train})

    # The cached embeddings belong to the row order of train.jsonl / eval.jsonl.
    assert [r["state"]["image_path"] for r in base_train] == [r["state"]["image_path"] for r in train], \
        "20-way rows must line up with the cached embeddings"
    assert [r["state"]["image_path"] for r in base_eval] == [r["state"]["image_path"] for r in evaluation]

    x_train = torch.from_numpy(np.load(os.path.join(CIFAR, "x_train.npy")))
    x_eval = torch.from_numpy(np.load(os.path.join(CIFAR, "x_eval.npy")))
    gold = np.array([labels.index(r["expected"]["class"]) for r in evaluation])
    gold_train = np.array([labels.index(r["expected"]["class"]) for r in train])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipModel.from_pretrained(MODEL, torch_dtype=torch.float32).to(device).eval()
    tower_patch.patch_tower(model)
    proc = SiglipProcessor.from_pretrained(MODEL)

    text = []
    with torch.no_grad():
        for label in labels:
            inputs = proc(text=[t.format(label) for t in TEMPLATES], return_tensors="pt",
                          padding="max_length", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = model.get_text_features(**inputs).pooler_output.float().cpu()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(0)
            text.append(mean / mean.norm())
    text = torch.stack(text)
    # SigLIP's learned logit scale, so the probabilities (and therefore the ECE) mean something.
    scale = float(model.logit_scale.exp())
    # `report` reads these as probabilities, so a 20-way softmax over the scaled logits -- the same
    # normalisation Laya applies to a choice question. Raw sigmoid scores sum well past 1.
    logits = torch.softmax((x_eval @ text.T) * scale, dim=1).numpy()
    print("zero-shot logit scale %.1f" % scale)
    out = {"zero_shot": _floats(report("zero-shot", logits, gold))}

    torch.manual_seed(0)
    head = torch.nn.Linear(x_train.shape[1], len(labels)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=0.05, weight_decay=1e-4)
    xt, yt = x_train.to(device), torch.from_numpy(gold_train).to(device)
    for _ in range(300):
        perm = torch.randperm(xt.shape[0], device=device)
        for start in range(0, xt.shape[0], 512):
            idx = perm[start:start + 512]
            loss = torch.nn.functional.cross_entropy(head(xt[idx]) * 10.0, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        probs = torch.softmax(head(x_eval.to(device)) * 10.0, dim=1).cpu().numpy()
    out["linear_probe"] = _floats(report("linear probe", probs, gold))

    out["labels"] = labels
    out["chance"] = 1.0 / len(labels)
    with open(os.path.join(CIFAR, "controls20.json"), "w") as handle:
        json.dump(out, handle, indent=2)
    print("chance %.4f | wrote controls20.json" % out["chance"])


if __name__ == "__main__":
    main()
