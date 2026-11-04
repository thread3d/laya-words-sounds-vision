"""Controls for the CIFAR-100 run: what the same frozen SigLIP2 tower gets with its *own* heads.

Three rungs on one ladder, all on the exact eval split `laya-train` will be scored on:

  * zero-shot, SigLIP2's published protocol (`a photo of a {class}.`, averaged over templates);
  * linear probe on the frozen image embedding -- the standard "trained head on a frozen tower";
  * a 12-parameter mean-RGB centroid, the trivial floor.

Writes `$LAYA_BENCH_DATA/cifar100/controls.json` plus the cached embeddings, so a later analysis can
report calibration and risk-coverage for the controls and Laya on identical rows.

    source benchmarks/cifar100/gpuenv.sh && python3 benchmarks/cifar100/controls.py [--limit N]
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import SiglipModel, SiglipProcessor

import _paths

# This machine's MIOpen segfaults on every Conv2d; the patch-embed conv of SigLIP2 is one, so the
# tower cannot run on the GPU without `tower_patch.py` (beside this script). Environment
# workaround, not product code.
_paths.add_harness_to_path()
import tower_patch  # noqa: E402

CIFAR = _paths.CIFAR
MODEL = _paths.SIGLIP

# CLIP's standard CIFAR prompt set, trimmed to seven; averaging their embeddings is the published
# zero-shot protocol. SigLIP 2 was trained on long captions, so the plain ones are kept first.
TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of a large {}.",
    "a blurry photo of a {}.",
    "a photo of a {} in a forest.",
    "a photo of a {} on a white background.",
]


def _feature(output):
    """The contrastive embedding out of `get_{image,text}_features`.

    transformers 5.13 returns a `BaseModelOutputWithPooling` here rather than a tensor; the
    projected, pooled vector is `.pooler_output` (there is no separate `visual_projection` module
    in this version, the head is baked in).
    """
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


def load_rows(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_set(rows):
    return sorted({r["expected"]["class"] for r in rows})


def embed_images(model, proc, paths, device, batch=64):
    """Projected image embeddings, L2-normalised (SigLIP's own space)."""
    out = []
    with torch.no_grad():
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            pixels = [proc.image_processor(images=__import__("PIL.Image", fromlist=["Image"]).open(p).convert("RGB"),
                                           return_tensors="pt")["pixel_values"][0] for p in chunk]
            pixels = torch.stack(pixels).to(device)
            out.append(_feature(model.get_image_features(pixel_values=pixels)).float().cpu())
            if start % (batch * 20) == 0:
                print("  embedded %d/%d" % (start + len(chunk), len(paths)), flush=True)
    emb = torch.cat(out)
    return emb / emb.norm(dim=-1, keepdim=True)


def text_embeddings(model, proc, labels, device):
    """One embedding per label, averaged over templates, L2-normalised."""
    out = []
    with torch.no_grad():
        for label in labels:
            texts = [t.format(label) for t in TEMPLATES]
            inputs = proc(text=texts, return_tensors="pt", padding="max_length", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = _feature(model.get_text_features(**inputs)).float().cpu()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(0)
            out.append(mean / mean.norm())
    return torch.stack(out)


def zero_shot(image_emb, text_emb, y, labels):
    logits = image_emb @ text_emb.T
    pred = logits.argmax(1)
    gold = torch.tensor([labels.index(l) for l in y])
    acc = float((pred == gold).float().mean())
    probs = torch.softmax(logits, dim=1).numpy()
    return acc, probs


def linear_probe(x_train, y_train, x_eval, y_eval, labels, epochs=400, lr=0.05, wd=1e-4):
    """Multinomial logistic regression on frozen features -- the standard frozen-tower head."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xt = x_train.to(device)
    xe = x_eval.to(device)
    yt = torch.tensor([labels.index(l) for l in y_train], device=device)
    ye = torch.tensor([labels.index(l) for l in y_eval], device=device)
    head = torch.nn.Linear(xt.shape[1], len(labels)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = xt.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, 512):
            idx = perm[start:start + 512]
            loss = torch.nn.functional.cross_entropy(head(xt[idx]) * 10.0, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits = head(xe) * 10.0
        acc = float((logits.argmax(1) == ye).float().mean())
    return acc, head


def centroid_baseline(train_paths, train_labels, eval_paths, eval_labels, labels, size=16):
    from PIL import Image

    def mean_rgb(path):
        im = Image.open(path).convert("RGB").resize((size, size))
        return np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(0)

    xtr = np.array([mean_rgb(p) for p in train_paths], dtype=np.float32)
    xev = np.array([mean_rgb(p) for p in eval_paths], dtype=np.float32)
    cents = np.array([xtr[[i for i, l in enumerate(train_labels) if l == c]].mean(0) for c in labels])
    pred = np.array([labels[i] for i in np.argmin(((xev[:, None, :] - cents[None]) ** 2).sum(-1), 1)])
    return float((pred == np.array(eval_labels)).mean()), cents.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-limit", type=int, default=0, help="cap the training split (0 = all)")
    ap.add_argument("--eval-limit", type=int, default=0, help="cap the eval split (0 = all)")
    args = ap.parse_args()

    train = load_rows(os.path.join(CIFAR, "train.jsonl"))
    evaluation = load_rows(os.path.join(CIFAR, "eval.jsonl"))
    labels = label_set(train)
    if args.train_limit:
        train = stratified(train, args.train_limit)
    if args.eval_limit:
        evaluation = stratified(evaluation, args.eval_limit)
    print("labels %d | train %d | eval %d" % (len(labels), len(train), len(evaluation)))

    device = "cuda"
    model = SiglipModel.from_pretrained(MODEL, torch_dtype=torch.float32).to(device).eval()
    n = tower_patch.patch_tower(model)
    print("tower_patch: %d conv(s) rerouted through im2col+matmul" % n, flush=True)
    proc = SiglipProcessor.from_pretrained(MODEL)

    train_paths = [r["state"]["image_path"] for r in train]
    eval_paths = [r["state"]["image_path"] for r in evaluation]
    train_y = [r["expected"]["class"] for r in train]
    eval_y = [r["expected"]["class"] for r in evaluation]

    print("embedding train ...", flush=True)
    x_train = embed_images(model, proc, train_paths, device)
    print("embedding eval ...", flush=True)
    x_eval = embed_images(model, proc, eval_paths, device)
    np.save(os.path.join(CIFAR, "x_train.npy"), x_train.numpy())
    np.save(os.path.join(CIFAR, "x_eval.npy"), x_eval.numpy())

    print("text embeddings ...", flush=True)
    t_emb = text_embeddings(model, proc, labels, device)

    zs_acc, zs_probs = zero_shot(x_eval, t_emb, eval_y, labels)
    print("zero-shot accuracy            %.4f" % zs_acc, flush=True)

    lp_acc, head = linear_probe(x_train, train_y, x_eval, eval_y, labels)
    print("frozen linear probe accuracy  %.4f" % lp_acc, flush=True)

    cb_acc, cb_params = centroid_baseline(train_paths, train_y, eval_paths, eval_y, labels)
    print("mean-RGB centroid (%d params) %.4f" % (cb_params, cb_acc), flush=True)

    np.save(os.path.join(CIFAR, "zs_probs.npy"), zs_probs)
    result = {"labels": len(labels), "train": len(train), "eval": len(evaluation),
              "zero_shot": zs_acc, "linear_probe": lp_acc,
              "centroid": cb_acc, "centroid_params": cb_params,
              "templates": TEMPLATES}
    with open(os.path.join(CIFAR, "controls.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


def stratified(rows, per_class):
    counts, out = {}, []
    for row in rows:
        label = row["expected"]["class"]
        if counts.get(label, 0) < per_class:
            counts[label] = counts.get(label, 0) + 1
            out.append(row)
    return out


if __name__ == "__main__":
    main()
