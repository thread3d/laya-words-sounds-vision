"""Same CIFAR-100 images and the same selection, with the *parquet's* coarse labels.

Authoritative on purpose: the coarse label comes out of the dataset, not out of a hand-typed
fine->coarse table, so a typo cannot silently relabel 500 items.

The 100-way run collapsed to the uniform prior. This 20-way version splits the explanations: if a
20-way head learns, the failure is cardinality / marker-to-visual distance; if it also collapses,
the head does not learn from real images at all.
"""
import json, os
import pyarrow.parquet as pq

import _paths

CIFAR = _paths.CIFAR
DATA = _paths.PARQUET
PER_CLASS_TRAIN, PER_CLASS_EVAL = 100, 50

COARSE = ["aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
          "household electrical devices", "household furniture", "insects", "large carnivores",
          "large man-made outdoor things", "large natural outdoor scenes",
          "large omnivores and herbivores", "medium mammals", "non-insect invertebrates", "people",
          "reptiles", "small mammals", "trees", "vehicles 1", "vehicles 2"]
assert len(COARSE) == 20


def emit(split, parquet, per_class, out_name):
    table = pq.read_table(parquet)
    taken = {i: None for i in range(100)}
    seen = {}
    rows = []
    for batch in table.to_batches(max_chunksize=8192):
        cols = batch.to_pydict()
        for img, fine, coarse in zip(cols["img"], cols["fine_label"], cols["coarse_label"]):
            n = seen.get(fine, 0)
            if n >= per_class:
                continue
            seen[fine] = n + 1
            path = os.path.join(CIFAR, "images", split, "%03d_%03d.png" % (fine, n))
            assert os.path.exists(path), path
            rows.append({"state": {"text": "a photograph", "image_path": path},
                         "questions": {"class": {
                             "type": "choice",
                             "instructions": "Which CIFAR-100 coarse class is shown in this photograph?",
                             "criteria": COARSE}},
                         "expected": {"class": COARSE[coarse]}})
    with open(os.path.join(CIFAR, out_name), "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    labels = sorted({r["expected"]["class"] for r in rows})
    print("%-14s %5d items, %2d distinct coarse labels: %s"
          % (out_name, len(rows), len(labels), ", ".join(labels)))


emit("train", os.path.join(DATA, "train.parquet"), PER_CLASS_TRAIN, "train20.jsonl")
emit("eval", os.path.join(DATA, "test.parquet"), PER_CLASS_EVAL, "eval20.jsonl")
