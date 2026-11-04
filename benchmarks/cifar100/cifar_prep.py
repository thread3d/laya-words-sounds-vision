"""CIFAR-100 -> Laya JSONL, as a 100-way image choice task.

Real photographs, real public benchmark, and a task shape Laya is actually built for: one `choice`
question with 100 criteria, the answer carried by the pixels only.

    python benchmarks/cifar100/cifar_prep.py     # -> $LAYA_BENCH_DATA/cifar100/{train,eval}.jsonl

The PNG bytes in the HF parquet are the original 32x32 images; they are written through unchanged
and `laya.vision.load_pixels` does the 224 resize, so every control sees the same pixels.
"""
import json
import os

import pyarrow.parquet as pq

import _paths

DATA = _paths.PARQUET
OUT = _paths.CIFAR

# The 100 fine labels, in the official index order (`meta` file of cifar-100-python).
FINE = """apple aquarium_fish baby bear beaver bed bee beetle bicycle bottle bowl boy bridge bus
butterfly camel can castle caterpillar cattle chair chimpanzee clock cloud cockroach couch crab
crocodile cup dinosaur dolphin elephant flatfish forest fox girl hamster house kangaroo keyboard
lamp lawn_mower leopard lion lizard lobster man maple_tree motorcycle mountain mouse mushroom
oak_tree orange orchid otter palm_tree pear pickup_truck pine_tree plain plate poppy porcupine
possum rabbit raccoon ray road rocket rose sea seal shark shrew skunk skyscraper snail snake
spider squirrel streetcar sunflower sweet_pepper table tank telephone television tiger tractor
train trout tulip turtle wardrobe whale willow_tree wolf woman worm""".split()
assert len(FINE) == 100, len(FINE)

PER_CLASS_TRAIN = 100
PER_CLASS_EVAL = 50


def rows(path):
    """Every row of the split, as (png_bytes, fine_label)."""
    table = pq.read_table(path)
    out = []
    for batch in table.to_batches(max_chunksize=8192):
        cols = batch.to_pydict()
        out.extend(zip(cols["img"], cols["fine_label"]))
    return out


def write_split(name, path, per_class):
    taken = {i: 0 for i in range(100)}
    image_dir = os.path.join(OUT, "images", name)
    os.makedirs(image_dir, exist_ok=True)
    records = []
    for blob, label in rows(path):
        if taken[label] >= per_class:
            continue
        index = taken[label]
        taken[label] += 1
        image_path = os.path.join(image_dir, "%03d_%03d.png" % (label, index))
        with open(image_path, "wb") as handle:
            handle.write(blob["bytes"])
        records.append({
            "state": {"text": "a photograph", "image_path": image_path},
            "questions": {"class": {
                "type": "choice",
                "instructions": "Which CIFAR-100 class is shown in this photograph?",
                "criteria": [c.replace("_", " ") for c in FINE],
            }},
            "expected": {"class": FINE[label].replace("_", " ")},
        })
    assert all(n == per_class for n in taken.values()), [n for n in taken.values() if n != per_class]
    out_path = os.path.join(OUT, name + ".jsonl")
    with open(out_path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print("%-6s %5d items -> %s" % (name, len(records), out_path))
    return records


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    train = write_split("train", os.path.join(DATA, "train.parquet"), PER_CLASS_TRAIN)
    evaluation = write_split("eval", os.path.join(DATA, "test.parquet"), PER_CLASS_EVAL)
    labels = {r["expected"]["class"] for r in train}
    print("distinct labels:", len(labels), "| image size on disk:", os.path.getsize(train[0]["state"]["image_path"]), "bytes")
