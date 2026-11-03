# CIFAR-100 as a typed decision — Laya's vision head on a public benchmark

The vision decision system's only numbers used to be synthetic: four colours on noisy backgrounds.
This directory replaces them with a real public benchmark, asked in Laya's own shape — one `choice`
question per image, the answer carried by the pixels only — and scores it against the three controls
that make the number mean something.

Full write-up: [`docs/vision-benchmark-results.md`](../../../docs/vision-benchmark-results.md).

## Result

Held-out split, 5 000 rows per column, identical rows for every arm. Chance is 1 % / 5 %.

| arm | 100-way | 20-way |
|---|---|---|
| mean-RGB centroid (300 parameters, no learning) | 4.68 % | — |
| SigLIP2 zero-shot (the tower's own head) | 73.64 % | 57.50 % |
| linear probe on the frozen embedding | 79.54 % | 84.44 % |
| **Laya's vision head** (27.3 M trained) | **1.02 %** | **87.28 %** |

At 20 options Laya beats both controls and is well calibrated (ECE 0.018, gate 1 gain +0.819), but
**fails gate 2b at 16.9 % option-order flips**. At 100 options the head collapses to the uniform
prior in four independent configurations. Both facts, and the code-level reason for the second, are
in `results.json` and the write-up.

## Layout

| file | role |
|---|---|
| `cifar_prep.py` | CIFAR-100 parquet → PNGs + Laya JSONL, 100-way |
| `make_coarse.py` | the same images with the parquet's `coarse_label`, 20-way |
| `controls.py` | embeds the data with the frozen tower; zero-shot, linear probe, centroid |
| `controls_calib.py`, `controls20.py` | the controls on the calibration and risk-coverage axes |
| `train_cifar.sh`, `train20.sh` | fine-tune through `laya-train` (GPU, unsandboxed) |
| `train_gpu.py` | `laya-train` with this machine's GPU facts applied and `tower_patch` installed |
| `eval_cifar.py` | accuracy, ECE, risk-coverage, gate 1, gate 2b — from the saved artifact |
| `tower_patch.py` | MIOpen workaround: the tower's patch-embed conv as im2col + matmul |
| `gpuenv.sh` | `LD_LIBRARY_PATH`, `HSA_OVERRIDE_GFX_VERSION` |

Data, images, weights and checkpoints live in `$VISION_CIFAR_DATA` (default
`<repo>/.vision-cifar100`, gitignored). About 18 GB with checkpoints.

## Reproduce

```bash
# 1. data: the HuggingFace mirror; the .toronto.edu tarball measured ~60 KB/s from here
mkdir -p .vision-cifar100/data
curl -sSLo .vision-cifar100/data/train.parquet \
  https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet
curl -sSLo .vision-cifar100/data/test.parquet \
  https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet

# 2. items, both cardinalities
python3 research/benchmarks/vision_cifar100/cifar_prep.py
python3 research/benchmarks/vision_cifar100/make_coarse.py

# 3. the frozen tower, its own heads (GPU; needs an unsandboxed shell, /dev/kfd)
bash -c 'source research/benchmarks/vision_cifar100/gpuenv.sh && \
         python3 research/benchmarks/vision_cifar100/controls.py'
python3 research/benchmarks/vision_cifar100/controls_calib.py   # 100-way calibration (CPU)
python3 research/benchmarks/vision_cifar100/controls20.py       # 20-way calibration

# 4. Laya
bash research/benchmarks/vision_cifar100/train20.sh \
     .vision-cifar100/ckpt_cifar20_frozen 4 8 --freeze-encoder

# 5. score the artifact (never the trainer's console report)
python3 research/benchmarks/vision_cifar100/eval_cifar.py \
        .vision-cifar100/ckpt_cifar20_frozen --data eval20.jsonl --tag _20
```

`controls.py` needs the SigLIP2 processor and tokenizer files next to the weights: the checkpoint
the repo already caches ships only `config.json` and `model.safetensors`, so
`preprocessor_config.json` and `tokenizer*.json` are fetched from
`google/siglip2-base-patch16-224` and all five files are placed in `.vision-cifar100/siglip2/`
with `model.safetensors` symlinked to the cached blob.

## Cost

One `gfx1030` (Radeon Pro W6800-class, 32 GiB), fp32, micro-batch 8 × grad-accum 2: 10 000 items ×
3 epochs at 100 options ≈ 34 min, × 4 epochs at 20 options ≈ 18 min, scoring 5 000 held-out rows
plus both gates ≈ 5 min. The frozen-patch cache is ~0.6 MB per item.

## What to know before trusting a number here

* **Single seed** except one check: 20 options × 64 visual tokens collapsed at seed 31 and learned
  normally at seed 77 (epoch means 1.185 → 0.575 → 0.426 → 0.312), so that failure was an artefact.
  The 100-way collapse reproduced in all four configurations, so that one is not seed variance.
* **`--freeze-encoder` trains 27.3 M parameters** (projector + heads); the 394 M ModernBERT encoder
  and the 92.9 M tower stay frozen. Without the flag the encoder trains too, at ~3× the wall clock,
  and it did not change the 100-way outcome.
* **The gates run one state per forward pass** here: `Agent.predict_batch` raises
  `KeyError: 'visual'` when a batch mixes an image state with a text-only one, and an image ablation
  is exactly that mixture.

## Dataset

CIFAR-100 (Krizhevsky, 2009), downloaded from the `uoft-cs/cifar100` HuggingFace mirror and used
unmodified; images are written through as the PNG bytes the parquet carries. Check the dataset's own
terms before redistributing anything derived from it.
