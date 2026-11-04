# CIFAR-100 as a typed multiple-choice decision

The real-data benchmark for the vision path: one `choice` question per photograph, the class names
as criteria, an uninformative state text (`"a photograph"`), and the answer carried by the pixels
only. Two cardinalities, same images — 100 fine classes and 20 coarse ones — because the two
behave differently and only the pair separates the explanations.

It exists to answer three questions no synthetic task can:

1. does the vision head work on real images, and does it beat what the frozen tower already gives
   anything else (a linear probe, the tower's own zero-shot head)?
2. do the two release gates hold on real data — **gate 1**, the image is necessary
   (`laya-evals ablate`), and **gate 2b**, option order does not move the answer (`laya-evals flip`)?
3. where does it stop working? The 20-option head learns; the 100-option head does not.

The published numbers, and the analysis behind them, live in
[`docs/vision-benchmark-results.md`](../../docs/vision-benchmark-results.md). Read that first; this
page is the recipe.

## What is here

| file | role |
|---|---|
| `_paths.py` | where the data, the base checkpoint and the reports live (repo + two env vars) |
| `cifar_prep.py` | parquet → PNGs + the 100-way JSONL splits |
| `make_coarse.py` | parquet → the 20-way JSONL splits, from the dataset's own `coarse_label` |
| `controls.py`, `controls20.py`, `controls_calib.py` | zero-shot, linear probe, mean-RGB centroid, with calibration and risk–coverage |
| `train_gpu.py` | `laya-train` with this machine's ROCm facts applied, refusing to fall back to CPU |
| `tower_patch.py` | runs SigLIP2's patch-embed conv as GEMM (MIOpen segfaults here); harness-only |
| `train.sh` | a training run: dataset, option layout, visual-token budget, seed |
| `eval_cifar.py` | accuracy, ECE, risk–coverage and both gates from the **saved checkpoint** |
| `gpuenv.sh` | `LD_LIBRARY_PATH` for `libcaffe2_nvrtc.so`, `HSA_OVERRIDE_GFX_VERSION` unset |

Nothing here is imported by the package, and no test depends on it.

## Paths

No absolute path is written into any script. `_paths.py` derives the repository from its own
location and reads two variables:

| variable | default | holds |
|---|---|---|
| `LAYA_BENCH_DATA` | `<repo>/.vision-cifar100`, else the sibling scratch `.benchprobe/` | parquet, PNGs, JSONL splits, cached SigLIP2 weights, trained checkpoints |
| `LAYA_MODELS` | `<repo>/models` | the base checkpoints `--base` starts from |

Both directories are gitignored. `python3 benchmarks/cifar100/_paths.py`-style checks are not
needed: every script prints what it resolved, and `train.sh` echoes the same three lines.

## Reproduce

```bash
# 0. one interpreter that has torch, PIL and pyarrow (a HIP build if you want the GPU path)
python3 -c "import torch; print(torch.__version__, torch.cuda.device_count())"

# 1. data: the HuggingFace parquet mirror (the .toronto.edu tarball measured ~60 KB/s)
mkdir -p "$LAYA_BENCH_DATA/data"
curl -sSLo "$LAYA_BENCH_DATA/data/train.parquet" \
  https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet
curl -sSLo "$LAYA_BENCH_DATA/data/test.parquet" \
  https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet
python3 benchmarks/cifar100/cifar_prep.py     # 100-way splits
python3 benchmarks/cifar100/make_coarse.py    # 20-way splits

# 2. controls, on the identical held-out rows
source benchmarks/cifar100/gpuenv.sh
python3 benchmarks/cifar100/controls.py
HIP_VISIBLE_DEVICES="" python3 benchmarks/cifar100/controls_calib.py
HIP_VISIBLE_DEVICES="" python3 benchmarks/cifar100/controls20.py

# 3. train (frozen tower *and* frozen text encoder): ~18 min at 20 options, ~34 min at 100
bash benchmarks/cifar100/train.sh "$LAYA_BENCH_DATA/ckpt_cifar20_frozen" 4 8 --dataset 20 --freeze-encoder
bash benchmarks/cifar100/train.sh "$LAYA_BENCH_DATA/ckpt_cifar_frozen" 3 8 --dataset 100 \
     --summary-over 0 --freeze-encoder

# 4. score the artifact -- never the trainer's console report
python3 benchmarks/cifar100/eval_cifar.py "$LAYA_BENCH_DATA/ckpt_cifar20_frozen" --data eval20.jsonl --tag _20
python3 benchmarks/cifar100/eval_cifar.py "$LAYA_BENCH_DATA/ckpt_cifar_frozen"   --data eval.jsonl   --tag _100
```

`--dataset 100` needs `--summary-over 0` today: `visual_plan` asks for one more visual token above
`summary_for_options_over=96` and `PooledProjector` does not emit it, so the run refuses with
`plan expects 17 visual tokens, got 16`. That is the `[VIS_SUMMARY]` gap
`docs/vision-benchmark-results.md` §4 describes, not a harness bug.

## Gate 1 needs `--batch 1`

`eval_cifar.py` scores the gate-1 ablation one state per forward pass by default: a batch that
mixes an image state with a text-only one raises `KeyError: 'visual'` in `collate_visual`, which
is the defect recorded in `VISION_PROGRESS.md`. The accuracy pass above it does batch images.

## This machine's GPU facts

Four obstacles, all environmental, all worked around here rather than in the package:

1. **`/dev/kfd` is invisible inside a file sandbox** (`bwrap --dev /dev` mounts a fresh `/dev`), so
   `torch.cuda.device_count()` is 0 there and 3 outside it. GPU work runs unsandboxed.
2. **`libcaffe2_nvrtc.so` is dlopened by bare name** — `gpuenv.sh` puts `torch/lib` on
   `LD_LIBRARY_PATH`.
3. **`HSA_OVERRIDE_GFX_VERSION=` forces every card to `gfx1030`**, which the gfx906 card cannot
   execute; `gpuenv.sh` unsets it.
4. **MIOpen's `Conv2d` segfaults on all three cards**, and SigLIP2's patch embedding is a conv, so
   `tower_patch.py` rebinds that conv's `forward` to `unfold` + matmul — the same arithmetic when
   kernel equals stride (checked against the real conv: max |diff| 1.1e-06).

`train_gpu.py` and `eval_cifar.py` install the patch themselves. It is an environment workaround
and stays in the harness: making the package conv-free is a decision for a machine where MIOpen
works.
