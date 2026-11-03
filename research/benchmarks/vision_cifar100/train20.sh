#!/bin/bash
# Train Laya's vision head on the 20-way (coarse) CIFAR-100 task -- the cardinality that learns.
#
#   bash train20.sh <out-dir> <epochs> <micro-batch> [extra laya-train flags...]
#   bash train20.sh ../../../../.vision-cifar100/ckpt_cifar20_frozen 4 8 --freeze-encoder
#
# The vision block is left at its defaults here: 20 options stay under `summary_for_options_over`
# (96 rendered option tokens), so this script exercises the intended path. A 64-token variant is
# one edit away (`n_visual_tokens`); it did not change the outcome in the runs recorded in
# results.json.
set -u
BENCH=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$BENCH/../../.." && pwd)
cd "$REPO"
source "$BENCH/gpuenv.sh"
DATA="${VISION_CIFAR_DATA:-$REPO/.vision-cifar100}"
export HF_HOME="${HF_HOME:-$DATA/hf}"
export LAYA_MODELS="${LAYA_MODELS:-$REPO/models}"

OUT=$1; EPOCHS=$2; MB=$3; shift 3
python3 "$BENCH/train_gpu.py" \
  --data "$DATA/cifar100/train20.jsonl" \
  --eval "$DATA/cifar100/eval20_300.jsonl" \
  --base "$LAYA_MODELS/laya" \
  --out "$OUT" \
  --vision '{"n_visual_tokens":16}' \
  --option-layout sequential --loss soft-ce \
  --head-max-len 256 --max-len 384 \
  --epochs "$EPOCHS" --micro-batch "$MB" --grad-accum 2 \
  --encoder-lr 2e-5 --head-lr 1e-3 --seed 31 "$@"
