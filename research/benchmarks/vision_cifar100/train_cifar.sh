#!/bin/bash
# Train Laya's vision head on CIFAR-100 as a 100-way image choice decision.
#
#   bash train_cifar.sh <out-dir> <epochs> <micro-batch> [extra laya-train flags...]
#   bash train_cifar.sh ../../../../.vision-cifar100/ckpt_cifar_frozen 3 8 --freeze-encoder
#
# `--freeze-encoder` trains only the projector and the heads (27.3 M), which is what the published
# numbers use; without it the 394 M ModernBERT encoder trains too, at ~3x the wall clock.
#
# Note `summary_for_options_over: 0`: without it a 100-option head cannot be built at all, because
# `visual_plan` asks for one more visual token than the projector emits. See section 4 of
# docs/vision-benchmark-results.md.
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
  --data "$DATA/cifar100/train.jsonl" \
  --eval "$DATA/cifar100/eval300.jsonl" \
  --base "$LAYA_MODELS/laya" \
  --out "$OUT" \
  --vision '{"n_visual_tokens":16,"summary_for_options_over":0}' \
  --option-layout sequential --loss soft-ce \
  --head-max-len 512 --max-len 640 \
  --epochs "$EPOCHS" --micro-batch "$MB" --grad-accum 2 \
  --encoder-lr 2e-5 --head-lr 1e-3 --seed 31 "$@"
