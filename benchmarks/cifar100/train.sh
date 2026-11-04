#!/bin/bash
# Train Laya's vision head on CIFAR-100, from a source checkout, on a GPU.
#
#   bash benchmarks/cifar100/train.sh OUT EPOCHS MICRO_BATCH [options] [laya-train flags...]
#
# Options consumed here, with their defaults:
#   --dataset 20|100     coarse (20-option) or fine (100-option) split          [20]
#   --layout NAME        sequential | parallel | state_conditioned              [sequential]
#   --visual-tokens N    n_visual_tokens in the vision block                    [16]
#   --summary-over N     summary_for_options_over                               [default]
#   --seed N             training seed                                          [31]
#
# Anything else is forwarded to `laya-train`, so `--freeze-encoder`, the learning rates and a
# larger `--max-len` all reach it. Our own defaults are emitted first, so a forwarded flag wins.
#
# `summary_for_options_over` is left at its default unless asked: at 20 options that is the
# intended path, while at 100 options the default currently *refuses* (`plan expects 17 visual
# tokens, got 16` -- the [VIS_SUMMARY] token is not implemented), so a 100-way run passes
# `--summary-over 0` until it is.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/.venv/bin/python" ]; then PYTHON="$REPO/.venv/bin/python"; else PYTHON=python3; fi
fi
# shellcheck source=gpuenv.sh
. "$HERE/gpuenv.sh"

export LAYA_MODELS="${LAYA_MODELS:-$REPO/models}"
export HF_HOME="${HF_HOME:-$REPO/.hf-cache}"
# One source of truth for where the data lives: ask the harness, do not guess here.
BENCH="$("$PYTHON" -c "import sys; sys.path.insert(0, '$HERE'); import _paths; print(_paths.CIFAR)")"

OUT="${1:?usage: train.sh OUT EPOCHS MICRO_BATCH [options]}"
EPOCHS="${2:?usage: train.sh OUT EPOCHS MICRO_BATCH [options]}"
MICRO_BATCH="${3:?usage: train.sh OUT EPOCHS MICRO_BATCH [options]}"
shift 3

DATASET=20
LAYOUT=sequential
VISUAL_TOKENS=16
SUMMARY_OVER=""
SEED=31
TRAIN_ROWS=""
PASSTHROUGH=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)        DATASET="$2"; shift 2 ;;
    --layout)         LAYOUT="$2"; shift 2 ;;
    --visual-tokens)  VISUAL_TOKENS="$2"; shift 2 ;;
    --summary-over)   SUMMARY_OVER="$2"; shift 2 ;;
    --seed)           SEED="$2"; shift 2 ;;
    --train-rows)     TRAIN_ROWS="$2"; shift 2 ;;
    *)                PASSTHROUGH+=("$1"); shift ;;
  esac
done

case "$DATASET" in
  20)  TRAIN=train20.jsonl;  FULL_EVAL=eval20.jsonl; EVAL=eval20_300.jsonl
       HEAD_MAX_LEN=256; MAX_LEN=384 ;;
  100) TRAIN=train.jsonl;    FULL_EVAL=eval.jsonl;   EVAL=eval300.jsonl
       HEAD_MAX_LEN=512; MAX_LEN=640 ;;
  *)   echo "train.sh: --dataset must be 20 or 100, got '$DATASET'" >&2; exit 2 ;;
esac

# `laya-train` evaluates on a subset while it trains; the held-out scoring is `eval_cifar.py` on
# the full split. The subset is derived rather than checked in, so it cannot go stale when the
# data moves.
if [ ! -f "$BENCH/$EVAL" ]; then
  head -n 300 "$BENCH/$FULL_EVAL" > "$BENCH/$EVAL"
  echo "train.sh: derived $EVAL (300 rows) from $FULL_EVAL"
fi

# A strided subset, for sizing a layout that costs one row per option: `state_conditioned` puts
# `k` rows in a batch where `sequential` puts one, so a 20-option run is ~20x the steps of a joint
# run on the same data. Striding (not head) keeps the class balance.
if [ -n "$TRAIN_ROWS" ]; then
  STRIDE=$(( (10000 + TRAIN_ROWS - 1) / TRAIN_ROWS ))
  SUBSET="train_rows${TRAIN_ROWS}.jsonl"
  awk -v s="$STRIDE" 'NR % s == 1' "$BENCH/$TRAIN" > "$BENCH/$SUBSET"
  echo "train.sh: derived $SUBSET ($(wc -l < "$BENCH/$SUBSET") rows, stride $STRIDE) from $TRAIN"
  TRAIN="$SUBSET"
fi

if [ -n "$SUMMARY_OVER" ]; then
  VISION="{\"n_visual_tokens\": $VISUAL_TOKENS, \"summary_for_options_over\": $SUMMARY_OVER}"
else
  VISION="{\"n_visual_tokens\": $VISUAL_TOKENS}"
fi

echo "train.sh: repo=$REPO"
echo "train.sh: data=$BENCH/$TRAIN  base=$LAYA_MODELS/laya  out=$OUT"
echo "train.sh: dataset=$DATASET layout=$LAYOUT vision=$VISION"

"$PYTHON" "$HERE/train_gpu.py" \
  --data "$BENCH/$TRAIN" \
  --eval "$BENCH/$EVAL" \
  --base "$LAYA_MODELS/laya" \
  --out "$OUT" \
  --vision "$VISION" \
  --option-layout "$LAYOUT" --loss soft-ce \
  --head-max-len "$HEAD_MAX_LEN" --max-len "$MAX_LEN" \
  --epochs "$EPOCHS" --micro-batch "$MICRO_BATCH" --grad-accum 2 \
  --encoder-lr 2e-5 --head-lr 1e-3 --seed "$SEED" "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"
