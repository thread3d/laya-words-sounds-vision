#!/usr/bin/env bash
# Train and score the engine arms back to back, on a CPU, in one background run.
#
#   ./run_engine_arms.sh [--epochs-sc 8] [--epochs-seq 3]
#
# The gate arms are the expensive part: a decision takes ~4 s singly at 50 options, so both passes
# are batched (--batch-size), which is the same arithmetic in 6-8x fewer forward passes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
export AUDIO_BENCH_DATA="${AUDIO_BENCH_DATA:-$REPO/.audio-bench}"
export HF_HOME="${HF_HOME:-$REPO/.hf-cache}"
epochs_sc=8
epochs_seq=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs-sc) epochs_sc="$2"; shift 2 ;;
    --epochs-seq) epochs_seq="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "=== [$(date +%H:%M:%S)] arm: esc10-sc (10 options, one row per option, both gates)"
"$HERE/train.sh" --arm esc10-sc --fold 1 --epochs "$epochs_sc"
"$REPO/.venv/bin/python" "$HERE/eval_gate.py" \
  --checkpoint "$AUDIO_BENCH_DATA/checkpoints/esc10-sc_fold1" \
  --rows esc10_fold1_eval.jsonl --batch-size 10 --min-gain 0.35 --max-flip-rate 0.02

echo "=== [$(date +%H:%M:%S)] arm: esc50-seq (50 options, joint head, accuracy + gate 1)"
"$HERE/train.sh" --arm esc50-seq --fold 1 --epochs "$epochs_seq"
"$REPO/.venv/bin/python" "$HERE/eval_gate.py" \
  --checkpoint "$AUDIO_BENCH_DATA/checkpoints/esc50-seq_fold1" \
  --rows esc50_fold1_eval.jsonl --batch-size 8 --min-gain 0.35 --skip-flip

echo "=== [$(date +%H:%M:%S)] done"
