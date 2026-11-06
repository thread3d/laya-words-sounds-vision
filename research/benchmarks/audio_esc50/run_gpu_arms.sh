#!/usr/bin/env bash
# The deferred benchmark arms, on the cards. Two arms in parallel, one per GPU:
#
#   * esc50 50-way, joint layout, 20 epochs -- the literature-comparable accuracy arm;
#   * FSDD 10-way, state_conditioned, 2000 recordings, 15 epochs -- the gate-2b arm on public data.
#
#   source research/benchmarks/audio_esc50/gpuenv.sh
#   ./research/benchmarks/audio_esc50/run_gpu_arms.sh [--epochs-seq 20] [--epochs-sc 15]
#
# `--encode-workers 1` on purpose: `encode_media`'s thread pool is for a CPU tower, and eight threads
# issuing GPU work just queue on the device.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
export AUDIO_BENCH_DATA="${AUDIO_BENCH_DATA:-$REPO/.audio-bench}"
export HF_HOME="${HF_HOME:-$REPO/.hf-cache}"
export DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-$HOME/miniconda3/bin/python3}"

epochs_seq=20
epochs_sc=15
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs-seq) epochs_seq="$2"; shift 2 ;;
    --epochs-sc) epochs_sc="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "=== [$(date +%H:%M:%S)] esc50 50-way, joint layout, ${epochs_seq} epochs (GPU 0)"
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 PYTHON="$PYTHON" DEVICE="$DEVICE" \
  "$HERE/train.sh" --arm esc50-seq --fold 1 --epochs "$epochs_seq" \
    -- --grad-accum 1 --micro-batch 32 --encode-workers 1 --checkpoint-every 0 \
  > "$AUDIO_BENCH_DATA/esc50-seq-gpu.log" 2>&1 &
seq_pid=$!

echo "=== [$(date +%H:%M:%S)] fsdd state_conditioned, ${epochs_sc} epochs (GPU 1)"
HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 PYTHON="$PYTHON" DEVICE="$DEVICE" \
  "$HERE/train.sh" --arm fsdd-sc --epochs "$epochs_sc" \
    -- --grad-accum 1 --micro-batch 16 --encode-workers 1 --checkpoint-every 0 \
  > "$AUDIO_BENCH_DATA/fsdd-sc-gpu.log" 2>&1 &
sc_pid=$!

wait "$seq_pid"; echo "=== [$(date +%H:%M:%S)] esc50 arm done"
wait "$sc_pid";  echo "=== [$(date +%H:%M:%S)] fsdd sc arm done"
