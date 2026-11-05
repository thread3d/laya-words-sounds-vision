#!/usr/bin/env bash
# The engine arms: Laya itself, trained and scored on the same rows the probe uses.
#
#   ./train.sh --arm esc50-seq --fold 1 --epochs 3     # 50 options, joint head, accuracy arm
#   ./train.sh --arm esc10-sc  --fold 1 --epochs 3     # 10 options, one row per option, gate arm
#
# Everything is CPU: the audio tower is frozen and the decision encoder is frozen too
# (`--freeze-encoder`), so the step trains the projector and the heads only -- which is the
# configuration a small-data run wants, and the only one this machine can afford.
#
# Data lives in `$AUDIO_BENCH_DATA` (default `<repo>/.audio-bench`); `prep.py` writes the rows.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
DATA_ROOT="${AUDIO_BENCH_DATA:-$REPO/.audio-bench}"
ROWS="$DATA_ROOT/esc50"
BASE="${AUDIO_BENCH_BASE:-$REPO/models/laya}"
export HF_HOME="${HF_HOME:-$REPO/.hf-cache}"

arm="esc50-seq"
fold="1"
dataset="esc50"
train_override=""
eval_override=""
epochs="3"
audio_block='{"n_audio_tokens": 32}'
layout="sequential"
extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) arm="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --train-rows) train_override="$2"; shift 2 ;;
    --eval-rows) eval_override="$2"; shift 2 ;;
    --fold) fold="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --audio-block) audio_block="$2"; shift 2 ;;
    --base) BASE="$2"; shift 2 ;;
    --micro-batch) extra+=(--micro-batch "$2"); shift 2 ;;
    --limit) extra+=(--limit "$2"); shift 2 ;;
    --) shift; extra+=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$arm" in
  esc50-seq) subset="esc50"; layout="sequential" ;;
  esc50-sc)  subset="esc50"; layout="state_conditioned" ;;
  esc10-sc)  subset="esc10"; layout="state_conditioned" ;;
  esc10-seq) subset="esc10"; layout="sequential" ;;
  fsdd-seq)  dataset="fsdd"; layout="sequential" ;;
  fsdd-sc)   dataset="fsdd"; layout="state_conditioned" ;;
  *) echo "unknown arm: $arm (esc50-seq, esc50-sc, esc10-sc, esc10-seq, fsdd-seq, fsdd-sc)" >&2
     exit 2 ;;
esac

if [[ "$dataset" == "fsdd" ]]; then
  FSDD="$DATA_ROOT/fsdd"
  train_rows="$FSDD/fsdd_train.jsonl"
  eval_rows="$FSDD/fsdd_eval.jsonl"
  out="$DATA_ROOT/checkpoints/${arm}"
else
  train_rows="$ROWS/${subset}_fold${fold}_train.jsonl"
  eval_rows="$ROWS/${subset}_fold${fold}_eval.jsonl"
  out="$DATA_ROOT/checkpoints/${arm}_fold${fold}"
fi
for path in "$train_rows" "$eval_rows"; do
  [[ -f "$path" ]] || { echo "missing $path -- run prep.py first" >&2; exit 2; }
done

# Explicit row files win over the dataset defaults, so a subset run (a smaller training set for a
# layout that costs k rows per item) is one flag rather than a copied script.
[[ -n "$train_override" ]] && train_rows="$train_override"
[[ -n "$eval_override" ]] && eval_rows="$eval_override"

echo "arm=$arm fold=$fold layout=$layout epochs=$epochs base=$BASE"
echo "train=$train_rows"
echo "eval=$eval_rows"
echo "out=$out"

"$REPO/.venv/bin/python" -m laya.train_cli \
  --data "$train_rows" \
  --base "$BASE" \
  --out "$out" \
  --audio "$audio_block" \
  --option-layout "$layout" \
  --freeze-encoder \
  --epochs "$epochs" \
  --device cpu \
  --eval "$eval_rows" \
  "${extra[@]}"
