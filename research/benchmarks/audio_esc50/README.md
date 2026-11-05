# ESC-50 as a typed decision — Laya's audio head on a public benchmark

The sound path shipped without a public number. This directory produces one, in Laya's own shape:
one `choice` question per recording, the answer carried by the audio alone, measured against the
controls that make an accuracy figure mean something.

Background and the comparable-systems table: [`docs/audio-benchmark-assessment.md`](../../../docs/audio-benchmark-assessment.md).
Measured results: [`docs/audio-benchmark-results.md`](../../../docs/audio-benchmark-results.md).

## What it measures

| arm | what it is |
|---|---|
| `chance` | 1 / classes |
| `centroid` | nearest class mean of the frozen embedding (no learning) |
| `ridge` | linear probe on the frozen embedding, penalty picked by inner CV on the training fold |
| **Laya** | frozen wav2vec2-base → projector → audio block spliced after the option head, trained with `laya-train --audio` |

The two media gates run on the same rows: `ablate --modality audio` (does the answer depend on the
recording?) and `flip` (does reordering the options change it?).

## Layout

| file | role |
|---|---|
| `prep.py` | unzip ESC-50 (download if missing) and write Laya rows for all five folds: 50-way and the ESC-10 subset |
| `probe.py` | the control ladder: frozen tower features (cached on disk) → centroid and ridge probes, 5-fold |
| `train.sh` | one engine arm through `laya-train`: `--arm esc50-seq\|esc50-sc\|esc10-sc\|esc10-seq --fold N --epochs E` |
| `run_engine_arms.sh` | both engine arms back to back, each followed by its gates |
| `eval_gate.py` | accuracy, gate 1, gate 2b and per-decision latency for one checkpoint |

Data and checkpoints live in `$AUDIO_BENCH_DATA` (default `<repo>/.audio-bench`, gitignored):
616 MB of ESC-50, ~760 MB of cached frames, and the trained checkpoints.

## Reproduce

```bash
# 1. data (~616 MB download, then unpack)
python3 research/benchmarks/audio_esc50/prep.py

# 2. the controls: one frozen tower pass per clip, then the probes (CPU: ~3 min for 2 000 clips)
python3 research/benchmarks/audio_esc50/probe.py --folds 1 2 3 4 5

# 3. the engine, on a CPU: 10-way state_conditioned + both gates, then 50-way + gate 1
./research/benchmarks/audio_esc50/run_engine_arms.sh

# 4. a smoke test of either half, in under a minute
python3 research/benchmarks/audio_esc50/probe.py --limit 40 --workers 4
python3 research/benchmarks/audio_esc50/eval_gate.py --checkpoint <ckpt> --rows esc10_fold1_eval.jsonl --limit 8 --skip-flip
```

## Notes that cost time to learn

* A 50-option head renders to 150 tokens, all 50 markers on a `[MASK]`, and crosses
  `summary_for_options_over` (139 option tokens) so the audio block carries its summary row —
  the high-cardinality path the vision work fixed is exercised here.
* ESC-50 clips are 44.1 kHz mono, so every clip goes through the product's own resampling
  (`laya.audio.load_waveform`) rather than a benchmark-specific decoder.
* The tower pass is embarrassingly parallel and is where the build spends its time: 9.2 clips/s
  serially against 18.9 with eight threads on the 56-core box used here. That measurement is why
  `laya-train` grew `--encode-workers`.
* A gate at 50 options costs ~4 s per decision singly; `--batch-size` re-answers with several rows
  per forward pass and is what makes a 400-row gate take minutes rather than an hour.
