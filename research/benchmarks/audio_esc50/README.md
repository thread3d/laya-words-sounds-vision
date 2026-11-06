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
| `zeroshot_clap.py` | the CLAP zero-shot comparator, measured locally: 0.8955 on all 2 000 clips |
| `cv_esc10.py` | the published protocol at engine scale: ESC-10 five-fold cross-validation in the shipping layout, each fold scored with both gates. ~2 minutes per epoch on one card, so a five-fold run is about two hours; the fold-level arms in the results page are what this session ran |
| `gpu_patch.py`, `train_gpu.py`, `gpuenv.sh`, `run_gpu_arms.sh` | the cards: the MIOpen conv workaround, a `laya-train` entry point that installs it, the environment, and both arms one per GPU |

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

## Running on the GPUs

This machine has 3x AMD gfx1030 (34 GB each) reached through the ROCm build in the miniconda
interpreter; the repository's `.venv` is the CPU build the CI gates use.

```bash
source research/benchmarks/audio_esc50/gpuenv.sh     # PYTHON, LD_LIBRARY_PATH, HF_HOME, HSA override
python3 research/benchmarks/audio_esc50/gpu_patch.py # tower on a card: 3 passes, prints timings
./research/benchmarks/audio_esc50/run_gpu_arms.sh    # the two arms, one per card, in parallel
```

**The conv workaround is not optional on this build.** `torch.nn.Conv1d` segfaults here — a trivial
single-layer conv on a 1-second input exits 139 before it can raise — and wav2vec2's feature
extractor is eight `Conv1d` layers, so every audio run dies before its first epoch. The same defect
is why the vision harness next door needs `tower_patch.py` for `Conv2d`; MIOpen reaches its workspace
query and dumps core, and no MIOpen knob changes it. `gpu_patch.py` evaluates each conv as
unfold + `bmm` (verified layer by layer against `conv1d` on the real weights, max |diff| ≤ 2e-6) and
`train_gpu.py` installs it before handing over to `laya-train`. Inference through `laya.load` needs
the same patch; the package does not carry it, because the defect is this machine's MIOpen and not a
property of Laya.

Measured here (one card of three, 34 GB):

| step | CPU (56 cores) | 1 GPU |
|---|---|---|
| tower, 5 s clip (after warm-up) | ~1.7 s | **26 ms** (43 clips/s batched) |
| training, 50-option row (~195 tokens) | 5.9 rows/s | **27.8-30.9 rows/s** |
| training, 10-option row (~80 tokens) | 12.7-25.4 rows/s | **60-92 rows/s** |
| training, 10-option `state_conditioned` (10 rows/item) | 42 rows/s | **145 rows/s** |
| peak VRAM, encoder + tower + batch 48 | — | 18.3 GB |

The tower gets its ~65x; the training step gets 3-5x, because what dominates there is the 335 M
encoder's forward *and* backward, not the tower.

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
