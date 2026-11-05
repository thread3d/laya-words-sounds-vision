# Changelog

All notable changes to Laya (the Python package) are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Releases are tagged `v<version>`; the language SDKs
under `sdk/`, `laya-ts/`, `laya-go/`, `laya-java/`, `laya-rust/`, `laya-cpp/` and `laya-dotnet/`
are versioned and released separately.

## [2.1.3] - 2026-11-05

The high-cardinality control, and the 50-way arm reported as it ran.

### Added

- `synth.py --classes N`: the synthetic control at 100 options, which crosses
  `summary_for_options_over` and therefore exercises the summary row. Measured: 0.1000 → **0.3000**
  held out at 6 epochs (84 updates), chance 0.0100 — the path that the vision program once shipped
  broken builds and learns here.
- The 50-way ESC-50 arm's run, reported at its real budget: 2 epochs (360 updates), accuracy
  **0.0200** against a linear probe of **0.5265** on the same frozen features. It did not learn, and
  the page says so and why, rather than dropping the arm; the number is what an untuned benchmark
  looks like, and the FSDD arm shows what the same path does given epochs.

## [2.1.2] - 2026-11-05

The audio path's first public numbers, and the two knobs that produce them.

### Added

- **The engine measured on public data.** FSDD, 2 000 training recordings and 1 000 speaker-disjoint
  held-out clips, frozen encoder: **0.7550** accuracy at 10-way after 15 epochs (the linear probe on
  the same frozen features is 0.7950), **gate 1 +0.655** — removing the recording drops accuracy to
  chance 0.1000 — and **69 ms** median CPU latency per decision (p95 111 ms). Gate 2b fails at
  **27.4 %** option-order flips on the joint layout, which is the failure the vision program measured
  at 28.3 % and fixed with `state_conditioned`.
- **`research/benchmarks/audio_esc50/synth.py`**: the two-minute control that separates "the audio
  block cannot learn" from "the head was not trained" — 64 tone recordings through the real path,
  0.5000 → 1.0000 accuracy in four epochs.

### Fixed

- Every page that said the audio capability had no public benchmark now points at the measured one:
  the README, `docs/index.md`, `docs/audio-decision-system.md` §8, and the fine-tuning guide.
- `docs/finetune.md` records what the runs taught about the knobs: the default epoch budget is too
  small on a few hundred/thousand media rows (0.1490 at 3 epochs against 0.7550 at 15, same recipe),
  and raising `head_lr` to 1e-3 makes the loss rise where the default 1e-4 descends.

## [2.1.1] - 2026-11-05

Diagnosis, not a new number: the first ESC-10 arm landed at chance and this release settles why.

### Added

- `research/benchmarks/audio_esc50/synth.py` and
  `test_the_audio_block_can_learn_a_task_end_to_end`: a synthetic two-class tone task through the
  real path, which reaches 1.0000 and proves the block carries learnable signal. The test's stub
  tower keys on the zero-crossing rate rather than amplitude, because the loader peak-normalises
  every waveform — a property worth pinning.

### Fixed

- `eval_gate.py`: the timing wrapper only knew `predict`, so a batched run reported zero decisions
  and divided by zero; an accuracy of exactly 0.0 was printed as `nan` (`x or nan` treats 0.0 as
  falsy). Both fixed; batched and single-row gates verified equal on real rows (0.0833 both ways).
- `probe.py` gained `--dataset fsdd`, stratified smoke sampling, and a question-id-agnostic reader;
  `prep_fsdd.py` and the `fsdd-seq`/`fsdd-sc` arms were added.

## [2.1.0] - 2026-11-05

The first public audio benchmark, and the three defects it found on the way. ESC-50 measured through
the shipped path: frozen wav2vec2-base probe **0.5265 ± 0.0368** over five folds (chance 0.02), and
the engine's own arms in `research/benchmarks/audio_esc50/`.

### Added

- **The ESC-50 benchmark harness** (`research/benchmarks/audio_esc50/`): `prep.py` builds Laya rows
  for all five folds (50-way and the ESC-10 subset), `probe.py` measures the control ladder
  (chance → centroid → ridge probe on cached frozen features), `train.sh` runs one engine arm
  through `laya-train --audio`, `run_engine_arms.sh` runs both arms plus their gates, and
  `eval_gate.py` reports accuracy, gate 1, gate 2b and per-decision latency.
- **`--encode-workers`** on `laya-train` (and `TrainConfig.encode_workers`,
  `laya.train.encode_media`): the media inputs of a dataset are encoded in threads instead of one at
  a time. Measured on this 56-core machine: 9.2 clips/s serially against 18.9 with eight threads,
  with the torch thread count split across the workers (N workers asking for every core measured
  *slower* than serial). The items are identical either way.
- **`--batch-size`** on `laya-evals ablate` and `flip`: the gate passes can re-answer several rows
  per forward pass, grouped by question set. A 50-option audio question costs ~4 s per decision
  singly against ~0.5 s batched, which is the difference between an hour and seven minutes for a
  400-row gate. Both passes are batched identically, so the comparison holds.
- `docs/audio-benchmark-assessment.md` (candidate datasets, the published-systems table with its
  protocols, and what a Laya number can honestly be compared to) and
  `docs/audio-benchmark-results.md` (the measured controls).

### Fixed

- **A held-out eval set of recordings was reported as overlapping its training data.** The input
  signature the trainer uses to detect overlap included the picture's content hash but not the
  recording's, and every ESC-50 row shares the state text "a recording of an environmental sound" —
  so every eval row matched every training row, `eval_mode` became `overlapping_eval`, and the run
  refused to call its own numbers held out. The signature now includes the recording.
  `tests/test_train_audio.py` holds it with a same-text/different-waveform pair.
- **`docs/evals.md` described a batching the gate did not do.** It claimed both ablation passes go
  through one `predict_batch`; the implementation answered one row at a time. The batching now
  exists (`--batch-size`) and the page describes both modes and their measurements.

## [2.0.0] - 2026-11-05

Words, sounds or vision. A state may now carry a **recording** as well as text or a picture, and a
checkpoint trained with `--audio` answers from it — through the same block, layout, projector,
transport limits and evaluation gates the picture already used.

**On the major number.** The Python API is additive: a text checkpoint and a vision checkpoint are
byte-identical after this release, and no call site has to change. The major version marks the
contract change — a *state* now has three modalities, a checkpoint config may carry an `"audio"`
block, and `MODALITIES` has a third tag — plus the internal reorganisation that made one layout
serve both. [Upgrading to 2.0](docs/upgrading.md) lists every name that moved and what a caller who
was reaching into a private helper should do instead.

### Added

- **The sound path** (`laya/audio.py`): `AudioConfig` and `parse_audio` for a checkpoint's `"audio"`
  block; `audio_plan`/`AudioPlan` for where the block sits; `AudioEncoder` with a frozen,
  **injected** tower and the trained projector; `load_tower`, `build_audio_encoder`,
  `local_tower_dir` and `bundle_tower` for offline checkpoints; `save_audio_weights` and
  `load_audio_weights` (`audio.safetensors`), with a checkpoint that declares audio and ships none
  refused rather than answered from noise.
- **Decoding and transport limits**: `load_waveform`, `decode_waveform` and `resample` (mono
  downmix, resampling to the tower's rate, peak normalisation, sample cap); `audio_info` reads
  WAV/FLAC/Ogg/MP3 headers without a decoder; `check_audio_bytes` refuses oversize, over-long or
  unmeasurable payloads on the header; `read_audio_bytes` reads bytes, a path, base64 or an
  opted-in URL; `AudioRefused` carries the reason; `AudioCache` is the content-addressed LRU.
  `AUDIO_KEYS`, `has_audio`, `split_state` and `collate_audio` are the state surface.
- **A sound in an `Agent`**: a state carrying `audio`/`audio_path`/`audio_base64`/`audio_waveform`
  is split, decoded, encoded once per distinct recording and spliced after the option head;
  `predict_batch` accepts a batch that mixes sound states with text states.
- **Routing**: `Router(audio_model="audio")` sends a state carrying a sound to the audio checkpoint,
  on the vision rule's terms — opt-in by registration, above the language rules, below
  `model=`/`task=`.
- **Training**: `TrainConfig.audio`, `laya-train --audio` (JSON block, defaults to wav2vec2-base /
  32 audio tokens / 16 kHz), `attach_audio_projector`, audio rows in `items_from_rows` /
  `eval_items_from_rows`, `encode_item(..., audio=, drop_audio=)`, and modality dropout that drops
  the recording. A checkpoint may declare both `--vision` and `--audio`.
- **Serving**: a request state may carry `audio`, `audio_path`, `audio_url` or `audio_base64`,
  resolved by the server to raw container bytes; `LAYA_MAX_AUDIO_BYTES` (12 MB),
  `LAYA_MAX_AUDIO_SECONDS` (120) and `LAYA_ALLOW_AUDIO_URLS` (off); the NixOS module exposes
  `maxAudioBytes`, `maxAudioSeconds` and `allowAudioUrls`.
- **MCP**: state validation accepts `audio_path`/`audio_url` and refuses `audio_base64` and
  `{"audio": {"base64": ...}}`, as it already did for images.
- **Gates**: `laya.evals_audio` (`audio_ablation`, `strip_audio`) and
  `laya-evals ablate --modality image|audio`. The measurement is shared with the vision gate
  (`laya.evals_media.ablation`), so a vision report and a sound report are produced by the same
  arithmetic.
- **Shared machinery**, now used by both modalities: `laya/block.py` (`BlockPlan`, `plan_block`,
  `splice_embeddings`, `reserved_positions`, `collate_block`), `laya/projector.py`
  (`PooledProjector`, `tower_width`, `truncate_tower`), `laya/transport.py` (`Refused`,
  `read_reference`, `looks_base64`) and `laya/cache.py` (`ContentCache`).
- **Packaging**: the `audio` extra (`pip install laya[audio]`, which adds `soundfile` for
  FLAC/Ogg/MP3 and float WAV; PCM WAV decodes with the standard library, and a caller with its own
  decoder passes `{"audio": {"waveform": <tensor>}}` and skips decoding entirely).
- Documentation: a [sounds reference](docs/reference/audio.md) and the
  [sound decision system](docs/audio-decision-system.md) page (contract, layout, tower knobs, data
  contract, gates, limits, non-goals, and an explicit statement of what is not yet measured),
  sound fine-tuning in the [fine-tuning guide](docs/finetune.md#sounds-teaching-a-checkpoint-to-listen),
  media gates in the [evaluation harness](docs/evals.md#media-gates-does-it-actually-look-at-the-picture-or-listen-to-the-recording),
  and the sound request surface and limits in [the HTTP API](docs/http-api.md).

### Changed

- `laya.vision.PooledProjector` is now `laya.projector.PooledProjector`. It is still importable
  from `laya.vision`, and `n_visual_tokens` is now a read-only property over `n_tokens`.
- `laya.vision`'s layout arithmetic delegates to `laya.block.plan_block`. Every public name
  (`visual_plan`, `VisualPlan`, `assemble_inputs_embeds`, `describe_plan`) keeps its signature and
  its results: `tests/test_audio.py` asserts the audio and vision plans agree number for number, and
  `tests/test_vision.py` is the no-regression net.
- `laya.vision.ImageRefused` is now a subclass of `laya.transport.Refused`. The `.reason` codes and
  `.detail` text are the ones it had; the class is still a `ValueError`, so `except ValueError` and
  `except ImageRefused` both keep working.
- `laya.train.TrainConfig.resolve_loss` takes an optional `has_audio` argument (default `False`), so
  its existing one-argument calls are unchanged.
- `laya.calibrate.MODALITIES` is `("text", "image", "audio")`. A per-modality temperature map may
  now carry an `"audio"` entry, and an unknown tag is still refused.
- The `--option-layout parallel` refusal, `fast=True` and `compile=` now name either modality: the
  reasons (masks built from token ids, the five-input TileLang forward, the compiled `input_ids`
  path) are unchanged.

### Fixed

- `laya.__version__`, `pyproject.toml` and `nix/package.nix` now agree (`tests/test_packaging.py`
  holds all three equal). They had drifted apart in the 1.1.0 tag: that release bumped
  `pyproject.toml` only, so the three reported different versions.
- `Agent._decode_answers`'s cited line range in `tests/test_confidence.py` follows the function, as
  `tests/test_doc_citations.py` requires.

### Notes

- **No audio accuracy is claimed.** There is no labelled audio benchmark in this repository yet, so
  no page quotes an accuracy figure for a sound checkpoint. The vision figures come from the
  CIFAR-100 harness in `research/benchmarks/`; the audio equivalent — a labelled sound set, both
  gates, and a published result — is the named next step.
- A state may carry a picture **or** a sound, not both, and a batch may not mix picture rows with
  sound rows: one batch is one forward pass with one spliced block per row.
- ONNX export of an audio checkpoint is not implemented (`laya/vision_onnx.py` has no audio twin).

## 1.1.x tags

The sound modality was developed and shipped as seven incremental tags, each with its own docs and
tests; 2.0.0 is the release that consolidates them.

| tag | what it added |
|---|---|
| `v1.1.0` | the audio block: config, shared layout (`laya.block`), projector, encoder, weights |
| `v1.1.1` | measurement and decoding: header readers, caps, `load_waveform`, `laya.transport`, `laya.cache`, the `audio` extra |
| `v1.1.2` | sound states in `Agent`, `collate_audio`, `MODALITY_AUDIO`, the `Router(audio_model=)` rule |
| `v1.1.3` | training: `--audio`, projector restore on resume, `audio.safetensors` + bundled tower |
| `v1.1.4` | the HTTP surface and MCP media references, the `LAYA_MAX_AUDIO_*` limits and nix options |
| `v1.1.5` | the sound-necessity gate (`laya-evals ablate --modality audio`) |
| `v1.1.6` | the capability written down: the sound decision system page, README, site navigation |
| `v1.1.7` | the shared positive-integer limit reader named for what it checks |

## [1.0.0] - 2026-11-04

The vision path released: a checkpoint trained with `--vision` answers from a picture, and the first
artifact here passed both gates that decide whether a vision head may ship — gate 1 (removing the
image drops CIFAR-100 accuracy from 0.8130 held out to chance) and gate 2b (reordering the options
moves 0.0 % of answers). [The measured results](docs/vision-benchmark-results.md) hold the tables;
`research/benchmarks/vision_cifar100/` reproduces them.

## Earlier releases

Everything before 1.0.0 is recorded in the git tags (`v0.4.x`) and their annotated messages; this
changelog starts at the release that made the multimodal path a capability rather than an
experiment.
