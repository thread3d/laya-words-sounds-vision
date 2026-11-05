# Changelog

All notable changes to Laya (the Python package) are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Releases are tagged `v<version>`; the language SDKs
under `sdk/`, `laya-ts/`, `laya-go/`, `laya-java/`, `laya-rust/`, `laya-cpp/` and `laya-dotnet/`
are versioned and released separately.

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
