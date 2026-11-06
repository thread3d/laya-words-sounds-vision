# Upgrading

This page is the per-release answer to "do I have to change anything?". The newest release comes
first and the older sections stay, because a user can be arriving from any version. Release history,
including the 2.1 tags this page does not repeat line by line, is in
[CHANGELOG.md](https://github.com/thread3d/laya-words-sounds-vision/blob/main/CHANGELOG.md).

## 3.0 — the measured release

**Nothing to change.** 2.0 added the sound modality and 2.1 measured it on public data (ESC-50, ESC-10
and FSDD, with both media gates); no call site moves for text, vision or audio, and no checkpoint
needs rebuilding. The numbers, the controls and the like-for-like caveats are in
[the audio results](audio-benchmark-results.md).

Two things worth knowing if you upgrade:

* **What the trainer calls held out.** A held-out eval set of recordings is now reported as
  `eval_mode: "held_out"` / `is_held_out: true`. Before, every recording looked like a training row —
  the overlap detector keyed on the image's content hash but not the recording's, and every ESC-50
  row shares its state text — so a run refused to call its own numbers held out. Code that keys on
  those fields for a media run will see the corrected value.
* **Three new optional flags**, all additive: `laya-train --encode-workers` (encode a media dataset's
  inputs in threads), `laya-train --checkpoint-every` (a media checkpoint is ~1.2 GB; `0` keeps only
  the final one), and `laya-evals ablate|flip --batch-size` (re-answer several rows per forward pass).

## 2.0 — a third modality

2.0 adds a third modality — a state may now carry a **sound** as well as words or a picture — and
moves the layout and projector the two media share into modules of their own. This section is the
short version for someone arriving from 1.0.x: what changes for you, what does not, and what to do
if you were reaching into something internal.

### Do I have to change anything?

**No, for a text or vision deployment.** A checkpoint whose config carries no `"audio"` block parses
to `None`, every sound branch is skipped, and a text checkpoint's sequences and a vision
checkpoint's visual blocks are byte-identical to 1.0.2 — `tests/test_vision.py` and
`tests/test_audio.py` are the nets that hold that, and they run on every change.

**Nothing at the command line.** `laya`, `laya-serve`, `laya-evals`, `laya-mcp-server` and
`laya-train` take the flags they took before. `laya-evals ablate` grows an optional
`--modality image|audio` that defaults to `image`, which is its old behaviour.

**One optional install.** Decoding a compressed recording (FLAC, Ogg, MP3, or a float WAV) needs
`soundfile`:

```bash
pip install "laya[audio]"
```

It is optional on purpose: the declared dependencies stay `torch`, `transformers`, `safetensors`,
`huggingface_hub` and `numpy`, PCM WAV decodes with the standard library, and a caller with its own
decoder passes `{"audio": {"waveform": <tensor>}}` and skips the codec entirely.

### What is new, in one table

| you want to | do this | read |
|---|---|---|
| ask a question about a recording | `agent.predict({"audio_path": "clip.wav", "text": "…"}, questions)` | [Sounds](reference/audio.md) |
| route sound traffic automatically | `Router(models={"audio": "…"})` | [Routing](routing.md#states-that-carry-a-sound) |
| train a checkpoint on recordings | `laya-train --audio '{"n_audio_tokens": 32}' …` | [Sound fine-tuning](finetune.md#sounds-teaching-a-checkpoint-to-listen) |
| send a recording over HTTP | `{"state": {"audio_base64": …, "text": …}}` | [HTTP API](http-api.md) |
| gate a checkpoint on the recording mattering | `laya-evals ablate eval.jsonl --modality audio --min-gain 0.35` | [Media gates](evals.md#media-gates-does-it-actually-look-at-the-picture-or-listen-to-the-recording) |
| understand the whole design | — | [The sound decision system](audio-decision-system.md) |

### Names that moved

The layout and the projector are now shared, so they live where both modalities can import them.
Every public name below is either still re-exported from its old home or was private:

| 1.0.2 | 2.0 | status |
|---|---|---|
| `laya.vision.PooledProjector` | `laya.projector.PooledProjector` | **still importable from `laya.vision`**; the class is the same |
| `PooledProjector.n_visual_tokens` (attribute) | `.n_tokens`, with `.n_visual_tokens` a read-only property | reading it is unchanged; assigning to it does not exist any more |
| `laya.vision.visual_plan`, `VisualPlan`, `assemble_inputs_embeds`, `describe_plan` | unchanged | signatures and results identical for the same arguments |
| `laya.vision.SLOTS` | `laya.block.SLOTS` | still importable from `laya.vision` |
| `laya.vision.ImageRefused` | `laya.transport.Refused` superclass, `ImageRefused` unchanged | `.reason` codes and `.detail` text unchanged; still a `ValueError` |
| `laya.vision._looks_base64` | `laya.transport.looks_base64` | private |
| `laya.vision._read_path`, `_decode_base64`, `_read_url` | `laya.transport.read_reference` | private; `read_image_bytes` keeps its signature, so call it |
| `laya.serve._image_limit` | `laya.serve._positive_limit` | private; it now also reads the sound limits |
| `laya.train.TrainConfig.resolve_loss(has_vision)` | `resolve_loss(has_vision, has_audio=False)` | one-argument calls unchanged |

`laya.vision`'s layout arithmetic now delegates to `laya.block.plan_block`, which is why the two
modalities cannot drift apart on the splice point. If you have a test that pins `visual_plan`'s
output for a given config, it passes unchanged.

### Behaviour that is new, and deliberately strict

Three rules were added rather than inherited, and each refuses loudly:

* **One block per state.** A state carrying both a picture and a sound raises, naming both keys.
  One sequence carries one spliced block; send two requests.
* **One modality per batch.** A `predict_batch` (or a training chunk) that mixes picture rows with
  sound rows raises too, for the same reason: one batch is one forward pass with one block per row.
* **A checkpoint must ship its projector.** A checkpoint that declares audio and carries no
  `audio.safetensors` is refused rather than answered from a random projector — the rule vision
  already had, now applied to sound.

A recording is also refused, not truncated, when it exceeds the byte or duration cap. The caps are
checked on the container header before any decoder runs (`check_audio_bytes`).

### If you are on 1.1.0 specifically

That tag bumped `pyproject.toml` without `laya.__version__` and `nix/package.nix`, so the three
disagreed. 1.1.1 brought them in step and `tests/test_packaging.py` holds them equal from then on;
upgrade past it or straight to 2.0.

### Checking your upgrade

```bash
python -c "import laya; print(laya.__version__)"        # 2.0.0
python -m pytest tests/test_vision.py tests/test_audio.py -q   # if you vendor the suites
```

A text checkpoint needs nothing more. If you own a vision checkpoint, the fastest confirmation that
nothing moved is to run your own evaluation set before and after: the before/after numbers must be
identical, because the layout, the positions and the splice are.
