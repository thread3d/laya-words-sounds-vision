# Sounds — the audio decision system

Laya answers typed questions about a **state**. A state is words. It may also be a picture, and
since 2.0 a sound: a recording is an input the answer can depend on, exactly as a photograph is. This page is the written-down contract for the sound path — what a recording becomes,
where it goes in the sequence, what is enforced before a decoder runs, and what has and has not been
measured.

The implementation is `laya/audio.py` (the block, the tower, the codec surface) and
`laya/block.py` / `laya/projector.py` (the layout and projector it shares with vision). The API
reference is [Sounds](reference/audio.md); the training recipe is in the
[fine-tuning guide](finetune.md#sounds-teaching-a-checkpoint-to-listen).

## 1. What a sound becomes

```
{"audio_path": "clip.wav", "text": "is this a dog barking?"}
      │
      ├─ decode ───────────► mono waveform at 16 kHz, peak-normalised      (laya.audio.load_waveform)
      │                       byte and duration caps checked on the header first
      ├─ frozen tower ─────► frame tokens [N, C]                            (laya.audio.AudioEncoder)
      ├─ projector ────────► n_audio_tokens rows in the encoder's width     (laya.projector.PooledProjector)
      └─ splice ───────────► after the option head, positions contiguous    (laya.block.plan_block)
                                  [CLS] question [MASK] opt0 [MASK] opt1 [SEP] │ audio… │ state…
```

The decision head is unchanged: it still scores `[MASK]` markers against each other. The audio block
is inserted where the option markers can attend to it, which is the whole design — a block appended
after a long state would be reachable only through the encoder's ten global layers.

**There is no speech recognition anywhere in this path.** The tower sees the waveform — prosody,
tone, a bark, a door, the words too if the tower learned them — and the head learns which of those
the options are about. That is what keeps the path non-autoregressive and fast, and it is a real
limitation: a question whose answer is *what was said* is a transcription problem, not a decision
problem, and this path is the wrong tool for it.

## 2. The contract

Seven invariants, each of which has a test:

1. **A text checkpoint is byte-identical.** A config with no `"audio"` block parses to `None` and
   every branch is skipped; nothing about a text state changes.
2. **A vision checkpoint is unaffected.** The layout is shared, not renamed: `laya.vision` keeps
   every public name it had, and `tests/test_vision.py` is the regression net.
3. **One block per state.** A state may carry a picture or a sound, not both, and a batch may not
   mix picture rows with sound rows. Both are refused by name rather than scored without one input.
4. **The decoder is optional.** PCM WAV decodes with the standard library, a compressed container
   needs `laya[audio]` (soundfile), and `{"audio": {"waveform": <tensor>}}` skips decoding entirely.
5. **Limits are enforced on the header.** Bytes and duration are refused before any decoder runs:
   an hour of 48 kHz stereo is 690 MB decoded inside a 12 MB file.
6. **The sample rate is part of the model's contract.** A 16 kHz tower fed 44.1 kHz samples hears
   the sound stretched, so decoding resamples to `sample_rate`.
7. **A checkpoint that declares audio and ships no projector is refused**, never answered from a
   randomly initialised one.

## 3. The layout

The audio block is spliced after the option head, with positions that continue the text's numbering
(`position_policy: "contiguous"`). That is not a stylistic choice: the encoder is RoPE-based with a
128-token local window in 18 of its 28 layers, so *sequence adjacency without positional adjacency*
leaves the block unreachable from the markers. The vision path measured the reserved-position
alternative at 0.708 on a probe and made contiguous the default; the audio path inherits the same
default from the same function.

Above `summary_for_options_over` rendered option tokens (default 96, roughly 30 options) the block
carries one extra row: the globally pooled frame vector. It is the only row that has seen the whole
recording, which is what a head whose last markers sit outside the local window needs.

Because the arithmetic lives in `laya.block.plan_block`, the trainer and the server cannot disagree
about where the block goes — the failure mode that would silently make a trained checkpoint worse
than no checkpoint at all.

## 4. The tower

| field | default | meaning |
|---|---|---|
| `tower` | `facebook/wav2vec2-base` | a frozen encoder that takes the waveform itself, so no second artifact (a feature extractor) has to be downloaded |
| `sample_rate` | 16000 | the rate the tower was trained on; decoding resamples to it |
| `n_audio_tokens` | 32 | rows the block costs in the sequence |
| `pooling` | `adaptive_avg` | how frames are folded to `n_audio_tokens`; `none` passes them through |
| `keep_layers` | `None` | read the frames N blocks early — the last blocks of a deep transformer tend to be worse frozen features |
| `input` | `waveform` | what the tower eats; a spectrogram-taking tower is a future value, not a current one |

The tower is **injected** into `AudioEncoder`, never loaded by it, which is what lets the whole path
be tested with a stub and lets the caller decide what to download. `laya-train` bundles the tower
into the checkpoint (`audio_tower/`), so a checkpoint answers offline.

## 5. Training data contract

A row is `{state, questions, gold|expected}` exactly as for a text fine-tune; the state may carry a
recording instead of text, or beside it.

* Recognised keys: `audio`, `audio_path`, `audio_base64`, `audio_waveform`.
* A row naming **both** a picture and a sound is skipped as `both_modalities` rather than trained as
  whichever half was parsed first.
* A recording with no text at all is a legal row: the recording is the evidence.
* A row whose text budget would be less than 16 tokens after the block is skipped as
  `audio_too_large`, mirroring the image path.
* One recording asked five questions costs one tower pass (`AudioCache`, keyed on the samples).

## 6. The gates

The sound path ships with the same two gates the vision path is held to, through the same
measurement (`laya.evals_media.ablation`):

```bash
laya-evals ablate eval.jsonl --modality audio --min-gain 0.35 --json report.json
laya-evals flip   eval.jsonl --max-flip-rate 0.02 --json report.json
```

`ablate --modality audio` removes the recording and requires accuracy to fall: a checkpoint that
barely notices is a text model with an expensive tower. `flip` permutes the options and requires the
answer not to move — a calibrated probability that changes when a caller reorders a list is not one
the caller can act on. See [the evaluation harness](evals.md#media-gates-does-it-actually-look-at-the-picture-or-listen-to-the-recording).

## 7. Limits, by layer

| layer | limit | where |
|---|---|---|
| reference | 12 MB, 120 s | `laya.audio.check_audio_bytes`, checked on the WAV/FLAC/Ogg/MP3 header |
| server | `LAYA_MAX_AUDIO_BYTES`, `LAYA_MAX_AUDIO_SECONDS` | `laya.serve`, `413` for either |
| server | `LAYA_ALLOW_AUDIO_URLS` (off) | fetching an `audio_url` is opt-in, for the image path's SSRF reason |
| MCP | references only | `audio_base64` is refused by argument validation |
| long-document scan | `max_len - n_audio_tokens` | `Agent.predict_long` sizes its windows against the block |

## 8. What is measured, and what is not

**Measured** (and re-runnable here): the layout arithmetic is shared with the vision path and
identical by construction (`tests/test_audio.py` asserts the two plans agree number for number); an
audio block changes the option logits end to end on a tiny model; a training step moves the head and
the projector through the spliced-embeddings path; the header readers, the caps and the refusals.

**Not measured, and stated as such:** there is no public-data audio benchmark in this repository
yet. No accuracy figure for a sound checkpoint appears anywhere in these docs, because there is no
run behind one. The vision page's numbers come from a CIFAR-100 harness in `research/benchmarks/`;
the audio equivalent — a labelled sound set, the two gates, and a published result — is the next
step, and until it exists the [fine-tuning guide](finetune.md#sounds-teaching-a-checkpoint-to-listen)
says so rather than borrowing the vision figures.

## 9. Non-goals

* **A state that carries both a picture and a sound.** One sequence carries one spliced block; two
  blocks would need a layout the checkpoint does not have. Refused by name.
* **Streaming audio.** A recording is decoded, measured and encoded as a whole; a live stream is a
  different product with different limits.
* **Speech recognition.** See §1.
* **Multi-channel preservation.** The decode downmixes to mono so that the number of frames a tower
  sees — and therefore the block's content — does not depend on the channel layout a caller
  recorded in.
