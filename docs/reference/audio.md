# Sounds

`laya.audio` is the sound half of the decision system: how a recording is measured, what the frozen
tower turns it into, and how the resulting block is laid out against the option markers. It is
imported from that module — `from laya.audio import parse_audio` — and nothing here is re-exported
from the top-level package, so a text-only deployment never pays for it.

`import laya.audio` does not import torch. The tower is injected rather than built, which is what
makes the module testable with a stub and lets the caller decide what to download.

The layout is shared with vision: `laya.block` owns where a block goes, what it costs and how it is
spliced, and both modalities delegate to it, so a vision checkpoint and an audio checkpoint cannot
disagree about the splice point the trainer used.

Nothing here is speech recognition. The tower sees the waveform; the decision head learns which
qualities of it the options are asking about. A sound, like a picture, is an input the answer
depends on, not a transcript.

## Configuration

A checkpoint opts in with an `"audio"` block in its `config.json`. A checkpoint without one parses
to `None` and the text path is byte-identical.

::: laya.audio.AudioConfig

::: laya.audio.parse_audio

## The tower and the projector

The tower is a frozen audio encoder, injected rather than loaded here. The projector is the part
that is trained, and it is the same class vision uses: `laya.projector.PooledProjector`, documented
on the [vision page](vision.md).

::: laya.audio.AudioEncoder

::: laya.audio.build_audio_encoder

::: laya.audio.load_tower

::: laya.audio.local_tower_dir

::: laya.audio.bundle_tower

::: laya.audio.tower_parameters

## Persisting the projector

The projector is not in `DecisionModel`'s `state_dict` — a new parameter there would break
`load_state_dict(..., strict=True)` for every text checkpoint — so it travels in its own file beside
the weights, and a checkpoint that declares audio without it is refused rather than guessed at.

::: laya.audio.save_audio_weights

::: laya.audio.load_audio_weights

::: laya.audio.AUDIO_WEIGHTS_FILE

## The audio block in the sequence

::: laya.audio.audio_plan

::: laya.audio.AudioPlan

::: laya.audio.describe_plan

::: laya.block.splice_embeddings

## A sound inside a state

A state carries a sound the way it carries a picture: a reference the caller names, or samples the
caller already decoded. Decoding is optional — `pip install laya[audio]` adds `soundfile` for a
compressed container, PCM WAV goes through the standard library either way, and
`{"waveform": <tensor>}` skips both.

::: laya.audio.load_waveform

::: laya.audio.decode_waveform

::: laya.audio.resample

::: laya.audio.AudioCache

## Transport limits

A recording has two costs, and only one of them is bytes: an hour of 48 kHz stereo is 690 MB decoded
inside a 12 MB compressed file. Both are answered from the container header, before any decoder
runs, which is also why they work on an install without the audio extra.

::: laya.audio.MAX_AUDIO_BYTES

::: laya.audio.MAX_AUDIO_SECONDS

A recording is fetched over the network only when the caller asks
(`read_audio_bytes(..., allow_url=True)`) or when the deployment sets `LAYA_ALLOW_AUDIO_URLS=1`
(`true`, `yes` and `on` are accepted too). It is off by default for the reason the image path gives:
a self-hosted engine that fetches whatever URL a request names is an egress proxy pointed at the
operator's own network, and only `http`/`https` are fetched even when it is on.

::: laya.audio.audio_info

::: laya.audio.check_audio_bytes

::: laya.audio.read_audio_bytes

::: laya.audio.AudioRefused

::: laya.audio.allow_audio_urls

::: laya.transport.read_reference
