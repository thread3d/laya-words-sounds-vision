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
