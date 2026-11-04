# Vision

`laya.vision` is the vision half of the decision system: how a state that carries a picture is
split, what the picture becomes inside the sequence, and how the block is laid out against the
option markers. It is imported from that module — `from laya.vision import parse_vision` — and
nothing here is re-exported from the top-level package, so a text-only deployment never pays for it.

`import laya.vision` does not import torch. The tower is injected rather than built, which is what
makes the module testable with a stub and lets the caller decide what to download.

The concepts, the layouts and the measured numbers are in the
[fine-tuning guide](../finetune.md#vision-teaching-a-checkpoint-to-look-at-a-picture); this page is
the surface.

## Configuration

::: laya.vision.VisionConfig

::: laya.vision.parse_vision

## The tower and the projector

::: laya.vision.VisionEncoder

::: laya.vision.PooledProjector

::: laya.vision.build_vision_encoder

::: laya.vision.load_tower

::: laya.vision.local_tower_dir

::: laya.vision.bundle_tower

## Persisting the projector

The projector is not in `DecisionModel`'s `state_dict` — a new parameter there would break
`load_state_dict(..., strict=True)` for every text checkpoint — so it travels in its own file
beside the weights, and a checkpoint that declares vision without it is refused rather than guessed
at.

::: laya.vision.save_vision_weights

::: laya.vision.load_vision_weights

::: laya.vision.VISION_WEIGHTS_FILE

## The visual block in the sequence

::: laya.vision.visual_plan

::: laya.vision.VisualPlan

::: laya.vision.assemble_inputs_embeds

::: laya.vision.collate_visual

::: laya.vision.describe_plan

## Images inside a state

::: laya.vision.split_state

::: laya.vision.has_image

::: laya.vision.load_pixels

::: laya.vision.VisualCache

## Transport limits

Checked on the header, before any decoder runs: a 40-byte PNG can declare a 30 000 × 30 000
picture, and decoding it first is a decompression bomb.

::: laya.vision.image_info

::: laya.vision.check_image_bytes

::: laya.vision.read_image_bytes

::: laya.vision.ImageRefused
