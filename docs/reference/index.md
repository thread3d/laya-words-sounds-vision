# Python API

These pages are generated from the docstrings in `laya/`, so they change with the code. Apart
from `ONNXAgent` and `laya.train`, every name below can be imported from the top-level package,
for example `from laya import Router`.

- [Agent](agent.md): `Agent` and `load` run one checkpoint; `ONNXAgent` runs an exported ONNX
  model.
- [Router](router.md): `Router` picks the checkpoint for each request; `RouteDecision` records
  the choice.
- [Helpers](helpers.md): language detection, email cleaning, question presets, shortlisting
  and calibration utilities.
- [Fine-tuning](train.md): `finetune` and `TrainConfig` from `laya.train`, imported from that
  module rather than the top-level package, and the pieces they are built from.
- [Vision](vision.md): `laya.vision` — how a state carrying an image is split, what the picture
  becomes inside the sequence, and how the block is laid out against the option markers.
- [Sounds](audio.md): `laya.audio` — how a recording is measured and decoded, what the frozen
  tower turns it into, and how the block is laid out against the option markers.
- [LangChain components](langchain.md): `LayaRouter`, `LayaGuardrail`, `LayaTriage` and
  `LayaEvaluator`.

Prediction hooks have a hand-written [API reference](../hooks/api.md) with the rest of the
[hooks guide](../hooks/index.md).
