"""ONNX shapes of a vision decision model: the two graphs, and their contracts.

A vision checkpoint exports as **two** graphs rather than one, because the pipeline is genuinely
two pipelines:

```
pixel_values --[tower + projector]--> visual_embeds
inputs_embeds + position_ids + ... --[encoder + head]--> logits, act_logits
```

Splitting there is not an optimisation detail, it is where the cost is: the tower pass is per
*image*, and every question asked about that image reuses its output. One fused graph would have
to re-run the tower for every question, which is the single largest avoidable cost in the serving
path (spec §4.6). It also means a deployment can cache `visual_embeds` across requests, or compute
them on a different device entirely.

Nothing in this module imports `onnx` or `onnxruntime`; it holds the torch wrappers whose forward
signatures *are* the graph contracts, plus the name tables both the exporter and the runtime
check against.
"""
from typing import Any, Dict, List

from .vision import VisionEncoder

# The wrapper's signature is the graph's contract, so these lists and the `forward`s below have to
# be kept in step. `check_*_contract` is what turns a drift into an error instead of a silent
# positional shuffle -- ONNX Runtime matches inputs by name, so a renamed input is a load failure,
# but a *reordered* one would be a wrong answer.
VISION_INPUT_NAMES = ["pixel_values"]
VISION_OUTPUT_NAMES = ["visual_embeds"]

DECISION_INPUT_NAMES = ["inputs_embeds", "attention_mask", "position_ids",
                        "marker_pos", "marker_mask", "qtype"]
DECISION_OUTPUT_NAMES = ["logits", "act_logits"]


def _torch():
    import torch
    return torch


class VisionEncoderExport:
    """`pixel_values [B, 3, H, W]` -> `visual_embeds [B, n_visual, hidden]`.

    Repeats `VisionEncoder.encode` without its `no_grad` guard: export already runs under
    `torch.no_grad`, and a guard inside the module body is one more thing a tracer has to reason
    about.
    """

    def __init__(self, encoder: VisionEncoder):
        self.encoder = encoder

    # torch.nn.Module is not subclassed here on purpose: the wrapper must stay importable on a
    # deployment with no torch at all (the router and `laya.serve` must not pull it in), so the
    # module is built lazily by `as_module()` when something actually exports.
    def as_module(self):
        torch = _torch()

        class _Module(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.tower = encoder.tower
                self.projector = encoder.projector
                self._mean = float(encoder.config.pixel_mean)
                self._std = float(encoder.config.pixel_std)

            def forward(self, pixel_values):
                normalised = (pixel_values - self._mean) / self._std
                patches = self.tower(pixel_values=normalised).last_hidden_state
                return self.projector(patches)

        return _Module(self.encoder)


class VisionDecisionExport:
    """`inputs_embeds` + masks -> `logits, act_logits`, over an already-spliced sequence.

    The visual block is *in* `inputs_embeds`; this graph does not know or care where. That keeps
    the exported decision graph identical to the text one apart from its first input, which is the
    reason a vision checkpoint can serve a text-only request from the same graph: the caller
    simply passes token embeddings and a `position_ids` without a reserved block.
    """

    def __init__(self, model):
        self.model = model

    def as_module(self):
        torch = _torch()

        class _Module(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, inputs_embeds, attention_mask, position_ids,
                        marker_pos, marker_mask, qtype):
                logits, act_logits = self.model(
                    None, attention_mask, marker_pos, marker_mask, qtype,
                    position_ids=position_ids, inputs_embeds=inputs_embeds)
                return logits, act_logits

        return _Module(self.model)


def example_vision_inputs(config, batch: int = 2):
    """Deterministic `pixel_values` for tracing, sized from the checkpoint's vision config."""
    torch = _torch()
    size = int(config.image_size)
    # Varied, not constant: a constant image is annihilated by the tower's own normalisation and
    # then by the encoder's embedding LayerNorm (see `laya.vision`), so a constant example would
    # trace a graph whose visual path carries no signal and still verify.
    base = torch.arange(3 * size * size, dtype=torch.float32).reshape(1, 3, size, size)
    return (base / max(1.0, base.max().item())).repeat(batch, 1, 1, 1)


def example_decision_inputs(hidden_size: int, n_visual: int = 3, seq_len: int = 12,
                            num_markers: int = 2, batch: int = 2):
    """Positional inputs for the decision graph, ragged past the first row like the text one."""
    torch = _torch()
    if num_markers >= seq_len:
        raise ValueError("num_markers=%d needs seq_len > %d" % (num_markers, seq_len))
    embeds = torch.arange(batch * seq_len * hidden_size, dtype=torch.float32)
    embeds = (embeds / max(1.0, embeds.max().item())).reshape(batch, seq_len, hidden_size)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    marker_pos = torch.arange(num_markers, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    marker_mask = torch.ones(batch, num_markers, dtype=torch.bool)
    qtype = torch.zeros(batch, dtype=torch.long)
    if batch > 1:
        attention_mask[1, -2:] = 0
        marker_mask[1, -1] = False
    return embeds, attention_mask, position_ids, marker_pos, marker_mask, qtype


def check_names(session, expected_inputs: List[str], expected_outputs: List[str],
                what: str = "graph") -> None:
    """Refuse a loaded session whose signature is not the contract above.

    ONNX Runtime resolves feeds by name, so a renamed input fails loudly at load. This is for the
    other half: a graph built by a different version of these wrappers may have the right *names*
    in a different *order*, and a caller feeding positionally would then hand the encoder a mask.
    """
    if hasattr(session, "get_inputs"):
        got_inputs = [i.name for i in session.get_inputs()]
        got_outputs = [o.name for o in session.get_outputs()]
    else:
        got_inputs = list(getattr(session, "inputs", []))
        got_outputs = list(getattr(session, "outputs", []))
    if got_inputs != list(expected_inputs):
        raise ValueError("%s inputs are %s; this runtime feeds %s -- re-export the checkpoint"
                         % (what, got_inputs, list(expected_inputs)))
    if got_outputs != list(expected_outputs):
        raise ValueError("%s outputs are %s; this runtime reads %s"
                         % (what, got_outputs, list(expected_outputs)))


def contract() -> Dict[str, Any]:
    """The two graphs' contracts, for a report or a `--dry-run` print."""
    return {"tower": {"inputs": list(VISION_INPUT_NAMES), "outputs": list(VISION_OUTPUT_NAMES)},
            "decision": {"inputs": list(DECISION_INPUT_NAMES),
                         "outputs": list(DECISION_OUTPUT_NAMES)}}
