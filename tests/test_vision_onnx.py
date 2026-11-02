"""Tests for the two-graph ONNX contract of a vision checkpoint.

The shape tests run on torch alone, because the wrappers' forward signatures *are* the graph
contract and a drift there should fail before anybody has an onnxruntime installed. The export and
parity test needs the `onnx` extra and skips without it.
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import DecisionModel
from laya.vision import build_vision_encoder, parse_vision
from laya.vision_onnx import (DECISION_INPUT_NAMES, DECISION_OUTPUT_NAMES, VISION_INPUT_NAMES,
                              VISION_OUTPUT_NAMES, VisionDecisionExport, VisionEncoderExport,
                              check_names, contract, example_decision_inputs, example_vision_inputs)


class _StubTower(torch.nn.Module):
    """A stand-in tower: [B, 3, H, W] -> 49 patches of width 32."""

    class config:
        hidden_size = 32

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 32)

    def forward(self, pixel_values):
        b = pixel_values.size(0)
        flat = pixel_values.mean(-1).mean(-1)
        return type("O", (), {"last_hidden_state": self.proj(flat).unsqueeze(1).repeat(1, 49, 1)})()


def _model(hidden=16):
    cfg = AutoConfig.for_model("bert", hidden_size=hidden, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32, vocab_size=50,
                               pad_token_id=0)
    model = DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2)
    return model.eval()


def _encoder(image_size=32, n_visual=3, hidden=16):
    vision = parse_vision({"vision": {"n_visual_tokens": n_visual, "image_size": image_size}})
    enc = build_vision_encoder(vision, hidden_size=hidden, tower=_StubTower())
    return enc.eval()


def test_the_tower_wrapper_emits_one_block_per_image():
    enc = _encoder()
    module = VisionEncoderExport(enc).as_module().eval()
    out = module(example_vision_inputs(enc.config, batch=2))
    assert tuple(out.shape) == (2, 3, 16), out.shape


def test_the_decision_wrapper_takes_embeddings_not_token_ids():
    model = _model()
    module = VisionDecisionExport(model).as_module().eval()
    feed = example_decision_inputs(16, n_visual=3)
    logits, act = module(*feed)
    assert tuple(logits.shape) == (2, 2), logits.shape
    assert act.shape[0] == 2


def test_the_example_inputs_are_varied_because_a_constant_one_carries_no_signal():
    """LayerNorm maps a constant block to zero, so a constant example traces a dead visual path."""
    pixels = example_vision_inputs(_encoder().config, batch=2)
    assert pixels.std().item() > 0, "a constant example would trace a graph with no visual signal"
    embeds = example_decision_inputs(16)[0]
    assert embeds[0].std().item() > 0


def test_the_example_decision_feed_is_ragged_past_the_first_row():
    embeds, att, pos, mpos, mmask, qtype = example_decision_inputs(16, seq_len=12, batch=2)
    assert int(att[1].sum()) < int(att[0].sum()), "row 2 must exercise the padding mask"
    assert not bool(mmask[1].all()), "row 2 must exercise a masked-off marker"


def test_the_name_table_is_the_wrapper_signature():
    """The two halves of the contract have to agree; ONNX resolves feeds by name."""
    import inspect
    sig = list(inspect.signature(
        VisionDecisionExport(_model()).as_module().forward).parameters)
    assert sig == DECISION_INPUT_NAMES, (sig, DECISION_INPUT_NAMES)
    tower_sig = list(inspect.signature(
        VisionEncoderExport(_encoder()).as_module().forward).parameters)
    assert tower_sig == VISION_INPUT_NAMES, (tower_sig, VISION_INPUT_NAMES)
    assert contract()["tower"]["outputs"] == VISION_OUTPUT_NAMES


def test_a_reordered_or_renamed_session_signature_is_refused():
    class _IO:
        def __init__(self, names):
            self.name = names

    class _Session:
        def __init__(self, ins, outs):
            self._ins, self._outs = [_IO(n) for n in ins], [_IO(n) for n in outs]

        def get_inputs(self):
            return self._ins

        def get_outputs(self):
            return self._outs

    check_names(_Session(DECISION_INPUT_NAMES, DECISION_OUTPUT_NAMES),
                DECISION_INPUT_NAMES, DECISION_OUTPUT_NAMES)
    swapped = list(DECISION_INPUT_NAMES)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    try:
        check_names(_Session(swapped, DECISION_OUTPUT_NAMES),
                    DECISION_INPUT_NAMES, DECISION_OUTPUT_NAMES)
    except ValueError as exc:
        assert "re-export" in str(exc), exc
    else:
        raise AssertionError("a reordered signature must be refused, not fed positionally")


def test_both_graphs_export_and_agree_with_torch():
    """The real thing, when the `onnx` extra is installed."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:                                     # pragma: no cover - optional extra
        print("SKIP: onnxruntime is not installed")
        return
    import tempfile

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "scripts"))
    import export_onnx

    model = _model()
    enc = _encoder()
    with tempfile.TemporaryDirectory() as tmp:
        tower, decision = export_onnx.export_vision_graphs(enc, model, tmp)
        assert os.path.exists(tower) and os.path.exists(decision)
        export_onnx.verify_vision_graphs(enc, model, tower, decision, atol=1e-3)


if __name__ == "__main__":
    test_the_tower_wrapper_emits_one_block_per_image()
    test_the_decision_wrapper_takes_embeddings_not_token_ids()
    test_the_example_inputs_are_varied_because_a_constant_one_carries_no_signal()
    test_the_example_decision_feed_is_ragged_past_the_first_row()
    test_the_name_table_is_the_wrapper_signature()
    test_a_reordered_or_renamed_session_signature_is_refused()
    test_both_graphs_export_and_agree_with_torch()
    print("all vision onnx tests passed")
