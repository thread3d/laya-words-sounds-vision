"""Tests for the vision half of the ONNX runtime (laya/onnx_agent.py).

The runtime is numpy-only, so it cannot reuse the torch collator: it assembles `inputs_embeds`
itself, splicing the tower's output into a token-embedding lookup read straight from the
checkpoint. The parity test below is the one that matters -- the numpy assembly is fed to a real
exported decision graph and compared against the torch path on the same rows.
"""
import os
import sys
import tempfile

import numpy as np
import torch
from transformers import AutoConfig, AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from laya.common import DecisionModel
from laya.onnx_agent import (_collate_vision_items, _load_embedding_table, _vision_tower_path)
from laya.vision import build_vision_encoder, parse_vision


class _StubTower(torch.nn.Module):
    class config:
        hidden_size = 32

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 32)

    def forward(self, pixel_values):
        flat = pixel_values.mean(-1).mean(-1)
        return type("O", (), {"last_hidden_state": self.proj(flat).unsqueeze(1).repeat(1, 49, 1)})()


def _model(hidden=16):
    cfg = AutoConfig.for_model("bert", hidden_size=hidden, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32, vocab_size=50,
                               pad_token_id=0)
    return DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2).eval()


def _vision(n_visual=3, image_size=32):
    return parse_vision({"vision": {"n_visual_tokens": n_visual, "image_size": image_size}})


def _items(vision, model, n=2):
    """Two rows with a visual block, built the way `ONNXAgent._encode_state` builds them."""
    from laya.vision import VisualCache, visual_plan
    enc = build_vision_encoder(vision, hidden_size=model.encoder.config.hidden_size,
                               tower=_StubTower()).eval()
    with torch.no_grad():
        block = enc.encode(torch.zeros(1, 3, 32, 32))[0].numpy()
    items = []
    for i in range(n):
        ids = [5, 6, 7, 8, 9 + i, 10, 11]
        markers = [2, 4]
        plan = visual_plan(vision, text_len=len(ids), markers=markers, head_len=6,
                           max_len=512, max_position_embeddings=512, option_tokens=4)
        items.append({"ids": ids, "markers": plan.marker_positions, "qtype": 0,
                      "visual": block, "plan": plan,
                      "state_stats": {"state_tokens": len(ids), "state_tokens_used": len(ids),
                                      "state_tokens_dropped": 0, "truncated": False}})
    return items, VisualCache(), block


def test_the_numpy_collation_puts_the_block_at_the_splice_point():
    model = _model()
    vision = _vision()
    items, _cache, block = _items(vision, model)
    table = np.random.RandomState(0).rand(50, 16).astype(np.float32)
    batch = _collate_vision_items(items, table, 0)
    width = max(len(it["ids"]) for it in items) + vision.n_visual_tokens
    assert batch["inputs_embeds"].shape == (2, width, 16), batch["inputs_embeds"].shape
    assert batch["attention_mask"].sum() > 0
    # the block's first row lands immediately after the splice point
    splice = items[0]["plan"].splice
    assert np.allclose(batch["inputs_embeds"][0, splice:splice + 3], block, atol=1e-6)
    # and a text token before the splice still matches the table
    assert np.allclose(batch["inputs_embeds"][0, 0], table[items[0]["ids"][0]], atol=1e-6)


def test_the_numpy_collation_pads_rows_of_different_length():
    from laya.vision import visual_plan
    model = _model()
    vision = _vision()
    items, _, _ = _items(vision, model, n=2)
    # A genuinely shorter row: its plan has to be rebuilt with it, since the plan carries the
    # position ids and the splice point for exactly this sequence length.
    shorter = items[1]["ids"][:6]
    items[1]["ids"] = shorter
    items[1]["plan"] = visual_plan(vision, text_len=len(shorter),
                                   markers=items[1]["markers"], head_len=6, max_len=512,
                                   max_position_embeddings=512, option_tokens=4)
    table = np.random.RandomState(0).rand(50, 16).astype(np.float32)
    batch = _collate_vision_items(items, table, 0)
    lengths = batch["attention_mask"].sum(axis=1)
    assert lengths[1] < lengths[0], lengths
    assert batch["inputs_embeds"][1, lengths[1]:].sum() == 0, "padding must be zeros"


def test_the_numpy_assembly_agrees_with_the_torch_one():
    """The runtime's own collation, through a real exported graph, against the torch path."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:                                     # pragma: no cover - optional extra
        print("SKIP: onnxruntime is not installed")
        return

    import export_onnx
    from laya.vision import collate_visual
    from laya.vision_onnx import VisionDecisionExport

    model = _model()
    vision = _vision()
    items, _, _ = _items(vision, model)
    table = model.encoder.embeddings.word_embeddings.weight.detach().numpy().astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        decision_path = os.path.join(tmp, "decision.onnx")
        torch.onnx.export(
            VisionDecisionExport(model).as_module().eval(),
            # Batch 2, not 1: torch specializes any example dimension of extent 1 whatever the
            # dynamic_axes declaration says (the repo's TRACE_BATCH note, `scripts/export_onnx.py`),
            # and a batch-1 trace bakes a Reshape to {10,16} that a batch-2 feed cannot satisfy.
            (torch.zeros(2, 10, 16), torch.ones(2, 10, dtype=torch.long),
             torch.arange(10).unsqueeze(0).repeat(2, 1),
             torch.tensor([[2, 4], [2, 4]]), torch.ones(2, 2, dtype=torch.bool),
             torch.zeros(2, dtype=torch.long)),
            decision_path,
            input_names=["inputs_embeds", "attention_mask", "position_ids", "marker_pos",
                         "marker_mask", "qtype"],
            output_names=["logits", "act_logits"],
            opset_version=export_onnx.default_opset_version(),
            # The runtime feeds a whole batch, so the graph has to be dynamic in it -- the same
            # declaration `export_vision_graphs` ships.
            dynamic_axes={"inputs_embeds": {0: "batch_size"},
                          "attention_mask": {0: "batch_size"},
                          "position_ids": {0: "batch_size"},
                          "marker_pos": {0: "batch_size"},
                          "marker_mask": {0: "batch_size"},
                          "qtype": {0: "batch_size"},
                          "logits": {0: "batch_size"},
                          "act_logits": {0: "batch_size"}},
        )
        import onnxruntime as ort
        session = ort.InferenceSession(decision_path, providers=["CPUExecutionProvider"])

        numpy_batch = _collate_vision_items(items, table, 0)
        got_logits, _got_act = session.run(["logits", "act_logits"], numpy_batch)

        # `collate_visual` is the torch collator, so its items need torch blocks; the numpy
        # collation under test consumed the same numbers.
        torch_items = [dict(it, visual=torch.as_tensor(it["visual"])) for it in items]
        torch_batch = collate_visual(torch_items,
                                     model.encoder.embeddings.word_embeddings, 0)
        with torch.no_grad():
            want, _ = VisionDecisionExport(model).as_module().eval()(
                torch_batch["inputs_embeds"], torch_batch["attention_mask"],
                torch_batch["position_ids"], torch_batch["marker_pos"],
                torch_batch["marker_mask"], torch_batch["qtype"])
        assert np.allclose(got_logits, want.numpy(), atol=1e-4), \
            "numpy assembly disagrees with the torch one: %r" % (
                float(np.abs(got_logits - want.numpy()).max()),)


def test_a_missing_tower_graph_is_named():
    with tempfile.TemporaryDirectory() as tmp:
        decision = os.path.join(tmp, "decision.onnx")
        open(decision, "wb").close()
        try:
            _vision_tower_path(decision)
        except FileNotFoundError as exc:
            assert "tower.onnx" in str(exc) and "export_vision_graphs" in str(exc), exc
        else:
            raise AssertionError("a missing tower graph must be named, not guessed at")


def test_the_embedding_table_is_read_from_the_checkpoint():
    model_dir = os.path.join(ROOT, "models", "laya")
    if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
        print("SKIP: no local checkpoint")
        return
    table = _load_embedding_table(model_dir)
    assert table.ndim == 2 and table.shape[0] > 1000, table.shape
    assert table.dtype == np.float32


if __name__ == "__main__":
    test_the_numpy_collation_puts_the_block_at_the_splice_point()
    test_the_numpy_collation_pads_rows_of_different_length()
    test_the_numpy_assembly_agrees_with_the_torch_one()
    test_a_missing_tower_graph_is_named()
    test_the_embedding_table_is_read_from_the_checkpoint()
    print("all onnx vision runtime tests passed")
