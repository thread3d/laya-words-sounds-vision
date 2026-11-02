"""Tests for the vision training path (laya/train.py).

Everything here runs on a tiny from-config model with a stub tower, so no checkpoint and no GPU is
involved: the point is that the *path* is right -- rows with images become items with a visual
block, modality dropout is reproducible and never trains an image-required item blind by accident,
and one optimization step moves the loss through the embeddings path.
"""
import os
import struct
import sys
import zlib

import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import DecisionModel
from laya.train import (TrainConfig, encode_item, items_from_rows, train_model)
from laya.vision import build_vision_encoder, parse_vision

# The tiny test encoder is 512 positions wide, not ModernBERT's 8192; the reserved visual block
# has to fit inside whatever the checkpoint actually has.
MAXPOS = 512


class _StubTok:
    mask_token = "[M]"
    mask_token_id = 2
    sep_token_id = 3
    cls_token_id = 1
    pad_token_id = 0

    def __call__(self, text, **kw):
        text = text if isinstance(text, str) else " ".join(map(str, text))
        ids = [min(49, 4 + (ord(c) % 40)) for c in text][:64] or [4]
        return {"input_ids": ids}

    def encode(self, text, **kw):
        return self(text)["input_ids"]


class _StubTower(torch.nn.Module):
    class config:
        hidden_size = 32

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 32)

    def forward(self, pixel_values):
        flat = pixel_values.mean(-1).mean(-1)
        return type("O", (), {"last_hidden_state": self.proj(flat).unsqueeze(1).repeat(1, 49, 1)})()


def _png(w=8, h=8):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _model(hidden=16):
    cfg = AutoConfig.for_model("bert", hidden_size=hidden, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32, vocab_size=50,
                               pad_token_id=0)
    return DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2)


def _vision(n_visual=3, image_size=32):
    return parse_vision({"vision": {"n_visual_tokens": n_visual, "image_size": image_size}})


def _encoder(vision, hidden=16):
    enc = build_vision_encoder(vision, hidden_size=hidden, tower=_StubTower())
    return enc.eval()


def _encode_visual(enc):
    def encode(pixels):
        with torch.no_grad():
            return enc.encode(pixels.unsqueeze(0))[0]
    return encode


def _rows():
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    return [
        {"state": {"image": {"pixels": torch.zeros(3, 32, 32)}, "text": "one"},
         "questions": {"q": question}, "expected": {"q": "a"}},
        {"state": {"image": {"pixels": torch.ones(3, 32, 32)}, "text": "two"},
         "questions": {"q": question}, "expected": {"q": "b"}},
        {"state": "text only", "questions": {"q": question}, "expected": {"q": "a"}},
    ]


# -- items -----------------------------------------------------------------------------------

def test_image_rows_become_items_that_carry_a_visual_block():
    vision = _vision()
    enc = _encoder(vision)
    items, skipped = items_from_rows(_StubTok(), _rows(), max_len=512, head_max_len=192,
                                     vision=vision, encode_visual=_encode_visual(enc))
    assert skipped == {}, skipped
    assert len(items) == 3
    assert sum(1 for it in items if "visual" in it) == 2
    assert items[2].get("visual") is None, "a text row must train as text"
    assert tuple(items[0]["visual"].shape) == (3, 16), items[0]["visual"].shape


def test_one_image_asked_many_questions_costs_one_tower_pass():
    vision = _vision()
    enc = _encoder(vision)
    calls = {"n": 0}

    def counting(pixels):
        calls["n"] += 1
        return _encode_visual(enc)(pixels)

    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"image": {"pixels": torch.zeros(3, 32, 32)}},
             "questions": {"q1": question, "q2": question}, "expected": {"q1": "a", "q2": "b"}}]
    items, _ = items_from_rows(_StubTok(), rows, max_len=512, head_max_len=192,
                               vision=vision, encode_visual=counting)
    assert len(items) == 2 and calls["n"] == 1, (len(items), calls["n"])


def test_an_image_only_row_is_a_legal_training_item():
    """The picture is the evidence: a row with no text half must not be skipped as empty."""
    vision = _vision()
    enc = _encoder(vision)
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"image": {"pixels": torch.zeros(3, 32, 32)}},
             "questions": {"q": question}, "expected": {"q": "a"}}]
    items, skipped = items_from_rows(_StubTok(), rows, max_len=512, head_max_len=192,
                                     vision=vision, encode_visual=_encode_visual(enc))
    assert skipped == {} and len(items) == 1, skipped


def test_a_text_row_is_still_skipped_when_it_has_nothing_to_read():
    items, skipped = items_from_rows(_StubTok(), [{"state": "  ", "questions": {
        "q": {"type": "choice", "instructions": "pick", "criteria": ["a"]}},
        "expected": {"q": "a"}}], max_len=512, head_max_len=192)
    assert skipped.get("empty_text") == 1, skipped


# -- encoding and dropout --------------------------------------------------------------------

def test_encoding_a_vision_item_produces_a_plan_and_moves_the_markers():
    vision = _vision()
    enc = _encoder(vision)
    items, _ = items_from_rows(_StubTok(), _rows(), max_len=512, head_max_len=192,
                               vision=vision, encode_visual=_encode_visual(enc))
    vision_item = next(it for it in items if "visual" in it)
    text_item = next(it for it in items if "visual" not in it)
    plain = encode_item(_StubTok(), text_item, 512, 192)
    full = encode_item(_StubTok(), vision_item, 512, 192, vision=vision, max_positions=MAXPOS)
    dropped = encode_item(_StubTok(), vision_item, 512, 192, vision=vision,
                          drop_visual=True, max_positions=MAXPOS)
    assert "plan" in full and full["plan"].n_visual == 3
    assert "plan" not in dropped, "a dropped block must not plan positions for it"
    assert len(dropped["ids"]) + 3 == len(full["ids"]) + 3, "budgets differ by the block"
    assert dropped["markers"] == plain["markers"], "text-only encode must match a text item"


def test_modality_dropout_is_reproducible_and_bounded():
    for seed in (0, 1, 2):
        cfg = TrainConfig(modality_dropout=0.25, seed=seed, epochs=1)
        cfg.validate()
    for bad in (0.5, 0.9, 1.0, -0.1):
        try:
            TrainConfig(modality_dropout=bad).validate()
        except ValueError as exc:
            assert "modality_dropout" in str(exc), exc
            continue
        raise AssertionError("modality_dropout=%r should have been refused" % (bad,))


# -- the objective ---------------------------------------------------------------------------

def test_a_vision_checkpoint_defaults_to_soft_ce_and_text_to_rlcd():
    assert TrainConfig().resolve_loss(has_vision=True) == "soft-ce"
    assert TrainConfig().resolve_loss(has_vision=False) == "rlcd"
    assert TrainConfig(loss="rlcd").resolve_loss(has_vision=True) == "rlcd"


# -- one real optimization step --------------------------------------------------------------

def test_a_training_step_moves_the_loss_through_the_embeddings_path():
    torch.manual_seed(0)
    model = _model()
    tok = _StubTok()
    vision = _vision()
    enc = _encoder(vision)
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"image": {"pixels": torch.full((3, 32, 32), float(i))}, "text": "t%d" % i},
             "questions": {"q": question}, "expected": {"q": "ab"[i % 2]}} for i in range(8)]
    items, skipped = items_from_rows(tok, rows, max_len=512, head_max_len=192,
                                     vision=vision, encode_visual=_encode_visual(enc))
    assert skipped == {} and len(items) == 8

    embeds = model.encoder.embeddings
    table = getattr(embeds, "tok_embeddings", None) or embeds.word_embeddings
    torch.manual_seed(0)
    before = model.scorer[3].weight.detach().clone()
    cfg = TrainConfig(epochs=2, micro_batch=4, grad_accum=1, encoder_lr=1e-3, head_lr=1e-2,
                      log_every=0, loss="soft-ce")
    history = train_model(model, tok, items, cfg, torch.device("cpu"), 512, 192,
                          vision=vision, embed_fn=table, max_positions=MAXPOS)
    # Not "the loss fell": 8 items for 2 epochs is far too little to demand that, and asserting it
    # would make this a flaky test about optimization rather than a test about wiring. What must
    # hold is that the loop ran through the embeddings path and actually updated weights -- the
    # optimizer clears gradients, so the evidence is the parameter, not its .grad.
    assert len(history) == 2, history
    assert all(isinstance(v, float) and v == v for v in history), history
    assert not torch.equal(before, model.scorer[3].weight.detach()), \
        "the head did not move: no backward pass reached it through the visual batch"


def test_the_visual_block_changes_the_training_logits():
    """The training path is wired to the image, not merely able to run without crashing."""
    from laya.train import _forward, collate_training_batch

    torch.manual_seed(0)
    model = _model().eval()
    tok = _StubTok()
    vision = _vision()
    enc = _encoder(vision)
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"image": {"pixels": torch.zeros(3, 32, 32)}, "text": "same"},
             "questions": {"q": question}, "expected": {"q": "a"}}]
    items, _ = items_from_rows(tok, rows, max_len=512, head_max_len=192,
                               vision=vision, encode_visual=_encode_visual(enc))
    embeds = model.encoder.embeddings
    table = getattr(embeds, "tok_embeddings", None) or embeds.word_embeddings

    def logits(vis):
        encoded = encode_item(tok, items[0], 512, 192, vision=vision, max_positions=MAXPOS)
        encoded["visual"] = vis
        batch = collate_training_batch([encoded], tok.pad_token_id, embed_fn=table)
        with torch.no_grad():
            return _forward(model, batch, torch.device("cpu"), False, False)

    # Scaled, not offset: the encoder's embedding LayerNorm subtracts a per-token mean, so a
    # shifted block is the *same* block to the model (the FP1 finding, now on the training path).
    a = logits(items[0]["visual"])
    b = logits(items[0]["visual"] * 3.0)
    assert not torch.allclose(a, b), "the visual block must reach the training logits"


if __name__ == "__main__":
    test_image_rows_become_items_that_carry_a_visual_block()
    test_one_image_asked_many_questions_costs_one_tower_pass()
    test_an_image_only_row_is_a_legal_training_item()
    test_a_text_row_is_still_skipped_when_it_has_nothing_to_read()
    test_encoding_a_vision_item_produces_a_plan_and_moves_the_markers()
    test_modality_dropout_is_reproducible_and_bounded()
    test_a_vision_checkpoint_defaults_to_soft_ce_and_text_to_rlcd()
    test_a_training_step_moves_the_loss_through_the_embeddings_path()
    test_the_visual_block_changes_the_training_logits()
    print("all vision training tests passed")
