"""Tests for the vision foundation: checkpoint config, visual-token layout, embedding assembly.

Weight-free and tower-free by construction -- nothing here downloads SigLIP2 or needs a checkpoint
on disk, so these run in the same fast lane as `tests/test_decision_model.py`. The one model test
builds a tiny from-config BERT and asserts that the `inputs_embeds` path gives the same hidden
states as the `input_ids` path when the embeddings are the encoder's own token embeddings, which is
the property the whole vision path rests on.
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import DecisionModel
from laya.vision import (VisionConfig, assemble_inputs_embeds, parse_vision,
                         visual_plan)


# -- config ---------------------------------------------------------------------------------

def test_no_block_means_no_vision_config():
    assert parse_vision(None) is None
    assert parse_vision({}) is None
    assert parse_vision({"encoder": "x", "max_len": 512}) is None, \
        "a text checkpoint must parse to None, not to a default vision config (C6)"


def test_explicitly_disabled_block_parses_to_none():
    assert parse_vision({"vision": {"enabled": False, "tower": "x"}}) is None


def test_block_is_validated_and_defaulted():
    v = parse_vision({"vision": {"n_visual_tokens": 128}})
    assert v.n_visual_tokens == 128
    assert v.slot == "after_options"
    assert v.pooling == "adaptive_avg"
    assert "siglip2-base" in v.tower, v.tower


def test_unknown_key_is_refused_with_the_known_keys():
    try:
        parse_vision({"vision": {"n_viusal_tokens": 64}})
    except ValueError as e:
        assert "n_viusal_tokens" in str(e) and "n_visual_tokens" in str(e), e
    else:
        raise AssertionError("a typo in a vision key must not be silently ignored")


def test_bad_values_are_refused():
    for block in ({"slot": "after_state"}, {"pooling": "mean"}, {"n_visual_tokens": 0},
                  {"n_visual_tokens": -4}, {"image_size": 0}, {"keep_layers": 0}):
        try:
            parse_vision({"vision": block})
        except (ValueError, TypeError):
            continue
        raise AssertionError("block %r should have been refused" % (block,))


def test_non_mapping_block_is_refused():
    try:
        parse_vision({"vision": ["nope"]})
    except TypeError as e:
        assert "mapping" in str(e), e
    else:
        raise AssertionError("a list is not a vision block")


# -- position ids ---------------------------------------------------------------------------

def test_visual_positions_are_a_reserved_block_at_the_top():
    v = parse_vision({"vision": {"n_visual_tokens": 16}})
    pos = v.position_ids_for_visual(128)
    assert pos == list(range(112, 128)), pos


def test_visual_positions_never_collide_with_text_positions():
    v = parse_vision({"vision": {"n_visual_tokens": 64}})
    pos = set(v.position_ids_for_visual(8192))
    assert pos == set(range(8192 - 64, 8192))
    assert not pos & set(range(0, 4096)), "the reserved block must be clear of ordinary text positions"


def test_explicit_offset_is_honoured_and_bounds_checked():
    v = parse_vision({"vision": {"n_visual_tokens": 8, "position_offset": 1000}})
    assert v.position_ids_for_visual(2048) == list(range(1000, 1008))
    try:
        parse_vision({"vision": {"n_visual_tokens": 8, "position_offset": 2044}},
                     max_position_embeddings=2048)
    except ValueError as e:
        assert "positions" in str(e), e
    else:
        raise AssertionError("an offset that overruns the positional budget must be refused")


# -- layout ---------------------------------------------------------------------------------

def _plan(**block):
    v = parse_vision({"vision": dict({"n_visual_tokens": 8, "summary_for_options_over": 96}, **block)})
    return v


def test_block_lands_after_the_head_and_markers_do_not_move():
    v = _plan()
    # [CLS] q [SEP] [MASK] a [MASK] b [SEP] | state...
    markers = [3, 5]
    p = visual_plan(v, text_len=20, markers=markers, head_len=7, max_len=512,
                    option_tokens=4)
    assert p.splice == 7, p.splice
    assert p.n_visual == 8
    assert p.marker_positions == [3, 5], "markers sit before the splice, so they must not shift"
    assert p.length == 28


def test_summary_token_shifts_markers_and_sits_before_the_first_option():
    v = _plan()
    markers = [3, 5]
    p = visual_plan(v, text_len=20, markers=markers, head_len=7, max_len=512, option_tokens=400)
    assert p.summary is True
    assert p.n_visual == 9, p.n_visual            # 8 patch tokens + 1 summary
    assert p.marker_positions == [4, 6], p.marker_positions
    assert p.splice == 8, p.splice


def test_position_ids_keep_text_monotone_and_visual_reserved():
    v = _plan()
    p = visual_plan(v, text_len=20, markers=[3, 5], head_len=7, max_len=512,
                    option_tokens=4, max_position_embeddings=8192)
    assert len(p.position_ids) == 28
    text_pos = p.position_ids[:7] + p.position_ids[7 + 8:]
    assert text_pos == list(range(20)), "text keeps its own monotone numbering"
    vis_pos = p.position_ids[7:7 + 8]
    assert vis_pos == list(range(8192 - 8, 8192)), vis_pos


def test_budget_overflow_is_refused_with_the_arithmetic():
    v = _plan()
    try:
        visual_plan(v, text_len=505, markers=[3], head_len=7, max_len=512, option_tokens=4)
    except ValueError as e:
        msg = str(e)
        assert "505" in msg and "8" in msg and "512" in msg, msg
    else:
        raise AssertionError("505 + 8 > 512 must be refused, not silently truncated")


def test_head_len_cannot_exceed_the_sequence():
    v = _plan()
    for bad in (-1, 21):
        try:
            visual_plan(v, text_len=20, markers=[3], head_len=bad, max_len=512, option_tokens=4)
        except ValueError:
            continue
        raise AssertionError("head_len=%d should have been refused" % bad)


# -- assembly -------------------------------------------------------------------------------

def test_assembly_splices_visual_rows_at_the_splice_point():
    v = _plan()
    p = visual_plan(v, text_len=20, markers=[3, 5], head_len=7, max_len=512, option_tokens=4)
    tok = torch.zeros(1, 20, 4)
    tok[0, :, 0] = torch.arange(20)             # text token i carries the value i
    vis = torch.ones(1, 8, 4) * 99.0             # the visual block is unmistakable
    out = assemble_inputs_embeds(tok, vis, p)
    assert tuple(out.shape) == (1, 28, 4), out.shape
    assert out[0, :7, 0].tolist() == list(range(7))
    assert out[0, 7:15, 0].tolist() == [99.0] * 8
    assert out[0, 15:, 0].tolist() == list(range(7, 20))


def test_assembly_refuses_shape_and_width_mismatches():
    v = _plan()
    p = visual_plan(v, text_len=20, markers=[3], head_len=7, max_len=512, option_tokens=4)
    tok = torch.zeros(2, 20, 4)
    cases = [
        (tok, torch.zeros(2, 8, 4), "fine"),
        (tok, torch.zeros(1, 8, 4), "batch"),
        (tok, torch.zeros(2, 8, 5), "width"),
        (tok, torch.zeros(2, 7, 4), "plan expects"),
    ]
    for t, vis, expect in cases:
        try:
            assemble_inputs_embeds(t, vis, p)
        except ValueError as e:
            assert expect in str(e), (expect, str(e))
            continue
        assert expect == "fine", "shape %s should have been refused" % (tuple(vis.shape),)


def test_gradient_reaches_the_visual_block():
    v = _plan()
    p = visual_plan(v, text_len=12, markers=[3], head_len=6, max_len=512, option_tokens=4)
    tok = torch.zeros(1, 12, 4)
    vis = torch.zeros(1, 8, 4, requires_grad=True)
    out = assemble_inputs_embeds(tok, vis, p)
    out.sum().backward()
    assert vis.grad is not None and vis.grad.abs().sum().item() == 8 * 4, "no gradient path"


# -- the encoder accepts the spliced embeddings ---------------------------------------------

def _tiny_model() -> DecisionModel:
    cfg = AutoConfig.for_model("bert", hidden_size=16, num_hidden_layers=1, num_attention_heads=1,
                               intermediate_size=32, vocab_size=50)
    model = DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2)
    model.eval()
    return model


@torch.no_grad()
def test_inputs_embeds_path_matches_the_input_ids_path():
    """The property the vision path rests on: a pre-norm splice is a no-op.

    The embeddings passed in must be the raw *token* embeddings, not
    `encoder.embeddings(...)`, because `BertEmbeddings`/`ModernBertEmbeddings` apply their own
    LayerNorm (+dropout) to `inputs_embeds` too. Splicing the post-norm output normalises the
    sequence twice and the hidden states drift; see the test below for the size of that drift.
    """
    model = _tiny_model()
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    att = torch.ones_like(ids)
    emb = _raw_token_embeddings(model, ids)
    mpos = torch.tensor([[2, 4]])
    mmask = torch.ones(1, 2, dtype=torch.bool)
    qt = torch.tensor([0])

    a, a_act = model.forward(input_ids=ids, attention_mask=att, marker_pos=mpos,
                             marker_mask=mmask, qtype=qt)
    b, b_act = model.forward(input_ids=None, attention_mask=att, marker_pos=mpos,
                             marker_mask=mmask, qtype=qt, inputs_embeds=emb,
                             position_ids=torch.arange(ids.size(1)).unsqueeze(0))
    assert torch.equal(a, b), "a pre-norm splice must reproduce the input_ids path exactly"
    assert torch.equal(a_act, b_act)


@torch.no_grad()
def test_post_norm_embeddings_do_not_reproduce_the_input_ids_path():
    """Guards the trap above: splicing `embeddings(...)` output double-normalises.

    This test asserts the *difference* rather than equality on purpose. If a future refactor makes
    the two agree, that is not a fix -- it means the encoder stopped normalising `inputs_embeds`,
    which would be a transformers behaviour change worth noticing.
    """
    model = _tiny_model()
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    att = torch.ones_like(ids)
    mpos = torch.tensor([[2, 4]])
    mmask = torch.ones(1, 2, dtype=torch.bool)
    qt = torch.tensor([0])
    raw = _raw_token_embeddings(model, ids)
    post = model.encoder.embeddings(ids)
    a, _ = model.forward(input_ids=ids, attention_mask=att, marker_pos=mpos, marker_mask=mmask, qtype=qt)
    b, _ = model.forward(input_ids=None, attention_mask=att, marker_pos=mpos, marker_mask=mmask,
                         qtype=qt, inputs_embeds=post,
                         position_ids=torch.arange(ids.size(1)).unsqueeze(0))
    c, _ = model.forward(input_ids=None, attention_mask=att, marker_pos=mpos, marker_mask=mmask,
                         qtype=qt, inputs_embeds=raw,
                         position_ids=torch.arange(ids.size(1)).unsqueeze(0))
    assert torch.equal(a, c), "pre-norm is the correct splice space"
    assert not torch.equal(a, b), "post-norm embeddings should NOT match; if they do, see docstring"


def _raw_token_embeddings(model, ids):
    """Pre-norm token embeddings, the space the visual block must be spliced into."""
    emb = model.encoder.embeddings
    table = getattr(emb, "tok_embeddings", None)
    if table is None:
        table = emb.word_embeddings
    return table(ids)


@torch.no_grad()
def test_parallel_layout_refuses_inputs_embeds():
    model = _tiny_model()
    ids = torch.tensor([[5, 6, 7, 8]])
    try:
        model.forward(input_ids=ids, attention_mask=torch.ones_like(ids),
                      marker_pos=torch.tensor([[1]]), marker_mask=torch.ones(1, 1, dtype=torch.bool),
                      qtype=torch.tensor([0]), option_ids=torch.zeros_like(ids),
                      inputs_embeds=model.encoder.embeddings(ids))
    except ValueError as e:
        assert "parallel" in str(e), e
    else:
        raise AssertionError("the parallel layout cannot carry visual tokens and must say so")


@torch.no_grad()
def test_a_spliced_visual_block_changes_the_option_logits():
    """End to end on a tiny model: the block reaches the head, so this is not a dead code path."""
    model = _tiny_model()
    ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])
    qt = torch.tensor([0])
    mpos = torch.tensor([[2, 4]])
    mmask = torch.ones(1, 2, dtype=torch.bool)
    tok_emb = _raw_token_embeddings(model, ids)          # [1, 8, D], pre-norm
    att = torch.ones(1, ids.size(1) + 4)                 # 8 text + 4 visual

    def logits(vis_value):
        # The encoder's embedding LayerNorm removes two things that look like differences and are
        # not: a constant vector (it maps to zero) and any per-token constant *offset* (the mean is
        # subtracted). So the block has to differ in its per-feature pattern, which is also the only
        # thing a real projector can meaningfully change. Scaling the pattern is the cheapest way to
        # write that down.
        pattern = torch.arange(4 * tok_emb.size(-1), dtype=tok_emb.dtype).reshape(
            1, 4, tok_emb.size(-1)) / 7.0
        vis = pattern * (1.0 + vis_value)
        emb = assemble_inputs_embeds(tok_emb, vis, _p(splice=6, n_visual=4))
        # text keeps 0..7; the visual block takes 8..11 (a reserved region in the real layout)
        pos = torch.cat([torch.arange(6), torch.arange(8, 12), torch.arange(6, 8)]).unsqueeze(0)
        out, _ = model.forward(input_ids=None, attention_mask=att, marker_pos=mpos,
                               marker_mask=mmask, qtype=qt, inputs_embeds=emb, position_ids=pos)
        return out

    a, b = logits(0.0), logits(3.0)
    assert not torch.allclose(a, b), "changing the visual block must move the logits"
    assert a.shape == (1, 2), a.shape


def _p(splice, n_visual):
    from laya.vision import VisualPlan
    return VisualPlan(splice=splice, n_visual=n_visual, summary=False, marker_positions=[],
                      position_ids=[], length=splice + n_visual)


if __name__ == "__main__":
    test_no_block_means_no_vision_config()
    test_explicitly_disabled_block_parses_to_none()
    test_block_is_validated_and_defaulted()
    test_unknown_key_is_refused_with_the_known_keys()
    test_bad_values_are_refused()
    test_non_mapping_block_is_refused()
    test_visual_positions_are_a_reserved_block_at_the_top()
    test_visual_positions_never_collide_with_text_positions()
    test_explicit_offset_is_honoured_and_bounds_checked()
    test_block_lands_after_the_head_and_markers_do_not_move()
    test_summary_token_shifts_markers_and_sits_before_the_first_option()
    test_position_ids_keep_text_monotone_and_visual_reserved()
    test_budget_overflow_is_refused_with_the_arithmetic()
    test_head_len_cannot_exceed_the_sequence()
    test_assembly_splices_visual_rows_at_the_splice_point()
    test_assembly_refuses_shape_and_width_mismatches()
    test_gradient_reaches_the_visual_block()
    test_inputs_embeds_path_matches_the_input_ids_path()
    test_post_norm_embeddings_do_not_reproduce_the_input_ids_path()
    test_parallel_layout_refuses_inputs_embeds()
    test_a_spliced_visual_block_changes_the_option_logits()
    print("all vision foundation tests passed")
