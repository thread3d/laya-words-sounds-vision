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
from laya.vision import (PooledProjector, VisionConfig, assemble_inputs_embeds,
                         build_vision_encoder, parse_vision, visual_plan)


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


# -- tower + projector ----------------------------------------------------------------------

class _StubTower:
    """A stand-in for SigLIP2: [B, 3, H, W] -> last_hidden_state [B, N, C].

    Injected rather than downloaded so these tests stay weight-free and offline, which is also why
    `VisionEncoder` takes a tower instead of building one.
    """

    class config:
        hidden_size = 32

    def __init__(self, n_patches=196):
        self.n_patches = n_patches
        self.trained = True
        self.seen = []

    def parameters(self):
        return []

    def __call__(self, pixel_values):
        self.seen.append(tuple(pixel_values.shape))
        b = pixel_values.size(0)
        patches = torch.arange(b * self.n_patches * 32, dtype=pixel_values.dtype).reshape(
            b, self.n_patches, 32) / 1000.0
        return type("Out", (), {"last_hidden_state": patches})()

    def train(self, mode=True):
        self.trained = mode

    def eval(self):
        self.trained = False


def test_projector_compresses_and_projects():
    proj = PooledProjector(32, 16, 8, pooling="adaptive_avg")
    out = proj(torch.randn(2, 196, 32))
    assert tuple(out.shape) == (2, 8, 16), out.shape


def test_projector_passthrough_keeps_every_patch():
    proj = PooledProjector(32, 16, 196, pooling="none")
    out = proj(torch.randn(1, 196, 32))
    assert tuple(out.shape) == (1, 196, 16), out.shape


def test_projector_emits_a_varied_pattern_not_a_constant():
    """LayerNorm removes a constant row, so a projector must not collapse to one."""
    proj = PooledProjector(32, 16, 8, pooling="adaptive_avg")
    out = proj(torch.randn(1, 196, 32))
    assert out[0].std(dim=-1).min().item() > 1e-6, "rows collapsed to a constant"
    assert not torch.allclose(out[0, 0], out[0, 1]), "visual tokens are indistinguishable"


def test_type_vector_is_learnable_and_applied_after_the_linear():
    proj = PooledProjector(32, 16, 4, pooling="adaptive_avg")
    assert any(p is proj.type_vector for p in proj.parameters())
    with torch.no_grad():
        before = proj(torch.zeros(1, 32, 32))
        proj.type_vector.add_(1.0)
        after = proj(torch.zeros(1, 32, 32))
    assert torch.allclose(after - before, torch.ones(1, 4, 16)), "type vector not applied"


def test_encoder_normalises_pixels_the_way_the_tower_was_trained():
    """[0,1] pixels must arrive at the tower as (x - 0.5) / 0.5, not raw."""
    seen = []

    class _Spy(_StubTower):
        def __call__(self, pixel_values):
            seen.append((round(pixel_values.min().item(), 3), round(pixel_values.max().item(), 3)))
            return _StubTower.__call__(self, pixel_values)

    v = parse_vision({"vision": {"n_visual_tokens": 4}})
    enc = build_vision_encoder(v, hidden_size=16, tower=_Spy())
    enc.encode(torch.zeros(1, 3, 224, 224))
    enc.encode(torch.ones(1, 3, 224, 224))
    assert seen == [(-1.0, -1.0), (1.0, 1.0)], seen


def test_encoder_produces_the_plan_sized_block():
    v = parse_vision({"vision": {"n_visual_tokens": 8}})
    enc = build_vision_encoder(v, hidden_size=16, tower=_StubTower())
    out = enc.encode(torch.zeros(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 8, 16), out.shape
    assert enc.tower.seen[-1] == (2, 3, 224, 224), enc.tower.seen


def test_encoder_gradient_reaches_the_projector_not_the_tower():
    v = parse_vision({"vision": {"n_visual_tokens": 4}})
    enc = build_vision_encoder(v, hidden_size=16, tower=_StubTower())
    out = enc.encode(torch.zeros(1, 3, 224, 224))
    out.sum().backward()
    assert enc.projector.proj.weight.grad is not None
    assert enc.projector.proj.weight.grad.abs().sum().item() > 0


def test_encoder_train_eval_toggles_the_tower():
    v = parse_vision({"vision": {"n_visual_tokens": 4}})
    enc = build_vision_encoder(v, hidden_size=16, tower=_StubTower())
    enc.eval()
    assert enc.tower.trained is False
    enc.train(True)
    assert enc.tower.trained is True


def test_truncating_a_tower_validates_the_layer_count():
    import torch.nn as nn
    from laya.vision import _truncate_tower

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)

    class _Enc(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList([_Layer() for _ in range(n)])

    class _Tower(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.vision_model = nn.Module()
            self.vision_model.encoder = _Enc(n)

    t = _Tower(6)
    _truncate_tower(t, 4)
    assert len(t.vision_model.encoder.layers) == 4
    for bad in (0, 7):
        try:
            _truncate_tower(_Tower(6), bad)
        except ValueError as e:
            assert "out of range" in str(e), e
            continue
        raise AssertionError("keep_layers=%d should have been refused" % bad)


def test_a_vision_encoder_feeds_the_decision_model_end_to_end():
    """The whole path on a tiny model: pixels -> tower -> projector -> sequence -> option logits."""
    model = _tiny_model()
    v = parse_vision({"vision": {"n_visual_tokens": 4, "summary_for_options_over": 0}})
    enc = build_vision_encoder(v, hidden_size=model.encoder.config.hidden_size, tower=_StubTower())
    ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])
    head_len = 6
    plan = visual_plan(v, text_len=ids.size(1), markers=[2, 4], head_len=head_len, max_len=512,
                       max_position_embeddings=512, option_tokens=6)
    tok_emb = _raw_token_embeddings(model, ids)
    vis = enc.encode(torch.zeros(1, 3, 224, 224))
    emb = assemble_inputs_embeds(tok_emb, vis, plan)
    att = torch.ones(1, plan.length)
    with torch.no_grad():
        logits, _ = model.forward(input_ids=None, attention_mask=att,
                                  marker_pos=torch.tensor([plan.marker_positions]),
                                  marker_mask=torch.ones(1, 2, dtype=torch.bool),
                                  qtype=torch.tensor([0]), inputs_embeds=emb,
                                  position_ids=torch.tensor([plan.position_ids]))
    assert tuple(logits.shape) == (1, 2), logits.shape


# -- an Agent that can see ----------------------------------------------------------------

def _vision_agent():
    """A tiny Agent built without a checkpoint, with a stub tower.

    `Agent.__new__` skips the disk/weights path while keeping every attribute the vision code
    touches, so this exercises the real `_encode_state` -> `collate_visual` -> `_infer` plumbing
    on CPU with no checkpoint and no tower download.
    """
    from laya.agent import Agent
    from laya.common import build_sequence as _bs  # noqa: F401  (import kept for parity)

    model = _tiny_model()
    agent = Agent.__new__(Agent)
    agent.model = model
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.amp_enabled = False
    agent._fast = None
    agent._compiled = False
    agent._reduce_overhead = False
    agent._amp_failures = 0
    agent.cpu_fallback_count = 0
    agent.last_fallback_reason = None
    agent.parallel_options = False
    agent.tok = _stub_tokenizer()
    agent.cfg = {"max_len": 512, "head_max_len": 192}
    agent.model_id = "test-vision"
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.lang_temperatures = {}
    agent.binning_map = None
    from laya.vision import VisionConfig, VisualCache
    agent.vision = VisionConfig({"n_visual_tokens": 4, "summary_for_options_over": 0})
    agent._visual_cache = VisualCache(8)
    agent.vision_encoder = build_vision_encoder(
        agent.vision, hidden_size=model.encoder.config.hidden_size, tower=_StubTower())
    agent.vision_encoder.eval()
    return agent


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

    def __getitem__(self, k):
        return self.__dict__[k]


def _stub_tokenizer():
    return _StubTok()


def test_agent_runs_a_state_without_an_image():
    agent = _vision_agent()
    items = agent._encode_state("plain text state", ["q"],
                                {"q": agent._to_internal(
                                    {"type": "choice", "instructions": "pick",
                                     "criteria": ["a", "b"]})})
    assert "visual" not in items[0], "a text state must not grow a visual block"
    assert items[0]["ids"], "text state still tokenizes"


def test_agent_attaches_a_visual_block_and_shifts_nothing_else():
    agent = _vision_agent()
    internal = {"q": agent._to_internal(
        {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]})}
    text_items = agent._encode_state("state", ["q"], internal)
    img_items = agent._encode_state({"text": "state", "image": {"pixels": torch.zeros(3, 224, 224)}},
                                    ["q"], internal)
    assert "visual" in img_items[0]
    assert tuple(img_items[0]["visual"].shape) == (4, agent.model.encoder.config.hidden_size)
    assert img_items[0]["plan"].n_visual == 4
    assert img_items[0]["markers"] == text_items[0]["markers"], "no summary token -> markers hold"


def test_the_tower_runs_once_for_many_questions_on_one_image():
    agent = _vision_agent()
    internal = {q: agent._to_internal({"type": "choice", "instructions": "pick",
                                       "criteria": ["a", "b"]}) for q in ("q1", "q2", "q3")}
    before = agent._visual_cache.misses
    agent._encode_state({"image": {"pixels": torch.zeros(3, 224, 224)}}, ["q1", "q2", "q3"], internal)
    assert agent._visual_cache.misses - before == 1, "one image, one tower pass (spec 4.6)"
    agent._encode_state({"image": {"pixels": torch.zeros(3, 224, 224)}}, ["q1"], internal)
    assert agent._visual_cache.hits == 1, "the same pixels must hit the cache"


def test_visual_collate_builds_inputs_embeds():
    agent = _vision_agent()
    internal = {"qa": agent._to_internal({"type": "choice", "instructions": "pick",
                                          "criteria": ["a", "b"]}),
                "qb": agent._to_internal({"type": "choice", "instructions": "pick",
                                          "criteria": ["a", "b", "c"]})}
    items = agent._encode_state({"text": "hello", "image": {"pixels": torch.zeros(3, 224, 224)}},
                                ["qa", "qb"], internal)
    from laya.vision import collate_visual
    b = collate_visual(items, agent._token_embeddings, agent.tok.pad_token_id)
    n_vis = items[0]["plan"].n_visual
    assert "inputs_embeds" in b and "input_ids" not in b
    assert b["inputs_embeds"].shape[0] == 2
    assert b["inputs_embeds"].shape[1] == max(len(it["ids"]) for it in items) + n_vis
    # Ragged option counts pad the marker mask, so compare per row rather than asserting all(True).
    per_row = [int(m.sum()) for m in b["marker_mask"]]
    assert per_row == [len(it["markers"]) for it in items], per_row
    assert b["attention_mask"].sum() <= b["attention_mask"].numel()


def test_the_batch_forward_runs_and_answers():
    agent = _vision_agent()
    internal = {"q": agent._to_internal({"type": "choice", "instructions": "pick",
                                         "criteria": ["a", "b"]})}
    items = agent._encode_state({"image": {"pixels": torch.zeros(3, 224, 224)}}, ["q"], internal)
    from laya.vision import collate_visual
    b = collate_visual(items, agent._token_embeddings, agent.tok.pad_token_id)
    with torch.no_grad():
        logits, act = agent._infer(b)
    assert tuple(logits.shape) == (1, 2), logits.shape
    assert act.shape[0] == 1


def test_a_vision_checkpoint_refuses_the_fast_backend_message():
    """The guard lives in Agent.__init__; assert the message a caller would see."""
    import inspect
    from laya import agent as agent_mod
    src = inspect.getsource(agent_mod.Agent.__init__)
    assert "cannot use fast=True" in src, "the fast path has no visual-token support"


def test_split_state_keeps_the_other_keys_as_data():
    from laya.vision import split_state
    text, spec = split_state({"image": "x.png", "vendor": "acme"})
    assert spec == "x.png"
    assert text == {"vendor": "acme"}, "non-image keys are still state"


# -- calibration on the modality axis (spec 7.4 / 8) ---------------------------------------

def _records(n=5200, seed=0):
    """Enough records that each *channel* clears MIN_BUCKET_N (2000), not just the pooled bucket.

    That is the real cost of the modality axis: an image bucket reaches the floor only after the
    pooled bucket did, at roughly twice the records, because each channel sees half of them.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    recs, mods = [], []
    for i in range(n):
        k = 4
        z = rng.randn(k) * 4.0
        y = np.zeros(k)
        y[i % k] = 1.0
        recs.append((0, z, y))
        mods.append("image" if i % 2 else "text")
    return recs, mods


def test_a_text_only_fit_gains_no_modality_key():
    from laya.calibrate import fit_temperature_map
    recs, _ = _records()
    out = fit_temperature_map(recs)
    assert "temperature_by_modality" not in out, "a text fit must not change shape (C6)"
    assert "temperature_by_options" in out


def test_the_modality_fit_refines_per_channel_and_leaves_the_pooled_map_alone():
    from laya.calibrate import fit_temperature_map
    recs, mods = _records()
    pooled = fit_temperature_map(recs)
    split = fit_temperature_map(recs, modalities=mods)
    assert split["temperature"] == pooled["temperature"]
    assert split["temperature_by_options"] == pooled["temperature_by_options"], \
        "the pooled map is the refinement's parent, not its replacement"
    by_mod = split["temperature_by_modality"]
    assert set(by_mod) == {"text", "image"}, by_mod
    assert "choice:3-5" in by_mod["image"], by_mod["image"]
    assert split["n_by_modality_bucket"]["image"]["choice:3-5"] == 2600


def test_a_small_channel_bucket_is_omitted_not_guessed():
    """The MIN_BUCKET_N floor applies per (modality, bucket), exactly as per bucket."""
    from laya.calibrate import fit_temperature_map, MIN_BUCKET_N
    recs, mods = _records(n=MIN_BUCKET_N // 2)
    out = fit_temperature_map(recs, modalities=mods)
    assert out["temperature_by_modality"]["image"] == {}, \
        "below the floor the channel must contribute nothing, not a noisy fit"


def test_unknown_modality_tags_are_refused():
    from laya.calibrate import fit_temperature_map
    recs, mods = _records(n=40)
    bad = list(mods)
    bad[0] = "video"
    try:
        fit_temperature_map(recs, modalities=bad)
    except ValueError as e:
        assert "modality" in str(e), e
    else:
        raise AssertionError("an unknown modality must be refused")
    try:
        fit_temperature_map(recs, modalities=mods[:-1])
    except ValueError as e:
        assert "entries" in str(e), e
    else:
        raise AssertionError("a short modalities sequence must be refused")


def test_modality_map_round_trips_through_the_calibration_file():
    from laya.calibrate import apply_calibration_payload, calibration_payload, fit_temperature_map

    class _Obj:
        temperature = None
        temperature_by_options = None
        temperature_by_modality = None

    recs, mods = _records()
    fit = fit_temperature_map(recs, modalities=mods)
    payload = calibration_payload(fit["temperature"], fit["temperature_by_options"],
                                  temperature_by_modality=fit["temperature_by_modality"])
    obj = _Obj()
    apply_calibration_payload(obj, payload)
    assert obj.temperature_by_modality == fit["temperature_by_modality"], obj.temperature_by_modality

    text_payload = calibration_payload(fit["temperature"], fit["temperature_by_options"])
    assert "temperature_by_modality" not in text_payload, "a text payload keeps its keys (C6)"
    obj2 = _Obj()
    apply_calibration_payload(obj2, text_payload)
    assert obj2.temperature_by_modality is None, "an absent key must not clear or invent a map"


def test_a_bad_modality_payload_is_refused_by_name():
    from laya.calibrate import apply_calibration_payload

    class _Obj:
        pass

    base = {"version": 2, "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}
    for bad, expect in (({"image": [1.0]}, "object of"), ({"video": {}}, "known modalities")):
        try:
            apply_calibration_payload(_Obj(), dict(base, temperature_by_modality=bad))
        except ValueError as e:
            assert expect in str(e), (expect, str(e))
            continue
        raise AssertionError("temperature_by_modality=%r should have been refused" % (bad,))


def test_decode_uses_the_modality_temperature_for_image_rows_only():
    agent = _vision_agent()
    internal = {"q": agent._to_internal({"type": "choice", "instructions": "pick",
                                         "criteria": ["a", "b"]})}
    agent.temperature_by_options = {"choice:2": 2.0}
    agent.temperature_by_modality = {"image": {"choice:2": 4.0}}
    img = agent._encode_state({"image": {"pixels": torch.zeros(3, 224, 224)}}, ["q"], internal)
    txt = agent._encode_state("plain", ["q"], internal)
    assert img[0]["modality"] == "image" and txt[0]["modality"] == "text"

    import numpy as np
    logits = np.array([[2.0, 0.0]], dtype=np.float32)
    act = np.array([[0.0, 1.0]], dtype=np.float32)
    img_ans = agent._decode_answers(logits, act, img, ["q"], internal, 0)
    txt_ans = agent._decode_answers(logits, act, txt, ["q"], internal, 0)
    # a larger temperature flattens the distribution: p(a) moves toward 0.5.
    # softmax([2, 0] / 2) = 0.7311 and softmax([2, 0] / 4) = 0.6225.
    assert img_ans["q"]["probabilities"]["a"] < txt_ans["q"]["probabilities"]["a"]
    assert abs(txt_ans["q"]["probabilities"]["a"] - 0.7311) < 1e-3, txt_ans["q"]
    assert abs(img_ans["q"]["probabilities"]["a"] - 0.6225) < 1e-3, img_ans["q"]


def test_an_empty_modality_map_leaves_text_confidences_untouched():
    """The C6 guarantee at the decode site: no map, byte-identical probabilities."""
    agent = _vision_agent()
    internal = {"q": agent._to_internal({"type": "choice", "instructions": "pick",
                                         "criteria": ["a", "b"]})}
    items = agent._encode_state("plain", ["q"], internal)
    import numpy as np
    logits = np.array([[2.0, 0.0]], dtype=np.float32)
    act = np.array([[0.0, 1.0]], dtype=np.float32)
    before = agent._decode_answers(logits, act, items, ["q"], internal, 0)
    agent.temperature_by_modality = {}
    after = agent._decode_answers(logits, act, items, ["q"], internal, 0)
    assert before == after


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
    test_projector_compresses_and_projects()
    test_projector_passthrough_keeps_every_patch()
    test_projector_emits_a_varied_pattern_not_a_constant()
    test_type_vector_is_learnable_and_applied_after_the_linear()
    test_encoder_normalises_pixels_the_way_the_tower_was_trained()
    test_encoder_produces_the_plan_sized_block()
    test_encoder_gradient_reaches_the_projector_not_the_tower()
    test_encoder_train_eval_toggles_the_tower()
    test_truncating_a_tower_validates_the_layer_count()
    test_a_vision_encoder_feeds_the_decision_model_end_to_end()
    test_agent_runs_a_state_without_an_image()
    test_agent_attaches_a_visual_block_and_shifts_nothing_else()
    test_the_tower_runs_once_for_many_questions_on_one_image()
    test_visual_collate_builds_inputs_embeds()
    test_the_batch_forward_runs_and_answers()
    test_a_vision_checkpoint_refuses_the_fast_backend_message()
    test_split_state_keeps_the_other_keys_as_data()
    test_a_text_only_fit_gains_no_modality_key()
    test_the_modality_fit_refines_per_channel_and_leaves_the_pooled_map_alone()
    test_a_small_channel_bucket_is_omitted_not_guessed()
    test_unknown_modality_tags_are_refused()
    test_modality_map_round_trips_through_the_calibration_file()
    test_a_bad_modality_payload_is_refused_by_name()
    test_decode_uses_the_modality_temperature_for_image_rows_only()
    test_an_empty_modality_map_leaves_text_confidences_untouched()
    print("all vision foundation tests passed")
