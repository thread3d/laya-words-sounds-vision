"""Tests for the audio foundation: checkpoint config, audio-token layout, the tower and projector.

Weight-free and tower-free by construction -- nothing here downloads Wav2Vec2 or needs a checkpoint
on disk, so these run in the same fast lane as `tests/test_vision.py`. The one model test builds a
tiny from-config BERT and asserts that a spliced audio block reaches the option logits, which is the
property the whole audio path rests on.

The layout tests are deliberately a mirror of the vision suite's: `laya.block` is shared, so the
same numbers must come out of both modalities, and a difference here is a difference in the model.
"""
import os
import subprocess
import sys
import tempfile

import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.audio import (AUDIO_WEIGHTS_FILE, audio_plan, build_audio_encoder, bundle_tower,
                        describe_plan, load_audio_weights, load_tower, local_tower_dir, parse_audio,
                        save_audio_weights, tower_parameters)
from laya.block import splice_embeddings
from laya.common import DecisionModel
from laya.projector import PooledProjector
from laya.vision import parse_vision, visual_plan


# -- config ---------------------------------------------------------------------------------

def test_no_block_means_no_audio_config():
    assert parse_audio(None) is None
    assert parse_audio({}) is None
    assert parse_audio({"encoder": "x", "max_len": 512}) is None, \
        "a text checkpoint must parse to None, not to a default audio config"


def test_explicitly_disabled_block_parses_to_none():
    assert parse_audio({"audio": {"enabled": False, "tower": "x"}}) is None


def test_block_is_validated_and_defaulted():
    a = parse_audio({"audio": {"n_audio_tokens": 64}})
    assert a.n_audio_tokens == 64
    assert a.slot == "after_options"
    assert a.pooling == "adaptive_avg"
    assert a.sample_rate == 16000
    assert a.input == "waveform"
    assert "wav2vec2" in a.tower, a.tower


def test_unknown_key_is_refused_with_the_known_keys():
    try:
        parse_audio({"audio": {"n_adio_tokens": 64}})
    except ValueError as e:
        assert "n_adio_tokens" in str(e) and "n_audio_tokens" in str(e), e
    else:
        raise AssertionError("a typo in an audio key must not be silently ignored")


def test_bad_values_are_refused():
    for block in ({"slot": "after_state"}, {"pooling": "mean"}, {"n_audio_tokens": 0},
                  {"n_audio_tokens": -4}, {"sample_rate": 0}, {"keep_layers": 0},
                  {"feature_std": 0.0}, {"input": "spectrogram"}, {"cache_size": 0}):
        try:
            parse_audio({"audio": block})
        except (ValueError, TypeError):
            continue
        raise AssertionError("block %r should have been refused" % (block,))


def test_non_mapping_block_is_refused():
    try:
        parse_audio({"audio": ["nope"]})
    except TypeError as e:
        assert "mapping" in str(e), e
    else:
        raise AssertionError("a list is not an audio block")


# -- position ids ---------------------------------------------------------------------------

def test_audio_positions_are_a_reserved_block_at_the_top():
    a = parse_audio({"audio": {"n_audio_tokens": 16}})
    assert a.position_ids_for_audio(128) == list(range(112, 128))


def test_explicit_offset_is_honoured_and_bounds_checked():
    a = parse_audio({"audio": {"n_audio_tokens": 8, "position_offset": 1000}})
    assert a.position_ids_for_audio(2048) == list(range(1000, 1008))
    try:
        parse_audio({"audio": {"n_audio_tokens": 8, "position_offset": 2044}},
                    max_position_embeddings=2048)
    except ValueError as e:
        assert "positions" in str(e), e
    else:
        raise AssertionError("an offset that overruns the positional budget must be refused")


# -- layout ---------------------------------------------------------------------------------

def _plan(**block):
    return parse_audio({"audio": dict({"n_audio_tokens": 8, "summary_for_options_over": 96},
                                      **block)})


def test_the_block_lands_after_the_head_and_the_markers_do_not_move():
    a = _plan()
    markers = [3, 5]
    p = audio_plan(a, text_len=20, markers=markers, head_len=7, max_len=512, option_tokens=4)
    assert p.splice == 7, p.splice
    assert p.n_audio == 8, p.n_audio
    assert p.marker_positions == [3, 5], "markers sit before the splice, so they must not shift"
    assert p.length == 28


def test_the_summary_row_extends_the_block_and_leaves_the_markers_alone():
    a = _plan()
    p = audio_plan(a, text_len=20, markers=[3, 5], head_len=7, max_len=512, option_tokens=400)
    assert p.summary is True
    assert p.n_audio == 9, p.n_audio            # 8 frame tokens + 1 summary row
    assert p.marker_positions == [3, 5], p.marker_positions
    assert p.splice == 7, p.splice


def test_the_default_position_policy_keeps_the_block_adjacent():
    a = _plan()
    p = audio_plan(a, text_len=20, markers=[3, 5], head_len=7, max_len=512,
                   option_tokens=4, max_position_embeddings=8192)
    assert a.position_policy == "contiguous"
    vis_pos = p.position_ids[7:15]
    assert vis_pos == list(range(7, 15)), vis_pos
    assert p.position_ids[:7] == list(range(7))
    assert p.position_ids[15:] == list(range(15, 28)), p.position_ids[15:]
    assert min(vis_pos) - max(p.marker_positions) < 10


def test_the_reserved_policy_is_available_and_is_far_away():
    a = _plan(position_policy="reserved")
    p = audio_plan(a, text_len=20, markers=[3, 5], head_len=7, max_len=512,
                   option_tokens=4, max_position_embeddings=8192)
    assert p.position_ids[7:15] == list(range(8192 - 8, 8192))
    assert min(p.position_ids[7:15]) - max(p.marker_positions) > 8000


def test_a_summary_row_under_the_reserved_policy_sits_one_past_the_block():
    a = _plan(position_policy="reserved")
    p = audio_plan(a, text_len=20, markers=[3], head_len=7, max_len=512, option_tokens=400,
                   max_position_embeddings=8192)
    assert p.n_audio == 9
    assert p.position_ids[7:16] == list(range(8192 - 8, 8192)) + [8192], p.position_ids[7:16]


def test_the_budget_error_names_the_audio_knob():
    a = _plan()
    try:
        audio_plan(a, text_len=505, markers=[3], head_len=7, max_len=512, option_tokens=4)
    except ValueError as e:
        msg = str(e)
        assert "505" in msg and "8" in msg and "512" in msg, msg
        assert "audio.n_audio_tokens" in msg, msg
        assert "vision" not in msg, msg
    else:
        raise AssertionError("505 + 8 > 512 must be refused, not silently truncated")


def test_the_audio_plan_matches_the_vision_plan_layout():
    """The two modalities share `laya.block`; the same shape of config must give the same layout.

    A difference here means one modality would train against one splice point and infer at another,
    which is the class of bug the shared plan exists to make impossible.
    """
    a = _plan()
    v = parse_vision({"vision": {"n_visual_tokens": 8, "summary_for_options_over": 96}})
    for option_tokens, policy in ((4, "contiguous"), (400, "contiguous"), (4, "reserved"),
                                  (400, "reserved")):
        a.position_policy = policy
        v.position_policy = policy
        pa = audio_plan(a, text_len=40, markers=[3, 5], head_len=9, max_len=512,
                        option_tokens=option_tokens, max_position_embeddings=8192)
        pv = visual_plan(v, text_len=40, markers=[3, 5], head_len=9, max_len=512,
                         option_tokens=option_tokens, max_position_embeddings=8192)
        assert (pa.splice, pa.n_audio, pa.summary, pa.length) == \
               (pv.splice, pv.n_visual, pv.summary, pv.length), (option_tokens, policy)
        assert pa.marker_positions == pv.marker_positions
        assert pa.position_ids == pv.position_ids, (option_tokens, policy)


def test_describe_plan_names_the_modality():
    a = _plan()
    p = audio_plan(a, text_len=20, markers=[3], head_len=7, max_len=512, option_tokens=4)
    facts = describe_plan(p)
    assert facts[1].startswith("8 audio tokens"), facts


# -- assembly -------------------------------------------------------------------------------

def test_assembly_splices_audio_rows_at_the_splice_point():
    a = _plan()
    p = audio_plan(a, text_len=20, markers=[3, 5], head_len=7, max_len=512, option_tokens=4)
    tok = torch.zeros(1, 20, 4)
    tok[0, :, 0] = torch.arange(20)
    block = torch.ones(1, 8, 4) * 99.0
    out = splice_embeddings(tok, block, p, label="audio")
    assert tuple(out.shape) == (1, 28, 4), out.shape
    assert out[0, :7, 0].tolist() == list(range(7))
    assert out[0, 7:15, 0].tolist() == [99.0] * 8
    assert out[0, 15:, 0].tolist() == list(range(7, 20))


def test_assembly_refuses_shape_and_width_mismatches():
    a = _plan()
    p = audio_plan(a, text_len=20, markers=[3], head_len=7, max_len=512, option_tokens=4)
    tok = torch.zeros(2, 20, 4)
    cases = [
        (tok, torch.zeros(2, 8, 4), "fine"),
        (tok, torch.zeros(1, 8, 4), "batch"),
        (tok, torch.zeros(2, 8, 5), "width"),
        (tok, torch.zeros(2, 7, 4), "plan expects"),
    ]
    for t, block, expect in cases:
        try:
            splice_embeddings(t, block, p, label="audio")
        except ValueError as e:
            assert expect in str(e), (expect, str(e))
            continue
        assert expect == "fine", "shape %s should have been refused" % (tuple(block.shape),)


def test_gradient_reaches_the_audio_block():
    a = _plan()
    p = audio_plan(a, text_len=12, markers=[3], head_len=6, max_len=512, option_tokens=4)
    block = torch.zeros(1, 8, 4, requires_grad=True)
    out = splice_embeddings(torch.zeros(1, 12, 4), block, p, label="audio")
    out.sum().backward()
    assert block.grad is not None and block.grad.abs().sum().item() == 8 * 4, "no gradient path"


# -- the tower and the projector ------------------------------------------------------------

class _StubTower:
    """A stand-in for a wav2vec-style tower: [B, T] samples -> last_hidden_state [B, N, C]."""

    class config:
        hidden_size = 32

    def __init__(self, n_frames=19):
        self.n_frames = n_frames
        self.trained = False
        self.seen = []
        self.received = None

    def parameters(self):
        return []

    def __call__(self, input_values):
        self.seen.append(tuple(input_values.shape))
        self.received = input_values.detach()
        b = input_values.size(0)
        frames = torch.arange(b * self.n_frames * 32, dtype=input_values.dtype).reshape(
            b, self.n_frames, 32) / 1000.0
        return type("Out", (), {"last_hidden_state": frames})()

    def train(self, mode=True):
        self.trained = mode

    def eval(self):
        self.trained = False

    def to(self, device):
        self.moved_to = str(device)
        return self


def _encoder(**block):
    a = parse_audio({"audio": dict({"n_audio_tokens": 8}, **block)})
    tower = _StubTower()
    return build_audio_encoder(a, hidden_size=16, tower=tower), tower


def test_the_encoder_produces_a_plan_sized_block():
    enc, tower = _encoder()
    out = enc.encode(torch.zeros(1, 16000))
    assert tuple(out.shape) == (1, 8, 16), out.shape
    assert tower.seen == [(1, 16000)], tower.seen


def test_the_summary_row_is_the_last_row_and_same_width():
    enc, _ = _encoder()
    assert tuple(enc.encode(torch.zeros(2, 16000), summary=True).shape) == (2, 9, 16)


def test_the_encoder_applies_the_configs_input_normalisation():
    """feature_mean/std belong to the tower, so the encoder must apply them before it runs."""
    enc, tower = _encoder(feature_mean=0.5, feature_std=0.5)
    enc.encode(torch.zeros(1, 64) + 1.0)
    assert torch.allclose(tower.received, torch.ones(1, 64)), tower.received


def test_encoder_to_moves_the_tower_as_well_as_the_projector():
    enc, tower = _encoder()
    moved = enc.to("cpu")
    assert moved is enc, "to() must return the encoder for chaining"
    assert getattr(tower, "moved_to", None) == "cpu", getattr(tower, "moved_to", None)
    for parameter in enc.projector.parameters():
        assert parameter.device.type == "cpu", parameter.device


def test_encoder_gradient_reaches_the_projector_not_the_tower():
    enc, tower = _encoder()
    out = enc.encode(torch.randn(2, 400))
    out.sum().backward()
    weight = enc.projector.proj.weight
    assert weight.grad is not None and weight.grad.abs().sum().item() > 0, \
        "the projector must receive a gradient"
    assert not any(getattr(p, "grad", None) is not None for p in tower.parameters())


def test_encoder_train_eval_toggles_the_tower():
    enc, tower = _encoder()
    enc.train(True)
    assert tower.trained is True
    enc.eval()
    assert tower.trained is False


def test_projector_compresses_and_projects():
    proj = PooledProjector(32, 16, 8, pooling="adaptive_avg")
    assert tuple(proj(torch.randn(2, 19, 32)).shape) == (2, 8, 16)


def test_projector_emits_a_varied_pattern_not_a_constant():
    """LayerNorm removes a constant row, so a projector must not collapse to one."""
    proj = PooledProjector(32, 16, 8, pooling="adaptive_avg")
    out = proj(torch.randn(1, 19, 32))
    assert out[0].std(dim=-1).min().item() > 1e-6, "rows collapsed to a constant"
    assert not torch.allclose(out[0, 0], out[0, 1]), "audio tokens are indistinguishable"


def test_projector_keeps_the_vision_paths_name_for_its_width():
    proj = PooledProjector(32, 16, 8)
    assert proj.n_tokens == 8 and proj.n_visual_tokens == 8


def test_truncating_a_tower_validates_the_layer_count():
    from laya.projector import truncate_tower

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

    class _Enc(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = torch.nn.ModuleList([_Layer() for _ in range(n)])

    class _Tower(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.encoder = _Enc(n)

    t = _Tower(6)
    truncate_tower(t, 4, name="audio tower")
    assert len(t.encoder.layers) == 4
    for bad in (0, 7):
        try:
            truncate_tower(_Tower(6), bad, name="audio tower")
        except ValueError as e:
            assert "out of range" in str(e), e
            continue
        raise AssertionError("keep_layers=%d should have been refused" % bad)


# -- persisting the projector ---------------------------------------------------------------

def test_the_projector_round_trips_through_a_checkpoint():
    enc, _ = _encoder()
    with torch.no_grad():
        enc.projector.proj.weight.add_(0.25)
        enc.projector.type_vector.add_(0.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_audio_weights(enc, tmp)
        assert os.path.basename(path) == AUDIO_WEIGHTS_FILE
        other, _ = _encoder()
        load_audio_weights(other, tmp)
        assert torch.allclose(other.projector.proj.weight, enc.projector.proj.weight.half().float(),
                              atol=1e-3)
        assert torch.allclose(other.projector.type_vector, enc.projector.type_vector.half().float(),
                              atol=1e-3)


def test_a_checkpoint_without_a_projector_is_refused_by_name():
    enc, _ = _encoder()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_audio_weights(enc, tmp)
        except FileNotFoundError as exc:
            assert AUDIO_WEIGHTS_FILE in str(exc), exc
            assert "randomly" in str(exc), exc
        else:
            raise AssertionError("a checkpoint that declares audio and ships no projector must be "
                                 "refused, not answered from noise")


def test_a_bundled_tower_is_preferred_over_the_hub_id():
    enc, _ = _encoder()

    class _Savable(_StubTower):
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "config.json"), "w") as fh:
                fh.write("{}")

    enc.tower = _Savable()
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "checkpoint")
        os.makedirs(target)
        bundled = bundle_tower(enc, target)
        assert os.path.isdir(bundled), bundled
        assert local_tower_dir(target) == bundled
        assert local_tower_dir(os.path.join(tmp, "empty")) is None


def test_an_unbundleable_tower_fails_loudly():
    enc, _ = _encoder()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            bundle_tower(enc, tmp)
        except TypeError as exc:
            assert "save_pretrained" in str(exc), exc
        else:
            raise AssertionError("a tower that cannot be saved must not ship silently")


def test_tower_parameters_are_the_projector_only():
    enc, _ = _encoder()
    assert tower_parameters(enc) == enc.projector.parameters()


def test_load_tower_builds_a_frozen_wav2vec_tower_without_the_hub():
    """`load_tower` is the one place that names an audio architecture; it must freeze what it loads.

    The tower is built from a tiny local config written here, so this needs no download: the point
    is the freezing and the `keep_layers` plumbing, not the weights.
    """
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    cfg = Wav2Vec2Config(hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=16, conv_dim=(8, 8), conv_stride=(5, 2),
                         conv_kernel=(10, 3), num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=4, vocab_size=4)
    with tempfile.TemporaryDirectory() as tmp:
        Wav2Vec2Model(cfg).save_pretrained(tmp)
        tower = load_tower(tmp, keep_layers=1)
        assert len(tower.encoder.layers) == 1, len(tower.encoder.layers)
        assert not any(p.requires_grad for p in tower.parameters()), "the tower must be frozen"
        out = tower(input_values=torch.zeros(1, 640))
        assert out.last_hidden_state.dim() == 3


# -- the model end to end -------------------------------------------------------------------

def _tiny_model() -> DecisionModel:
    cfg = AutoConfig.for_model("bert", hidden_size=16, num_hidden_layers=1, num_attention_heads=1,
                               intermediate_size=32, vocab_size=50)
    model = DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2)
    model.eval()
    return model


@torch.no_grad()
def test_an_audio_block_reaches_the_option_logits():
    """The whole path on a tiny model: samples -> tower -> projector -> splice -> option logits.

    A block that enters the sequence but cannot change an answer is the failure mode this whole
    design is against, so the assertion is that the *answers move*.
    """
    model = _tiny_model()
    a = parse_audio({"audio": {"n_audio_tokens": 4, "summary_for_options_over": 0}})
    enc = build_audio_encoder(a, hidden_size=model.encoder.config.hidden_size, tower=_StubTower())
    ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])
    head_len = 6
    plan = audio_plan(a, text_len=ids.size(1), markers=[3, 5], head_len=head_len, max_len=64,
                      option_tokens=4)
    tok_emb = model.encoder.embeddings.word_embeddings(ids)
    block = enc.encode(torch.randn(1, 400))
    embeds = splice_embeddings(tok_emb, block, plan, label="audio")
    att = torch.ones(1, embeds.size(1), dtype=torch.long)
    mpos = torch.tensor([plan.marker_positions])
    mmask = torch.ones(1, 2, dtype=torch.bool)
    qt = torch.tensor([0])
    vmask = torch.zeros(1, embeds.size(1), dtype=torch.bool)
    vmask[0, plan.splice:plan.splice + plan.n_audio] = True

    pos = torch.tensor([plan.position_ids])
    with_block, _ = model.forward(None, att, mpos, mmask, qt, position_ids=pos,
                                  inputs_embeds=embeds, visual_mask=vmask)

    no_block = model.forward(input_ids=ids, attention_mask=torch.ones_like(ids),
                             marker_pos=mpos, marker_mask=mmask, qtype=qt)[0]
    assert with_block.shape == no_block.shape, (with_block.shape, no_block.shape)
    assert not torch.allclose(with_block, no_block, atol=1e-6), \
        "the audio block must change the option logits"


# -- import discipline ----------------------------------------------------------------------

def test_importing_laya_audio_does_not_import_torch():
    """A text-only deployment must never pay for the audio stack (`tests/test_lazy_import.py`)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probe = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import laya.audio, laya.block, laya.projector\n"
        "assert 'torch' not in sys.modules, 'importing laya.audio pulled in torch'\n"
        "print('ok')\n" % (root,)
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ok" in out.stdout, out.stdout


if __name__ == "__main__":
    test_no_block_means_no_audio_config()
    test_explicitly_disabled_block_parses_to_none()
    test_block_is_validated_and_defaulted()
    test_unknown_key_is_refused_with_the_known_keys()
    test_bad_values_are_refused()
    test_non_mapping_block_is_refused()
    test_audio_positions_are_a_reserved_block_at_the_top()
    test_explicit_offset_is_honoured_and_bounds_checked()
    test_the_block_lands_after_the_head_and_the_markers_do_not_move()
    test_the_summary_row_extends_the_block_and_leaves_the_markers_alone()
    test_the_default_position_policy_keeps_the_block_adjacent()
    test_the_reserved_policy_is_available_and_is_far_away()
    test_a_summary_row_under_the_reserved_policy_sits_one_past_the_block()
    test_the_budget_error_names_the_audio_knob()
    test_the_audio_plan_matches_the_vision_plan_layout()
    test_describe_plan_names_the_modality()
    test_assembly_splices_audio_rows_at_the_splice_point()
    test_assembly_refuses_shape_and_width_mismatches()
    test_gradient_reaches_the_audio_block()
    test_the_encoder_produces_a_plan_sized_block()
    test_the_summary_row_is_the_last_row_and_same_width()
    test_the_encoder_applies_the_configs_input_normalisation()
    test_encoder_to_moves_the_tower_as_well_as_the_projector()
    test_encoder_gradient_reaches_the_projector_not_the_tower()
    test_encoder_train_eval_toggles_the_tower()
    test_projector_compresses_and_projects()
    test_projector_emits_a_varied_pattern_not_a_constant()
    test_projector_keeps_the_vision_paths_name_for_its_width()
    test_truncating_a_tower_validates_the_layer_count()
    test_the_projector_round_trips_through_a_checkpoint()
    test_a_checkpoint_without_a_projector_is_refused_by_name()
    test_a_bundled_tower_is_preferred_over_the_hub_id()
    test_an_unbundleable_tower_fails_loudly()
    test_tower_parameters_are_the_projector_only()
    test_load_tower_builds_a_frozen_wav2vec_tower_without_the_hub()
    test_an_audio_block_reaches_the_option_logits()
    test_importing_laya_audio_does_not_import_torch()
    print("all audio foundation tests passed")
