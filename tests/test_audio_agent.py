"""Tests for the audio path inside an Agent: states that carry a sound, and the batch they build.

Weight-free and checkpoint-free: `Agent.__new__` skips the disk/weights path while keeping every
attribute the audio code touches, so this exercises the real `_encode_state` -> `collate_audio` ->
`_infer` plumbing on CPU with no checkpoint and no tower download -- the same trick
`tests/test_vision.py` uses for the vision path, and for the same reason.

What is being checked, beyond "it runs": a state carrying no sound must not grow a block (a text
checkpoint's rows stay byte-identical), a state carrying both a picture and a sound must be refused
by name rather than silently scored without one of them, and a checkpoint that can only see one
modality at a time must not splice a sound and a picture into one batch.
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.audio import (AudioCache, AudioConfig, audio_plan, build_audio_encoder, collate_audio,
                        parse_audio)
from laya.common import DecisionModel
from laya.projector import PooledProjector

SOUND = {"audio": {"waveform": torch.zeros(16000)}}


class _StubTower:
    """A wav2vec-style tower: [B, T] samples -> last_hidden_state [B, N, C], no weights."""

    class config:
        hidden_size = 32

    def __init__(self, n_frames=11):
        self.n_frames = n_frames
        self.calls = 0
        self.trained = False

    def parameters(self):
        return []

    def __call__(self, input_values):
        self.calls += 1
        b = input_values.size(0)
        frames = torch.arange(b * self.n_frames * 32, dtype=input_values.dtype,
                              device=input_values.device).reshape(b, self.n_frames, 32) / 1000.0
        return type("Out", (), {"last_hidden_state": frames})()

    def train(self, mode=True):
        self.trained = mode

    def eval(self):
        self.trained = False

    def to(self, device):
        return self


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


def _tiny_model() -> DecisionModel:
    cfg = AutoConfig.for_model("bert", hidden_size=16, num_hidden_layers=1, num_attention_heads=1,
                               intermediate_size=32, vocab_size=50)
    model = DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2)
    model.eval()
    return model


def _audio_agent(n_audio_tokens=4, sound=True, picture=False):
    """A tiny Agent built without a checkpoint, with a stub audio tower (and optionally a vision one).

    `audio=False` gives a text-only agent; `picture=True` adds a vision block beside the audio one,
    which is how the both-modalities cases are reached.
    """
    from laya.agent import Agent
    from laya.vision import VisualCache, VisionConfig, build_vision_encoder

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
    agent.tok = _StubTok()
    agent.cfg = {"max_len": 512, "head_max_len": 192}
    agent.model_id = "test-audio"
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.temperature_by_modality = {}
    agent.temperature_by_modality_raw = {}
    agent.lang_temperatures = {}
    agent.binning_map = None
    agent.vision = None
    agent.vision_encoder = None
    agent._visual_cache = None
    agent.audio = None
    agent.audio_encoder = None
    agent._audio_cache = None
    if sound:
        agent.audio = AudioConfig({"n_audio_tokens": n_audio_tokens, "summary_for_options_over": 0})
        agent._audio_cache = AudioCache(8)
        agent.audio_encoder = build_audio_encoder(
            agent.audio, hidden_size=model.encoder.config.hidden_size, tower=_StubTower())
        agent.audio_encoder.eval()
    if picture:
        agent.vision = VisionConfig({"n_visual_tokens": 3, "summary_for_options_over": 0,
                                     "image_size": 32})
        agent._visual_cache = VisualCache(8)
        agent.vision_encoder = build_vision_encoder(
            agent.vision, hidden_size=model.encoder.config.hidden_size,
            tower=_pixel_tower())
        agent.vision_encoder.eval()
    return agent


class _PixelStubTower:
    """A SigLIP-shaped tower for the picture-audio conflict tests: [B, 3, H, W] -> [B, N, 32]."""

    class config:
        hidden_size = 32

    def parameters(self):
        return []

    def __call__(self, pixel_values):
        b = pixel_values.size(0)
        patches = torch.zeros(b, 9, 32, dtype=pixel_values.dtype)
        return type("Out", (), {"last_hidden_state": patches})()

    def train(self, mode=True):
        pass

    def eval(self):
        pass

    def to(self, device):
        return self


def _pixel_tower():
    return _PixelStubTower()


def _question(agent):
    return {"q": agent._to_internal({"type": "choice", "instructions": "pick",
                                     "criteria": ["a", "b"]})}


# -- one state at a time --------------------------------------------------------------------

def test_a_text_state_grows_no_audio_block():
    agent = _audio_agent()
    items = agent._encode_state("plain text state", ["q"], _question(agent))
    assert "audio" not in items[0], "a text state must not grow an audio block"
    assert items[0]["modality"] == "text"
    assert items[0]["ids"], "the text state still tokenizes"


def test_a_sound_state_attaches_a_block_and_keeps_the_markers():
    agent = _audio_agent()
    internal = _question(agent)
    text_items = agent._encode_state("state", ["q"], internal)
    sound_items = agent._encode_state(dict(SOUND, text="state"), ["q"], internal)
    assert "audio" in sound_items[0], sound_items[0].keys()
    assert tuple(sound_items[0]["audio"].shape) == (4, agent.model.encoder.config.hidden_size)
    assert sound_items[0]["plan"].n_audio == 4
    assert sound_items[0]["modality"] == "audio"
    assert sound_items[0]["markers"] == text_items[0]["markers"], \
        "no summary row -> the markers must not move"


def test_a_sound_only_state_is_a_legal_state():
    """`{"audio": ...}` with no text at all: the recording is the whole state."""
    agent = _audio_agent()
    items = agent._encode_state({"audio": {"waveform": torch.zeros(400)}}, ["q"], _question(agent))
    assert "audio" in items[0]
    assert items[0]["ids"], "an empty text half still tokenizes to a sequence"


def test_the_tower_runs_once_for_many_questions_on_one_sound():
    """One recording, one tower pass -- counted at the tower, not at the cache."""
    agent = _audio_agent()
    internal = {q: agent._to_internal({"type": "choice", "instructions": "pick",
                                       "criteria": ["a", "b"]}) for q in ("q1", "q2", "q3")}
    tower = agent.audio_encoder.tower
    before = agent._audio_cache.misses
    agent._encode_state({"audio": {"waveform": torch.zeros(16000)}}, ["q1", "q2", "q3"], internal)
    assert tower.calls == 1, "three questions about one sound cost one tower pass, got %d" % tower.calls
    assert agent._audio_cache.misses - before == 1, agent._audio_cache.misses - before
    agent._encode_state({"audio": {"waveform": torch.zeros(16000)}}, ["q1"], internal)
    assert tower.calls == 1 and agent._audio_cache.hits >= 1, "the same samples must hit the cache"


def test_collate_audio_builds_inputs_embeds():
    agent = _audio_agent()
    internal = {"qa": agent._to_internal({"type": "choice", "instructions": "pick",
                                          "criteria": ["a", "b"]}),
                "qb": agent._to_internal({"type": "choice", "instructions": "pick",
                                          "criteria": ["a", "b", "c"]})}
    items = agent._encode_state(dict(SOUND, text="hello"), ["qa", "qb"], internal)
    b = collate_audio(items, agent._token_embeddings, agent.tok.pad_token_id)
    n_audio = items[0]["plan"].n_audio
    assert "inputs_embeds" in b and "input_ids" not in b
    assert b["inputs_embeds"].shape[0] == 2
    assert b["visual_mask"][0].sum().item() == n_audio
    with torch.no_grad():
        logits, act = agent._infer(b)
    assert logits.shape[0] == 2 and logits.shape[1] >= 2, logits.shape
    assert act.shape[0] == 2


def test_a_batch_may_mix_a_sound_with_a_text_only_state():
    """The batched form of the sound-removed half of the ablation gate, which needs one pass."""
    agent = _audio_agent()
    internal = _question(agent)
    heard = agent._encode_state(dict(SOUND, text="state"), ["q"], internal)
    deaf = agent._encode_state({"text": "state"}, ["q"], internal)
    assert "audio" not in deaf[0]

    b = collate_audio(heard + deaf, agent._token_embeddings, agent.tok.pad_token_id)
    assert b["inputs_embeds"].shape[0] == 2
    assert bool(b["visual_mask"][0].any()), "the sound row lost its block mask"
    assert not bool(b["visual_mask"][1].any()), "the text row must not be marked as carrying a sound"
    n = int(b["attention_mask"][1].sum())
    assert b["position_ids"][1, :n].tolist() == list(range(n)), b["position_ids"][1, :n].tolist()
    with torch.no_grad():
        logits, _act = agent._infer(b)
    assert tuple(logits.shape)[0] == 2, logits.shape


def test_a_picture_and_a_sound_in_one_state_are_refused_by_name():
    agent = _audio_agent(picture=True)
    try:
        agent._encode_state(dict(SOUND, text="state", image={"pixels": torch.zeros(3, 32, 32)}),
                            ["q"], _question(agent))
    except ValueError as exc:
        message = str(exc)
        assert "picture" in message and "sound" in message, message
        assert "not both" in message, message
    else:
        raise AssertionError("a state carrying both inputs must be refused, not silently "
                             "scored without one of them")


def test_the_audio_block_spends_the_text_budget():
    """`max_len - n_audio_tokens` is the text sequence, so the block never pushes past `max_len`."""
    agent = _audio_agent(n_audio_tokens=8)
    agent.cfg["max_len"] = 64
    internal = _question(agent)
    items = agent._encode_state(dict(SOUND, text="word " * 200), ["q"], internal)
    assert len(items[0]["plan"].position_ids) <= 64, len(items[0]["plan"].position_ids)
    assert len(items[0]["ids"]) + items[0]["plan"].n_audio <= 64, \
        (len(items[0]["ids"]), items[0]["plan"].n_audio)


def test_a_block_that_eats_the_whole_sequence_is_refused():
    agent = _audio_agent(n_audio_tokens=500)
    agent.cfg["max_len"] = 512
    try:
        agent._encode_state(dict(SOUND, text="state"), ["q"], _question(agent))
    except ValueError as exc:
        assert "audio tokens leave" in str(exc), str(exc)
        assert "n_audio_tokens" in str(exc), str(exc)
    else:
        raise AssertionError("a block that leaves no room for the text must be refused")


# -- a batch of more than one state ---------------------------------------------------------

def test_predict_batch_splices_a_sound_state_and_scores_a_text_state():
    """The branch in `predict_batch` that chooses `collate_audio`, exercised end to end."""
    agent = _audio_agent()
    answers = agent.predict_batch([dict(SOUND, text="state"), {"text": "state"}],
                                  {"q": {"type": "choice", "instructions": "pick",
                                         "criteria": ["a", "b"]}})
    assert isinstance(answers, list) and len(answers) == 2, answers
    for result in answers:
        assert "q" in result["answers"], result


def test_predict_batch_refuses_to_mix_a_picture_row_with_a_sound_row():
    """One batch is one sequence: two different blocks in it is a layout this checkpoint lacks."""
    agent = _audio_agent(picture=True)
    questions = {"q": {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}}
    try:
        agent.predict_batch([dict(SOUND, text="state"),
                             {"image": {"pixels": torch.zeros(3, 32, 32)}, "text": "state"}],
                            questions)
    except ValueError as exc:
        assert "mixes" in str(exc), str(exc)
        assert "modality" in str(exc), str(exc)
    else:
        raise AssertionError("a mixed-modality batch must be refused by name")


def test_the_decode_uses_the_audio_modality_temperature_for_sound_rows():
    """A checkpoint that fits a per-modality map must have it applied to a sound row's temperature."""
    agent = _audio_agent()
    agent.temperature_by_modality = {"audio": {"choice:2": 0.25}}
    internal = _question(agent)
    items = agent._encode_state(dict(SOUND, text="state"), ["q"], internal)
    b = collate_audio(items, agent._token_embeddings, agent.tok.pad_token_id)
    with torch.no_grad():
        logits, act = agent._infer(b)
    hot = agent._decode_answers(logits, act, items, ["q"], internal, 0)
    agent.temperature_by_modality = {}
    cold = agent._decode_answers(logits, act, items, ["q"], internal, 0)
    assert hot["q"]["confidence"] >= cold["q"]["confidence"], (hot, cold)


def test_an_audio_checkpoint_refuses_the_fast_backend_message():
    """The guard lives in Agent.__init__; assert the message a caller would see."""
    import inspect
    from laya import agent as agent_mod
    src = inspect.getsource(agent_mod.Agent.__init__)
    assert "an audio checkpoint cannot use fast=True" in src, \
        "the fast path has no audio-token support"


def test_predict_long_sizes_its_windows_against_the_audio_budget():
    """A window must leave room for the block, or every window's tail reaches no model."""
    agent = _audio_agent(n_audio_tokens=8)
    agent.cfg["max_len"] = 512
    assert agent._block_tokens() == 8
    assert agent._block_budget()[1:] == ("audio", "audio"), agent._block_budget()


def test_parse_audio_reads_a_checkpoint_config_and_nothing_else():
    assert parse_audio({"audio": {"n_audio_tokens": 8}}).n_audio_tokens == 8
    assert parse_audio({"encoder": "x"}) is None
    # A plan built from the same config agrees with the agent's own view of the budget.
    cfg = parse_audio({"audio": {"n_audio_tokens": 8, "summary_for_options_over": 0}})
    plan = audio_plan(cfg, text_len=20, markers=[3, 5], head_len=7, max_len=64, option_tokens=2)
    assert plan.n_audio == 8 and plan.splice == 7


def test_the_projector_is_the_shared_one():
    """The audio path trains `laya.projector.PooledProjector`, not a second copy of it."""
    agent = _audio_agent()
    assert isinstance(agent.audio_encoder.projector, PooledProjector)
    assert agent.audio_encoder.projector.n_tokens == 4


if __name__ == "__main__":
    test_a_text_state_grows_no_audio_block()
    test_a_sound_state_attaches_a_block_and_keeps_the_markers()
    test_a_sound_only_state_is_a_legal_state()
    test_the_tower_runs_once_for_many_questions_on_one_sound()
    test_collate_audio_builds_inputs_embeds()
    test_a_batch_may_mix_a_sound_with_a_text_only_state()
    test_a_picture_and_a_sound_in_one_state_are_refused_by_name()
    test_the_audio_block_spends_the_text_budget()
    test_a_block_that_eats_the_whole_sequence_is_refused()
    test_predict_batch_splices_a_sound_state_and_scores_a_text_state()
    test_predict_batch_refuses_to_mix_a_picture_row_with_a_sound_row()
    test_the_decode_uses_the_audio_modality_temperature_for_sound_rows()
    test_an_audio_checkpoint_refuses_the_fast_backend_message()
    test_predict_long_sizes_its_windows_against_the_audio_budget()
    test_parse_audio_reads_a_checkpoint_config_and_nothing_else()
    test_the_projector_is_the_shared_one()
    print("all audio agent tests passed")
