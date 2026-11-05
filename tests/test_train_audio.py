"""Tests for the audio training path (`laya/train.py`).

Everything here runs on a tiny from-config model with a stub tower, so no checkpoint, no GPU and no
audio codec is involved: the point is that the *path* is right -- rows with a recording become items
with an audio block, the projector is trained inside the step rather than before it, and the
before/after eval table scores the sound instead of the state text alone.

The tests mirror `tests/test_train_vision.py` deliberately: the audio path is the vision path with
a different tower, and a difference between the two suites is a difference in the model.
"""
import os
import sys
import tempfile

import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.audio import build_audio_encoder, parse_audio, save_audio_weights
from laya.common import DecisionModel
from laya.train import TrainConfig, encode_item, items_from_rows, train_model

# The tiny test encoder is 512 positions wide, not ModernBERT's 8192.
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


class _StubAudioTower(torch.nn.Module):
    """A wav2vec-shaped tower: [B, T] samples -> [B, N, 32] frames that depend on the samples."""

    class config:
        hidden_size = 32

    def __init__(self, n_frames=13):
        super().__init__()
        self.n_frames = n_frames
        self.proj = torch.nn.Linear(1, 32)

    def forward(self, input_values):
        flat = input_values.mean(-1, keepdim=True)
        frames = self.proj(flat).unsqueeze(1).repeat(1, self.n_frames, 1)
        return type("O", (), {"last_hidden_state": frames})()


def _model(hidden=16):
    cfg = AutoConfig.for_model("bert", hidden_size=hidden, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32, vocab_size=50,
                               pad_token_id=0)
    return DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2)


def _audio(n_audio=3):
    return parse_audio({"audio": {"n_audio_tokens": n_audio, "sample_rate": 16000,
                                  "summary_for_options_over": 0}})


def _encoder(audio, hidden=16):
    enc = build_audio_encoder(audio, hidden_size=hidden, tower=_StubAudioTower())
    return enc.eval()


def _encode_audio(enc):
    """Mirror of `train._audio_encoder_fn`: the tower's **frames**, not the projected block."""
    def encode(waveform):
        with torch.no_grad():
            return enc.encode_frames(waveform.unsqueeze(0))[0]
    return encode


def _rows():
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    return [
        {"state": {"audio": {"waveform": torch.zeros(16000)}, "text": "one"},
         "questions": {"q": question}, "expected": {"q": "a"}},
        {"state": {"audio": {"waveform": torch.ones(16000)}, "text": "two"},
         "questions": {"q": question}, "expected": {"q": "b"}},
        {"state": "text only", "questions": {"q": question}, "expected": {"q": "a"}},
    ]


# -- the eval report ---------------------------------------------------------------------------

def test_eval_rows_are_built_through_the_audio_path():
    """`--eval` rows carry the recording, because the report is about the recording."""
    from laya.train import eval_items_from_rows

    audio = _audio()
    enc = _encoder(audio)
    items, skipped = eval_items_from_rows(_StubTok(), _rows(), max_len=512, head_max_len=192,
                                          audio=audio, encode_audio=_encode_audio(enc))
    assert skipped == {}, skipped
    assert sum(1 for it in items if "audio_frames" in it) == 2, \
        "the sound rows lost their audio block on the way to the eval report"
    assert items[2].get("audio_frames") is None, "a text row must evaluate as text"


def test_the_eval_report_scores_the_sound_not_just_the_state_text():
    """The two item sets produce different logits, so the report's number moves with the audio."""
    from laya.train import calibration_records, eval_items_from_rows

    torch.manual_seed(0)
    model = _model().eval()
    tok = _StubTok()
    audio = _audio()
    enc = _encoder(audio)
    embeds = model.encoder.embeddings
    table = getattr(embeds, "tok_embeddings", None) or embeds.word_embeddings

    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"audio": {"waveform": torch.full((16000,), float(i) / 8.0)},
                       "text": "same"},
             "questions": {"q": question}, "expected": {"q": "ab"[i % 2]}} for i in range(4)]
    seen, _ = eval_items_from_rows(tok, rows, max_len=512, head_max_len=192,
                                   audio=audio, encode_audio=_encode_audio(enc))
    # The baseline is the same text with no recording: a waveform tensor cannot be serialized into
    # the text-only state (which is the point -- a text model is never handed one), so the blind
    # side is built from the text half alone.
    blind, _ = items_from_rows(
        tok, [{"state": "same", "questions": r["questions"], "expected": r["expected"]}
              for r in rows],
        max_len=512, head_max_len=192, label_smoothing=0.0)

    kwargs = dict(audio=audio, embed_fn=table, max_positions=MAXPOS, audio_projector=enc.projector)
    with torch.no_grad():
        a = calibration_records(model, tok, seen, torch.device("cpu"), 512, 192, batch_size=4,
                                **kwargs)
        b = calibration_records(model, tok, blind, torch.device("cpu"), 512, 192, batch_size=4,
                                audio=audio, embed_fn=table, max_positions=MAXPOS)
    assert len(a) == len(b) == len(rows), (len(a), len(b))
    differs = any(abs(float(x) - float(y)) > 1e-6 for x, y in zip(a[0][1], b[0][1]))
    assert differs, "the eval report's logits ignore the audio block"


# -- the projector on resume ------------------------------------------------------------------

def test_a_resumed_audio_checkpoint_keeps_its_trained_projector():
    """A run started from an audio checkpoint must continue from its projector, not from noise."""
    from laya.train import attach_audio_projector

    audio = _audio()
    trained = _encoder(audio)
    with torch.no_grad():
        trained.projector.proj.weight.add_(1.0)
        trained.projector.proj.bias.add_(2.0)
        trained.projector.type_vector.add_(3.0)
    tmp = tempfile.mkdtemp()
    save_audio_weights(trained, tmp)           # stores float16, hence the tolerance below

    fresh = _encoder(audio)
    note = attach_audio_projector(fresh, tmp, base_declares_audio=True)
    assert "continuing from the base checkpoint's projector" in note, note
    assert torch.allclose(fresh.projector.proj.weight, trained.projector.proj.weight, atol=1e-3)
    assert torch.allclose(fresh.projector.type_vector, trained.projector.type_vector, atol=1e-3)


def test_an_audio_checkpoint_without_projector_weights_is_refused():
    """Declaring audio and shipping no projector is a corrupt checkpoint, not a fresh start."""
    from laya.train import attach_audio_projector

    try:
        attach_audio_projector(_encoder(_audio()), tempfile.mkdtemp(), base_declares_audio=True)
    except FileNotFoundError as exc:
        assert "audio.safetensors" in str(exc), exc
    else:
        raise AssertionError("an audio checkpoint with no projector weights was accepted")


def test_a_text_base_checkpoint_starts_a_new_audio_projector():
    """The first audio fine-tune is where a projector is created; that is not an error."""
    from laya.train import attach_audio_projector

    note = attach_audio_projector(_encoder(_audio()), tempfile.mkdtemp(), base_declares_audio=False)
    assert "trains a new one" in note, note
    assert "n_audio_tokens=3" in note, note


# -- items -----------------------------------------------------------------------------------

def test_sound_rows_become_items_that_carry_an_audio_block():
    audio = _audio()
    enc = _encoder(audio)
    items, skipped = items_from_rows(_StubTok(), _rows(), max_len=512, head_max_len=192,
                                     audio=audio, encode_audio=_encode_audio(enc))
    assert skipped == {}, skipped
    assert len(items) == 3
    assert sum(1 for it in items if "audio_frames" in it) == 2
    assert items[2].get("audio_frames") is None, "a text row must train as text"
    # Frames, not a projected block: the projector is trained, so it has to run inside the step.
    assert tuple(items[0]["audio_frames"].shape) == (13, 32), items[0]["audio_frames"].shape


def test_one_recording_asked_many_questions_costs_one_tower_pass():
    audio = _audio()
    enc = _encoder(audio)
    calls = {"n": 0}

    def counting(waveform):
        calls["n"] += 1
        return _encode_audio(enc)(waveform)

    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"audio": {"waveform": torch.zeros(16000)}},
             "questions": {"q1": question, "q2": question}, "expected": {"q1": "a", "q2": "b"}}]
    items, _ = items_from_rows(_StubTok(), rows, max_len=512, head_max_len=192,
                               audio=audio, encode_audio=counting)
    assert len(items) == 2 and calls["n"] == 1, (len(items), calls["n"])


def test_a_sound_only_row_is_a_legal_training_item():
    """The recording is the evidence: a row with no text half must not be skipped as empty."""
    audio = _audio()
    enc = _encoder(audio)
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"audio": {"waveform": torch.zeros(16000)}},
             "questions": {"q": question}, "expected": {"q": "a"}}]
    items, skipped = items_from_rows(_StubTok(), rows, max_len=512, head_max_len=192,
                                     audio=audio, encode_audio=_encode_audio(enc))
    assert skipped == {} and len(items) == 1, skipped


def test_a_row_with_both_a_picture_and_a_sound_is_skipped_by_name():
    """One sequence carries one block; a dataset row naming both cannot be trained as either."""
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"audio": {"waveform": torch.zeros(16000)},
                       "image": {"pixels": torch.zeros(3, 32, 32)}, "text": "both"},
             "questions": {"q": question}, "expected": {"q": "a"}}]
    from laya.vision import build_vision_encoder, parse_vision

    vision = parse_vision({"vision": {"n_visual_tokens": 2, "image_size": 32}})
    vision_encoder = build_vision_encoder(vision, hidden_size=16, tower=_StubPixelTower()).eval()
    audio = _audio()
    enc = _encoder(audio)
    items, skipped = items_from_rows(
        _StubTok(), rows, max_len=512, head_max_len=192,
        vision=vision,
        encode_visual=lambda px: torch.zeros(4, 32), audio=audio,
        encode_audio=_encode_audio(enc))
    assert items == [] and skipped.get("both_modalities") == 1, skipped


class _StubPixelTower(torch.nn.Module):
    class config:
        hidden_size = 32

    def forward(self, pixel_values):
        b = pixel_values.size(0)
        return type("O", (), {"last_hidden_state": torch.zeros(b, 4, 32)})()


def test_two_recordings_are_not_the_same_input():
    """The held-out overlap check must compare the *recording*, not the state text around it.

    Every ESC-50 row shares the state text "a recording of an environmental sound", exactly as every
    CIFAR row shares "a photograph". The vision path learned this the hard way and keyed its input
    signature on the image; the audio path inherited only half of the fix, so an eval set built from
    held-out folds was reported as "overlapping training data" and the run refused to call its own
    numbers held out.
    """
    audio = _audio()
    enc = _encoder(audio)
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    # The ESC-50 shape: identical state text, different recordings.
    rows = [{"state": {"audio": {"waveform": torch.zeros(16000)}, "text": "same"},
             "questions": {"q": question}, "expected": {"q": "a"}},
            {"state": {"audio": {"waveform": torch.ones(16000)}, "text": "same"},
             "questions": {"q": question}, "expected": {"q": "b"}}]
    items, skipped = items_from_rows(_StubTok(), rows, max_len=512, head_max_len=192,
                                     audio=audio, encode_audio=_encode_audio(enc))
    assert skipped == {}, skipped
    keys = [it["audio_key"] for it in items]
    assert keys[0] != keys[1], "two different recordings must not share an input signature"
    # ... and the text half alone cannot tell them apart, which is the bug's shape.
    assert len({tuple(it["state_ids"]) for it in items}) == 1


def test_parallel_encoding_gives_the_same_items_as_serial():
    """`encode_workers` is a speed knob: the items must be identical, block for block.

    The build is where a media run spends most of its time (a frozen tower over every input), so the
    trainer encodes the distinct inputs in threads. The measurement behind the default: 9.2 clips/s
    serially against 18.9 with eight threads on this 56-core machine.
    """
    audio = _audio()
    enc = _encoder(audio)
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"audio": {"waveform": torch.full((16000,), float(i) / 8.0)}, "text": "t"},
             "questions": {"q": question}, "expected": {"q": "ab"[i % 2]}} for i in range(6)]
    serial, skipped_a = items_from_rows(_StubTok(), rows, max_len=512, head_max_len=192,
                                        audio=audio, encode_audio=_encode_audio(enc),
                                        encode_workers=1)
    parallel, skipped_b = items_from_rows(_StubTok(), rows, max_len=512, head_max_len=192,
                                          audio=audio, encode_audio=_encode_audio(enc),
                                          encode_workers=4)
    assert skipped_a == skipped_b == {}, (skipped_a, skipped_b)
    assert len(serial) == len(parallel) == 6
    for a, b in zip(serial, parallel):
        assert torch.equal(a["audio_frames"], b["audio_frames"])
        assert a["state_ids"] == b["state_ids"] and a["target"] == b["target"]
        assert a["audio_key"] == b["audio_key"]


def test_parallel_encoding_still_costs_one_tower_pass_per_recording():
    audio = _audio()
    enc = _encoder(audio)

    class _Counting:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def __call__(self, waveform):
            self.calls += 1
            return self.inner(waveform)

    counting = _Counting(_encode_audio(enc))
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"audio": {"waveform": torch.zeros(16000)}},
             "questions": {"q1": question, "q2": question}, "expected": {"q1": "a", "q2": "b"}}]
    items, _ = items_from_rows(_StubTok(), rows, max_len=512, head_max_len=192,
                               audio=audio, encode_audio=counting, encode_workers=4)
    assert len(items) == 2 and counting.calls == 1, (len(items), counting.calls)


def test_a_text_row_is_still_skipped_when_it_has_nothing_to_read():
    items, skipped = items_from_rows(_StubTok(), [{"state": "  ", "questions": {
        "q": {"type": "choice", "instructions": "pick", "criteria": ["a"]}},
        "expected": {"q": "a"}}], max_len=512, head_max_len=192)
    assert skipped.get("empty_text") == 1, skipped


# -- encoding and dropout --------------------------------------------------------------------

def test_encoding_an_audio_item_produces_a_plan_and_keeps_the_markers():
    audio = _audio()
    enc = _encoder(audio)
    items, _ = items_from_rows(_StubTok(), _rows(), max_len=512, head_max_len=192,
                               audio=audio, encode_audio=_encode_audio(enc))
    from laya.train import _project_audio

    sound_item = next(it for it in items if "audio_frames" in it)
    text_item = next(it for it in items if "audio_frames" not in it)
    plain = encode_item(_StubTok(), text_item, 512, 192)
    projected = _project_audio(sound_item, enc.projector)
    full = encode_item(_StubTok(), projected, 512, 192, audio=audio, max_positions=MAXPOS)
    dropped = encode_item(_StubTok(), projected, 512, 192, audio=audio,
                          drop_audio=True, max_positions=MAXPOS)
    assert "plan" in full and full["plan"].n_audio == 3
    assert "plan" not in dropped, "a dropped block must not plan positions for it"
    assert dropped["markers"] == plain["markers"], "text-only encode must match a text item"
    assert full["markers"] == plain["markers"], "the block is spliced after the markers"


def test_a_chunk_mixing_pictures_and_sounds_is_refused():
    from laya.train import collate_training_batch

    chunk = [{"ids": [1, 2], "markers": [1], "qtype": 0, "target": [1.0],
              "visual": torch.zeros(2, 4), "plan": None},
             {"ids": [1, 2], "markers": [1], "qtype": 0, "target": [1.0],
              "audio": torch.zeros(2, 4), "plan": None}]
    try:
        collate_training_batch(chunk, 0, embed_fn=lambda ids: torch.zeros(len(ids), 4))
    except ValueError as exc:
        assert "picture" in str(exc) and "sound" in str(exc), str(exc)
    else:
        raise AssertionError("a chunk mixing the two modalities must be refused")


# -- the objective ---------------------------------------------------------------------------

def test_an_audio_checkpoint_defaults_to_soft_ce_and_text_to_rlcd():
    assert TrainConfig().resolve_loss(has_vision=False, has_audio=True) == "soft-ce"
    assert TrainConfig().resolve_loss(has_vision=False, has_audio=False) == "rlcd"
    assert TrainConfig(loss="rlcd").resolve_loss(has_vision=False, has_audio=True) == "rlcd"


def test_a_media_run_can_throttle_its_recovery_checkpoint():
    """`checkpoint_every` is the knob for the largest non-compute cost of a media run.

    A checkpoint carries the encoder *and* the bundled tower -- ~1.2 GB -- and the trainer wrote one
    after every epoch. Measured here: ~6 minutes per write against 2.4 minutes of training per
    epoch, so the recovery copy cost more than the epoch it protects. `--checkpoint-every 0` keeps
    the final checkpoint and drops the per-epoch ones.
    """
    assert TrainConfig().checkpoint_every == 1
    for bad in (-1, True, 1.5):
        try:
            TrainConfig(checkpoint_every=bad).validate()
        except ValueError as exc:
            assert "checkpoint_every" in str(exc), exc
            continue
        raise AssertionError("checkpoint_every=%r should have been refused" % (bad,))
    TrainConfig(checkpoint_every=0).validate()
    TrainConfig(checkpoint_every=5).validate()


def test_a_bad_audio_block_is_refused_at_argument_time():
    for block in ({"n_audio_tokens": 0}, {"sample_rate": 0}, {"nonsense": 1}):
        try:
            TrainConfig(audio=block).validate()
        except (ValueError, TypeError):
            continue
        raise AssertionError("audio block %r should have been refused" % (block,))
    try:
        TrainConfig(audio=["not", "a", "mapping"]).validate()
    except ValueError as exc:
        assert "mapping" in str(exc), exc
    else:
        raise AssertionError("a non-mapping audio block must be refused")


# -- one real optimization step --------------------------------------------------------------

def test_a_training_step_moves_the_loss_through_the_embeddings_path():
    torch.manual_seed(0)
    model = _model()
    tok = _StubTok()
    audio = _audio()
    enc = _encoder(audio)
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"audio": {"waveform": torch.full((16000,), float(i))}, "text": "t%d" % i},
             "questions": {"q": question}, "expected": {"q": "ab"[i % 2]}} for i in range(8)]
    items, skipped = items_from_rows(tok, rows, max_len=512, head_max_len=192,
                                     audio=audio, encode_audio=_encode_audio(enc))
    assert skipped == {} and len(items) == 8

    embeds = model.encoder.embeddings
    table = getattr(embeds, "tok_embeddings", None) or embeds.word_embeddings
    torch.manual_seed(0)
    before = model.scorer[3].weight.detach().clone()
    cfg = TrainConfig(epochs=2, micro_batch=4, grad_accum=1, encoder_lr=1e-3, head_lr=1e-2,
                      log_every=0, loss="soft-ce")
    history = train_model(model, tok, items, cfg, torch.device("cpu"), 512, 192,
                          audio=audio, embed_fn=table, max_positions=MAXPOS)
    assert len(history) == 2, history
    assert all(isinstance(v, float) and v == v for v in history), history
    assert not torch.equal(before, model.scorer[3].weight.detach()), \
        "the head did not move: no backward pass reached it through the audio batch"


def test_the_audio_block_changes_the_training_logits():
    """The training path is wired to the sound, not merely able to run without crashing."""
    from laya.train import _forward, _project_audio, collate_training_batch

    torch.manual_seed(0)
    model = _model().eval()
    tok = _StubTok()
    audio = _audio()
    enc = _encoder(audio)
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"audio": {"waveform": torch.zeros(16000)}, "text": "same"},
             "questions": {"q": question}, "expected": {"q": "a"}}]
    items, _ = items_from_rows(tok, rows, max_len=512, head_max_len=192,
                               audio=audio, encode_audio=_encode_audio(enc))
    embeds = model.encoder.embeddings
    table = getattr(embeds, "tok_embeddings", None) or embeds.word_embeddings

    def logits(frames):
        encoded = encode_item(tok, _project_audio(dict(items[0], audio_frames=frames),
                                                  enc.projector),
                              512, 192, audio=audio, max_positions=MAXPOS)
        batch = collate_training_batch([encoded], tok.pad_token_id, embed_fn=table)
        with torch.no_grad():
            return _forward(model, batch, torch.device("cpu"), False, False)

    a = logits(items[0]["audio_frames"])
    b = logits(items[0]["audio_frames"] * 3.0)
    assert not torch.allclose(a, b), "the audio block must reach the training logits"


class _SignalTower(torch.nn.Module):
    """A tower whose frames carry the class: the synthetic-learning test's stand-in.

    A real wav2vec2 would make this a benchmark rather than a check, so the frames encode the class
    directly: two fixed, *differently directed* patterns.

    Direction, not scale, deliberately. The encoder applies its own LayerNorm to the spliced block,
    which subtracts a per-token mean and normalises length -- so a class encoded as "louder" is the
    same block to the model as one encoded as "quieter" (the trap `laya.vision` documents at
    length). Encoding the class as a different direction is what makes the task learnable at all.
    """

    class config:
        hidden_size = 32

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.patterns = torch.randn(2, 8, 32, generator=generator)

    def forward(self, input_values):
        # Count zero crossings, not amplitude: the loader peak-normalises every waveform, so a
        # "loud" versus "quiet" class is one class to this path (a real finding, not a test bug).
        crossings = (torch.sign(input_values[:, 1:]) != torch.sign(input_values[:, :-1])).sum(-1)
        which = (crossings > 10).long()                              # [B]
        return type("O", (), {"last_hidden_state": self.patterns[which]})()


def test_the_audio_block_can_learn_a_task_end_to_end():
    """The check the benchmark needed: a loss that falls through the block is not the same as a task
    the block can learn.

    On ESC-50 the first engine arm sat at chance, which had two possible causes -- a broken path, or
    a recipe that never trained the head. A two-class synthetic task separated them: this test is
    the same experiment in miniature, and it has to reach high accuracy, not merely move a weight.
    """
    torch.manual_seed(0)
    model = _model()
    tok = _StubTok()
    audio = _audio()
    enc = build_audio_encoder(audio, hidden_size=16, tower=_SignalTower()).eval()
    model.audio_encoder = enc

    question = {"type": "choice", "instructions": "Which tone is this?", "criteria": ["low", "high"]}
    rows = []
    for i in range(24):
        label = "low" if i % 2 == 0 else "high"
        rate = 4 if label == "low" else 40
        wave = torch.sign(torch.sin(torch.linspace(0, 2 * 3.14159 * rate, 1600)))
        rows.append({"state": {"audio": {"waveform": wave}, "text": "a tone"},
                     "questions": {"q": question}, "expected": {"q": label}})
    items, skipped = items_from_rows(tok, rows, max_len=512, head_max_len=192,
                                     audio=audio, encode_audio=_encode_audio(enc))
    assert skipped == {} and len(items) == 24, (skipped, len(items))
    embeds = model.encoder.embeddings
    table = getattr(embeds, "tok_embeddings", None) or embeds.word_embeddings

    cfg = TrainConfig(epochs=6, micro_batch=8, grad_accum=1, head_lr=1e-2, log_every=0,
                      loss="soft-ce", freeze_encoder=True, seed=0)
    history = train_model(model, tok, items, cfg, torch.device("cpu"), 512, 192,
                          audio=audio, embed_fn=table, max_positions=MAXPOS)
    from laya.train import calibration_records, evaluate_records
    records = calibration_records(model, tok, items, torch.device("cpu"), 512, 192,
                                  audio=audio, embed_fn=table, max_positions=MAXPOS,
                                  audio_projector=enc.projector)
    report = evaluate_records(records, [1.0, 1.0, 1.0], {})
    assert history[-1] < history[0], history
    assert report["accuracy"] > 0.75, (report["accuracy"], history)


def test_training_with_an_audio_encoder_attached_projects_inside_the_step():
    """The wiring `finetune` uses: the projector has to be found on the model to be optimised."""
    torch.manual_seed(0)
    model = _model()
    tok = _StubTok()
    audio = _audio()
    enc = _encoder(audio)
    model.audio_encoder = enc                        # exactly what finetune does
    question = {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}
    rows = [{"state": {"audio": {"waveform": torch.full((16000,), float(i))}, "text": "t%d" % i},
             "questions": {"q": question}, "expected": {"q": "ab"[i % 2]}} for i in range(6)]
    items, _ = items_from_rows(tok, rows, max_len=512, head_max_len=192,
                               audio=audio, encode_audio=_encode_audio(enc))
    embeds = model.encoder.embeddings
    table = getattr(embeds, "tok_embeddings", None) or embeds.word_embeddings

    before = enc.projector.proj.weight.detach().clone()
    cfg = TrainConfig(epochs=1, micro_batch=3, grad_accum=1, head_lr=1e-2, log_every=0,
                      loss="soft-ce", freeze_encoder=True)
    train_model(model, tok, items, cfg, torch.device("cpu"), 512, 192,
                audio=audio, embed_fn=table, max_positions=MAXPOS)
    assert not torch.equal(before, enc.projector.proj.weight.detach()), \
        "the projector did not move: it is not in the optimizer"


if __name__ == "__main__":
    test_eval_rows_are_built_through_the_audio_path()
    test_the_eval_report_scores_the_sound_not_just_the_state_text()
    test_a_resumed_audio_checkpoint_keeps_its_trained_projector()
    test_an_audio_checkpoint_without_projector_weights_is_refused()
    test_a_text_base_checkpoint_starts_a_new_audio_projector()
    test_sound_rows_become_items_that_carry_an_audio_block()
    test_one_recording_asked_many_questions_costs_one_tower_pass()
    test_a_sound_only_row_is_a_legal_training_item()
    test_a_row_with_both_a_picture_and_a_sound_is_skipped_by_name()
    test_two_recordings_are_not_the_same_input()
    test_parallel_encoding_gives_the_same_items_as_serial()
    test_parallel_encoding_still_costs_one_tower_pass_per_recording()
    test_a_text_row_is_still_skipped_when_it_has_nothing_to_read()
    test_encoding_an_audio_item_produces_a_plan_and_keeps_the_markers()
    test_a_chunk_mixing_pictures_and_sounds_is_refused()
    test_an_audio_checkpoint_defaults_to_soft_ce_and_text_to_rlcd()
    test_a_media_run_can_throttle_its_recovery_checkpoint()
    test_a_bad_audio_block_is_refused_at_argument_time()
    test_a_training_step_moves_the_loss_through_the_embeddings_path()
    test_the_audio_block_changes_the_training_logits()
    test_training_with_an_audio_encoder_attached_projects_inside_the_step()
    test_the_audio_block_can_learn_a_task_end_to_end()
    print("all audio training tests passed")
