"""Tests for the sound routing rule.

The rule is deliberately narrow: a state carrying a sound goes to the audio checkpoint, but only when
one is registered. Everything here is weight-free -- `Router.route` decides without loading anything
-- so these run in the same fast lane as `tests/test_router.py`, and mirror the vision rule's tests
because the two rules are the same shape.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.audio import has_audio
from laya.router import Router

AUDIO = {"audio": "acme/laya-audio"}


def test_has_audio_recognises_every_reference_shape():
    assert has_audio({"audio_path": "a.wav"})
    assert has_audio({"audio": {"waveform": 1}})
    assert has_audio({"audio_url": "https://example.com/a.wav"})
    assert has_audio({"audio_base64": "AAAA"})
    assert has_audio({"audio_waveform": [0.0, 0.1]})
    assert not has_audio({"message": "no sound here"})
    assert not has_audio("plain text")
    assert not has_audio(["a", "list"])


def test_a_sound_state_routes_to_the_audio_checkpoint():
    router = Router(models=AUDIO)
    decision = router.route({"audio_path": "a.wav", "text": "is this a dog barking?"})
    assert decision.model == "audio", decision
    assert "sound" in decision.reason, decision.reason


def test_an_explicit_model_still_wins():
    router = Router(models=AUDIO)
    assert router.route({"audio_path": "a.wav"}, model="multilingual").model == "multilingual"
    assert router.route({"audio_path": "a.wav"}, task="typed-decisions").model == "typed-decisions"


def test_a_text_state_is_unaffected():
    router = Router(models=AUDIO)
    assert router.route({"message": "I was charged twice"}).model == "english"
    assert router.route({"message": "Mein Konto wurde zweimal belastet"}).model == "multilingual"


def test_without_a_registered_audio_checkpoint_nothing_changes():
    """The rule is opt-in in the sense that matters: no checkpoint, no change (C6)."""
    router = Router()
    decision = router.route({"audio_path": "a.wav", "message": "hello there"})
    assert decision.model == "multilingual", decision
    assert "sound" not in decision.reason, decision.reason


def test_the_rule_can_be_turned_off():
    router = Router(models=AUDIO, audio_model=None)
    decision = router.route({"audio_path": "a.wav", "message": "hello there"})
    assert decision.model == "multilingual", decision


def test_picking_a_checkpoint_beats_the_sound_rule_but_picking_a_language_does_not():
    """`model=`/`task=` name a checkpoint and win; `lang=` names a *language* and does not.

    The split is the same one the image rule makes: a text checkpoint cannot hear a recording,
    whatever language the text beside it is in, so a language hint does not override which
    checkpoint can answer at all.
    """
    router = Router(models=AUDIO)
    assert router.route({"audio_path": "a.wav", "message": "bonjour"}).model == "audio"
    assert router.route({"audio_path": "a.wav"}, lang="de").model == "audio"
    assert router.route({"audio_path": "a.wav"}, lang="de",
                        model="multilingual").model == "multilingual"


def test_a_picture_rule_wins_over_a_sound_rule():
    """A state carrying both is out of scope at the model, so routing has to pick one deterministically."""
    router = Router(models={"vision": "acme/laya-vision", "audio": "acme/laya-audio"})
    decision = router.route({"image_path": "a.png", "audio_path": "a.wav"})
    assert decision.model == "vision", decision
    assert "image" in decision.reason, decision.reason


if __name__ == "__main__":
    test_has_audio_recognises_every_reference_shape()
    test_a_sound_state_routes_to_the_audio_checkpoint()
    test_an_explicit_model_still_wins()
    test_a_text_state_is_unaffected()
    test_without_a_registered_audio_checkpoint_nothing_changes()
    test_the_rule_can_be_turned_off()
    test_picking_a_checkpoint_beats_the_sound_rule_but_picking_a_language_does_not()
    test_a_picture_rule_wins_over_a_sound_rule()
    print("all audio routing tests passed")
