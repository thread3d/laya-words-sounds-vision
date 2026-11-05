"""Tests for the sound gates (`laya/evals_audio.py`).

A scripted runner stands in for a checkpoint, so these run weight-free and need no decoder: each
test states the behaviour it wants to detect and then checks the gate reports it. The mirrors of
`tests/test_evals_vision.py` are deliberate -- the two gates share `laya.evals_media.ablation`, and
a difference between these suites is a difference in the measurement.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.evals import Dataset, Example
from laya.evals_audio import (audio_ablation, option_order_flip_rate, permuted_questions,
                              strip_audio)


class _Scripted:
    """A runner that answers from a function of (state, question id)."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def predict(self, state, questions, model=None):
        self.calls.append(state)
        return {"answers": {qid: self.answer(state, qid) for qid in questions}}


def _choice(value, probabilities=None):
    return {"type": "choice", "choice": value,
            "probabilities": probabilities or {value: 1.0}, "confidence": 1.0}


def _row(state, expected="a", tags=()):
    return Example(state=state, questions={"q": {"type": "choice", "instructions": "pick",
                                                 "criteria": ["a", "b"]}},
                   expected={"q": expected}, tags=tags)


def _hears(state) -> bool:
    return isinstance(state, dict) and "audio" in state


# -- ablation --------------------------------------------------------------------------------

def test_a_model_that_listens_shows_a_gain():
    runner = _Scripted(lambda state, qid: _choice("a" if _hears(state) else "b"))
    report = audio_ablation(runner, Dataset([_row({"audio": {"waveform": 1}, "text": "t"})]),
                            min_gain=0.35)
    assert report["gate"] == "audio-necessity", report
    assert report["gain"] == 1.0, report
    assert report["passed"] is True


def test_a_model_that_ignores_the_recording_shows_no_gain_and_fails_the_gate():
    """The documented default failure: the answer does not move when the sound goes away."""
    runner = _Scripted(lambda state, qid: _choice("a"))
    report = audio_ablation(runner, Dataset([_row({"audio_path": "x.wav", "text": "t"})]),
                            min_gain=0.35)
    assert report["gain"] == 0.0, report
    assert report["passed"] is False, "a text model in an audio costume must not pass the gate"


def test_text_only_rows_are_excluded_from_the_gain():
    """They have no recording to remove; including them would dilute the number toward zero."""
    runner = _Scripted(lambda state, qid: _choice("a" if _hears(state) else "b"))
    dataset = Dataset([_row({"audio": {"waveform": 1}, "text": "t"}), _row("plain text")])
    report = audio_ablation(runner, dataset)
    assert report["rows_with_audio"] == 1 and report["rows_text_only"] == 1, report
    assert report["gain"] == 1.0, report


def test_a_sound_only_state_is_stripped_to_text_not_to_an_empty_object():
    assert strip_audio({"audio": {"waveform": 1}}) == "", "an empty dict would still be serialized"
    assert strip_audio({"audio": {"waveform": 1}, "text": "hi"}) == {"text": "hi"}
    assert strip_audio("plain") == "plain"


def test_every_sound_key_is_stripped():
    for key in ("audio", "audio_path", "audio_base64", "audio_waveform"):
        assert strip_audio({key: "x"}) == "", key
        assert strip_audio({key: "x", "text": "t"}) == {"text": "t"}, key


def test_a_non_deterministic_pass_is_reported_rather_than_taken_as_a_finding():
    """If the no-op pass moves a text row, every other number in the report is noise."""
    counter = {"n": 0}

    def flaky(state, qid):
        counter["n"] += 1
        return _choice("a" if counter["n"] % 2 else "b")

    report = audio_ablation(_Scripted(flaky), Dataset([_row("plain text")]))
    assert report["text_only_unchanged"] is False, report
    assert report["passed"] is None or report["passed"] is False


def test_per_tag_slices_are_reported():
    runner = _Scripted(lambda state, qid: _choice("a" if _hears(state) else "b"))
    dataset = Dataset([_row({"audio": {"waveform": 1}, "text": "t"}, tags=("speech.urgent",))])
    report = audio_ablation(runner, dataset)
    assert report["per_tag"]["speech.urgent"]["gain"] == 1.0, report["per_tag"]
    assert report["per_tag"]["all"]["with_audio"] == 1.0, report["per_tag"]


def test_a_run_below_the_threshold_fails_and_one_above_it_passes():
    runner = _Scripted(lambda state, qid: _choice("a" if _hears(state) else "b"))
    dataset = Dataset([_row({"audio": {"waveform": 1}, "text": "t"})] * 3)
    assert audio_ablation(runner, dataset, min_gain=0.9)["passed"] is True
    # An input that makes the answer *worse* is a negative gain, and it must fail the gate just
    # as a gain of zero does -- the gate asks for a minimum, not for movement.
    confused = _Scripted(lambda state, qid: _choice("a" if _hears(state) else "b"))
    report = audio_ablation(confused,
                            Dataset([_row({"audio": {"waveform": 1}}, expected="b")] * 2),
                            min_gain=0.35)
    assert report["gain"] == -1.0, report
    assert report["passed"] is False, report


# -- the option-order gate is modality-agnostic ----------------------------------------------

def test_permuting_a_sound_question_is_the_same_question():
    """The gate permutes the *question*, so it does not care what input the state carries."""
    questions = {"q": {"type": "choice", "instructions": "pick", "criteria": ["a", "b", "c"]}}
    import random
    permuted = permuted_questions(questions, random.Random(0))
    assert sorted(permuted["q"].get("option_order", [])) == [0, 1, 2]
    assert list(permuted["q"]["criteria"]) == ["a", "b", "c"], "the labels must not be re-indexed"


def test_a_sound_runner_s_flip_rate_is_reported():
    def flipping(state, qid):
        # Answer from the first option in the order it was given: the classic order-dependent bug.
        return _choice("a" if not _hears(state) else "b")

    report = option_order_flip_rate(_Scripted(flipping), Dataset([_row({"audio": "x"})]))
    assert report["gate"] == "option-order-invariance"
    assert report["questions_compared"] == 1, report
    assert report["passed"] is None, "no threshold means reported, not judged"


if __name__ == "__main__":
    test_a_model_that_listens_shows_a_gain()
    test_a_model_that_ignores_the_recording_shows_no_gain_and_fails_the_gate()
    test_text_only_rows_are_excluded_from_the_gain()
    test_a_sound_only_state_is_stripped_to_text_not_to_an_empty_object()
    test_every_sound_key_is_stripped()
    test_a_non_deterministic_pass_is_reported_rather_than_taken_as_a_finding()
    test_per_tag_slices_are_reported()
    test_a_run_below_the_threshold_fails_and_one_above_it_passes()
    test_permuting_a_sound_question_is_the_same_question()
    test_a_sound_runner_s_flip_rate_is_reported()
    print("all audio gate tests passed")
