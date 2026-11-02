"""Tests for the two vision gates (laya/evals_vision.py).

A scripted runner stands in for a checkpoint, so these run weight-free: each test states the
behaviour it wants to detect and then checks the gate reports it. The three cases worth pinning
are the ones a gate exists to catch -- a model that ignores the image, a model whose answers move
when the caller reorders a list, and a pass whose own noise would be mistaken for a finding.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.evals import Dataset, Example
from laya.evals_vision import (image_ablation, option_order_flip_rate, permuted_questions,
                               strip_image)


class _Scripted:
    """A runner that answers from a function of (state, question id)."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def predict(self, state, questions, model=None):
        self.calls.append(state)
        return {"answers": {qid: self.answer(state, qid) for qid in questions}}


def _choice(value):
    return {"type": "choice", "choice": value, "probabilities": {value: 1.0}, "confidence": 1.0}


def _row(state, expected="a", tags=()):
    return Example(state=state, questions={"q": {"type": "choice", "instructions": "pick",
                                                 "criteria": ["a", "b"]}},
                   expected={"q": expected}, tags=tags)


# -- ablation --------------------------------------------------------------------------------

def test_a_model_that_reads_the_image_shows_a_gain():
    runner = _Scripted(lambda state, qid: _choice("a" if isinstance(state, dict) and "image" in state
                                                  else "b"))
    report = image_ablation(runner, Dataset([_row({"image": "x.png", "text": "t"})]), min_gain=0.35)
    assert report["gain"] == 1.0, report
    assert report["passed"] is True


def test_a_model_that_ignores_the_image_shows_no_gain_and_fails_the_gate():
    """The documented default failure: the answer does not move when the pixels go away."""
    runner = _Scripted(lambda state, qid: _choice("a"))
    report = image_ablation(runner, Dataset([_row({"image": "x.png", "text": "t"})]), min_gain=0.35)
    assert report["gain"] == 0.0, report
    assert report["passed"] is False, "a text model in a vision costume must not pass gate 1"


def test_text_only_rows_are_excluded_from_the_gain():
    """They have no image to remove; including them would dilute the number toward zero."""
    runner = _Scripted(lambda state, qid: _choice("a" if isinstance(state, dict) and "image" in state
                                                  else "b"))
    dataset = Dataset([_row({"image": "x.png", "text": "t"}), _row("plain text")])
    report = image_ablation(runner, dataset)
    assert report["rows_with_image"] == 1 and report["rows_text_only"] == 1
    assert report["gain"] == 1.0, report


def test_an_image_only_state_is_stripped_to_text_not_to_an_empty_object():
    assert strip_image({"image": "x.png"}) == "", "an empty dict would still be serialized"
    assert strip_image({"image": "x.png", "text": "hi"}) == {"text": "hi"}
    assert strip_image("plain") == "plain"


def test_a_non_deterministic_pass_is_reported_rather_than_taken_as_a_finding():
    """If the no-op pass moves a text row, every other number in the report is noise."""
    counter = {"n": 0}

    def flaky(state, qid):
        counter["n"] += 1
        return _choice("a" if counter["n"] % 2 else "b")

    report = image_ablation(_Scripted(flaky), Dataset([_row("plain text")]))
    assert report["text_only_unchanged"] is False, report
    assert report["passed"] is None or report["passed"] is False


def test_per_tag_slices_are_reported():
    runner = _Scripted(lambda state, qid: _choice("a" if isinstance(state, dict) and "image" in state
                                                  else "b"))
    dataset = Dataset([_row({"image": "x", "text": "t"}, tags=("doc.field",)),
                       _row({"image": "x", "text": "t"}, tags=("chart.read",))])
    report = image_ablation(runner, dataset)
    assert set(report["per_tag"]) == {"all", "doc.field", "chart.read"}, report["per_tag"]
    assert report["per_tag"]["doc.field"]["gain"] == 1.0


def test_a_missing_answer_is_not_scored_but_a_malformed_one_is():
    """Two different failures, deliberately scored differently.

    A question the runner returned nothing for is not evidence about the model, so it leaves the
    denominator alone. An answer object that *is* a choice but carries no chosen label is a broken
    response and counts as wrong -- silently dropping it would let a runner that answers garbage on
    every row publish `questions: 0` and look clean.
    """
    absent = _Scripted(lambda state, qid: None)
    report = image_ablation(absent, Dataset([_row({"image": "x"})]))
    assert report["per_tag"]["all"]["questions"] == 0, report

    malformed = _Scripted(lambda state, qid: {"type": "choice"})
    report = image_ablation(malformed, Dataset([_row({"image": "x"})]))
    assert report["per_tag"]["all"]["questions"] == 1, report
    assert report["per_tag"]["all"]["with_image"] == 0.0, report


# -- option-order flip -----------------------------------------------------------------------

def test_a_stable_model_has_a_zero_flip_rate():
    runner = _Scripted(lambda state, qid: _choice("a"))
    report = option_order_flip_rate(runner, Dataset([_row({"image": "x"})]), max_flip_rate=0.02)
    assert report["flip_rate"] == 0.0, report
    assert report["passed"] is True


def test_a_model_that_reads_the_option_order_is_caught():
    """The failure the gate exists for: the answer follows position, not content."""
    def positional(state, qid):
        return _choice("a")

    class _ReorderSensitive(_Scripted):
        def predict(self, state, questions, model=None):
            q = questions["q"]
            if q.get("option_order"):
                # whatever was in slot 0 after the shuffle becomes the answer
                names = list(q["criteria"].keys()) if isinstance(q["criteria"], dict) else list(q["criteria"])
                chosen = names[q["option_order"][0]]
                return {"answers": {"q": _choice(chosen)}}
            return {"answers": {"q": _choice("a")}}

    report = option_order_flip_rate(_ReorderSensitive(lambda s, q: _choice("a")),
                                    Dataset([_row({"image": "x"})]), seed=1, max_flip_rate=0.02)
    assert report["flip_rate"] == 1.0, report
    assert report["passed"] is False


def test_only_choice_questions_are_compared():
    def answer(state, qid):
        return {"type": "noul", "noul": 0.9}

    example = Example(state="s", questions={"n": {"type": "noul", "instructions": "x"}},
                      expected={"n": True})
    report = option_order_flip_rate(_Scripted(answer), Dataset([example]))
    assert report["questions_compared"] == 0, report
    assert report["flip_rate"] is None


def test_permuting_preserves_the_question_and_only_reorders():
    rng = __import__("random").Random(7)
    original = {"q": {"type": "choice", "instructions": "pick", "criteria": ["a", "b", "c"]}}
    out = permuted_questions(original, rng)
    assert out["q"]["instructions"] == "pick"
    assert sorted(out["q"]["option_order"]) == [0, 1, 2], out["q"]["option_order"]
    assert "option_order" not in original["q"], "the caller's questions must not be mutated"
    assert len(out["q"]["criteria"]) == 3


def test_a_single_option_question_is_passed_through_unpermuted():
    rng = __import__("random").Random(0)
    out = permuted_questions({"q": {"type": "choice", "criteria": ["only"]}}, rng)
    assert "option_order" not in out["q"]


def test_the_same_seed_gives_the_same_permutation():
    a = permuted_questions({"q": {"type": "choice", "criteria": ["a", "b", "c", "d"]}},
                           __import__("random").Random(11))
    b = permuted_questions({"q": {"type": "choice", "criteria": ["a", "b", "c", "d"]}},
                           __import__("random").Random(11))
    assert a == b


def test_a_bad_seed_is_refused():
    try:
        option_order_flip_rate(_Scripted(lambda s, q: _choice("a")), Dataset([]), seed=True)
    except ValueError as e:
        assert "seed" in str(e), e
    else:
        raise AssertionError("a bool seed must be refused")


if __name__ == "__main__":
    test_a_model_that_reads_the_image_shows_a_gain()
    test_a_model_that_ignores_the_image_shows_no_gain_and_fails_the_gate()
    test_text_only_rows_are_excluded_from_the_gain()
    test_an_image_only_state_is_stripped_to_text_not_to_an_empty_object()
    test_a_non_deterministic_pass_is_reported_rather_than_taken_as_a_finding()
    test_per_tag_slices_are_reported()
    test_a_missing_answer_is_not_scored_but_a_malformed_one_is()
    test_a_stable_model_has_a_zero_flip_rate()
    test_a_model_that_reads_the_option_order_is_caught()
    test_only_choice_questions_are_compared()
    test_permuting_preserves_the_question_and_only_reorders()
    test_a_single_option_question_is_passed_through_unpermuted()
    test_the_same_seed_gives_the_same_permutation()
    test_a_bad_seed_is_refused()
    print("all vision gate tests passed")
