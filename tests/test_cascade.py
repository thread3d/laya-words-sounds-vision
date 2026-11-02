"""Tests for the tier policy (spec §4.5).

A scripted runner stands in for checkpoints, so these run weight-free: the point is that the
*policy* escalates on the decision's own calibrated confidence, stops as soon as a tier is sure
enough, and reports what it did.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.cascade import Cascade, Tier, _weakest_confidence


def _answer(value, confidence=0.9):
    return {"type": "choice", "choice": value, "answer_confidence": confidence,
            "probabilities": {value: confidence}}


class _Runner:
    """A runner that answers from a per-model table of (value, confidence)."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def predict(self, state, questions, model=None, **kwargs):
        self.calls.append(model)
        value, confidence = self.table[model]
        return {"answers": {"q": _answer(value, confidence)}, "model": model}


QUESTION = {"q": {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}}


def _cascade(table, **tiers):
    return Cascade([Tier(**t) for t in tiers.get("tiers", [
        {"name": "fast", "model": "fast", "min_confidence": 0.70},
        {"name": "slow", "model": "slow"},
    ])], runner=_Runner(table))


# -- validation ------------------------------------------------------------------------------

def test_a_cascade_needs_at_least_one_tier():
    try:
        Cascade([])
    except ValueError as exc:
        assert "at least one tier" in str(exc), exc
    else:
        raise AssertionError("an empty cascade must be refused")


def test_a_tier_above_the_last_must_state_a_floor():
    """Without one it would always "pass", and the rest of the cascade would be dead weight."""
    try:
        Cascade([Tier("a", "m"), Tier("b", "n")])
    except ValueError as exc:
        assert "nothing below it" in str(exc), exc
    else:
        raise AssertionError("a non-terminal tier with no floor can never escalate")


def test_the_terminal_tier_must_not_state_a_floor():
    """There is nothing below it: a floor there would refuse answers it produced."""
    try:
        Cascade([Tier("a", "m", 0.5), Tier("b", "n", 0.6)])
    except ValueError as exc:
        assert "nothing below it" in str(exc), exc
    else:
        raise AssertionError("a floor on the terminal tier would refuse answers it produced")


def test_a_single_tier_cascade_is_allowed_and_is_just_the_model():
    """Degenerate but coherent: it always answers, and reports which rung did."""
    out = Cascade([Tier("only", "m")], runner=_Runner({"m": ("a", 0.01)})).predict({}, QUESTION)
    assert out["cascade"]["tier"] == "only"
    assert out["cascade"]["escalations"] == []


def test_tiers_are_validated():
    for bad in ([Tier("", "m", 0.5), Tier("b", "n")], [Tier("a", "", 0.5), Tier("b", "n")],
                [Tier("a", "m", 1.5), Tier("b", "n")], [Tier("a", "m", True), Tier("b", "n")]):
        try:
            Cascade(bad)
        except ValueError:
            continue
        raise AssertionError("tiers %r should have been refused" % (bad,))


def test_tier_names_must_be_distinct():
    try:
        Cascade([Tier("same", "m", 0.5), Tier("same", "n")])
    except ValueError as exc:
        assert "distinct" in str(exc), exc
    else:
        raise AssertionError("two tiers with one name cannot be told apart in a report")


def test_the_tier_ceiling_is_a_latency_guard():
    try:
        Cascade([Tier("t%d" % i, "m%d" % i, 0.5) for i in range(5)] + [Tier("last", "m")])
    except ValueError as exc:
        assert "latency" in str(exc), exc
    else:
        raise AssertionError("an unbounded cascade is a latency accident")


# -- serving ---------------------------------------------------------------------------------

def test_a_confident_tier_answers_and_nothing_escalates():
    cascade = _cascade({"fast": ("a", 0.95), "slow": ("b", 0.99)})
    out = cascade.predict({"text": "x"}, QUESTION)
    assert out["answers"]["q"]["choice"] == "a", "the fast tier answered; it must be the answer"
    assert out["cascade"]["tier"] == "fast"
    assert out["cascade"]["index"] == 0
    assert out["cascade"]["escalations"] == []
    assert cascade.runner.calls == ["fast"], "no second call for a confident answer"


def test_a_doubtful_tier_escalates_and_records_why():
    cascade = _cascade({"fast": ("a", 0.55), "slow": ("b", 0.91)})
    out = cascade.predict({"text": "x"}, QUESTION)
    assert out["answers"]["q"]["choice"] == "b", "the escalated tier's answer is the result"
    assert out["cascade"]["tier"] == "slow"
    assert cascade.runner.calls == ["fast", "slow"]
    record = out["cascade"]["escalations"][0]
    assert record["from"] == "fast" and record["to"] == "slow"
    assert record["min_confidence"] == 0.70 and record["answer_confidence"] == 0.55


def test_the_weakest_answer_decides_not_the_average():
    """Four confident answers do not excuse one doubtful one."""
    class _Mixed(_Runner):
        def predict(self, state, questions, model=None, **kwargs):
            self.calls.append(model)
            if model == "fast":
                return {"answers": {"a": _answer("x", 0.95), "b": _answer("y", 0.40)}}
            return {"answers": {"a": _answer("x", 0.99), "b": _answer("y", 0.98)}}

    cascade = Cascade([Tier("fast", "fast", 0.70), Tier("slow", "slow")],
                       runner=_Mixed({}))
    out = cascade.predict({}, {"a": QUESTION["q"], "b": QUESTION["q"]})
    assert out["cascade"]["tier"] == "slow", out["cascade"]
    assert out["cascade"]["escalations"][0]["answer_confidence"] == 0.40


def test_the_terminal_tier_answers_even_when_it_is_not_confident():
    """There is nothing below it; refusing would turn a low confidence into an error."""
    cascade = _cascade({"fast": ("a", 0.10), "slow": ("b", 0.20)})
    out = cascade.predict({"text": "x"}, QUESTION)
    assert out["cascade"]["tier"] == "slow"
    assert out["answers"]["q"]["choice"] == "b"
    assert len(out["cascade"]["escalations"]) == 1


def test_a_response_with_no_confidence_does_not_escalate_on_a_guess():
    class _NoConfidence(_Runner):
        def predict(self, state, questions, model=None, **kwargs):
            self.calls.append(model)
            return {"answers": {"q": {"type": "choice", "choice": "a"}}}

    cascade = Cascade([Tier("fast", "fast", 0.70), Tier("slow", "slow")], runner=_NoConfidence({}))
    out = cascade.predict({}, QUESTION)
    assert out["cascade"]["tier"] == "fast"
    assert out["cascade"]["escalations"] == [], "an unmeasured confidence is not a low one"


def test_the_cascade_adds_metadata_and_changes_nothing_else():
    runner = _Runner({"fast": ("a", 0.95), "slow": ("b", 0.99)})
    cascade = Cascade([Tier("fast", "fast", 0.70), Tier("slow", "slow")], runner=runner)
    plain = runner.predict({}, QUESTION, model="fast")
    out = cascade.predict({}, QUESTION)
    assert {k: v for k, v in out.items() if k != "cascade"} == plain, \
        "the tier's own response must come back untouched"


def test_weakest_confidence_reads_the_calibrated_number():
    assert _weakest_confidence({"answers": {"q": _answer("a", 0.8)}}) == 0.8
    assert _weakest_confidence({"answers": {}}) is None
    assert _weakest_confidence({}) is None
    assert _weakest_confidence({"answers": {"q": {"type": "choice"}}}) is None
    # `confidence` is a different quantity per question type; only answer_confidence compares.
    assert _weakest_confidence({"answers": {"q": {"type": "choice", "confidence": 0.1}}}) is None


if __name__ == "__main__":
    test_a_cascade_needs_at_least_one_tier()
    test_a_tier_above_the_last_must_state_a_floor()
    test_the_terminal_tier_must_not_state_a_floor()
    test_a_single_tier_cascade_is_allowed_and_is_just_the_model()
    test_tiers_are_validated()
    test_tier_names_must_be_distinct()
    test_the_tier_ceiling_is_a_latency_guard()
    test_a_confident_tier_answers_and_nothing_escalates()
    test_a_doubtful_tier_escalates_and_records_why()
    test_the_weakest_answer_decides_not_the_average()
    test_the_terminal_tier_answers_even_when_it_is_not_confident()
    test_a_response_with_no_confidence_does_not_escalate_on_a_guess()
    test_the_cascade_adds_metadata_and_changes_nothing_else()
    test_weakest_confidence_reads_the_calibrated_number()
    print("all cascade tests passed")
