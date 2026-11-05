"""The two vision gates from `docs/vision-decision-system.md`, §9.1 and §9.2b.

`laya.evals` scores answers against labels. These two functions ask different questions, and both
of them are the kind that a plausible-looking model can fail:

**Gate 1 — vision necessity.** Run the same items twice: once as given, once with the image
removed. If accuracy barely moves, the "vision" model is a text model with expensive extra
parameters and the image is not reaching the decision. That is the documented default failure of
multimodal fusion (models attend to text over images unless something forces them not to), and it
is the one failure a benchmark of accuracy alone cannot see.

**Gate 2b — option-order invariance.** Re-run each choice question with its options permuted. A
model that scores options jointly can learn to compare option *i* against option *j* rather than
against the image, and the symptom is published: 8-10% of answers flip under a permutation that
carries no information. A calibrated probability that moves when the caller reorders a list is not
a probability the caller can act on.

Neither function needs a vision checkpoint to exist to be useful: the ablation gate runs happily
against a text runner and reports a gain of zero, which is the correct answer for a text model.
Both call `runner.predict(state, questions)` one row at a time, deliberately: batching would make
the two passes differ in padding and numerics, and a gate whose two halves are not directly
comparable is not a gate.
"""
import random
from typing import Any, Dict, Optional

from .evals import Dataset
from .evals_media import ablation, ask as _ask
from .vision import IMAGE_KEYS

# The two passes a gate compares. Named so a report says which side a number came from.
WITH = "with"
WITHOUT = "without"


def _has_image(state: Any) -> bool:
    return isinstance(state, dict) and any(k in state for k in IMAGE_KEYS)


def strip_image(state: Any) -> Any:
    """`state` with its image keys removed.

    A dict that carried only an image becomes an empty string rather than an empty dict: an empty
    dict is still a JSON document to serialize, so the model would be asked the question against
    `{}` -- text it was never meant to see -- and the ablation would be measuring the wrong thing.
    """
    if not _has_image(state):
        return state
    rest = {k: v for k, v in state.items() if k not in IMAGE_KEYS}
    return rest if rest else ""


def image_ablation(runner, dataset: Dataset, *, min_gain: Optional[float] = None,
                   min_text_change: float = 0.01) -> Dict[str, Any]:
    """Gate 1: how much of the accuracy depends on the image actually being there.

    Runs every row twice -- as given, and with the image stripped -- and reports accuracy on both
    sides per tag slice, plus the drop. `min_gain` turns the report into a verdict: the run fails
    when the drop is smaller than that, which is the spec's gate for a family whose decisions are
    supposed to need the pixels.

    Rows that carry no image at all are counted under `text_only_rows` and excluded from the drop:
    there is nothing to remove, so including them would dilute the number toward zero and let a
    collapsed model pass.

    `min_text_change` is the companion check -- a text-only row must be answered the same with and
    without the ablation, because the ablation is a no-op for it. A row that moves anyway means the
    pass itself is non-deterministic, and every other number in the report is then noise.

    The measurement is `laya.evals_media.ablation`, shared with the sound gate, so the two
    modalities' numbers are produced by the same arithmetic.
    """
    return ablation(runner, dataset, gate="vision-necessity", noun="image", keys=IMAGE_KEYS,
                    min_gain=min_gain, min_text_change=min_text_change)


# ------------------------------------------------------------------------------------------------
# Gate 2b -- option-order invariance
# ------------------------------------------------------------------------------------------------

def permuted_questions(questions: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """`questions` with every choice question's options in a fresh order.

    The order travels as the `option_order` key the sequence builder already understands and the
    answers are still keyed by the original criteria, so a flip is a changed *label*, not a
    re-indexed one. Questions with fewer than two options are passed through: there is nothing to
    permute and `option_order` would be a no-op.
    """
    out: Dict[str, Any] = {}
    for qid, body in questions.items():
        body = dict(body)
        if body.get("type") == "choice":
            criteria = body.get("criteria") or []
            names = list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)
            if len(names) > 1:
                order = list(range(len(names)))
                rng.shuffle(order)
                body["option_order"] = order
        out[qid] = body
    return out


def _answer_value(answer: Dict[str, Any]) -> Any:
    """The decision an answer carries, for comparison across passes."""
    kind = answer.get("type")
    if kind == "choice":
        return answer.get("choice")
    if kind == "noul":
        return bool(answer.get("noul", 0.0) >= 0.5)
    if kind == "score":
        return answer.get("score")
    return None


def option_order_flip_rate(runner, dataset: Dataset, *, seed: int = 0,
                           max_flip_rate: Optional[float] = None) -> Dict[str, Any]:
    """Gate 2b: how often permuting a choice question's options changes the chosen label.

    Reports the flip rate overall and per tag, over the *choice* answers both passes produced: a
    question either pass failed to answer is counted in `unanswered` and left out of the rate
    rather than scored as agreement.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer, got %r" % (seed,))
    rng = random.Random(seed)

    flips = compared = unanswered = 0
    per_tag_total: Dict[str, int] = {}
    per_tag_flips: Dict[str, int] = {}
    for example in dataset.examples:
        first = _ask(runner, example.state, example.questions, example.model)
        second = _ask(runner, example.state, permuted_questions(example.questions, rng),
                      example.model)
        for qid, body in example.questions.items():
            if body.get("type") != "choice":
                continue
            a, b = first.get(qid), second.get(qid)
            if not isinstance(a, dict) or not isinstance(b, dict):
                unanswered += 1
                continue
            va, vb = _answer_value(a), _answer_value(b)
            compared += 1
            flipped = va != vb
            flips += int(flipped)
            for tag in ("all",) + tuple(str(t) for t in example.tags):
                per_tag_total[tag] = per_tag_total.get(tag, 0) + 1
                per_tag_flips[tag] = per_tag_flips.get(tag, 0) + int(flipped)

    rate = (flips / compared) if compared else None
    report: Dict[str, Any] = {
        "gate": "option-order-invariance",
        "seed": seed,
        "questions_compared": compared,
        "flips": flips,
        "unanswered": unanswered,
        "flip_rate": rate,
        "passed": None,
        "per_tag": {tag: {"questions": per_tag_total[tag],
                          "flips": per_tag_flips.get(tag, 0),
                          "flip_rate": per_tag_flips.get(tag, 0) / per_tag_total[tag]}
                    for tag in sorted(per_tag_total)},
    }
    if max_flip_rate is not None:
        report["max_flip_rate"] = float(max_flip_rate)
        report["passed"] = bool(rate is not None and rate <= max_flip_rate)
    return report
