"""The gate machinery two modalities share: the input-removed ablation, and per-tag accuracy.

A gate that asks "does the answer depend on the input" is the same question for a picture and for a
recording: run every row twice, once as given and once with the input removed, and compare accuracy.
Only two things differ between the modalities -- which keys are the input, and the word the report
uses for it -- so those are arguments here and the arithmetic lives in one place. A gate whose two
modalities were implemented separately is a gate whose numbers would eventually stop being
comparable.

The option-order gate is fully modality-agnostic (it permutes a question, not a state) and stays in
`laya.evals_vision` where it was first written, next to the report format its tests pin.
"""
from typing import Any, Dict, List, Optional, Sequence

from .evals import Dataset, EvalError, Example, _correct


def has_key(state: Any, keys: Sequence[str]) -> bool:
    """Whether a state carries any of `keys`."""
    return isinstance(state, dict) and any(key in state for key in keys)


def strip_keys(state: Any, keys: Sequence[str]) -> Any:
    """`state` with `keys` removed.

    A dict that carried only the input becomes an empty string rather than an empty dict: an empty
    dict is still a JSON document to serialize, so the model would be asked the question against
    `{}` -- text it was never meant to see -- and the ablation would be measuring the wrong thing.
    """
    if not has_key(state, keys):
        return state
    rest = {k: v for k, v in state.items() if k not in keys}
    return rest if rest else ""


def ask(runner, state: Any, questions: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
    """One example's answers, in the one shape the runners agree on."""
    kwargs = {"model": model} if model is not None else {}
    try:
        result = runner.predict(state, questions, **kwargs)
    except TypeError as exc:                            # a runner whose predict takes no `model`
        if "model" not in kwargs or "model" not in str(exc):
            raise
        result = runner.predict(state, questions)
    answers = (result or {}).get("answers")
    if not isinstance(answers, dict):
        raise EvalError("runner.predict returned no 'answers' object")
    return answers


def accuracy(examples: Sequence[Example], answers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fraction of questions answered, and the fraction of those answered correctly."""
    asked = correct = 0
    for example, got in zip(examples, answers):
        for qid, expected in example.expected.items():
            answer = got.get(qid)
            if not isinstance(answer, dict):
                continue
            verdict = _correct(answer, expected)
            if verdict is None:
                continue
            asked += 1
            correct += int(bool(verdict))
    return {"questions": asked, "correct": correct,
            "accuracy": (correct / asked) if asked else None}


def slices(examples: Sequence[Example]) -> Dict[str, List[int]]:
    """Indices per tag, with an `all` slice so a report always has a headline number."""
    out: Dict[str, List[int]] = {"all": list(range(len(examples)))}
    for index, example in enumerate(examples):
        for tag in example.tags:
            out.setdefault(str(tag), []).append(index)
    return out


def ablation(runner, dataset: Dataset, *, gate: str, noun: str, keys: Sequence[str],
             min_gain: Optional[float] = None,
             min_text_change: float = 0.01) -> Dict[str, Any]:
    """How much of the accuracy depends on `keys` actually being there.

    `noun` is the word the report uses for the input ("image", "audio") and names the per-slice keys
    (`with_image` / `without_image`, `with_audio` / `without_audio`); `gate` is the gate's name in
    the report. Everything else is the measurement the vision specification asks for, applied to
    whichever input the checkpoint is supposed to be reading.

    `min_gain` turns the report into a verdict: the run fails when the drop is smaller than that,
    which is the gate for a family whose decisions are supposed to need the input. Rows that carry
    no input at all are counted under ``rows_text_only`` and excluded from the drop: there is
    nothing to remove, so including them would dilute the number toward zero and let a collapsed
    model pass.

    `min_text_change` is the companion check -- a text-only row must be answered the same with and
    without the ablation, because the ablation is a no-op for it. A row that moves anyway means the
    pass itself is non-deterministic, and every other number in the report is then noise.
    """
    with_input: List[Example] = []
    without_input: List[Example] = []
    stripped: List[Example] = []
    for example in dataset.examples:
        if has_key(example.state, keys):
            with_input.append(example)
            stripped.append(Example(state=strip_keys(example.state, keys),
                                    questions=example.questions,
                                    expected=example.expected, tags=example.tags,
                                    language=example.language, model=example.model))
        else:
            without_input.append(example)

    answers_with = [ask(runner, ex.state, ex.questions, ex.model) for ex in with_input]
    answers_without = [ask(runner, ex.state, ex.questions, ex.model) for ex in stripped]

    per_tag: Dict[str, Any] = {}
    for tag, indices in slices(with_input).items():
        a = accuracy([with_input[i] for i in indices], [answers_with[i] for i in indices])
        b = accuracy([with_input[i] for i in indices], [answers_without[i] for i in indices])
        per_tag[tag] = {
            "with_" + noun: a["accuracy"],
            "without_" + noun: b["accuracy"],
            "questions": a["questions"],
            "gain": (a["accuracy"] - b["accuracy"])
            if a["accuracy"] is not None and b["accuracy"] is not None else None,
        }

    # The pass must be a no-op on rows that carry no input.
    text_answers = [ask(runner, ex.state, ex.questions, ex.model) for ex in without_input]
    text_moved = 0
    for example, first in zip(without_input, text_answers):
        second = ask(runner, strip_keys(example.state, keys), example.questions, example.model)
        if first != second:
            text_moved += 1
    text_change = (text_moved / len(without_input)) if without_input else 0.0

    headline = per_tag["all"]["gain"]
    report: Dict[str, Any] = {
        "gate": gate,
        "rows_with_%s" % noun: len(with_input),
        # Kept as the stable name both gates have always reported, so a caller comparing a vision
        # report with an audio one does not have to know which noun that report used.
        "rows_text_only": len(without_input),
        "per_tag": per_tag,
        "gain": headline,
        "text_only_unchanged": text_change <= min_text_change,
        "text_only_change_rate": text_change,
        # Always present, so a caller can branch on it without first asking whether a threshold
        # was set: None means "reported, not judged".
        "passed": None,
    }
    if min_gain is not None:
        report["min_gain"] = float(min_gain)
        report["passed"] = bool(headline is not None and headline >= min_gain
                                and text_change <= min_text_change)
    return report
