"""Serve a decision through a cascade of checkpoints, escalating on calibrated confidence.

The specification's §4.5, made executable. A vision checkpoint is not one model but a policy over
several: the fast one answers the easy majority, and a *calibrated* confidence decides which
requests are worth spending a more expensive one on. Two rules make that honest:

* **the gate is the decision's own confidence, not a second opinion about the input.** An
  answerability head is a documented negative result — it ordered the stream worse than the plain
  decision probability (AURC 0.0362 against 0.0456) — so this class reads `answer_confidence` and
  nothing else.
* **thresholds come from data, not from taste.** A tier's `min_confidence` is the number a conformal
  fit produces for a target coverage (§8.2); picking one by eye on a validation curve does not
  survive a domain shift.

The cascade never invents an answer: each tier's response is returned exactly as that tier produced
it, with cascade metadata added beside it. Typed answers, probabilities and temperature scaling are
the tier's own.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# What a tier may be asked to guarantee. The terminal tier takes no threshold: there is nothing
# below it to escalate to, so a floor there would only turn an answer into an error.
TERMINAL = None


@dataclass
class Tier:
    """One rung of the cascade: which checkpoint, and the confidence it must reach to stop.

    `min_confidence` is compared against the *weakest* answer the tier produced. A response that
    answered four questions confidently and one badly is not a response to act on, and averaging
    would hide exactly the one that needs a second look.

    `name` is what the response reports and what an escalation record names, so it has to be
    distinct and human-readable; `model` is any checkpoint name the runner accepts.
    """

    name: str
    model: str
    min_confidence: Optional[float] = None

    def validate(self, terminal: bool) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a tier needs a non-empty name, got %r" % (self.name,))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("tier %r needs a non-empty model, got %r" % (self.name, self.model))
        if terminal:
            if self.min_confidence is not TERMINAL:
                raise ValueError(
                    "tier %r is the last one, so it must not set min_confidence: there is nothing "
                    "below it to escalate to, and a floor there would refuse answers it produced"
                    % (self.name,))
            return
        value = self.min_confidence
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "tier %r needs min_confidence in [0, 1], got %r. Only the last tier may omit it -- "
                "there is nothing below it to escalate to, so a tier with no floor and somewhere "
                "to escalate would never escalate" % (self.name, value))


class Cascade:
    """Answer through `tiers`, in order, stopping at the first one that is sure enough.

        from laya.cascade import Cascade, Tier

        cascade = Cascade([
            Tier("fast", "vision-fast", min_confidence=0.70),
            Tier("slow", "vision"),
        ])
        result = cascade.predict({"image_path": "invoice.png"}, questions)
        result["cascade"]["tier"]          # which rung answered
        result["cascade"]["escalations"]   # and why it moved
    """

    def __init__(self, tiers: Sequence[Tier], runner: Any = None, max_tiers: int = 4):
        tiers = list(tiers)
        if not tiers:
            raise ValueError("a cascade needs at least one tier")
        if len(tiers) > max_tiers:
            raise ValueError("a cascade of %d tiers is a latency accident, not a policy; the "
                             "ceiling is %d" % (len(tiers), max_tiers))
        names = [t.name for t in tiers]
        if len(set(names)) != len(names):
            raise ValueError("tier names must be distinct, got %r" % (names,))
        for index, tier in enumerate(tiers):
            tier.validate(terminal=(index == len(tiers) - 1))
        self.tiers = tiers
        # Built lazily: a `Router()` constructs no checkpoint, but importing it here would put the
        # router's module in the import graph of anyone who only wanted the class.
        self._runner = runner

    # -- runner ---------------------------------------------------------------------------------
    @property
    def runner(self):
        if self._runner is None:
            from .router import Router

            self._runner = Router()
        return self._runner

    # -- serving --------------------------------------------------------------------------------
    def predict(self, state: Any, questions: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Answer `state`, escalating while the calibrated confidence stays under a tier's floor.

        Returns the answering tier's own result dict with a ``cascade`` key added:

            {"tier": name, "index": i, "escalations": [{"from", "to", "min_confidence",
             "answer_confidence"}]}

        The escalation list is empty when the first tier answered, so a caller can tell "fast and
        right" from "fast, doubted, then answered by a better model" without timing anything.
        """
        escalations: List[Dict[str, Any]] = []
        for index, tier in enumerate(self.tiers):
            result = self.runner.predict(state, questions, model=tier.model, **kwargs)
            confidence = _weakest_confidence(result)
            is_last = index == len(self.tiers) - 1
            if is_last or tier.min_confidence is None or confidence is None \
                    or confidence >= tier.min_confidence:
                out = dict(result)
                out["cascade"] = {"tier": tier.name, "index": index,
                                  "escalations": escalations}
                return out
            nxt = self.tiers[index + 1]
            escalations.append({"from": tier.name, "to": nxt.name,
                                "min_confidence": tier.min_confidence,
                                "answer_confidence": confidence})
        raise AssertionError("unreachable: the last tier always returns")   # pragma: no cover


def _weakest_confidence(result: Any) -> Optional[float]:
    """The smallest `answer_confidence` in a response, or None when it reports none.

    The weakest, not the mean: a response that answered four questions confidently and one badly
    is not a response to act on. `answer_confidence` is the calibrated number, the one every
    question type carries, and the one a conformal threshold is fitted for -- `confidence` means
    something different per type and is not comparable across them.
    """
    answers = (result or {}).get("answers")
    if not isinstance(answers, dict) or not answers:
        return None
    values = [a.get("answer_confidence") for a in answers.values() if isinstance(a, dict)]
    values = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return min(values) if values else None
