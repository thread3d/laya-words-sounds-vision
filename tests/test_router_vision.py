"""Tests for the vision routing rule (spec §10.4).

The rule is deliberately narrow: a state carrying an image goes to the vision checkpoint, but only
when one is registered. Everything here is weight-free -- `Router.route` decides without loading
anything -- so these run in the same fast lane as `tests/test_router.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.router import Router
from laya.vision import has_image

VISION = {"vision": "acme/laya-vision"}


def test_has_image_recognises_every_reference_shape():
    assert has_image({"image_path": "a.png"})
    assert has_image({"image": {"pixels": 1}})
    assert has_image({"image_url": "https://example.com/a.png"})
    assert has_image({"image_base64": "AAAA"})
    assert not has_image({"message": "no picture here"})
    assert not has_image("plain text")
    assert not has_image(["a", "list"])


def test_an_image_state_routes_to_the_vision_checkpoint():
    router = Router(models=VISION)
    decision = router.route({"image_path": "a.png", "text": "vendor says this is unpaid"})
    assert decision.model == "vision", decision
    assert "image" in decision.reason, decision.reason


def test_an_explicit_model_still_wins():
    router = Router(models=VISION)
    assert router.route({"image_path": "a.png"}, model="multilingual").model == "multilingual"
    assert router.route({"image_path": "a.png"}, task="typed-decisions").model == "typed-decisions"


def test_a_text_state_is_unaffected():
    router = Router(models=VISION)
    assert router.route({"message": "I was charged twice"}).model == "english"
    assert router.route({"message": "Mein Konto wurde zweimal belastet"}).model == "multilingual"


def test_without_a_registered_vision_checkpoint_nothing_changes():
    """The rule is opt-in in the sense that matters: no checkpoint, no change (C6)."""
    router = Router()
    decision = router.route({"image_path": "a.png", "message": "hello there"})
    assert decision.model == "multilingual", decision
    assert "image" not in decision.reason, decision.reason


def test_the_rule_can_be_turned_off():
    router = Router(models=VISION, vision_model=None)
    decision = router.route({"image_path": "a.png", "message": "hello there"})
    assert decision.model == "multilingual", decision


def test_picking_a_checkpoint_beats_the_image_rule_but_picking_a_language_does_not():
    """`model=`/`task=` name a checkpoint and win; `lang=` names a *language* and does not.

    That split is the spec's: "a state carrying an image goes to the vision checkpoint, and an
    explicit `model=` still wins" -- one stated exception, and the rule is otherwise at the top of
    the chain. It is worth stating plainly because it is the surprising case: `lang="de"` with an
    image still routes to the vision checkpoint. A caller who wants the German text model for an
    image-bearing state says so with `model="multilingual"`, which wins.
    """
    router = Router(models=VISION)
    assert router.route({"image_path": "a.png", "message": "bonjour"}).model == "vision"
    assert router.route({"image_path": "a.png"}, lang="de").model == "vision"
    assert router.route({"image_path": "a.png"}, lang="de", model="multilingual").model == "multilingual"


if __name__ == "__main__":
    test_has_image_recognises_every_reference_shape()
    test_an_image_state_routes_to_the_vision_checkpoint()
    test_an_explicit_model_still_wins()
    test_a_text_state_is_unaffected()
    test_without_a_registered_vision_checkpoint_nothing_changes()
    test_the_rule_can_be_turned_off()
    test_picking_a_checkpoint_beats_the_image_rule_but_picking_a_language_does_not()
    print("all vision routing tests passed")
