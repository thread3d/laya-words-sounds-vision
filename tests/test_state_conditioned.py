"""Tests for the state-conditioned option layout (spec §4.1.1).

The layout exists for one property: an answer must not depend on the order the caller lists the
options in. A joint layout lets each option see its competitors, so it can learn "compare A to B"
instead of "compare A to the image", and the measured symptom is 8-10% of answers flipping under a
reordering that carries no information. These tests state the property directly and check it on a
tiny model, so a regression in the layout shows up as a flip rather than as a shape error.
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import Agent
from laya.common import (DecisionModel, build_sequence, uses_state_conditioned_layout)
from laya.evals_vision import option_order_flip_rate


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

    def __getitem__(self, key):
        return self.__dict__[key]


def _agent(layout="state_conditioned", hidden=16):
    cfg = AutoConfig.for_model("bert", hidden_size=hidden, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32, vocab_size=50,
                               pad_token_id=0)
    model = DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2).eval()
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
    agent.state_conditioned = layout == "state_conditioned"
    agent.tok = _StubTok()
    agent.cfg = {"max_len": 512, "head_max_len": 192, "option_layout": layout}
    agent.model_id = "test"
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.lang_temperatures = {}
    agent.binning_module = None
    agent.binning_map = None
    agent.temperature_by_modality = {}
    agent.vision = None
    return agent


QUESTION = {"type": "choice", "instructions": "pick", "criteria": ["alpha", "beta", "gamma"]}


def test_the_layout_flag_reads_the_config():
    assert uses_state_conditioned_layout({"option_layout": "state_conditioned"}) is True
    assert uses_state_conditioned_layout({"option_layout": "sequential"}) is False
    assert uses_state_conditioned_layout({}) is False


def test_each_option_gets_its_own_row_with_one_marker():
    agent = _agent()
    internal = {"q": agent._to_internal(QUESTION)}
    items = agent._encode_state("some state", ["q"], internal)
    assert len(items) == 3, "one row per option, not one row per question"
    assert all(len(it["markers"]) == 1 for it in items)
    assert [it["option_group"] for it in items] == [("q", 0), ("q", 1), ("q", 2)]


def test_a_row_is_exactly_the_single_option_sequence():
    """The whole point: no row may contain a competitor's option tokens.

    Asserted by identity rather than by a token-overlap heuristic -- the test tokenizer maps
    characters into a 46-token vocabulary, so different option words share most of their ids and
    an overlap count says nothing.
    """
    agent = _agent()
    internal = {"q": agent._to_internal(QUESTION)}
    items = agent._encode_state("s", ["q"], internal)
    for slot, item in enumerate(items):
        solo = build_sequence(agent.tok, "s", internal["q"], 512, 192, only_option=slot)[0]
        assert item["ids"] == solo, "row %d is not the single-option sequence for its option" % slot
        assert len(items) == 3
        others = [build_sequence(agent.tok, "s", internal["q"], 512, 192, only_option=o)[0]
                  for o in range(3) if o != slot]
        assert all(solo != other for other in others), \
            "two options produced the same row; the layout cannot tell them apart"


def test_the_joint_layout_still_puts_everything_in_one_row():
    agent = _agent(layout="sequential")
    internal = {"q": agent._to_internal(QUESTION)}
    items = agent._encode_state("some state", ["q"], internal)
    assert len(items) == 1 and len(items[0]["markers"]) == 3


def test_a_state_conditioned_answer_is_invariant_to_option_order():
    """The property the layout is for: permuting the options must not change the label."""
    torch.manual_seed(0)
    agent = _agent()
    internal = {"q": agent._to_internal(QUESTION)}

    def answer(order):
        q = dict(internal["q"])
        if order is not None:
            q["option_order"] = order
        items = agent._encode_state("state text here", ["q"], {"q": q})
        from laya.common import collate_items
        batch = collate_items([items], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, act = agent._infer(batch)
        return agent._decode_answers(logits, act, items, ["q"], {"q": q}, 0)["q"]["choice"]

    baseline = answer(None)
    for order in ([1, 0, 2], [2, 1, 0], [0, 2, 1]):
        assert answer(order) == baseline, \
            "order %r changed the answer from %r to %r" % (order, baseline, answer(order))


def test_the_flip_gate_reports_zero_for_a_state_conditioned_agent():
    """The same property, measured through the gate a caller would actually run."""
    agent = _agent()
    from laya.evals import Dataset, Example

    dataset = Dataset([Example(state="state text", questions={"q": QUESTION},
                               expected={"q": "alpha"})])
    report = option_order_flip_rate(agent, dataset, seed=0, max_flip_rate=0.02)
    assert report["questions_compared"] == 1, report
    assert report["flip_rate"] == 0.0, report
    assert report["passed"] is True


if __name__ == "__main__":
    test_the_layout_flag_reads_the_config()
    test_each_option_gets_its_own_row_with_one_marker()
    test_a_row_is_exactly_the_single_option_sequence()
    test_the_joint_layout_still_puts_everything_in_one_row()
    test_a_state_conditioned_answer_is_invariant_to_option_order()
    test_the_flip_gate_reports_zero_for_a_state_conditioned_agent()
    print("all state-conditioned tests passed")
