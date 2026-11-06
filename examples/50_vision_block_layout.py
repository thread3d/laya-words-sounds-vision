"""Example 50 -- where the visual block goes, and what the splice actually does.

Lays out the projected patch block with the checkpoint's own tokenizer, then performs the splice on
a toy embedding tensor so the shape arithmetic is something you can see rather than something you
have to trust. No weights, no vision tower, no encoder.
"""
from _common import banner, head_for, heading, tokenizer

banner("50", "Where the visual block goes", """
    A frozen vision tower turns the picture into patches. A trainable projector compresses them
    into `n_visual_tokens` rows in the text encoder's hidden space, and `laya.block` splices those
    rows into the sequence -- after the option markers, never before.

    That placement is a measured decision, not a preference. ModernBERT attends locally with a
    128-token window in most of its layers, so a block appended after a long state could only be
    reached through the few global layers. Splicing at the end of the question head puts every
    option's `[MASK]` inside the same local window as the picture, which is the computation a
    multiple-choice answer needs.

    Everything on this page runs without loading a model: the tokenizer supplies the real head,
    and the splice is shown on a toy tensor.
    """)

tok = tokenizer()
print("   tokenizer: %s, [MASK] is id %d" % (type(tok).__name__, tok.mask_token_id))

from laya.vision import assemble_inputs_embeds, describe_plan, parse_vision, visual_plan

heading("a checkpoint opts in with a `vision` block")
print("   parse_vision({})                                  -> %r" % (parse_vision({}),))
print("   parse_vision({'vision': {'enabled': False}})       -> %r"
      % (parse_vision({"vision": {"enabled": False}}),))
cfg = parse_vision({"vision": {"n_visual_tokens": 16}})
print("   parse_vision({'vision': {'n_visual_tokens': 16}})  -> %s" % (cfg,))
print("   A checkpoint with no block parses to None, so every text checkpoint's forward pass stays")
print("   byte-identical -- that is the no-regression invariant this whole module is written under.")

QUESTION = {
    "type": "choice",
    "instructions": "Which coarse class is shown in this photograph?",
    "criteria": {"flowers": "blooms, petals, stems", "fish": "aquatic animals with fins",
                 "vehicles 1": "cars, trucks, road traffic",
                 "household furniture": "chairs, tables, cupboards"},
}
head_ids, markers, stats = head_for(tok, QUESTION, head_max_len=192)
option_tokens = len(head_ids) - markers[0]
state_tokens = 30
text_len = len(head_ids) + state_tokens

heading("the head, then the plan")
print("   question head : %d tokens, option markers at %s" % (len(head_ids), markers))
print("   state         : %d more tokens, so text_len = %d" % (state_tokens, text_len))
plan = visual_plan(cfg, text_len=text_len, markers=markers, head_len=len(head_ids),
                   max_len=512, option_tokens=option_tokens)
for line in describe_plan(plan):
    print("   %s" % line)
print("   the markers stay at %s: the block is spliced *at* the end of the head, so nothing"
      % (markers,))
print("   before the splice moves -- which is what keeps every marker sitting on its own [MASK].")

heading("what `n_visual_tokens` costs")
print("   %-18s %-8s %-10s %s" % ("n_visual_tokens", "block", "sequence", "share of max_len=512"))
for n in (4, 16, 64, 256):
    one = visual_plan(parse_vision({"vision": {"n_visual_tokens": n}}), text_len=text_len,
                       markers=markers, head_len=len(head_ids), max_len=512,
                       option_tokens=option_tokens)
    print("   %-18d %-8d %-10d %.0f%%" % (n, one.n_visual, one.length, 100.0 * one.length / 512))
print("   The block and the text draw on the same `max_len`. The shipped benchmark checkpoint uses")
print("   16 rows, because a coarse-class decision needs a summary of the picture, not its")
print("   every patch -- and because 256 rows would leave a long state nowhere to live.")

heading("the summary row a large head earns")
print("   threshold: `summary_for_options_over` = %d rendered option tokens"
      % cfg.summary_for_options_over)
print("   these options render to %d tokens -> summary row: %s" % (option_tokens, plan.summary))
BIG = {"type": "choice", "instructions": "Which is it?",
       "criteria": {"class %d" % i: None for i in range(40)}}
big_ids, big_markers, _ = head_for(tok, BIG, head_max_len=512)
big = visual_plan(cfg, text_len=len(big_ids) + state_tokens, markers=big_markers,
                  head_len=len(big_ids), max_len=1024,
                  option_tokens=len(big_ids) - big_markers[0])
print("   40 options render to %d tokens -> block %d rows (16 + 1 summary), sequence %d"
      % (len(big_ids) - big_markers[0], big.n_visual, big.length))
print("   One row of the block is the globally pooled patch vector -- the only token that has seen")
print("   every patch. A head whose last markers sit outside the 128-token window needs it to reach")
print("   the picture at all; before it existed, any head above ~30 options refused to build.")

heading("the splice, on a tensor you can count")
import torch

hidden = 8
text_embeds = torch.zeros(1, text_len, hidden)          # every text token marked 0.0
visual_embeds = torch.ones(1, plan.n_visual, hidden)    # every visual row marked 1.0
merged = assemble_inputs_embeds(text_embeds, visual_embeds, plan)
print("   text   [1, %d, %d] + visual [1, %d, %d] -> merged [%d, %d, %d]"
      % (text_len, hidden, plan.n_visual, hidden,
         merged.shape[0], merged.shape[1], merged.shape[2]))
row = merged[0, :, 0].tolist()
first_visual = row.index(1.0)
print("   the first 1.0 sits at index %d, which is `plan.splice` by construction." % first_visual)
print("   text tokens before it : %d" % row[:first_visual].count(0.0))
print("   visual rows           : %d" % row.count(1.0))
print("   text tokens after it  : %d" % row[first_visual:].count(0.0))
print("   It is a concatenation, not an arithmetic mix: a gradient reaches the block exactly as it")
print("   reaches the token embeddings, which is what makes the projector trainable and the tower")
print("   frozen in the same backward pass.")

print("""
   One layout, two modalities. `laya.vision` and `laya.audio` both delegate the arithmetic to
   `laya.block`, so a picture checkpoint and a sound checkpoint cannot disagree about where the
   block goes -- and a checkpoint trained at one splice point cannot be served at another. Example
   45 runs this page for a recording; the numbers have the same shape because the code is the same
   code.
   """)
