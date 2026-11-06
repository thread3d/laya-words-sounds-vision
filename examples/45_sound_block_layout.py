"""Example 45 -- where the sound block goes, and what it costs.

A checkpoint that reads a recording splices a projected block of rows into the token sequence. This
example lays that block out with the checkpoint's own tokenizer -- no weights, no audio tower, no
encoder -- so the arithmetic from the guides can be checked directly: the splice point, the token
cost of `n_audio_tokens`, the summary row a high-cardinality head earns, and the two position
policies.
"""
from _common import banner, head_for, heading, tokenizer

banner("45", "Where the sound block goes", """
    A frozen audio tower turns the waveform into frames. A trainable projector compresses those
    frames into `n_audio_tokens` rows in the text encoder's hidden space, and `laya.block`
    splices them into the sequence *after the option markers*.

    After, not before, and that is not an aesthetic choice: ModernBERT attends locally with a
    128-token window in most of its layers, so a block placed at the end of a long state could
    only reach the option markers through the few global layers. Splicing at the end of the head
    puts every option's `[MASK]` inside the same local window as the recording.

    This page builds the real head with the real tokenizer and plans the block against it. It
    loads no weights, so it costs a second rather than a minute.
    """)

tok = tokenizer()
print("   tokenizer: %s, [MASK] is id %d" % (type(tok).__name__, tok.mask_token_id))

from laya.audio import audio_plan, describe_plan, parse_audio

heading("a checkpoint opts in with an `audio` block")
print("   parse_audio({})                                    -> %r" % (parse_audio({}),))
print("   parse_audio({'audio': {'enabled': False}})          -> %r"
      % (parse_audio({"audio": {"enabled": False}}),))
cfg = parse_audio({"audio": {"n_audio_tokens": 32}})
print("   parse_audio({'audio': {'n_audio_tokens': 32}})      -> %s" % (cfg,))
print("   A checkpoint with no block parses to None and the text path stays byte-identical, so")
print("   this module existing costs a text deployment nothing.")

QUESTION = {
    "type": "choice",
    "instructions": "What is making this sound?",
    "criteria": {"a dog barking": "sharp repeated barks", "a door closing": "one low thud",
                 "a smoke alarm": "a steady high beep", "rain": "broadband hiss"},
}
head_ids, markers, stats = head_for(tok, QUESTION, head_max_len=192)
option_tokens = len(head_ids) - markers[0]
state_tokens = 40                                   # a short sentence about the recording
text_len = len(head_ids) + state_tokens

heading("the head, then the plan")
print("   question head : %d tokens, option markers at %s" % (len(head_ids), markers))
print("   state         : %d more tokens, so text_len = %d" % (state_tokens, text_len))
plan = audio_plan(cfg, text_len=text_len, markers=markers, head_len=len(head_ids),
                  max_len=512, option_tokens=option_tokens)
for line in describe_plan(plan):
    print("   %s" % line)
print("   markers after planning: %s -- unchanged, because the block is spliced *at* the end"
      % (markers,))
print("   of the head and everything before it keeps its position.")

heading("what `n_audio_tokens` costs")
print("   %-16s %-10s %-12s %s" % ("n_audio_tokens", "block", "sequence", "share of max_len=512"))
for n in (8, 32, 64, 128):
    one = audio_plan(parse_audio({"audio": {"n_audio_tokens": n}}), text_len=text_len,
                      markers=markers, head_len=len(head_ids), max_len=512,
                      option_tokens=option_tokens)
    print("   %-16d %-10d %-12d %.0f%%" % (n, one.n_audio, one.length, 100.0 * one.length / 512))
print("   The block is charged against the same `max_len` as the text: the two share one")
print("   sequence. A block that leaves under 16 tokens for the text is refused by name when a")
print("   real call is built, which is the kind of thing to know before picking a number.")

heading("the summary row a large head earns")
print("   threshold: `summary_for_options_over` = %d rendered option tokens"
      % cfg.summary_for_options_over)
print("   this question's options render to %d tokens -> summary row: %s"
      % (option_tokens, plan.summary))
BIG = {"type": "choice", "instructions": "Which of these is it?",
       "criteria": {"class %d" % i: None for i in range(40)}}
big_ids, big_markers, _ = head_for(tok, BIG, head_max_len=512)
big_option_tokens = len(big_ids) - big_markers[0]
big = audio_plan(cfg, text_len=len(big_ids) + state_tokens, markers=big_markers,
                 head_len=len(big_ids), max_len=1024, option_tokens=big_option_tokens)
print("   40 options render to %d tokens -> block %d rows (32 + 1 summary), sequence %d"
      % (big_option_tokens, big.n_audio, big.length))
print("   That one extra row is the only token in the block that has seen every frame, and a")
print("   head whose last markers sit outside the 128-token window needs it to reach the")
print("   recording at all. Set `summary_for_options_over: 0` to force the block back down.")

heading("the two position policies")
print("   `contiguous` (the default) numbers the block where the head ended and shifts the text")
print("   after it, so relative distances stay small. `reserved` parks it at the top of the")
print("   model's positional budget, far from everything:")
for policy in ("contiguous", "reserved"):
    policy_cfg = parse_audio({"audio": {"n_audio_tokens": 4, "position_policy": policy}})
    one = audio_plan(policy_cfg, text_len=text_len, markers=markers, head_len=len(head_ids),
                     max_len=512, option_tokens=option_tokens)
    block = list(one.position_ids[one.splice:one.splice + one.n_audio])
    print("   %-12s block positions %-26s text after it ends at %d"
          % (policy, str(block), one.position_ids[-1]))

print("""
   One layout, two modalities. `laya.vision` and `laya.audio` both delegate here to
   `laya.block`, so a picture checkpoint and a sound checkpoint cannot disagree about where the
   block goes -- and a checkpoint that was trained with one splice point cannot silently be
   served at another. Example 50 does this page for a picture; the numbers are the same shape
   because the layout is the same code.
   """)
