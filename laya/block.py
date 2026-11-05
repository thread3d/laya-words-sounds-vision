"""The modality-neutral half of a block splice: where a non-text block goes, and how it is wired in.

A checkpoint that reads something other than tokens -- a photograph, a sound -- hands the encoder a
short run of *projected* rows spliced into the token embeddings. Vision and audio differ in how the
rows are produced (a frozen tower and a projector, in both cases) and agree completely in where the
rows go and what the sequence looks like afterwards. That agreement lives here so the two paths
cannot drift apart on the one thing that has to match the trainer bit for bit.

Three pieces, all of them pure Python except :func:`splice_embeddings`:

* :class:`BlockPlan` -- where the block sits in one built sequence and what it costs.
* :func:`plan_block` -- the layout arithmetic: splice point, marker positions, position ids, length.
* :func:`splice_embeddings` -- the one tensor operation, `torch.cat` at the splice point.

Nothing here imports torch at module import: `import laya.block` has to stay cheap, and the router
must not pull the tensor stack in (``tests/test_lazy_import.py``).
"""
from typing import List, Optional, Sequence, Tuple

# Where a block may be inserted. `after_options` is the only one implemented; the others are named
# so a config naming them fails loudly at load with a list of what exists, rather than silently
# falling back to the default and moving the block somewhere the author did not ask for.
SLOTS = ("after_options",)
POSITION_POLICIES = ("contiguous", "reserved")


class BlockPlan:
    """Where a non-text block sits in one built sequence and what it costs.

    Produced by :func:`plan_block`; consumed by :func:`splice_embeddings` at inference and by the
    training loop, so both agree about the splice point and the marker shifts.

    The block's width is ``n_block``, and it already includes any summary row. A modality that
    wants its own name for the width (:attr:`laya.vision.VisualPlan.n_visual`) adds a property
    rather than a second counter, so there is exactly one number to keep right.
    """

    __slots__ = ("splice", "n_block", "summary", "marker_positions", "position_ids", "length")

    def __init__(self, splice: int, n_block: int, summary: bool, marker_positions: Sequence[int],
                 position_ids: Sequence[int], length: int):
        self.splice = splice
        self.n_block = n_block
        self.summary = summary
        self.marker_positions = list(marker_positions)
        self.position_ids = list(position_ids)
        self.length = length

    def __repr__(self) -> str:
        return ("BlockPlan(splice=%d, n_block=%d, summary=%s, length=%d)"
                % (self.splice, self.n_block, self.summary, self.length))


def reserved_positions(n_block: int, max_position_embeddings: Optional[int] = None,
                       offset: Optional[int] = None) -> List[int]:
    """The positions a reserved block occupies: `offset`, or the top of the positional budget.

    The default reserves the top so a block can never collide with an ordinary text position. It
    is not the default *policy* -- see :func:`plan_block` -- because a reserved block sits thousands
    of positions away from the option markers in RoPE terms, outside the local windows of most of
    the encoder's layers. It is kept because it is the policy a caller reaches for when positions
    must not move at all.
    """
    start = offset
    if start is None:
        start = max(0, (max_position_embeddings or 0) - n_block)
    if max_position_embeddings and start + n_block > max_position_embeddings:
        raise ValueError(
            "position_offset=%r plus %d block tokens exceeds this encoder's %d positions"
            % (offset, n_block, max_position_embeddings))
    return list(range(start, start + n_block))


def plan_block(name: str, noun: str, slot: str, n_block: int, text_len: int,
               markers: Sequence[int], head_len: int, max_len: Optional[int],
               summary: bool = False, max_position_embeddings: int = 8192,
               position_policy: str = "contiguous",
               position_offset: Optional[int] = None) -> BlockPlan:
    """Lay a block of `n_block` rows out inside a sequence of `text_len` tokens.

    `name` is the modality's own word ("vision", "audio") and `noun` is what its rows are called
    ("visual", "audio"); both appear in the errors, so a message about a sound never tells the
    caller to lower ``vision.n_visual_tokens``, and the config key it names --
    ``<name>.n_<noun>_tokens`` -- is the key that actually exists.

    `head_len` is ``len(build_head(...))`` -- the splice point. `markers` are ``build_head``'s
    marker positions, which sit *before* the splice and therefore never move: the whole block,
    summary row included, is inserted after them (`splice_embeddings` splices *at* `splice`).
    Shifting the markers for a summary row was wrong in both directions -- it compensated for an
    insertion in front of the options that never happens, and it moved every marker off its
    `[MASK]` token (measured on the real tokenizer: 100 options, 0/100 markers on a `[MASK]` with
    the shift, 100/100 without).
    """
    if slot not in SLOTS:                                   # pragma: no cover - guarded at load
        raise ValueError("unsupported %s slot %r" % (name, slot))
    if not 0 <= head_len <= text_len:
        raise ValueError("head_len=%d must be within the sequence (0..%d)" % (head_len, text_len))

    length = text_len + n_block
    if max_len is not None and length > max_len:
        raise ValueError(
            "%s: %d text tokens + %d %s tokens = %d, past max_len=%d. Lower "
            "%s.n_%s_tokens, shorten the state, or raise max_len for this checkpoint."
            % (name, text_len, n_block, noun, length, max_len, name, noun))

    splice = head_len
    marker_positions = list(markers)

    if position_policy == "contiguous":
        # The block continues the numbering where the head left off, and the text after it shifts
        # by the block's width. RoPE encodes *distance*, so this is what actually makes the block
        # adjacent to the option markers: the offsets are small, the local windows reach it, and
        # every relative distance stays inside the range the checkpoint was trained on.
        block_positions = list(range(head_len, head_len + n_block))
        after = [p + n_block for p in range(head_len, text_len)]
        position_ids = list(range(head_len)) + block_positions + after
    else:
        text_positions = list(range(text_len))
        # The summary row rides one position past the block's own run, exactly as the vision path
        # measured it: a reserved block of `n` rows occupies `n` reserved positions, and the extra
        # row is the next one.
        block_positions = reserved_positions(n_block - (1 if summary else 0),
                                             max_position_embeddings, position_offset)
        if summary:
            block_positions = block_positions + [block_positions[-1] + 1]
        position_ids = text_positions[:head_len] + block_positions + text_positions[head_len:]
    return BlockPlan(splice=splice, n_block=n_block, summary=summary,
                     marker_positions=marker_positions, position_ids=position_ids, length=length)


def splice_embeddings(tok_embeds, block_embeds, plan: BlockPlan, label: str = "block"):
    """Splice `block_embeds` ``[B, n_block, D]`` into `tok_embeds` ``[B, text_len, D]``.

    Returns the merged embeddings. The text embeddings are untouched and keep their order; only
    their position ids moved, which :func:`plan_block` already decided. No tensor arithmetic
    happens across the boundary -- this is a concatenation, so a gradient reaches the block exactly
    as it reaches the token embeddings.
    """
    if tok_embeds.dim() != 3 or block_embeds.dim() != 3:
        raise ValueError("expected [B, L, D] for both inputs, got %s and %s"
                         % (tuple(tok_embeds.shape), tuple(block_embeds.shape)))
    if tok_embeds.size(0) != block_embeds.size(0):
        raise ValueError("batch mismatch: %d text rows against %d %s rows"
                         % (tok_embeds.size(0), block_embeds.size(0), label))
    if tok_embeds.size(-1) != block_embeds.size(-1):
        raise ValueError("width mismatch: text %d against %s %d; the projector must emit the "
                         "encoder's hidden size"
                         % (tok_embeds.size(-1), label, block_embeds.size(-1)))
    if block_embeds.size(1) != plan.n_block:
        raise ValueError("plan expects %d %s tokens, got %d"
                         % (plan.n_block, label, block_embeds.size(1)))
    if plan.splice > tok_embeds.size(1):
        raise ValueError("splice point %d is past the sequence (%d tokens)"
                         % (plan.splice, tok_embeds.size(1)))
    return _cat(tok_embeds, block_embeds, plan.splice)


def _cat(tok_embeds, block_embeds, splice: int):
    import torch
    return torch.cat([tok_embeds[:, :splice], block_embeds, tok_embeds[:, splice:]], dim=1)


def collate_block(items, embed_fn, block_key: str, label: str = "block", pad_id: int = 0):
    """Collate already-encoded items that carry a non-text block into one batch.

    `items` is the flat list of question rows (every state's rows, in order). A row that carries an
    input has ``ids`` (its text sequence), ``markers`` (already placed by the modality's plan),
    ``<block_key>`` (``[n_block, D]``) and ``plan``; a row that does not has only ``ids`` and
    ``markers``.

    A mixed list is legitimate and is what `Agent.predict_batch` produces from
    ``[{"image_path": p}, {"text": "hello"}]``: one batch is one forward pass, and each row is its
    own sequence, so a row without an input is simply a text row spliced into nothing. This used
    to raise `KeyError: 'visual'` on the first text row, which also made a *batched* gate-1
    ablation -- the input-removed half of `laya-evals ablate` -- impossible by construction.

    The block is the same width on every row that has one, so right-padding the *text* means a
    row's splice point is its own and nothing needs shifting; the padded batch is therefore
    ``max(len(ids)) + n_block`` wide, and the trailing pad of a short row lands after its text.

    `visual_mask` in the output is this batch's "these positions are the input, not the words"
    tensor. It carries the vision path's name because `DecisionModel.forward` asks it by that name,
    and both modalities fill it identically.

    Returns the keys `DecisionModel.forward` needs, with `inputs_embeds` in place of `input_ids`.
    """
    import torch

    block_rows = [it for it in items if block_key in it]
    n_block = max((it["plan"].n_block for it in block_rows), default=0)
    width = max(len(it["ids"]) for it in items) + n_block
    k_max = max(len(it["markers"]) for it in items)
    hidden = block_rows[0][block_key].size(-1) if block_rows else None
    if hidden is None:
        raise ValueError("collate_%s needs at least one row carrying a %s block; use "
                         "`collate_items` for a batch that is entirely text" % (label, label))
    # The block is spliced from the same device it arrived on: a trained projector emits on its own
    # device (a GPU), while `embed_fn` is a CPU lookup because batches are built here, on the CPU.
    # Allocating the batch where the block already is makes `torch.cat` legal without a per-row
    # round trip, and `_forward`'s `.to(device)` stays a no-op for it.
    device = block_rows[0][block_key].device

    embeds = torch.zeros(len(items), width, hidden, device=device)
    att = torch.zeros(len(items), width, dtype=torch.long, device=device)
    pos = torch.zeros(len(items), width, dtype=torch.long, device=device)
    mpos = torch.zeros(len(items), k_max, dtype=torch.long, device=device)
    mmask = torch.zeros(len(items), k_max, dtype=torch.bool, device=device)
    # Which positions hold the block, for a checkpoint with an attention readout. The readout asks
    # "which tokens are the input", and nothing else in the batch says.
    block_mask = torch.zeros(len(items), width, dtype=torch.bool, device=device)
    # Training batches carry a target; inference batches do not. Mirrors `collate_items`, including
    # its refusal of a target longer than the row's own marker count.
    has_target = any("target" in it for it in items)
    target = torch.zeros(len(items), k_max, dtype=torch.float32, device=device) if has_target else None

    for i, it in enumerate(items):
        ids = it["ids"]
        tok = embed_fn(torch.tensor(ids, dtype=torch.long)).to(device)
        if block_key in it:
            row = splice_embeddings(tok.unsqueeze(0), it[block_key].unsqueeze(0), it["plan"],
                                    label=label)[0]
            positions = list(it["plan"].position_ids)
            splice, own_n = it["plan"].splice, it["plan"].n_block
        else:
            # A text row in a block batch: nothing is spliced, and its positions are the plain
            # 0..L-1 a text-only batch would get -- the padded batch keeps each row's numbering
            # to itself, so a neighbour's block cannot shift this row's RoPE distances.
            row = tok
            positions = list(range(len(ids)))
            splice, own_n = 0, 0
        n = row.size(0)
        embeds[i, :n] = row
        att[i, :n] = 1
        pos[i, :n] = torch.tensor(positions, dtype=torch.long, device=device)
        if own_n:
            block_mask[i, splice:splice + own_n] = True
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"], dtype=torch.long, device=device)
        mmask[i, :k] = True
        if has_target and "target" in it:
            t = it["target"]
            if len(t) > k:
                raise ValueError(
                    "collate_%s: item %d has %d target entries but only %d marker positions; "
                    "a target needs one entry per option" % (label, i, len(t), k))
            target[i, :len(t)] = torch.tensor([float(v) for v in t], dtype=torch.float32,
                                              device=device)
    out = {
        "inputs_embeds": embeds,
        "attention_mask": att,
        "position_ids": pos,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": torch.tensor([it["qtype"] for it in items], dtype=torch.long, device=device),
        "visual_mask": block_mask,
    }
    if target is not None:
        out["target"] = target
    return out


def describe_plan(plan: BlockPlan, noun: str = "block") -> Tuple[str, ...]:
    """Human-readable facts about a plan, for error messages and `--dry-run` style output."""
    return ("splice at token %d" % plan.splice,
            "%d %s tokens%s" % (plan.n_block, noun, " (with summary)" if plan.summary else ""),
            "sequence length %d" % plan.length)
