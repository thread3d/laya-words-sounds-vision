"""Vision support for the decision engine: checkpoint config, visual-token layout, assembly.

This module is the whole of what a vision checkpoint needs that a text checkpoint does not, and
nothing here changes the text path. A checkpoint whose config carries no ``"vision"`` block parses
to ``None`` and every function below refuses to invent a layout, so ``build_sequence`` and
``DecisionModel.forward`` stay byte-identical for the checkpoints that exist today (the
no-regression invariant, spec C6).

Three pieces:

* :class:`VisionConfig` -- the ``"vision"`` block, validated once at load instead of per request.
* :func:`visual_plan` -- where the visual block goes, how many tokens it costs, and the position
  ids it takes. Pure Python; no tensors, so it is testable without a model.
* :func:`assemble_inputs_embeds` -- the one tensor operation: splice the projected visual tokens
  into the token embeddings and shift nothing else.

Why the block goes *after* the question head (``slot="after_options"``): ``build_head`` emits
``[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]``, and ModernBERT attends
locally with a 128-token window in 18 of its 28 layers. Splicing at the end of that head puts every
option's ``[MASK]`` marker inside the same local window as the image, which is the computation a
multiple-choice decision needs; appending the image after a long state would leave the markers able
to reach it only through the 10 global layers.

Why the visual tokens get their own position block: text positions keep their own monotone
numbering (the sequence is numbered as if the block were not there), so "where in the sentence" and
"where in the image" are never the same axis and the text's relative geometry is untouched. The
reserved region sits at the top of the model's positional budget.

Two facts about splicing into a HF encoder that cost real debugging time and are easy to get wrong:

* **Splice in token-embedding space, not in `encoder.embeddings(...)` output space.** Both
  `BertEmbeddings` and `ModernBertEmbeddings` apply their own LayerNorm (and dropout) to
  `inputs_embeds`, so handing them the *post-norm* embedding normalises the sequence twice. Measured
  on a tiny BERT, that drift is ~3e-2 in the hidden states; feeding the raw token embedding table
  (`tok_embeddings` / `word_embeddings`) reproduces the `input_ids` path exactly. The projector
  therefore emits something that belongs in the same space as the token embedding table.
* **A projector must emit a *varied* pattern.** LayerNorm removes two things that look like signal
  and are not: a constant vector maps to zero, and a per-token constant offset is subtracted as the
  mean. A "zeroed visual block" ablation is therefore indistinguishable from a block of any constant,
  and a projector whose rows are near-identical carries no information into the encoder at all.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Slots the builder understands. `after_options` is the only one implemented; the others are named
# so a config naming them fails loudly at load with a list of what exists, rather than silently
# falling back to the default and moving the block somewhere the author did not ask for.
SLOTS = ("after_options",)
# How the projected patch tokens are compressed before they enter the text sequence.
POOLINGS = ("adaptive_avg", "none")

VISION_DEFAULTS: Dict[str, Any] = {
    "tower": "google/siglip2-base-patch16-224",
    "tower_revision": None,
    "tower_sha256": None,
    "image_size": 224,
    "patch_grid": None,          # [rows, cols] when the tower's grid is not square
    "n_visual_tokens": 64,
    "pooling": "adaptive_avg",
    "slot": "after_options",
    "position_offset": None,     # None -> reserved block at the top of the positional budget
    "keep_layers": None,         # None -> the tower's last block; N -> read N blocks early
    "summary_for_options_over": 96,   # rendered option tokens that trigger a [VIS_SUMMARY] token
}


class VisionConfig:
    """The validated ``"vision"`` block of a checkpoint config.

    Every field is copied out of the mapping and normalised here, so a malformed config fails at
    load with the offending key named rather than mid-request.
    """

    __slots__ = tuple(VISION_DEFAULTS) + ("raw",)

    def __init__(self, raw: Dict[str, Any], max_position_embeddings: Optional[int] = None):
        unknown = sorted(set(raw) - set(VISION_DEFAULTS))
        if unknown:
            raise ValueError("vision: unknown key(s) %s; known keys are %s"
                             % (", ".join(unknown), ", ".join(sorted(VISION_DEFAULTS))))
        self.raw = dict(raw)
        for key, default in VISION_DEFAULTS.items():
            setattr(self, key, raw.get(key, default))

        if self.slot not in SLOTS:
            raise ValueError("vision.slot must be one of %s, got %r" % (", ".join(SLOTS), self.slot))
        if self.pooling not in POOLINGS:
            raise ValueError("vision.pooling must be one of %s, got %r"
                             % (", ".join(POOLINGS), self.pooling))
        if not isinstance(self.n_visual_tokens, int) or isinstance(self.n_visual_tokens, bool):
            raise TypeError("vision.n_visual_tokens must be an int, got %s"
                            % type(self.n_visual_tokens).__name__)
        if self.n_visual_tokens < 1:
            raise ValueError("vision.n_visual_tokens must be >= 1, got %d" % self.n_visual_tokens)
        if not isinstance(self.image_size, int) or self.image_size < 1:
            raise ValueError("vision.image_size must be a positive int, got %r" % (self.image_size,))
        if self.keep_layers is not None and (not isinstance(self.keep_layers, int) or self.keep_layers < 1):
            raise ValueError("vision.keep_layers must be None or a positive int, got %r"
                             % (self.keep_layers,))
        if self.position_offset is not None:
            if not isinstance(self.position_offset, int) or self.position_offset < 0:
                raise ValueError("vision.position_offset must be None or a non-negative int, got %r"
                                 % (self.position_offset,))
            if max_position_embeddings is not None and \
                    self.position_offset + self.n_visual_tokens > max_position_embeddings:
                raise ValueError(
                    "vision: the visual block needs %d positions from offset %d, past this "
                    "encoder's %d"
                    % (self.n_visual_tokens, self.position_offset, max_position_embeddings))

    # -- helpers ---------------------------------------------------------------------------
    def position_ids_for_visual(self, max_position_embeddings: int) -> List[int]:
        """The reserved positions the visual block occupies.

        The default block is the top of the positional budget rather than the bottom: text keeps
        positions from 0, so a block at the top cannot collide with any text position the caller
        can still add, however long the state grows.
        """
        start = self.position_offset
        if start is None:
            start = max(0, max_position_embeddings - self.n_visual_tokens)
        if max_position_embeddings and start + self.n_visual_tokens > max_position_embeddings:
            raise ValueError(
                "vision: position_offset=%r plus %d visual tokens exceeds this encoder's %d "
                "positions" % (self.position_offset, self.n_visual_tokens, max_position_embeddings))
        return list(range(start, start + self.n_visual_tokens))

    def summary_wanted(self, option_tokens: int) -> bool:
        """Whether a high-cardinality head needs the summary token beside its markers."""
        limit = self.summary_for_options_over
        return bool(limit) and option_tokens > limit

    def __repr__(self) -> str:
        return ("VisionConfig(tower=%r, n_visual_tokens=%d, slot=%r, pooling=%r)"
                % (self.tower, self.n_visual_tokens, self.slot, self.pooling))


def parse_vision(cfg: Optional[Dict[str, Any]],
                 max_position_embeddings: Optional[int] = None) -> Optional[VisionConfig]:
    """Read the ``"vision"`` block out of a checkpoint config.

    Returns ``None`` for a config with no block -- which is every checkpoint that exists today --
    and ``None`` for a block that is explicitly disabled (``{"enabled": false}``), so a config can
    keep the block while shipping a text-only artifact.
    """
    if not cfg:
        return None
    block = cfg.get("vision")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise TypeError("config['vision'] must be a mapping, got %s" % type(block).__name__)
    if block.get("enabled") is False:
        return None
    return VisionConfig({k: v for k, v in block.items() if k != "enabled"},
                        max_position_embeddings=max_position_embeddings)


class VisualPlan:
    """Where the visual block sits in one built sequence and what it costs.

    Produced by :func:`visual_plan`; consumed by :func:`assemble_inputs_embeds` at inference and by
    the training loop, so both agree about the splice point and the marker shifts.
    """

    __slots__ = ("splice", "n_visual", "summary", "marker_positions", "position_ids", "length")

    def __init__(self, splice: int, n_visual: int, summary: bool, marker_positions: Sequence[int],
                 position_ids: Sequence[int], length: int):
        self.splice = splice
        self.n_visual = n_visual
        self.summary = summary
        self.marker_positions = list(marker_positions)
        self.position_ids = list(position_ids)
        self.length = length

    def __repr__(self) -> str:
        return ("VisualPlan(splice=%d, n_visual=%d, summary=%s, length=%d)"
                % (self.splice, self.n_visual, self.summary, self.length))


def visual_plan(vision: VisionConfig, text_len: int, markers: Sequence[int], head_len: int,
                max_len: int, max_position_embeddings: int = 8192,
                option_tokens: Optional[int] = None) -> VisualPlan:
    """Lay the visual block out inside a sequence of `text_len` tokens.

    `head_len` is ``len(build_head(...))`` -- the splice point. `markers` are ``build_head``'s
    marker positions, which sit *before* the splice and therefore never move, unless a summary
    token is emitted in front of the first option, which shifts all of them by one.
    """
    if vision.slot != "after_options":                      # pragma: no cover - guarded at load
        raise ValueError("unsupported vision slot %r" % (vision.slot,))
    if not 0 <= head_len <= text_len:
        raise ValueError("head_len=%d must be within the sequence (0..%d)" % (head_len, text_len))

    summary = vision.summary_wanted(
        option_tokens if option_tokens is not None else text_len - head_len)
    n_visual = vision.n_visual_tokens + (1 if summary else 0)
    length = text_len + n_visual
    if max_len is not None and length > max_len:
        raise ValueError(
            "vision: %d text tokens + %d visual tokens = %d, past max_len=%d. Lower "
            "vision.n_visual_tokens, shorten the state, or raise max_len for this checkpoint."
            % (text_len, n_visual, length, max_len))

    text_positions = list(range(text_len))
    visual_positions = vision.position_ids_for_visual(max_position_embeddings)
    if summary:
        # The summary token is a visual token, so it needs a visual position too; take the one
        # after the reserved block so the real patch slots keep their numbering.
        visual_positions = visual_positions + [visual_positions[-1] + 1]

    marker_positions = [m + (1 if summary else 0) for m in markers]
    splice = head_len + (1 if summary else 0)
    position_ids = text_positions[:head_len] + visual_positions + text_positions[head_len:]
    return VisualPlan(splice=splice, n_visual=n_visual, summary=summary,
                      marker_positions=marker_positions, position_ids=position_ids,
                      length=length)


def assemble_inputs_embeds(tok_embeds, visual_embeds, plan: VisualPlan):
    """Splice `visual_embeds` ``[B, n_visual, D]`` into `tok_embeds` ``[B, text_len, D]``.

    Returns the merged embeddings. The text embeddings are untouched and keep their order; only
    their position ids moved, which :func:`visual_plan` already decided. No tensor arithmetic
    happens across the boundary -- this is a concatenation, so a gradient reaches the visual block
    exactly as it reaches the token embeddings.
    """
    if tok_embeds.dim() != 3 or visual_embeds.dim() != 3:
        raise ValueError("expected [B, L, D] for both inputs, got %s and %s"
                         % (tuple(tok_embeds.shape), tuple(visual_embeds.shape)))
    if tok_embeds.size(0) != visual_embeds.size(0):
        raise ValueError("batch mismatch: %d text rows against %d visual rows"
                         % (tok_embeds.size(0), visual_embeds.size(0)))
    if tok_embeds.size(-1) != visual_embeds.size(-1):
        raise ValueError("width mismatch: text %d against visual %d; the projector must emit the "
                         "encoder's hidden size" % (tok_embeds.size(-1), visual_embeds.size(-1)))
    if visual_embeds.size(1) != plan.n_visual:
        raise ValueError("plan expects %d visual tokens, got %d"
                         % (plan.n_visual, visual_embeds.size(1)))
    if plan.splice + 0 > tok_embeds.size(1):
        raise ValueError("splice point %d is past the sequence (%d tokens)"
                         % (plan.splice, tok_embeds.size(1)))
    return _cat(tok_embeds, visual_embeds, plan.splice)


def _cat(tok_embeds, visual_embeds, splice: int):
    import torch
    return torch.cat([tok_embeds[:, :splice], visual_embeds, tok_embeds[:, splice:]], dim=1)


def describe_plan(plan: VisualPlan) -> Tuple[str, ...]:
    """Human-readable facts about a plan, for error messages and `--dry-run` style output."""
    return ("splice at token %d" % plan.splice,
            "%d visual tokens%s" % (plan.n_visual, " (with summary)" if plan.summary else ""),
            "sequence length %d" % plan.length)
