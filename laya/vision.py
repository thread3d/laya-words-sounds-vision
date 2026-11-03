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
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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
    "position_policy": "contiguous",   # "contiguous" | "reserved"
    "position_offset": None,     # only for "reserved": where the block starts (None -> the top)
    "keep_layers": None,         # None -> the tower's last block; N -> read N blocks early
    "summary_for_options_over": 96,   # rendered option tokens that trigger a [VIS_SUMMARY] token
    "pixel_mean": 0.5,          # SigLIP's processor rescales 1/255 then normalises by 0.5/0.5
    "pixel_std": 0.5,
    "cache_size": 64,           # distinct images kept in the per-agent visual-block cache
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
        for key in ("pixel_mean", "pixel_std"):
            val = getattr(self, key)
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not val > 0:
                raise ValueError("vision.%s must be a positive number, got %r" % (key, val))
        if not isinstance(self.cache_size, int) or self.cache_size < 1:
            raise ValueError("vision.cache_size must be a positive int, got %r" % (self.cache_size,))
        if self.keep_layers is not None and (not isinstance(self.keep_layers, int) or self.keep_layers < 1):
            raise ValueError("vision.keep_layers must be None or a positive int, got %r"
                             % (self.keep_layers,))
        if self.position_policy not in ("contiguous", "reserved"):
            raise ValueError("vision.position_policy must be 'contiguous' or 'reserved', got %r"
                             % (self.position_policy,))
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
    def position_ids_for_visual(self, max_position_embeddings: int,
                                start: Optional[int] = None) -> List[int]:
        """The positions the visual block occupies, for the `reserved` policy.

        `start` is the block's first position; `None` reserves the top of the positional budget.

        **This policy loses to `contiguous` on this encoder, and the measurement is the reason the
        default is what it is.** The block is *sequence*-adjacent to the option markers, but a
        reserved block puts it ~8000 positions away in RoPE terms -- outside the 128-token local
        window of 18 of ModernBERT's 28 layers and far outside the range the checkpoint was trained
        on. A frozen-feature probe on a trained checkpoint put the colour signal at 0.708 on the
        marker states with this policy; the same probe is the check for whatever replaces it.
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

    splice = head_len + (1 if summary else 0)
    marker_positions = [m + (1 if summary else 0) for m in markers]

    if vision.position_policy == "contiguous":
        # The block continues the numbering where the head left off, and the text after it shifts
        # by the block's width. RoPE encodes *distance*, so this is what actually makes the block
        # adjacent to the option markers: the offsets are small, the 128-token local windows reach
        # it, and every relative distance stays inside the range the checkpoint was trained on.
        visual_positions = list(range(head_len, head_len + n_visual))
        after = [p + n_visual for p in range(head_len, text_len)]
        position_ids = list(range(head_len)) + visual_positions + after
    else:
        text_positions = list(range(text_len))
        visual_positions = vision.position_ids_for_visual(
            max_position_embeddings, start=splice if vision.position_offset is None else None)
        if summary:
            visual_positions = visual_positions + [visual_positions[-1] + 1]
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


# ------------------------------------------------------------------------------------------------
# The tower and the projector
# ------------------------------------------------------------------------------------------------

def _torch():
    """Import torch here, not at module import.

    `import laya.vision` has to stay cheap for the text path: the router and `laya.serve` must not
    pull torch in (`tests/test_lazy_import.py`), and a text-only deployment should never pay for a
    module it does not use.
    """
    import torch
    return torch


class PooledProjector:
    """Compress the tower's patch tokens and project them into the encoder's width.

    Two operations, in this order:

    * **pool** -- ``adaptive_avg`` folds the patch grid down to ``n_visual_tokens`` positions
      (the documented baseline; a learned coarse-to-fine connector replaces this later behind the
      same interface). ``none`` passes the patches through untouched.
    * **project** -- a linear map into the text encoder's hidden size, plus a learned visual
      *type* vector.

    The type vector lives here rather than on ``DecisionModel`` on purpose. A new parameter on the
    decision model would break `load_state_dict(..., strict=True)` for every existing checkpoint;
    putting it in the projector keeps text checkpoints byte-identical and puts the "this is a
    patch, not a word" signal exactly where the patches are produced. It is added *after* the
    linear so it cannot be confused with a bias on a constant row.
    """

    def __init__(self, tower_dim: int, hidden_size: int, n_visual_tokens: int,
                 pooling: str = "adaptive_avg", dtype=None):
        torch = _torch()
        self.n_visual_tokens = n_visual_tokens
        self.pooling = pooling
        self.proj = torch.nn.Linear(tower_dim, hidden_size)
        self.type_vector = torch.nn.Parameter(torch.zeros(hidden_size))
        if dtype is not None:
            self.proj = self.proj.to(dtype)
            self.type_vector.data = self.type_vector.data.to(dtype)

    def parameters(self):
        return [self.proj.weight, self.proj.bias, self.type_vector]

    def to(self, device):
        self.proj = self.proj.to(device)
        self.type_vector.data = self.type_vector.data.to(device)
        return self

    def __call__(self, patches):
        return self.forward(patches)

    def forward(self, patches):
        """`patches` ``[B, N, C]`` -> ``[B, n_visual_tokens, hidden]``."""
        torch = _torch()
        x = patches
        if self.pooling == "adaptive_avg":
            x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.n_visual_tokens)
            x = x.transpose(1, 2)
        elif self.pooling != "none":
            raise ValueError("unknown pooling %r" % (self.pooling,))
        return self.proj(x) + self.type_vector


class VisionEncoder:
    """A frozen vision tower plus the projector, producing the visual block.

    The tower is **injected**, not loaded here, so this class is testable with a stub and so the
    caller decides what to download. :func:`load_tower` is the convenience that builds the real
    one, and it is the only place that imports a vision architecture.
    """

    #: Where the tower and projector live, set by :meth:`to`. ``None`` (the default) leaves input
    #: tensors untouched, which is the CPU path every caller had before devices entered the story.
    device = None

    def __init__(self, tower, projector: PooledProjector, config: VisionConfig):
        self.tower = tower
        self.projector = projector
        self.config = config

    def train(self, mode: bool = True):
        self.tower.train(mode) if mode else self.tower.eval()
        return self

    def eval(self):
        return self.train(False)

    def to(self, device):
        """Move the tower *and* the projector to `device`.

        `VisionEncoder` is deliberately not an `nn.Module` (the tower is injected and the
        projector is trained on its own), so `DecisionModel.to(device)` reaches neither half:
        `vision_encoder` is a plain attribute, and the projector lives beside the model. A caller
        that only moves the model therefore leaves the frozen tower on the CPU, which does not
        fail -- it just encodes every image ~100x slower while the trainer reports a GPU run.
        That was measured here at 2 images/s, and this method is the fix.

        Recording the device here also lets :meth:`encode_patches` place the pixels, which is what
        makes a CPU-decoded image safe to hand to a GPU tower.
        """
        torch = _torch()
        self.tower.to(device)
        self.projector.to(device)
        self.device = torch.device(device) if device is not None else None
        return self

    def encode_patches(self, pixel_values):
        """``[B, 3, H, W]`` -> the tower's patch tokens ``[B, N, C]``, unprojected.

        The split that makes training possible: the tower is frozen, so its output can be computed
        once per image and cached; the projector is *trained*, so it has to run inside the step
        where autograd can see it. `encode` is the two steps together, which is what inference
        wants.
        """
        torch = _torch()
        if self.device is not None:
            # A decoded image is a CPU tensor; the tower is wherever `to()` put it. Moving it here
            # rather than at each call site is the same rule the projector already follows, and it
            # is what keeps a GPU checkpoint from raising on its own inputs.
            pixel_values = pixel_values.to(self.device)
        normalised = (pixel_values - self.config.pixel_mean) / self.config.pixel_std
        with torch.no_grad() if not self._tower_trains() else _nullcontext():
            out = self.tower(pixel_values=normalised)
        patches = getattr(out, "last_hidden_state", None)
        if patches is None:
            raise TypeError("the vision tower must return `last_hidden_state` patch tokens; "
                            "got %s" % type(out).__name__)
        return patches

    def encode(self, pixel_values):
        """``[B, 3, H, W]`` pixels in [0, 1] -> ``[B, n_visual_tokens, hidden]`` embeddings.

        Pixel normalisation happens in :meth:`encode_patches`, because it belongs to the tower:
        SigLIP's processor rescales 1/255 and then maps [0, 1] through (x - mean) / std with
        0.5/0.5, so a caller handing over raw [0, 1] pixels would otherwise feed the tower values it
        was never trained on. Both constants are config fields so a different tower can move them.
        """
        return self.projector(self.encode_patches(pixel_values))

    def _tower_trains(self) -> bool:
        for p in self.tower.parameters():
            if getattr(p, "requires_grad", False):
                return True
        return False


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def load_tower(model_id: str, revision: Optional[str] = None, keep_layers: Optional[int] = None,
               dtype=None, trust_remote_code: bool = False):
    """Build the frozen vision tower. The only import of a vision architecture in the package.

    `keep_layers` reads the patch tokens `keep_layers` blocks early. Two independent groups found
    the last blocks of a vision tower produce *worse* features than intermediate ones (Meta PE
    measures -14.0 DAVIS J&F from its best layer to its last; InternVL 1.5 took the fourth-to-last
    layer and dropped three blocks), so this is a free latency saving that the prior says is also a
    free accuracy win. It is a config key, not a code change, so it can be swept.
    """
    from transformers import SiglipVisionModel

    kw = {}
    if revision:
        kw["revision"] = revision
    if dtype is not None:
        kw["torch_dtype"] = dtype
    if trust_remote_code:
        kw["trust_remote_code"] = True
    tower = SiglipVisionModel.from_pretrained(model_id, **kw)
    for p in tower.parameters():
        p.requires_grad_(False)
    tower.eval()
    if keep_layers:
        _truncate_tower(tower, keep_layers)
    return tower


def _truncate_tower(tower, keep_layers: int):
    """Keep only the first `keep_layers` blocks of the tower's encoder."""
    vision_model = getattr(tower, "vision_model", tower)
    enc = getattr(vision_model, "encoder", None)
    layers = getattr(enc, "layers", None)
    if layers is None:
        raise TypeError("keep_layers=%d was requested but this tower has no `.encoder.layers` to "
                        "truncate" % keep_layers)
    if not 1 <= keep_layers <= len(layers):
        raise ValueError("keep_layers=%d is out of range for a %d-block tower"
                         % (keep_layers, len(layers)))
    enc.layers = layers[:keep_layers]
    return tower


TOWER_DIR = "vision_tower"


def local_tower_dir(model_dir: Optional[str]) -> Optional[str]:
    """The bundled tower inside a checkpoint directory, when one is there.

    A vision checkpoint that names a Hub repository for its tower cannot serve a request offline,
    which contradicts the reason Laya exists. `laya-train` bundles the tower beside the weights
    (`save_vision_weights`'s neighbour), and this is the lookup that prefers it. A checkpoint with
    no bundle falls back to the Hub id in its config, so existing checkpoints are unaffected.
    """
    if not model_dir:
        return None
    candidate = os.path.join(model_dir, TOWER_DIR)
    return candidate if os.path.isdir(candidate) else None


def build_vision_encoder(config: VisionConfig, hidden_size: int, tower=None, dtype=None,
                         tower_dir: Optional[str] = None):
    """Assemble a :class:`VisionEncoder` for `config`.

    `tower` is injected when the caller already has one (tests, or a shared tower across
    checkpoints); otherwise it is loaded from `tower_dir` if given, and from `config.tower` last.
    A bundled directory is preferred over the Hub id because the bundled one is the checkpoint's
    own artifact: it cannot be re-tagged, re-uploaded, or disappear behind a network.
    """
    torch = _torch()
    if tower is None:
        if tower_dir:
            tower = load_tower(tower_dir, keep_layers=config.keep_layers, dtype=dtype)
        else:
            tower = load_tower(config.tower, revision=config.tower_revision,
                               keep_layers=config.keep_layers, dtype=dtype)
    tower_dim = _tower_width(tower)
    projector = PooledProjector(tower_dim, hidden_size, config.n_visual_tokens,
                                pooling=config.pooling, dtype=dtype)
    return VisionEncoder(tower, projector, config)


def _tower_width(tower) -> int:
    cfg = getattr(tower, "config", None)
    for attr in ("hidden_size", "projection_dim"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int):
            return val
    vision_cfg = getattr(cfg, "vision_config", None)
    val = getattr(vision_cfg, "hidden_size", None)
    if isinstance(val, int):
        return val
    raise TypeError("cannot determine the tower's patch width from %s" % type(cfg).__name__)


def tower_parameters(encoder: VisionEncoder):
    """The trainable parameters of a vision encoder (the tower stays frozen by default)."""
    return encoder.projector.parameters()


# ------------------------------------------------------------------------------------------------
# Images inside a state
# ------------------------------------------------------------------------------------------------

# Keys a state may use to carry an image. `text` is the companion text, and it may be absent: an
# image on its own is a legal state (spec §6.5).
IMAGE_KEYS = ("image", "image_path", "image_base64")


def has_image(state) -> bool:
    """Whether a state carries an image reference -- the routing question, answered cheaply.

    Deliberately narrower than `split_state`: it does not build anything, so the router can ask it
    on every request without paying for a state it may not need to split.
    """
    return isinstance(state, dict) and any(key in state for key in IMAGE_KEYS + ("image_url",))


def split_state(state):
    """Separate a state into its text half and its image spec.

    Returns ``(text_state, image_spec)``. For anything that carries no image this is
    ``(state, None)`` -- the caller only invokes it when the checkpoint has vision support, so a
    text checkpoint never has a state reshaped underneath it (spec C6).

    Recognised shapes::

        {"image": <spec>, "text": "..."}      # explicit
        {"image": <spec>}                     # the image is the whole state
        {"image_path": "a.png"}               # shorthand

    `<spec>` is a path string, a base64 string, or a mapping with a ``pixels`` entry holding an
    already-decoded tensor/array (which is how a caller with its own decoder, and the tests,
    bypass Pillow entirely).
    """
    if not isinstance(state, dict):
        return state, None
    spec = None
    for key in IMAGE_KEYS:
        if key in state:
            spec = state[key]
            break
    if spec is None:
        return state, None
    text = state.get("text")
    # Everything except the image key and the text companion is still state the model should see,
    # so it is not dropped: {"image": ..., "vendor": "acme"} keeps "vendor" as data.
    rest = {k: v for k, v in state.items() if k not in IMAGE_KEYS and k != "text"}
    if text is None:
        return (rest if rest else ""), spec
    if rest:
        merged = dict(rest)
        merged["text"] = text
        return merged, spec
    return text, spec


def load_pixels(spec, image_size: int = 224):
    """Decode an image spec into a ``[3, image_size, image_size]`` float tensor in [0, 1].

    Pillow is an optional dependency (`pip install laya[vision]`), imported here rather than at
    module import so a text-only deployment never pays for it. A spec carrying `pixels` bypasses
    decoding entirely.
    """
    torch = _torch()
    if isinstance(spec, dict) and "pixels" in spec:
        px = spec["pixels"]
        if not torch.is_tensor(px):
            px = torch.as_tensor(px)
        return _fit(px, image_size)
    if not isinstance(spec, (str, bytes)):
        raise TypeError("an image must be a path/base64 string or {'pixels': ...}, got %s"
                        % type(spec).__name__)
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:                                  # pragma: no cover - optional dep
        raise ImportError(
            "loading an image needs Pillow: `pip install laya[vision]`, or pass "
            "{'image': {'pixels': <tensor>}} with your own decoder") from exc
    import base64
    import io
    if isinstance(spec, str) and spec.startswith(("http://", "https://")):
        raise ValueError("laya does not fetch images: resolve the URL in the caller (the HTTP "
                         "server owns its own timeouts and limits) and pass a path or bytes")
    if isinstance(spec, bytes):
        # Raw image bytes, exactly what a server hands over after its own transport and limits.
        # They are *not* base64 text: decoding them as base64 was the earlier reading, and it
        # turned every served image into mojibake that Pillow then refused.
        src = io.BytesIO(spec)
    elif _looks_base64(spec):
        src = io.BytesIO(base64.b64decode(spec))
    else:
        src = spec
    with Image.open(src) as im:
        im = _to_rgb(im, Image, ImageOps)
        im = im.resize((image_size, image_size))
        arr = np.asarray(im, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _to_rgb(im, Image, ImageOps):
    """The normalisation C5 requires, in the order the specification lists it (§10.2).

    Three steps a naive decode skips, each of which silently changes the pixels:

    * **EXIF orientation.** A phone photo carries the rotation in its metadata and stores the
      sensor's pixels unrotated. Ignoring the tag hands the tower a sideways picture -- not a
      degraded one, a *different* one -- and the decision follows it.
    * **alpha onto white.** A transparent region has no colour; compositing it onto black rather
      than white turns "nothing here" into "a dark object", which is a different image and a
      different answer.
    * **sRGB.** A CMYK scan or a palette image is not laid out as RGB bytes, and `convert` maps it
      through the sRGB space the tower was trained on rather than reading the raw channels.

    The order matters: orientation first (so the composite happens on the upright image), alpha
    before the resize (so a transparent edge does not bleed into its neighbours), and the colour
    conversion last.
    """
    transposed = ImageOps.exif_transpose(im)
    if transposed is not None:                       # older Pillow returns None when there is no tag
        im = transposed
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        # White, not black: a transparent corner is absence of evidence, and black would make it
        # a very dark object.
        backdrop = Image.new("RGB", im.size, (255, 255, 255))
        backdrop.paste(im, mask=im.split()[-1])
        im = backdrop
    elif im.mode != "RGB":
        im = im.convert("RGB")
    return im


def _looks_base64(spec: str) -> bool:
    if not isinstance(spec, str) or os.path.exists(spec):       # a real path wins over a guess
        return False
    stripped = spec.strip()
    if len(stripped) < 64 or len(stripped) % 4:
        return False
    return all(c.isalnum() or c in "+/=\n\r" for c in stripped)


def _fit(px, image_size: int):
    torch = _torch()
    if px.dim() == 4:
        px = px[0]
    if px.dim() == 2:
        px = px.unsqueeze(0).repeat(3, 1, 1)
    if px.dim() != 3:
        raise ValueError("pixels must be [3, H, W] or [H, W], got %s" % (tuple(px.shape),))
    if px.size(0) == 1:
        px = px.repeat(3, 1, 1)
    elif px.size(0) > 3:
        px = px[:3]
    px = px.float()
    if px.max() > 1.5:                       # 0..255 input
        px = px / 255.0
    if px.size(1) != image_size or px.size(2) != image_size:
        px = torch.nn.functional.interpolate(px.unsqueeze(0), size=(image_size, image_size),
                                             mode="bilinear", align_corners=False).squeeze(0)
    return px


class VisualCache:
    """Content-addressed cache of the visual block, so one image is encoded once per call.

    The spec's §4.6: `predict_batch` already tokenizes a shared state once and reuses the ids
    across every question; the vision pass needs the same treatment and more urgently, because the
    tower is the expensive half. Keyed by a hash of the normalised pixels -- two encodings of the
    same picture are one image -- with a bounded size so a long-lived server cannot grow without
    limit.
    """

    def __init__(self, max_entries: int = 64):
        self.max_entries = max(1, int(max_entries))
        self._store = _ordered_dict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(pixels) -> str:
        import hashlib
        return hashlib.sha256(pixels.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def get_or_encode(self, pixels, encode):
        k = self.key(pixels)
        if k in self._store:
            self.hits += 1
            self._store.move_to_end(k)
            return self._store[k]
        self.misses += 1
        value = encode(pixels)
        self._store[k] = value
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
        return value

    def clear(self):
        self._store.clear()


def _ordered_dict():
    from collections import OrderedDict
    return OrderedDict()


def collate_visual(items, embed_fn, pad_id: int = 0):
    """Collate already-encoded items that carry a visual block into one batch.

    `items` is the flat list of question rows (every state's rows, in order). Each row carries
    ``ids`` (its text sequence), ``markers`` (already shifted by :func:`visual_plan`) and
    ``visual`` (``[n_visual, D]``) plus ``plan``.

    The visual block is the same width on every row, so right-padding the *text* means a row's
    splice point is its own and nothing needs shifting; the padded batch is therefore
    ``max(len(ids)) + n_visual`` wide, and the trailing pad of a short row lands after its text.

    Returns the keys `DecisionModel.forward` needs, with `inputs_embeds` in place of `input_ids`.
    """
    torch = _torch()
    n_visual = items[0]["plan"].n_visual
    width = max(len(it["ids"]) for it in items) + n_visual
    k_max = max(len(it["markers"]) for it in items)
    hidden = items[0]["visual"].size(-1)
    # The block is spliced from the same device the visual half arrived on: a trained projector
    # emits on its own device (a GPU), while `embed_fn` is a CPU lookup because batches are built
    # here, on the CPU. Allocating the batch where the block already is makes `torch.cat`
    # legal without a per-row round trip, and `_forward`'s `.to(device)` stays a no-op for it.
    device = items[0]["visual"].device

    embeds = torch.zeros(len(items), width, hidden, device=device)
    att = torch.zeros(len(items), width, dtype=torch.long, device=device)
    pos = torch.zeros(len(items), width, dtype=torch.long, device=device)
    mpos = torch.zeros(len(items), k_max, dtype=torch.long, device=device)
    mmask = torch.zeros(len(items), k_max, dtype=torch.bool, device=device)
    # Which positions hold the visual block, for a checkpoint with an attention readout. The
    # readout asks "which tokens are the picture", and nothing else in the batch says.
    vmask = torch.zeros(len(items), width, dtype=torch.bool, device=device)
    # Training batches carry a target; inference batches do not. Mirrors `collate_items`, including
    # its refusal of a target longer than the row's own marker count.
    has_target = any("target" in it for it in items)
    target = torch.zeros(len(items), k_max, dtype=torch.float32, device=device) if has_target else None

    for i, it in enumerate(items):
        ids = it["ids"]
        tok = embed_fn(torch.tensor(ids, dtype=torch.long)).to(device)
        row = assemble_inputs_embeds(tok.unsqueeze(0), it["visual"].unsqueeze(0), it["plan"])[0]
        n = row.size(0)
        embeds[i, :n] = row
        att[i, :n] = 1
        pos[i, :n] = torch.tensor(it["plan"].position_ids, dtype=torch.long, device=device)
        vmask[i, it["plan"].splice:it["plan"].splice + n_visual] = True
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"], dtype=torch.long, device=device)
        mmask[i, :k] = True
        if has_target and "target" in it:
            t = it["target"]
            if len(t) > k:
                raise ValueError(
                    "collate_visual: item %d has %d target entries but only %d marker positions; "
                    "a target needs one entry per option" % (i, len(t), k))
            target[i, :len(t)] = torch.tensor([float(v) for v in t], dtype=torch.float32, device=device)
    out = {
        "inputs_embeds": embeds,
        "attention_mask": att,
        "position_ids": pos,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": torch.tensor([it["qtype"] for it in items], dtype=torch.long, device=device),
        "visual_mask": vmask,
    }
    if target is not None:
        out["target"] = target
    return out


# ------------------------------------------------------------------------------------------------
# Transport limits (the serving layer's half)
# ------------------------------------------------------------------------------------------------
#
# A server cannot hand raw pixels to a model without first answering two questions: how many bytes
# arrived, and how big the picture claims to be. The second matters more than it looks -- a
# 40-byte PNG can declare 30000x30000 pixels, and decoding it before checking is a decompression
# bomb that costs gigabytes for a request that should have been refused in microseconds. Both
# checks are done on the header, before any decoder runs, which is also why they work on a
# deployment that has not installed the vision extra.

MAX_IMAGE_BYTES = 12 * 1024 * 1024        # 12 MB, override with LAYA_MAX_IMAGE_BYTES
MAX_IMAGE_PIXELS = 4096 * 4096            # ~16.8 MP, override with LAYA_MAX_IMAGE_PIXELS


class ImageRefused(ValueError):
    """A refused image, carrying the reason a caller should be told."""

    def __init__(self, reason: str, detail: str):
        super().__init__("%s: %s" % (reason, detail))
        self.reason = reason
        self.detail = detail


def image_info(data) -> Optional[Dict[str, Any]]:
    """Width, height and format read from the first bytes of an image, without decoding it.

    Covers PNG, JPEG, GIF and WebP -- the formats a caller actually sends. Returns ``None`` for
    anything else rather than guessing; the caller decides whether an unidentified header is a
    refusal or merely unmeasured.
    """
    if not isinstance(data, (bytes, bytearray)) or len(data) < 12:
        return None
    data = bytes(data)
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return {"format": "png", "width": width, "height": height}
    if data[:6] in (b"GIF87a", b"GIF89a"):
        width = int.from_bytes(data[6:8], "little")
        height = int.from_bytes(data[8:10], "little")
        return {"format": "gif", "width": width, "height": height}
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8 " and len(data) >= 30:
            width = int.from_bytes(data[26:28], "little") & 0x3FFF
            height = int.from_bytes(data[28:30], "little") & 0x3FFF
            return {"format": "webp", "width": width, "height": height}
        if chunk == b"VP8L" and len(data) >= 25:
            bits = int.from_bytes(data[21:25], "little")
            return {"format": "webp", "width": (bits & 0x3FFF) + 1,
                    "height": ((bits >> 14) & 0x3FFF) + 1}
        if chunk == b"VP8X" and len(data) >= 30:
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return {"format": "webp", "width": width, "height": height}
        return {"format": "webp", "width": None, "height": None}
    if data[:2] == b"\xff\xd8":
        # JPEG carries its dimensions in a start-of-frame segment, which is not at a fixed offset:
        # walk the marker chain until one shows up.
        i = 2
        n = len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            segment = int.from_bytes(data[i + 2:i + 4], "big")
            # SOF0..SOF15 minus the three non-frame markers in that range (DHT, JPG, DAC)
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height = int.from_bytes(data[i + 5:i + 7], "big")
                width = int.from_bytes(data[i + 7:i + 9], "big")
                return {"format": "jpeg", "width": width, "height": height}
            i += 2 + segment
        return {"format": "jpeg", "width": None, "height": None}
    return None


def check_image_bytes(data, max_bytes: Optional[int] = None,
                      max_pixels: Optional[int] = None) -> Dict[str, Any]:
    """Refuse an image that is too large to decode, before any decoder sees it.

    Returns what :func:`image_info` found, so a caller that wants the dimensions does not parse the
    header twice. An unidentified format is refused only when a pixel limit is in force and the
    header could not supply one -- "I could not measure it" is not the same as "it is small".
    """
    max_bytes = MAX_IMAGE_BYTES if max_bytes is None else int(max_bytes)
    max_pixels = MAX_IMAGE_PIXELS if max_pixels is None else int(max_pixels)
    if not isinstance(data, (bytes, bytearray)):
        raise ImageRefused("image", "expected bytes, got %s" % type(data).__name__)
    if not data:
        raise ImageRefused("image", "empty body")
    if len(data) > max_bytes:
        raise ImageRefused("image_too_large",
                           "%d bytes exceeds the %d-byte limit" % (len(data), max_bytes))
    info = image_info(data)
    if info is None:
        raise ImageRefused("image_unsupported",
                           "not a PNG, JPEG, GIF or WebP image")
    width, height = info.get("width"), info.get("height")
    if width is None or height is None:
        raise ImageRefused("image_unmeasurable",
                           "the %s header did not state its dimensions, so the %d-pixel limit "
                           "cannot be checked" % (info.get("format"), max_pixels))
    if width < 1 or height < 1:
        raise ImageRefused("image_invalid",
                           "the %s header states %dx%d" % (info.get("format"), width, height))
    if width * height > max_pixels:
        raise ImageRefused("image_too_large",
                           "%dx%d = %d pixels exceeds the %d-pixel limit"
                           % (width, height, width * height, max_pixels))
    return info


def read_image_bytes(spec, max_bytes: Optional[int] = None, max_pixels: Optional[int] = None,
                     allow_url: bool = False, timeout: float = 5.0) -> bytes:
    """Resolve an image reference into raw bytes, applying the transport limits.

    Accepted forms: raw ``bytes`` (already read), a filesystem ``path``, a base64 string, or --
    only when `allow_url` is set -- an ``http(s)`` URL.

    **URLs are off by default and that is not an oversight.** A self-hosted decision engine that
    fetches whatever URL a request names is a server-side request forgery primitive pointed at the
    operator's own network: ``http://169.254.169.254/`` and ``http://localhost:9200/`` are one
    request away. A deployment that wants it opts in and puts its own egress policy in front.
    """
    import base64
    from urllib.parse import urlparse

    if isinstance(spec, (bytes, bytearray)):
        data = bytes(spec)
    elif isinstance(spec, dict):
        if "bytes" in spec:
            data = spec["bytes"]
        elif "path" in spec:
            data = _read_path(spec["path"], max_bytes)
        elif "url" in spec:
            data = _read_url(spec["url"], max_bytes, allow_url, timeout, urlparse)
        elif "base64" in spec:
            data = _decode_base64(spec["base64"], max_bytes, base64)
        else:
            raise ImageRefused("image",
                               "a mapping must carry one of bytes, path, url or base64; got %s"
                               % ", ".join(sorted(spec)))
    elif isinstance(spec, str):
        if spec.startswith(("http://", "https://")):
            data = _read_url(spec, max_bytes, allow_url, timeout, urlparse)
        elif os.path.exists(spec):
            data = _read_path(spec, max_bytes)
        else:
            data = _decode_base64(spec, max_bytes, base64)
    else:
        raise ImageRefused("image", "expected bytes, a path, a URL or base64; got %s"
                           % type(spec).__name__)
    check_image_bytes(data, max_bytes=max_bytes, max_pixels=max_pixels)
    return data


def _read_path(path, max_bytes) -> bytes:
    if not isinstance(path, str):
        raise ImageRefused("image", "path must be a string, got %s" % type(path).__name__)
    limit = MAX_IMAGE_BYTES if max_bytes is None else int(max_bytes)
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise ImageRefused("image_unreadable", "%s: %s" % (path, exc.strerror or exc)) from exc
    if size > limit:
        raise ImageRefused("image_too_large",
                           "%s is %d bytes, over the %d-byte limit" % (path, size, limit))
    try:
        with open(path, "rb") as handle:
            return handle.read(limit + 1)
    except OSError as exc:
        raise ImageRefused("image_unreadable", "%s: %s" % (path, exc.strerror or exc)) from exc


def _decode_base64(text, max_bytes, base64) -> bytes:
    if not isinstance(text, str):
        raise ImageRefused("image", "base64 must be a string, got %s" % type(text).__name__)
    limit = MAX_IMAGE_BYTES if max_bytes is None else int(max_bytes)
    # A base64 string is 4/3 the size of what it decodes to, so the encoded length is already a
    # lower bound on the work; refusing here avoids materialising a huge buffer to measure it.
    if len(text) > (limit // 3 + 1) * 4:
        raise ImageRefused("image_too_large",
                           "the encoded image is longer than a %d-byte image can be" % limit)
    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:                                     # noqa: BLE001 - any decode error
        raise ImageRefused("image_invalid", "base64 did not decode: %s" % exc) from exc


def _read_url(url, max_bytes, allow_url, timeout, urlparse) -> bytes:
    if not allow_url:
        raise ImageRefused(
            "image_url_disabled",
            "fetching an image over the network is off by default; set LAYA_ALLOW_IMAGE_URLS=1 "
            "to enable it, or send the bytes")
    if not isinstance(url, str):
        raise ImageRefused("image", "url must be a string, got %s" % type(url).__name__)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ImageRefused("image_url_scheme",
                           "only http and https are fetched, got %r" % (parsed.scheme,))
    if not parsed.hostname:
        raise ImageRefused("image_url_invalid", "no host in %r" % (url,))
    limit = MAX_IMAGE_BYTES if max_bytes is None else int(max_bytes)
    import urllib.request
    request = urllib.request.Request(url, headers={"User-Agent": "laya-image-fetch"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:   # noqa: S310 - opt-in
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > limit:
                raise ImageRefused("image_too_large",
                                   "the server declares %s bytes, over the %d-byte limit"
                                   % (declared, limit))
            data = response.read(limit + 1)
    except ImageRefused:
        raise
    except Exception as exc:                                     # noqa: BLE001 - network errors
        raise ImageRefused("image_unreachable", "%s: %s" % (url, exc)) from exc
    if len(data) > limit:
        raise ImageRefused("image_too_large", "%s returned more than %d bytes" % (url, limit))
    return data


# ------------------------------------------------------------------------------------------------
# Persisting the trained projector
# ------------------------------------------------------------------------------------------------
#
# `DecisionModel` deliberately has no vision parameters -- a new one would break
# `load_state_dict(..., strict=True)` for every existing checkpoint -- so the projector lives beside
# the model and is *not* in `model.state_dict()`. That makes it the one part of a vision checkpoint
# that a naive save would drop, and the failure is silent in the worst way: every load would build a
# fresh random projector and answer confidently from it. So it gets its own file, and a checkpoint
# that declares vision but carries none is refused rather than guessed at.

VISION_WEIGHTS_FILE = "vision.safetensors"


def save_vision_weights(encoder: "VisionEncoder", path: str) -> str:
    """Write the projector's trained tensors into `path` (a directory)."""
    from safetensors.torch import save_file

    torch = _torch()
    os.makedirs(path, exist_ok=True)
    target = os.path.join(path, VISION_WEIGHTS_FILE)
    payload = {
        "projector.weight": encoder.projector.proj.weight.detach().cpu().contiguous(),
        "projector.bias": encoder.projector.proj.bias.detach().cpu().contiguous(),
        "type_vector": encoder.projector.type_vector.detach().cpu().contiguous(),
    }
    save_file({k: v.half() if v.is_floating_point() else v for k, v in payload.items()}, target)
    del torch
    return target


def load_vision_weights(encoder: "VisionEncoder", path: str, dtype=None) -> "VisionEncoder":
    """Load the projector's trained tensors from `path` (a directory) into `encoder`."""
    from safetensors.torch import load_file

    torch = _torch()
    target = os.path.join(path, VISION_WEIGHTS_FILE) if os.path.isdir(path) else path
    if not os.path.exists(target):
        raise FileNotFoundError(
            "%r is missing: this checkpoint declares vision support, so its projector has to come "
            "from somewhere. A checkpoint saved without it would answer from a randomly "
            "initialised projector, which is indistinguishable from a working one." % (target,))
    weights = load_file(target)
    weight = weights["projector.weight"].to(encoder.projector.proj.weight.device)
    bias = weights["projector.bias"].to(encoder.projector.proj.bias.device)
    type_vector = weights["type_vector"].to(encoder.projector.type_vector.device)
    target_dtype = dtype or encoder.projector.proj.weight.dtype
    encoder.projector.proj.weight.data = weight.to(target_dtype)
    encoder.projector.proj.bias.data = bias.to(target_dtype)
    encoder.projector.type_vector.data = type_vector.to(target_dtype)
    del torch
    return encoder


def bundle_tower(encoder: "VisionEncoder", path: str) -> str:
    """Copy the tower into `<path>/vision_tower/` so the checkpoint runs offline (§10.4).

    The tower is a few hundred megabytes beside a checkpoint that is already hundreds; that is the
    price of "local" meaning local. `save_pretrained` writes the config and weights the loader
    reads back, so the bundle is a normal Hugging Face directory and can be inspected with the
    usual tools.
    """
    tower = encoder.tower
    save = getattr(tower, "save_pretrained", None)
    if save is None:
        raise TypeError(
            "cannot bundle a %s tower: it has no save_pretrained, so the checkpoint would ship "
            "without the weights it needs to run offline" % type(tower).__name__)
    target = os.path.join(path, TOWER_DIR)
    os.makedirs(target, exist_ok=True)
    save(target)
    return target
