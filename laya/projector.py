"""The projector both non-text modalities share: pool a tower's rows, then project them.

A frozen tower emits one row per patch (vision) or per frame (audio). The decision encoder needs a
short, fixed-width run of rows in *its* embedding space, so both paths do the same two operations in
the same order:

* **pool** -- ``adaptive_avg`` folds the tower's rows down to ``n_tokens`` positions (the documented
  baseline; a learned coarse-to-fine connector replaces this later behind the same interface).
  ``none`` passes the tower's rows through untouched.
* **project** -- a linear map into the text encoder's hidden size, plus a learned *type* vector.

The type vector lives here rather than on ``DecisionModel`` on purpose. A new parameter on the
decision model would break ``load_state_dict(..., strict=True)`` for every existing checkpoint;
putting it in the projector keeps text checkpoints byte-identical and puts the "this is a patch, not
a word" signal exactly where the rows are produced. It is added *after* the linear so it cannot be
confused with a bias on a constant row.

The one row that carries a *summary* comes from the same map and the same type vector: it is a row
of this modality, and reusing them keeps the projector's parameter set -- and therefore every saved
``vision.safetensors`` / ``audio.safetensors`` -- unchanged.

torch is imported inside the functions, not at module import, so ``import laya.projector`` stays
cheap and a text-only process never pays for it.
"""
from typing import Optional


def _torch():
    """Import torch here, not at module import (see ``tests/test_lazy_import.py``)."""
    import torch
    return torch


class PooledProjector:
    """Compress a tower's rows and project them into the encoder's width.

    The same class serves vision and audio: a patch grid and a frame sequence are both ``[B, N, C]``
    and both are pooled along their single variable axis. `n_tokens` is how many rows the block
    occupies, and `summary` adds one more.
    """

    def __init__(self, tower_dim: int, hidden_size: int, n_tokens: int,
                 pooling: str = "adaptive_avg", dtype=None):
        torch = _torch()
        self.n_tokens = n_tokens
        self.pooling = pooling
        self.proj = torch.nn.Linear(tower_dim, hidden_size)
        self.type_vector = torch.nn.Parameter(torch.zeros(hidden_size))
        if dtype is not None:
            self.proj = self.proj.to(dtype)
            self.type_vector.data = self.type_vector.data.to(dtype)

    @property
    def n_visual_tokens(self) -> int:
        """The vision path's name for :attr:`n_tokens`, kept because it was public first."""
        return self.n_tokens

    def parameters(self):
        return [self.proj.weight, self.proj.bias, self.type_vector]

    def to(self, device):
        self.proj = self.proj.to(device)
        self.type_vector.data = self.type_vector.data.to(device)
        return self

    def __call__(self, rows, summary: bool = False):
        return self.forward(rows, summary=summary)

    def forward(self, rows, summary: bool = False):
        """`rows` ``[B, N, C]`` -> ``[B, n_tokens (+1), hidden]``.

        `summary=True` appends the block's summary row: the *globally* pooled tower vector,
        projected through the same map and type vector as the rows above it. It is the one row in
        the block that sees every patch or frame, which is what a high-cardinality head needs --
        above ~30 options the last markers sit past the encoder's 128-token local window and can
        only reach the input through a row that already pooled it.
        """
        torch = _torch()
        x = rows
        if self.pooling == "adaptive_avg":
            x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.n_tokens)
            x = x.transpose(1, 2)
        elif self.pooling != "none":
            raise ValueError("unknown pooling %r" % (self.pooling,))
        out = self.proj(x) + self.type_vector
        if summary:
            pooled = self.proj(rows.mean(dim=1, keepdim=True)) + self.type_vector
            out = torch.cat([out, pooled], dim=1)
        return out


def tower_width(tower, name: str = "tower") -> int:
    """The row width a tower emits, read from its config without running it."""
    cfg = getattr(tower, "config", None)
    for attr in ("hidden_size", "projection_dim"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int):
            return val
    vision_cfg = getattr(cfg, "vision_config", None)
    val = getattr(vision_cfg, "hidden_size", None)
    if isinstance(val, int):
        return val
    raise TypeError("cannot determine the %s's row width from %s" % (name, type(cfg).__name__))


def truncate_tower(tower, keep_layers: Optional[int], name: str = "tower"):
    """Keep only the first `keep_layers` blocks of a tower's encoder.

    Two independent groups found the last blocks of a vision tower produce *worse* features than
    intermediate ones (Meta PE measures -14.0 DAVIS J&F from its best layer to its last; InternVL
    1.5 took the fourth-to-last layer and dropped three blocks). Audio towers are the same shape of
    stack, so the same knob exists for both. A vision tower wraps its blocks in `vision_model`; a
    wav2vec-style audio tower keeps them directly on the model.
    """
    if keep_layers is None:
        return tower
    root = getattr(tower, "vision_model", tower)
    encoder = getattr(root, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise TypeError("keep_layers=%d was requested but this %s has no `.encoder.layers` to "
                        "truncate" % (keep_layers, name))
    if not 1 <= keep_layers <= len(layers):
        raise ValueError("keep_layers=%d is out of range for a %d-block %s"
                         % (keep_layers, len(layers), name))
    encoder.layers = layers[:keep_layers]
    return tower
