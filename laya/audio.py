"""Sound support for the decision engine: checkpoint config, audio-token layout, the encoder.

This is the audio half of what a checkpoint needs to answer about a *sound*, and it is deliberately
the same shape as :mod:`laya.vision`, because the two problems are the same problem: a frozen tower
turns the input into rows, a projector compresses and widens them into the text encoder's space, and
:mod:`laya.block` splices them into the sequence after the option head so every option marker can
attend to them.

A checkpoint whose config carries no ``"audio"`` block parses to ``None`` and every function below
refuses to invent a layout, so a text or vision checkpoint is untouched by this module existing.

Three pieces:

* :class:`AudioConfig` -- the ``"audio"`` block, validated once at load instead of per request.
* :func:`audio_plan` -- where the audio block goes, how many tokens it costs, and the position ids
  it takes. Pure Python; no tensors, so it is testable without a model or a WAV file.
* :class:`AudioEncoder` -- a frozen audio tower plus the projector, which is what turns a waveform
  into the block. The tower is **injected**, so this class is testable with a stub and the caller
  decides what to download.

What a sound is *not*, here: a transcript. Nothing in this path runs speech recognition. The tower
sees the waveform -- prosody, tone, music, a bark, a clap, the words too if the tower learned them --
and the decision head learns which of those the options are asking about. That keeps the path
non-autoregressive, which is the whole reason the engine exists.

The one fact about audio towers that is easy to get wrong: **a 16 kHz model fed 44.1 kHz samples
hears the sound stretched**, not degraded. Wav2Vec2's convolutional feature extractor has fixed
kernel strides, so the sample rate is part of the model's contract. :func:`audio_plan` cannot check
it (it never sees samples); the decoder does, and the config states the rate as `sample_rate`.
"""
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .block import BlockPlan, describe_plan as _describe_block_plan, plan_block, reserved_positions
from .block import POSITION_POLICIES, SLOTS
from .projector import PooledProjector, tower_width, truncate_tower, _torch

# How the projected frame tokens are compressed before they enter the text sequence.
POOLINGS = ("adaptive_avg", "none")
# What the tower eats. `waveform` is raw samples at `sample_rate`; `mel` is a log-mel spectrogram,
# which is what Whisper-style and AST-style towers expect.
INPUTS = ("waveform",)

AUDIO_DEFAULTS: Dict[str, Any] = {
    "tower": "facebook/wav2vec2-base",
    "tower_revision": None,
    "tower_sha256": None,
    "sample_rate": 16000,        # the rate the tower was trained on, not the caller's
    "n_audio_tokens": 32,
    "pooling": "adaptive_avg",
    "slot": "after_options",
    "position_policy": "contiguous",   # "contiguous" | "reserved"
    "position_offset": None,     # only for "reserved": where the block starts (None -> the top)
    "keep_layers": None,         # None -> the tower's last block; N -> read N blocks early
    "summary_for_options_over": 96,   # rendered option tokens that trigger a summary token
    "input": "waveform",
    "feature_mean": 0.0,         # the tower's own input normalisation, if it has one
    "feature_std": 1.0,
    "cache_size": 64,            # distinct sounds kept in the per-agent audio-block cache
}


class AudioConfig:
    """The validated ``"audio"`` block of a checkpoint config.

    Every field is copied out of the mapping and normalised here, so a malformed config fails at
    load with the offending key named rather than mid-request.
    """

    __slots__ = tuple(AUDIO_DEFAULTS) + ("raw",)

    def __init__(self, raw: Dict[str, Any], max_position_embeddings: Optional[int] = None):
        unknown = sorted(set(raw) - set(AUDIO_DEFAULTS))
        if unknown:
            raise ValueError("audio: unknown key(s) %s; known keys are %s"
                             % (", ".join(unknown), ", ".join(sorted(AUDIO_DEFAULTS))))
        self.raw = dict(raw)
        for key, default in AUDIO_DEFAULTS.items():
            setattr(self, key, raw.get(key, default))

        if self.slot not in SLOTS:
            raise ValueError("audio.slot must be one of %s, got %r" % (", ".join(SLOTS), self.slot))
        if self.pooling not in POOLINGS:
            raise ValueError("audio.pooling must be one of %s, got %r"
                             % (", ".join(POOLINGS), self.pooling))
        if self.input not in INPUTS:
            raise ValueError("audio.input must be one of %s, got %r"
                             % (", ".join(INPUTS), self.input))
        if not isinstance(self.n_audio_tokens, int) or isinstance(self.n_audio_tokens, bool):
            raise TypeError("audio.n_audio_tokens must be an int, got %s"
                            % type(self.n_audio_tokens).__name__)
        if self.n_audio_tokens < 1:
            raise ValueError("audio.n_audio_tokens must be >= 1, got %d" % self.n_audio_tokens)
        if not isinstance(self.sample_rate, int) or isinstance(self.sample_rate, bool) \
                or self.sample_rate < 1:
            raise ValueError("audio.sample_rate must be a positive int, got %r"
                             % (self.sample_rate,))
        for key in ("feature_mean", "feature_std"):
            val = getattr(self, key)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValueError("audio.%s must be a number, got %r" % (key, val))
        if not self.feature_std > 0:
            raise ValueError("audio.feature_std must be positive, got %r" % (self.feature_std,))
        if not isinstance(self.cache_size, int) or isinstance(self.cache_size, bool) \
                or self.cache_size < 1:
            raise ValueError("audio.cache_size must be a positive int, got %r"
                             % (self.cache_size,))
        if self.keep_layers is not None and (not isinstance(self.keep_layers, int)
                                             or isinstance(self.keep_layers, bool)
                                             or self.keep_layers < 1):
            raise ValueError("audio.keep_layers must be None or a positive int, got %r"
                             % (self.keep_layers,))
        if self.position_policy not in POSITION_POLICIES:
            raise ValueError("audio.position_policy must be one of %s, got %r"
                             % (", ".join(POSITION_POLICIES), self.position_policy))
        if self.position_offset is not None:
            if not isinstance(self.position_offset, int) or isinstance(self.position_offset, bool) \
                    or self.position_offset < 0:
                raise ValueError("audio.position_offset must be None or a non-negative int, got %r"
                                 % (self.position_offset,))
            if max_position_embeddings is not None and \
                    self.position_offset + self.n_audio_tokens > max_position_embeddings:
                raise ValueError(
                    "audio: the audio block needs %d positions from offset %d, past this "
                    "encoder's %d"
                    % (self.n_audio_tokens, self.position_offset, max_position_embeddings))

    # -- helpers ---------------------------------------------------------------------------
    def position_ids_for_audio(self, max_position_embeddings: int,
                               start: Optional[int] = None) -> List[int]:
        """The positions the audio block occupies, for the `reserved` policy.

        The same arithmetic as the visual block's (`laya.block.reserved_positions`), so "reserved"
        means one thing across the two modalities. `contiguous` is the default for the measured
        reason it is the default for vision: a reserved block sits thousands of positions away from
        the option markers in RoPE terms, outside the local windows of most of the encoder's layers.
        """
        return reserved_positions(self.n_audio_tokens, max_position_embeddings,
                                  self.position_offset)

    def summary_wanted(self, option_tokens: int) -> bool:
        """Whether a high-cardinality head needs the summary token beside its markers."""
        limit = self.summary_for_options_over
        return bool(limit) and option_tokens > limit

    def __repr__(self) -> str:
        return ("AudioConfig(tower=%r, n_audio_tokens=%d, slot=%r, pooling=%r)"
                % (self.tower, self.n_audio_tokens, self.slot, self.pooling))


def parse_audio(cfg: Optional[Dict[str, Any]],
                max_position_embeddings: Optional[int] = None) -> Optional[AudioConfig]:
    """Read the ``"audio"`` block out of a checkpoint config.

    Returns ``None`` for a config with no block -- which is every checkpoint that exists today --
    and ``None`` for a block that is explicitly disabled (``{"enabled": false}``), so a config can
    keep the block while shipping a text-only artifact.
    """
    if not cfg:
        return None
    block = cfg.get("audio")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise TypeError("config['audio'] must be a mapping, got %s" % type(block).__name__)
    if block.get("enabled") is False:
        return None
    return AudioConfig({k: v for k, v in block.items() if k != "enabled"},
                       max_position_embeddings=max_position_embeddings)


class AudioPlan(BlockPlan):
    """Where the audio block sits in one built sequence and what it costs.

    A :class:`laya.block.BlockPlan` with this path's own name for the block's width, so the vision
    and audio plans cannot disagree about the layout while each reads naturally in its own code.
    """

    __slots__ = ()

    def __init__(self, splice: int, n_audio: int, summary: bool, marker_positions: Sequence[int],
                 position_ids: Sequence[int], length: int):
        super().__init__(splice=splice, n_block=n_audio, summary=summary,
                         marker_positions=marker_positions, position_ids=position_ids,
                         length=length)

    @property
    def n_audio(self) -> int:
        """The block's width: `n_audio_tokens`, plus the summary row when one was asked for."""
        return self.n_block

    def __repr__(self) -> str:
        return ("AudioPlan(splice=%d, n_audio=%d, summary=%s, length=%d)"
                % (self.splice, self.n_audio, self.summary, self.length))


def audio_plan(audio: AudioConfig, text_len: int, markers: Sequence[int], head_len: int,
               max_len: int, max_position_embeddings: int = 8192,
               option_tokens: Optional[int] = None) -> AudioPlan:
    """Lay the audio block out inside a sequence of `text_len` tokens.

    `head_len` is ``len(build_head(...))`` -- the splice point, and `markers` are ``build_head``'s
    marker positions, which sit before it and therefore never move. The layout itself is
    :func:`laya.block.plan_block`, shared with the vision path.
    """
    summary = audio.summary_wanted(
        option_tokens if option_tokens is not None else text_len - head_len)
    n_audio = audio.n_audio_tokens + (1 if summary else 0)
    plan = plan_block("audio", "audio", audio.slot, n_audio, text_len, markers, head_len,
                      max_len, summary=summary,
                      max_position_embeddings=max_position_embeddings,
                      position_policy=audio.position_policy,
                      position_offset=audio.position_offset)
    return AudioPlan(splice=plan.splice, n_audio=plan.n_block, summary=plan.summary,
                     marker_positions=plan.marker_positions, position_ids=plan.position_ids,
                     length=plan.length)


def describe_plan(plan: AudioPlan) -> Tuple[str, ...]:
    """Human-readable facts about a plan, for error messages and `--dry-run` style output."""
    return _describe_block_plan(plan, noun="audio")


# ------------------------------------------------------------------------------------------------
# The tower and the projector
# ------------------------------------------------------------------------------------------------

class AudioEncoder:
    """A frozen audio tower plus the projector, producing the audio block.

    The tower is **injected**, not loaded here, so this class is testable with a stub and so the
    caller decides what to download. :func:`load_tower` is the convenience that builds the real one,
    and it is the only place in the package that names an audio architecture.
    """

    #: Where the tower and projector live, set by :meth:`to`. ``None`` (the default) leaves input
    #: tensors untouched, which is the CPU path every caller had before devices entered the story.
    device = None

    def __init__(self, tower, projector: PooledProjector, config: AudioConfig):
        self.tower = tower
        self.projector = projector
        self.config = config

    def train(self, mode: bool = True):
        if mode:
            self.tower.train(mode)
        else:
            self.tower.eval()
        return self

    def eval(self):
        return self.train(False)

    def to(self, device):
        """Move the tower *and* the projector to `device`.

        `AudioEncoder` is deliberately not an `nn.Module` (the tower is injected and the projector
        is trained on its own), so `DecisionModel.to(device)` reaches neither half. A caller that
        only moves the model leaves the frozen tower on the CPU, which does not fail -- it just
        encodes every sound ~100x slower while the trainer reports a GPU run. Recording the device
        here also lets :meth:`encode_frames` place the samples, which is what makes a CPU-decoded
        recording safe to hand to a GPU tower.
        """
        torch = _torch()
        self.tower.to(device)
        self.projector.to(device)
        self.device = torch.device(device) if device is not None else None
        return self

    def encode_frames(self, waveform):
        """``[B, T]`` samples in [-1, 1] -> the tower's frame tokens ``[B, N, C]``, unprojected.

        The split that makes training possible: the tower is frozen, so its output can be computed
        once per sound and cached; the projector is *trained*, so it has to run inside the step
        where autograd can see it. :meth:`encode` is the two steps together, which inference wants.

        `feature_mean` / `feature_std` are applied here rather than in the decoder, because they
        belong to the tower: a wav2vec-style model expects roughly zero-mean samples, and a caller
        handing over raw [-1, 1] audio would otherwise feed it values off its own training
        distribution. Both constants are config fields so a different tower can set them.
        """
        torch = _torch()
        if self.device is not None:
            waveform = waveform.to(self.device)
        normalised = (waveform - self.config.feature_mean) / self.config.feature_std
        with torch.no_grad() if not self._tower_trains() else _nullcontext():
            out = self.tower(input_values=normalised)
        frames = getattr(out, "last_hidden_state", None)
        if frames is None:
            raise TypeError("the audio tower must return `last_hidden_state` frame tokens; "
                            "got %s" % type(out).__name__)
        return frames

    def encode(self, waveform, summary: bool = False):
        """``[B, T]`` samples in [-1, 1] -> ``[B, n_audio_tokens (+1), hidden]`` embeddings."""
        return self.projector(self.encode_frames(waveform), summary=summary)

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
    """Build the frozen audio tower. The only import of an audio architecture in the package.

    `Wav2Vec2Model` because it takes the waveform itself: no feature extractor, no download of a
    second artifact, and a variable-length input rather than a fixed spectrogram window. A tower
    that wants log-mel features instead is injected through :func:`build_audio_encoder` with
    ``input: "mel"``, which this module will grow in the same place it grew for images.

    `keep_layers` reads the frame tokens `keep_layers` blocks early. The last blocks of a deep
    transformer tend to be *worse* frozen features than intermediate ones (Meta PE measures -14.0
    DAVIS J&F from its best layer to its last), so this is a free latency saving that the prior says
    is also a free accuracy win. It is a config key, not a code change, so it can be swept.
    """
    from transformers import Wav2Vec2Model

    kw = {}
    if revision:
        kw["revision"] = revision
    if dtype is not None:
        kw["torch_dtype"] = dtype
    if trust_remote_code:
        kw["trust_remote_code"] = True
    tower = Wav2Vec2Model.from_pretrained(model_id, **kw)
    for p in tower.parameters():
        p.requires_grad_(False)
    tower.eval()
    if keep_layers:
        truncate_tower(tower, keep_layers)
    return tower


TOWER_DIR = "audio_tower"


def local_tower_dir(model_dir: Optional[str]) -> Optional[str]:
    """The bundled tower inside a checkpoint directory, when one is there.

    An audio checkpoint that names a Hub repository for its tower cannot serve a request offline,
    which contradicts the reason Laya exists. `laya-train` bundles the tower beside the weights
    (`bundle_tower`), and this is the lookup that prefers it. A checkpoint with no bundle falls back
    to the Hub id in its config, so existing checkpoints are unaffected.
    """
    if not model_dir:
        return None
    candidate = os.path.join(model_dir, TOWER_DIR)
    return candidate if os.path.isdir(candidate) else None


def build_audio_encoder(config: AudioConfig, hidden_size: int, tower=None, dtype=None,
                        tower_dir: Optional[str] = None):
    """Assemble an :class:`AudioEncoder` for `config`.

    `tower` is injected when the caller already has one (tests, or a shared tower across
    checkpoints); otherwise it is loaded from `tower_dir` if given, and from `config.tower` last.
    A bundled directory is preferred over the Hub id because the bundled one is the checkpoint's
    own artifact: it cannot be re-tagged, re-uploaded, or disappear behind a network.
    """
    if tower is None:
        if tower_dir:
            tower = load_tower(tower_dir, keep_layers=config.keep_layers, dtype=dtype)
        else:
            tower = load_tower(config.tower, revision=config.tower_revision,
                               keep_layers=config.keep_layers, dtype=dtype)
    projector = PooledProjector(tower_width(tower, name="audio tower"), hidden_size,
                                config.n_audio_tokens, pooling=config.pooling, dtype=dtype)
    return AudioEncoder(tower, projector, config)


def tower_parameters(encoder: AudioEncoder):
    """The trainable parameters of an audio encoder (the tower stays frozen by default)."""
    return encoder.projector.parameters()


# ------------------------------------------------------------------------------------------------
# Persisting the trained projector
# ------------------------------------------------------------------------------------------------
#
# `DecisionModel` deliberately has no audio parameters -- a new one would break
# `load_state_dict(..., strict=True)` for every existing checkpoint -- so the projector lives beside
# the model and is *not* in `model.state_dict()`. That makes it the one part of an audio checkpoint
# that a naive save would drop, and the failure is silent in the worst way: every load would build a
# fresh random projector and answer confidently from it. So it gets its own file, and a checkpoint
# that declares audio but carries none is refused rather than guessed at.

AUDIO_WEIGHTS_FILE = "audio.safetensors"


def save_audio_weights(encoder: AudioEncoder, path: str) -> str:
    """Write the projector's trained tensors into `path` (a directory)."""
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    target = os.path.join(path, AUDIO_WEIGHTS_FILE)
    payload = {
        "projector.weight": encoder.projector.proj.weight.detach().cpu().contiguous(),
        "projector.bias": encoder.projector.proj.bias.detach().cpu().contiguous(),
        "type_vector": encoder.projector.type_vector.detach().cpu().contiguous(),
    }
    save_file({k: v.half() if v.is_floating_point() else v for k, v in payload.items()}, target)
    return target


def load_audio_weights(encoder: AudioEncoder, path: str, dtype=None) -> AudioEncoder:
    """Load the projector's trained tensors from `path` (a directory) into `encoder`."""
    from safetensors.torch import load_file

    target = os.path.join(path, AUDIO_WEIGHTS_FILE) if os.path.isdir(path) else path
    if not os.path.exists(target):
        raise FileNotFoundError(
            "%r is missing: this checkpoint declares audio support, so its projector has to come "
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
    return encoder


def bundle_tower(encoder: AudioEncoder, path: str) -> str:
    """Copy the tower into `<path>/audio_tower/` so the checkpoint runs offline.

    The tower is a few hundred megabytes beside a checkpoint that is already hundreds; that is the
    price of "local" meaning local. `save_pretrained` writes the config and weights the loader reads
    back, so the bundle is a normal Hugging Face directory and can be inspected with the usual
    tools.
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
