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
* :func:`load_waveform` -- the decode side: a reference becomes a mono waveform at the tower's rate,
  after the byte and duration caps in :func:`check_audio_bytes` have had their say.

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

import numpy as np

from .block import BlockPlan, describe_plan as _describe_block_plan, plan_block, reserved_positions
from .block import POSITION_POLICIES, SLOTS
from .cache import ContentCache
from .projector import PooledProjector, tower_width, truncate_tower, _torch
from .transport import Refused, read_reference

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
# Sounds inside a state
# ------------------------------------------------------------------------------------------------

# Keys a state may use to carry a sound. `text` is the companion text, and it may be absent: a
# recording on its own is a legal state.
AUDIO_KEYS = ("audio", "audio_path", "audio_base64", "audio_waveform")
# Accepted by `has_audio` but not by `split_state`: a URL is something the *caller* resolves, the
# way the HTTP server resolves an image URL before the model ever sees the state (see
# `read_audio_bytes`).
AUDIO_ROUTING_KEYS = AUDIO_KEYS + ("audio_url",)


def has_audio(state) -> bool:
    """Whether a state carries a sound reference -- the routing question, answered cheaply.

    Deliberately narrower than :func:`split_state`: it does not build anything, so the router can
    ask it on every request without paying for a state it may not need to split.
    """
    return isinstance(state, dict) and any(key in state for key in AUDIO_ROUTING_KEYS)


def split_state(state):
    """Separate a state into its text half and its sound spec.

    Returns ``(text_state, audio_spec)``. For anything that carries no sound this is
    ``(state, None)`` -- the caller only invokes it when the checkpoint has sound support, so a
    text checkpoint never has a state reshaped underneath it.

    Recognised shapes::

        {"audio": <spec>, "text": "..."}      # explicit
        {"audio": <spec>}                     # the sound is the whole state
        {"audio_path": "a.wav"}               # shorthand

    `<spec>` is a path string, a base64 string, raw bytes, or a mapping with a ``waveform`` entry
    holding an already-decoded tensor/array (which is how a caller with its own decoder, and the
    tests, bypass the codec and the container).
    """
    if not isinstance(state, dict):
        return state, None
    spec = None
    for key in AUDIO_KEYS:
        if key in state:
            spec = state[key]
            break
    if spec is None:
        return state, None
    text = state.get("text")
    # Everything except the sound key and the text companion is still state the model should see,
    # so it is not dropped: {"audio": ..., "vendor": "acme"} keeps "vendor" as data.
    rest = {k: v for k, v in state.items() if k not in AUDIO_KEYS and k != "text"}
    if text is None:
        return (rest if rest else ""), spec
    if rest:
        merged = dict(rest)
        merged["text"] = text
        return merged, spec
    return text, spec


def collate_audio(items, embed_fn, pad_id: int = 0):
    """Collate already-encoded items that carry an audio block into one batch.

    The same operation as `laya.vision.collate_visual` on a different key, through the shared
    `laya.block.collate_block`, so a mixed batch -- ``[{"audio_path": p}, {"text": "hello"}]`` --
    pads, splices and masks the same way a vision batch does.
    """
    from .block import collate_block
    return collate_block(items, embed_fn, block_key="audio", label="audio", pad_id=pad_id)


# ------------------------------------------------------------------------------------------------
# A sound inside a state: what is decoded, and what is refused before a decoder runs
# ------------------------------------------------------------------------------------------------
#
# The image path checks a header before it decodes, because a 40-byte PNG can declare 30000x30000
# pixels and decoding it first is a decompression bomb. A recording has the same shape of problem in
# two dimensions: bytes that were never going to fit in memory, and *duration*. An hour of 48 kHz
# stereo is 690 MB decoded, and it fits inside a 12 MB compressed file, so "the upload is small" is
# not the question -- the sample count is. Both are answered from the container header alone, before
# any decoder runs, which is also why they work on an install without the audio extra.

MAX_AUDIO_BYTES = 12 * 1024 * 1024       # 12 MB, the same request-body budget images get
MAX_AUDIO_SECONDS = 120.0                # two minutes, the default `max_audio_seconds`

# A deployment opts into fetching a URL for the same reason it opts into anything else that reaches
# its network: the engine runs on the operator's hardware and must not become an egress proxy.
ALLOW_AUDIO_URLS_ENV = "LAYA_ALLOW_AUDIO_URLS"


class AudioRefused(Refused):
    """A refused sound, carrying the reason a caller should be told.

    The reading policy itself is shared with the image path (`laya.transport`); this class only
    keeps the audio path's name and reason codes for it.
    """

    def __init__(self, reason: str, detail: str):
        super().__init__("audio", reason, detail)


def allow_audio_urls() -> bool:
    """Whether this deployment fetches an ``audio_url`` reference. Off unless explicitly asked.

    The name is spelled out in the lookup rather than read through `ALLOW_AUDIO_URLS_ENV`, because
    the repository's own checks derive the set of variables the package reads by scanning for
    ``environ.get("LAYA_...")``: a variable reached through a constant is read and invisible.
    """
    return os.environ.get("LAYA_ALLOW_AUDIO_URLS", "").strip().lower() in (
        "1", "true", "yes", "on")


def audio_info(data) -> Optional[Dict[str, Any]]:
    """Format, sample rate, channels and duration read from the container header, without decoding.

    Covers WAV, FLAC, Ogg (Vorbis and Opus) and MP3 -- the formats a caller actually sends. Returns
    ``None`` for anything else rather than guessing; the caller decides whether an unidentified
    header is a refusal or merely unmeasured.

    Durations are exact where the container states them (WAV's data chunk over its byte rate,
    FLAC's `total_samples`, Ogg's last granule position) and an *estimate* for MP3, whose header
    states a bit rate rather than a length: a variable-bit-rate file can be tens of percent off,
    which is why the decoded sample count is capped as well (`load_waveform`).
    """
    if not isinstance(data, (bytes, bytearray)) or len(data) < 12:
        return None
    data = bytes(data)
    for reader in (_wav_info, _flac_info, _ogg_info, _mp3_info):
        info = reader(data)
        if info is not None:
            return info
    return None


def _wav_info(data):
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    fmt: Optional[Dict[str, Any]] = None
    data_size: Optional[int] = None
    pos = 12
    while pos + 8 <= len(data):
        chunk = data[pos:pos + 4]
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        body = pos + 8
        if chunk == b"fmt " and size >= 16 and body + 16 <= len(data):
            fmt = {
                "audio_format": int.from_bytes(data[body:body + 2], "little"),
                "channels": int.from_bytes(data[body + 2:body + 4], "little"),
                "sample_rate": int.from_bytes(data[body + 4:body + 8], "little"),
                "byte_rate": int.from_bytes(data[body + 8:body + 12], "little"),
                "bits": int.from_bytes(data[body + 14:body + 16], "little"),
            }
        elif chunk == b"data":
            # A streamed WAV may write 0 or 0xFFFFFFFF as "unknown"; the bytes actually present are
            # then the honest answer.
            data_size = size if 0 < size <= len(data) - body else len(data) - body
            break
        pos = body + size + (size & 1)
    if fmt is None:
        return None
    duration = None
    if fmt["byte_rate"] > 0 and data_size:
        duration = data_size / fmt["byte_rate"]
    elif fmt["sample_rate"] > 0 and fmt["channels"] > 0 and fmt["bits"] > 0 and data_size:
        frames = data_size // max(1, fmt["channels"] * fmt["bits"] // 8)
        duration = frames / fmt["sample_rate"]
    return {"format": "wav", "sample_rate": fmt["sample_rate"], "channels": fmt["channels"],
            "bits": fmt["bits"], "duration_seconds": duration,
            # Which codec the container declares. `decode_waveform` uses it to decide whether the
            # standard library can read the file (PCM) or a real decoder is needed (IEEE float,
            # WAVE_FORMAT_EXTENSIBLE, ADPCM). Dropping it here made every WAV take the `soundfile`
            # path -- the check `info.get("audio_format") == 1` could never be true, so a CPU
            # install without the extra could not decode an ordinary recording.
            "audio_format": fmt["audio_format"]}


def _flac_info(data):
    if data[:4] != b"fLaC":
        return None
    pos = 4
    while pos + 4 <= len(data):
        header = data[pos]
        block_type = header & 0x7F
        length = int.from_bytes(data[pos + 1:pos + 4], "big")
        if block_type == 0 and length >= 34 and pos + 4 + 18 <= len(data):
            body = data[pos + 4:pos + 4 + length]
            # STREAMINFO: 16+16 bits of block size, 24+24 of frame size, then one 64-bit field
            # holding the sample rate (20), channels-1 (3), bits-1 (5) and total samples (36).
            packed = int.from_bytes(body[10:18], "big")
            rate = (packed >> 44) & 0xFFFFF
            channels = ((packed >> 41) & 0x7) + 1
            bits = ((packed >> 36) & 0x1F) + 1
            total = packed & ((1 << 36) - 1)
            return {"format": "flac", "sample_rate": rate, "channels": channels, "bits": bits,
                    "duration_seconds": (total / rate) if rate and total else None}
        pos += 4 + length
        if header & 0x80:
            break
    return None


def _ogg_info(data):
    if data[:4] != b"OggS":
        return None
    pos = 0
    rate: Optional[int] = None
    channels: Optional[int] = None
    pre_skip = 0
    while pos + 27 <= len(data) and data[pos:pos + 4] == b"OggS":
        segments = data[pos + 26]
        body = pos + 27 + segments
        size = sum(data[pos + 27:body])
        packet = data[body:body + size]
        if packet[1:7] == b"vorbis" and len(packet) >= 16:
            channels = packet[11]
            rate = int.from_bytes(packet[12:16], "little")
            break
        if packet[:8] == b"OpusHead" and len(packet) >= 16:
            channels = packet[9]
            pre_skip = int.from_bytes(packet[10:12], "little")
            rate = int.from_bytes(packet[12:16], "little") or 48000
            break
        if not segments:
            break
        pos = body + size
    if rate is None:
        return None
    # The last page's granule position is the stream's length in samples at `rate`.
    tail_start = max(0, len(data) - 65536)
    last = data.rfind(b"OggS", tail_start)
    duration = None
    if last >= 0 and last + 14 <= len(data):
        granule = int.from_bytes(data[last + 6:last + 14], "little")
        if granule and granule != 0xFFFFFFFFFFFFFFFF:
            duration = max(0.0, (granule - pre_skip) / rate)
    return {"format": "ogg", "sample_rate": rate, "channels": channels, "bits": None,
            "duration_seconds": duration}


_MP3_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}
_MP3_BITRATES = {
    3: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    0: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}


def _mp3_info(data):
    start = 0
    if data[:3] == b"ID3" and len(data) >= 10:
        size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) \
            | ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
        start = 10 + size
    # Find a frame sync: 11 set bits, and a version/layer pair that is not reserved.
    pos = start
    while pos + 4 <= len(data):
        if data[pos] != 0xFF or (data[pos + 1] & 0xE0) != 0xE0:
            pos += 1
            continue
        header = int.from_bytes(data[pos:pos + 4], "big")
        version = (header >> 19) & 0x3
        layer = (header >> 17) & 0x3
        rate_index = (header >> 10) & 0x3
        if version == 1 or layer == 0 or rate_index == 3:
            pos += 1
            continue
        rate = _MP3_RATES[version][rate_index]
        bitrate_index = (header >> 12) & 0xF
        bitrate = _MP3_BITRATES[version][bitrate_index] * 1000 if bitrate_index < 15 else 0
        channels = 1 if ((header >> 6) & 0x3) == 3 else 2
        payload = len(data) - pos
        # A Xing/Info header carries the frame count, which is the only exact duration an MP3 has.
        side = 32 if (version == 3 and channels == 2) else 17
        tag = data[pos + 4 + side:pos + 4 + side + 4]
        duration = None
        if tag[:4] in (b"Xing", b"Info") and pos + 4 + side + 12 <= len(data):
            flags = int.from_bytes(data[pos + 4 + side + 4:pos + 4 + side + 8], "big")
            if flags & 0x1:
                frames = int.from_bytes(data[pos + 4 + side + 8:pos + 4 + side + 12], "big")
                duration = frames * 1152 / rate
        if duration is None and bitrate:
            duration = payload * 8 / bitrate
        return {"format": "mp3", "sample_rate": rate, "channels": channels, "bits": None,
                "duration_seconds": duration}
    return None


def check_audio_bytes(data, max_bytes: Optional[int] = None,
                      max_seconds: Optional[float] = None) -> Dict[str, Any]:
    """Refuse a recording that is too large to decode, before any decoder sees it.

    Returns what :func:`audio_info` found, so a caller that wants the duration does not parse the
    header twice. An unidentified format is refused: unlike an image, there is no "decode it anyway
    and see" for a waveform, because the decoder is the thing being protected. A container that
    states no duration is refused while a duration cap is in force -- "I could not measure it" is
    not the same as "it is short". `max_seconds=0` disables the duration check for a deployment that
    trusts its own inputs; the byte cap still applies.
    """
    max_bytes = MAX_AUDIO_BYTES if max_bytes is None else int(max_bytes)
    max_seconds = MAX_AUDIO_SECONDS if max_seconds is None else float(max_seconds)
    if max_seconds <= 0:
        # Zero (or negative) is how a caller says "no duration limit": a deployment that serves
        # short clips it already trusts should not be forced to state a number it does not mean.
        max_seconds = None
    if not isinstance(data, (bytes, bytearray)):
        raise AudioRefused("audio", "expected bytes, got %s" % type(data).__name__)
    if not data:
        raise AudioRefused("audio", "empty body")
    if len(data) > max_bytes:
        raise AudioRefused("audio_too_large",
                           "%d bytes exceeds the %d-byte limit" % (len(data), max_bytes))
    info = audio_info(data)
    if info is None:
        raise AudioRefused("audio_unsupported", "not a WAV, FLAC, Ogg or MP3 file")
    rate, channels = info.get("sample_rate"), info.get("channels")
    if not rate or not channels:
        raise AudioRefused("audio_invalid",
                           "the %s header states %r Hz on %r channel(s)"
                           % (info.get("format"), rate, channels))
    duration = info.get("duration_seconds")
    if duration is None:
        if max_seconds:
            raise AudioRefused(
                "audio_unmeasurable",
                "the %s header does not state its duration, so the %g-second limit cannot be "
                "checked; re-encode to WAV or FLAC, or raise the limit for this deployment"
                % (info.get("format"), max_seconds))
    elif duration > max_seconds:
        raise AudioRefused("audio_too_long",
                           "%.1f seconds exceeds the %g-second limit"
                           % (duration, max_seconds))
    return info


def read_audio_bytes(spec, max_bytes: Optional[int] = None, max_seconds: Optional[float] = None,
                     allow_url: Optional[bool] = None, timeout: float = 5.0) -> bytes:
    """Resolve a sound reference into raw bytes, applying the transport limits.

    Accepted forms: raw ``bytes`` (already read), a filesystem ``path``, a base64 string, a mapping
    with one of ``bytes``/``path``/``url``/``base64``, or -- only when URL fetching is on -- an
    ``http(s)`` URL. URL fetching is off unless ``LAYA_ALLOW_AUDIO_URLS`` says otherwise, for the
    reason the image path gives: a self-hosted engine that fetches what a request names is an
    egress proxy pointed at the operator's own network.
    """
    def refuse(reason: str, detail: str):
        raise AudioRefused(reason, detail)

    if allow_url is None:
        allow_url = allow_audio_urls()
    data = read_reference(spec, kind="audio",
                          max_bytes=MAX_AUDIO_BYTES if max_bytes is None else int(max_bytes),
                          allow_url=allow_url, url_env=ALLOW_AUDIO_URLS_ENV, timeout=timeout,
                          refuse=refuse)
    check_audio_bytes(data, max_bytes=max_bytes, max_seconds=max_seconds)
    return data


def _soundfile(hint: str = "compressed audio"):
    """Import the optional decoder, naming the extra when it is missing.

    `soundfile` is `pip install laya[audio]`. It is optional on purpose: the package's declared
    dependencies stay torch/transformers/safetensors/huggingface_hub/numpy, so a text or vision
    install never grows a codec stack. A caller with its own decoder passes
    ``{"waveform": <tensor>}`` and skips this entirely, and PCM WAV decodes with the standard
    library either way.
    """
    try:
        import soundfile
    except ImportError as exc:                                  # pragma: no cover - optional dep
        raise ImportError(
            "decoding %s needs soundfile: `pip install laya[audio]`, or pass "
            "{'waveform': <tensor>} with your own decoder" % hint) from exc
    return soundfile


def decode_waveform(data, info: Optional[Dict[str, Any]] = None,
                    max_seconds: Optional[float] = None):
    """Decode container bytes into ``[channels, frames]`` float32 samples in [-1, 1].

    WAV goes through the standard library, so a PCM recording needs no extra installed at all; the
    compressed formats go through `soundfile` (`pip install laya[audio]`). `max_seconds` is enforced
    here as well as in :func:`check_audio_bytes`, because an MP3's duration is an estimate and the
    decoder is where a wrong estimate would become work.
    """
    torch = _torch()
    info = info if info is not None else audio_info(data)
    if info is None:
        raise AudioRefused("audio_unsupported", "not a WAV, FLAC, Ogg or MP3 file")
    fmt = info["format"]
    rate = info["sample_rate"]
    max_frames = int(max_seconds * rate) if max_seconds else None
    if fmt == "wav" and info.get("audio_format") == 1:
        return _decode_wav_pcm(data, info, max_frames, torch)
    if fmt == "wav":
        # A WAV the standard library cannot read: IEEE float (3), WAVE_FORMAT_EXTENSIBLE (0xFFFE),
        # ADPCM and friends. `soundfile` handles them; the message names the codec so a caller
        # knows why an ordinary-looking `.wav` needed the extra.
        soundfile = _soundfile(hint="a %s-format WAV" % {
            3: "IEEE float", 0xFFFE: "WAVE_FORMAT_EXTENSIBLE", 2: "ADPCM", 6: "A-law",
            7: "mu-law"}.get(info.get("audio_format"), "non-PCM"))
    else:
        soundfile = _soundfile()
    import io
    try:
        samples, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:                                     # noqa: BLE001 - codec errors
        raise AudioRefused("audio_invalid", "the %s decoder refused it: %s" % (fmt, exc)) from exc
    if samples.size == 0:
        raise AudioRefused("audio_invalid", "the %s decoder produced no samples" % fmt)
    if max_frames is not None:
        samples = samples[:max_frames]
    return torch.from_numpy(samples.T.copy()), rate


def _decode_wav_pcm(data, info, max_frames, torch):
    """PCM WAV through `wave`: 8/16/24/32-bit integers, no third-party decoder."""
    import io
    import wave

    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
            frames = handle.getnframes()
            if max_frames is not None:
                frames = min(frames, max_frames)
            raw = handle.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise AudioRefused("audio_invalid", "the WAV container refused it: %s" % exc) from exc
    if width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        raw24 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        packed = raw24[:, 0] | (raw24[:, 1] << 8) | (raw24[:, 2] << 16)
        samples = (packed.astype(np.float32) - (packed.astype(np.float32) >= 8388608) * 16777216) \
            / 8388608.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioRefused("audio_invalid", "unsupported PCM width %d bytes" % width)
    usable = (len(samples) // channels) * channels if channels else 0
    samples = samples[:usable].reshape(-1, channels).T
    if samples.size == 0:
        raise AudioRefused("audio_invalid", "the WAV file holds no frames")
    return torch.from_numpy(samples.copy()), rate


def resample(waveform, from_rate: int, to_rate: int):
    """Resample ``[channels, frames]`` between two rates, in plain torch.

    Linear interpolation, with a moving-average low-pass on the way *down*: decimating 48 kHz to
    16 kHz without one folds everything above 8 kHz back into the band the tower reads. It is not a
    polyphase resampler and the documentation says so -- the tower's own convolutional front end
    band-limits, and a decision engine cares about prosody and events, not about broadcasting.
    """
    torch = _torch()
    if from_rate == to_rate or waveform.size(-1) == 0:
        return waveform
    ratio = to_rate / from_rate
    if ratio < 1:
        window = max(1, int(round(1.0 / ratio)))
        if window > 1:
            kernel = torch.ones(1, 1, window, dtype=waveform.dtype) / window
            waveform = torch.nn.functional.conv1d(
                waveform.unsqueeze(0), kernel, padding=window // 2).squeeze(0)
    n_out = max(1, int(round(waveform.size(-1) * ratio)))
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0), size=n_out, mode="linear", align_corners=False).squeeze(0)


def load_waveform(spec, sample_rate: int = 16000, max_seconds: Optional[float] = None,
                  normalise: bool = True, allow_url: Optional[bool] = None):
    """Turn a sound reference into a mono ``[T]`` float32 tensor at `sample_rate`, in [-1, 1].

    A mapping carrying ``waveform`` bypasses decoding entirely -- a tensor ``[T]``/``[C, T]`` or
    anything array-like -- which is how a caller with its own decoder, and the tests, skip both the
    codec and the container. Everything else is resolved by :func:`read_audio_bytes` (so the byte
    and duration caps apply) and decoded by :func:`decode_waveform`.

    Mono is not a convenience. A decision about a sound has to be the same decision whatever channel
    layout the caller recorded in, and downmixing to mono is what makes the sample count -- and
    therefore the number of frames the tower sees -- independent of it.

    `normalise` scales the recording to its own peak. The point is gain invariance: the same bark
    recorded 20 dB quieter must not change the answer, and a tower that sees a normalized waveform
    does not have to learn loudness as a nuisance. Set it to ``False`` when level *is* the signal
    ("is this too loud?") and carry the level in the text instead.
    """
    torch = _torch()
    if isinstance(spec, dict) and "waveform" in spec:
        return _fit_waveform(spec["waveform"], sample_rate, max_seconds, normalise, torch)
    info = None
    data = read_audio_bytes(spec, max_seconds=max_seconds, allow_url=allow_url)
    info = audio_info(data)
    waveform, rate = decode_waveform(data, info, max_seconds=max_seconds)
    waveform = resample(waveform, rate, sample_rate)
    return _finish_waveform(waveform, max_seconds, normalise, torch, sample_rate=sample_rate)


def _fit_waveform(waveform, sample_rate: int, max_seconds: Optional[float], normalise: bool,
                  torch):
    """Place an already-decoded waveform: shape, dtype, device-free, no resampling guess.

    A caller handing over samples is making a claim about their rate, which is `sample_rate` -- the
    config's, not something inferred from the shape. Guessing here is how a 44.1 kHz recording ends
    up sounding a third slower than it is.
    """
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError("waveform must be [T] or [C, T], got %s" % (tuple(waveform.shape),))
    return _finish_waveform(waveform.float(), max_seconds, normalise, torch,
                            sample_rate=sample_rate)


def _finish_waveform(waveform, max_seconds: Optional[float], normalise: bool, torch,
                     sample_rate: Optional[int] = None):
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.size(0) == 0:
        raise AudioRefused("audio_invalid", "the recording has no channels")
    waveform = waveform.reshape(-1)
    if max_seconds and sample_rate:
        # A reference that is too long has already been refused by `check_audio_bytes`; this is the
        # cap for the `{"waveform": ...}` path, where there is no header to have refused it.
        waveform = waveform[:int(max_seconds * sample_rate)]
    if normalise and waveform.numel():
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
    return waveform.contiguous()


class AudioCache(ContentCache):
    """Content-addressed cache of the audio block, so one recording is encoded once per call.

    The tower pass is the expensive half of a decision about a sound, and one call that asks several
    questions about the same recording must not pay it several times. Keyed by a hash of the
    normalised samples -- two decodes of the same sound are one entry -- with `summary` in the key
    because it changes the block's width, and bounded so a long-lived server cannot grow without
    limit.
    """

    def get_or_encode(self, waveform, encode, summary: bool = False):
        return super().get_or_encode(waveform, encode, *(("summary",) if summary else ()))

    @staticmethod
    def key(waveform, summary: bool = False) -> str:
        """Content key for `waveform`; `summary` is part of it, for the vision path's reason."""
        return ContentCache.key(waveform, *(("summary",) if summary else ()))


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
