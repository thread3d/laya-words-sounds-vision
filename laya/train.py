"""Fine-tune a Laya checkpoint on typed decisions: one training loop the scripts can share.

The notebook, the Apple Silicon script and `research/scripts/finetune_single_device.py` each
carry their own copy of the same loop. This module is that loop once, with two additions the
copies do not have:

- **A choice of objective.** `loss="rlcd"` (the default) is the notebook's objective: a
  GRPO-style term over noisy logit samples rewarded by `proper_reward`, plus soft
  cross-entropy against the target distribution. `loss="soft-ce"` trains on the soft
  cross-entropy alone; #741 measured no gain from the extra term on the typed-decisions split.
- **Option-order augmentation (opt-in).** The scripts always present options in the order the
  question lists them, so the model can learn a position prior: the 20-option order-flip rate in
  `BENCHMARKS.md` and #131 are both that symptom. Each epoch, items of the types named in
  `shuffle_options` (for example `("choice",)`) are re-encoded with a random `option_order` -- the same path inference
  uses -- and the target is permuted with them, so slot `s` is trained on option
  `option_order[s]`.

Calibration goes through `laya.calibrate.fit_temperature_map`, so a fine-tuned checkpoint is
fitted with the same clamp and buckets the runtime applies instead of a local copy of the
fitter. `calibration_report` says per question type what that fit rests on, and `finetune` warns
when it rests on too little: a short run can otherwise save a checkpoint whose temperatures never
moved from 1.0, and nothing but the numbers would show it. It also warns when predictions have
collapsed to the class prior -- a near-constant logit (or calibration CE stuck at ln K) that
looks like a finished run on an unlearnable task, but is often just too few updates on a small
dataset (#963).

Items keep the question and the tokenized state rather than a finished sequence, because a
shuffled epoch needs to rebuild the head. States are tokenized once per row and shared by
every question on it.

No new dependency: torch, transformers and safetensors are already required by the package.
"""
import json
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .calibrate import MIN_TYPE_N, fit_abstention_thresholds, fit_temperature_map
from .vision import (VISION_WEIGHTS_FILE, VisualCache, build_vision_encoder, load_pixels,
                     load_vision_weights, local_tower_dir, parse_vision, split_state,
                     visual_plan)
from .audio import (AUDIO_WEIGHTS_FILE, AudioCache, audio_plan, build_audio_encoder,
                    load_audio_weights, load_waveform, parse_audio)
from .audio import local_tower_dir as local_audio_tower_dir
from .audio import split_state as split_audio_state
from .common import (
    uses_state_conditioned_layout,
    OPTION_LAYOUTS,
    QTYPE_NAMES,
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    build_head,
    build_model,
    build_sequence,
    collate_items,
    ece_score,
    encode_text,
    proper_reward,
    render_options,
    serialize_state,
    temp_bucket,
    uses_parallel_layout,
)

LOSSES = ("soft-ce", "rlcd")
# Below this many calibration items of a type, a fitted temperature is reported as resting on
# little evidence. MIN_TYPE_N (laya.calibrate) is the floor below which it is not fitted at all.
CALIB_WARN_N = 50
# Mean within-row logit range below this is the silent collapse to a constant (usually
# the class prior) measured in #963: 0.01 on a collapsed head vs 0.62 untuned.
COLLAPSE_LOGIT_RANGE = 0.05
# Calibration mean cross-entropy within this relative distance of ln K is the other
# collapse signal from #963 (training CE stuck at ln 4). The reported training loss is
# not used: the default `rlcd` objective is not cross-entropy.
COLLAPSE_CE_REL = 0.02
_MASKED = -1e4


@dataclass
class TrainConfig:
    """Training knobs. The defaults train the way the published notebook does.

    The notebook's effective batch of 64 is 8 per micro-batch x 2 GPUs x 4 accumulation steps;
    on one device that is `micro_batch=8, grad_accum=8`.

    `max_len` / `head_max_len` default to the checkpoint's own config. Whatever is used is
    written into the saved config, so inference sees the same budgets training did.
    `amp` defaults to on for CUDA only; `gradient_checkpointing` defaults to following `amp`.
    `option_layout` defaults to the checkpoint's own (`"sequential"` for every published one);
    `"parallel"` trains on `common.parallel_layout` and is written into the saved config.
    """

    epochs: int = 4
    micro_batch: int = 8
    grad_accum: int = 8
    encoder_lr: float = 2.5e-5
    head_lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    loss: Optional[str] = None
    rl_samples: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    w_rps: float = 1.0
    shuffle_options: Tuple[str, ...] = ()
    modality_dropout: float = 0.0
    # The checkpoint's `"vision"` block. `None` trains the checkpoint exactly as it is (a text
    # checkpoint, or one that already carries a block); a mapping *adds* the block to a text
    # checkpoint, which is the only way a vision checkpoint can come into existence -- the base
    # checkpoints ship without one.
    vision: Optional[Dict[str, Any]] = None
    # The checkpoint's `"audio"` block, on exactly the same terms: `None` leaves the checkpoint as
    # it is, and a mapping adds the block. A checkpoint may declare both blocks; a *state* carries
    # one input or the other, never both.
    audio: Optional[Dict[str, Any]] = None
    # Threads used to encode the media inputs while the dataset is built (0 = auto, 1 = serial).
    # A vision or audio run spends most of its build time in a frozen tower, and one tower pass per
    # input is embarrassingly parallel; this is the knob, and the result is identical either way.
    encode_workers: int = 0
    # How often the per-epoch recovery checkpoint is written: 0 = never (the run still saves its
    # final checkpoint), N = every N epochs. A media checkpoint is ~1.2 GB (the encoder plus the
    # bundled tower), so on a slow filesystem this is the largest single cost in a run that is
    # otherwise minutes per epoch -- measured here at ~6 minutes per write against 2.4 minutes of
    # training per epoch.
    checkpoint_every: int = 1
    option_layout: Optional[str] = None
    freeze_encoder: bool = False
    calib_max: int = 400
    calib_frac: float = 0.1
    calib_seed: int = 20260922
    seed: int = 0
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    amp: Optional[bool] = None
    gradient_checkpointing: Optional[bool] = None
    log_every: int = 100
    label_smoothing: float = 0.0
    text_column: str = "text"
    label_column: str = "label"
    question_id: str = "label"
    instructions: Optional[str] = None
    eval_data: Optional[str] = None
    target_error: float = 0.10
    min_abstain_n: int = 10

    def resolve_loss(self, has_vision: bool, has_audio: bool = False) -> str:
        """The objective to train with, given whether the checkpoint carries a block.

        `loss=None` means "the default for this checkpoint", and the default differs by design.
        `rlcd` is what the text checkpoints were trained with, but for a vision checkpoint the
        released recipe is measured to *trail* plain cross-entropy by 2.5-3.0 points, and after
        temperature scaling no objective is better calibrated than CE (spec §7.3). So a vision
        checkpoint starts from `soft-ce` unless the caller asks otherwise, and `rlcd` stays
        available as a small-weight auxiliary to be tested rather than a default to be inherited.
        """
        if self.loss is not None:
            return self.loss
        return "soft-ce" if (has_vision or has_audio) else "rlcd"

    def validate(self) -> None:
        if self.vision is not None:
            if not isinstance(self.vision, dict):
                raise ValueError("vision must be a mapping or None, got %r" % (self.vision,))
            from .vision import parse_vision
            # Parsed here so a typo in a key fails at argument parsing rather than after the
            # checkpoint has loaded and the dataset has been read.
            parse_vision({"vision": dict(self.vision)})
        if self.audio is not None:
            if not isinstance(self.audio, dict):
                raise ValueError("audio must be a mapping or None, got %r" % (self.audio,))
            from .audio import parse_audio
            parse_audio({"audio": dict(self.audio)})
        if self.modality_dropout:
            if isinstance(self.modality_dropout, bool) or not (0.0 <= self.modality_dropout < 1.0):
                raise ValueError("modality_dropout must be in [0, 1), got %r"
                                 % (self.modality_dropout,))
            if self.modality_dropout >= 0.5:
                raise ValueError(
                    "modality_dropout=%r would train most items without the input the decision "
                    "is about, which is how a multimodal checkpoint becomes a text one"
                    % (self.modality_dropout,))
        if isinstance(self.encode_workers, bool) or not isinstance(self.encode_workers, int) \
                or self.encode_workers < 0:
            raise ValueError("encode_workers must be a non-negative integer (0 = auto), got %r"
                             % (self.encode_workers,))
        if isinstance(self.checkpoint_every, bool) or not isinstance(self.checkpoint_every, int) \
                or self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be a non-negative integer (0 = only the final "
                             "checkpoint), got %r" % (self.checkpoint_every,))
        if self.loss is not None and self.loss not in LOSSES:
            raise ValueError("loss must be one of %s (or None for the checkpoint default), got %r"
                             % (", ".join(LOSSES), self.loss))
        unknown = sorted(set(self.shuffle_options) - set(QTYPES))
        if unknown:
            raise ValueError("shuffle_options names unknown question type(s): %s" % ", ".join(unknown))
        if self.option_layout is not None and self.option_layout not in OPTION_LAYOUTS:
            raise ValueError("option_layout must be one of %s, got %r"
                             % (", ".join(OPTION_LAYOUTS), self.option_layout))
        for name in ("epochs", "micro_batch", "grad_accum", "rl_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("%s must be a positive integer, got %r" % (name, value))
        if not 0.0 <= self.calib_frac < 1.0:
            raise ValueError("calib_frac must be in [0, 1), got %r" % (self.calib_frac,))
        if isinstance(self.label_smoothing, bool) or not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1), got %r" % (self.label_smoothing,))
        if isinstance(self.target_error, bool) or not (0.0 <= self.target_error <= 1.0):
            raise ValueError("target_error must be in [0, 1], got %r" % (self.target_error,))
        if isinstance(self.min_abstain_n, bool) or not isinstance(self.min_abstain_n, int) or self.min_abstain_n < 1:
            raise ValueError("min_abstain_n must be a positive integer, got %r" % (self.min_abstain_n,))
        if self.eval_data is not None and not isinstance(self.eval_data, str):
            raise ValueError("eval_data must be a string path, got %r" % (self.eval_data,))


# ------------------------------------------------------------------------------------- data

def to_internal(qid: Any, question: Dict[str, Any]) -> Dict[str, Any]:
    """The question as inference sees it: `Agent._check_question` then `Agent._to_internal`.

    Training on any other rendering would teach the model option texts it never sees at
    inference -- a list-form choice, custom noul `labels`, a non-string instruction.
    """
    from .agent import Agent

    Agent._check_question(qid, question)
    q = Agent._to_internal(question)
    q.pop("option_order", None)    # training draws its own order per epoch
    return q


def target_from_gold(q: Dict[str, Any], gold_q: Dict[str, Any]) -> List[float]:
    """The normalised target distribution for an internal question (`to_internal`), in option order.

    `gold_q["probabilities"]` is keyed by choice label, by `"false"`/`"true"` for noul, and by
    the level index as a string for score -- the `{state, questions, gold}` schema in
    `docs/finetune.md`. A choice label that is not a string is also looked up as its JSON key.
    A row whose probabilities sum to zero becomes uniform, as in the scripts. A negative or
    non-finite probability raises `ValueError`.
    """
    probs = gold_q["probabilities"]

    def get(key, default):
        return probs.get(key, probs.get(str(key), default))

    t, crit = q["t"], q["crit"]
    if t == "choice":
        target = [float(get(k, 0.0)) for k in crit.keys()]
    elif t == "noul":
        target = [float(get("false", 0.5)), float(get("true", 0.5))]
    else:
        target = [float(get(str(i), 0.0)) for i in range(len(crit))]
    if any(not math.isfinite(v) or v < 0 for v in target):
        raise ValueError("probabilities must be finite and non-negative, got %r" % (target,))
    total = sum(target)
    if total > 0:
        return [v / total for v in target]
    return [1.0 / len(target)] * len(target)


def target_from_expected(q: Dict[str, Any], expected_val: Any,
                         label_smoothing: float = 0.0) -> List[float]:
    """The normalised target distribution from a single expected answer, in option order.

    `expected_val` is:
    - a choice label (str or value matching a criterion) for choice questions
    - a boolean for noul questions
    - an integer level index or level string for score questions

    When `label_smoothing > 0`, smooths probability mass uniformly:
    target[i] = (1 - eps) * one_hot[i] + eps / K.
    """
    if isinstance(label_smoothing, bool) or not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must be in [0, 1), got %r" % (label_smoothing,))

    t, crit = q["t"], q["crit"]
    k = len(render_options(q))
    idx = None
    target = None

    if t == "choice":
        keys = list(crit.keys())
        if expected_val in crit:
            idx = keys.index(expected_val)
        else:
            str_expected = str(expected_val)
            for i, key in enumerate(keys):
                if str(key) == str_expected:
                    idx = i
                    break
        if idx is None:
            raise ValueError("expected answer %r not found in choice options %r" % (expected_val, keys))
    elif t == "noul":
        if isinstance(expected_val, bool):
            idx = 1 if expected_val else 0
        elif isinstance(expected_val, (int, float)):
            if expected_val in (0, 1) or expected_val in (0.0, 1.0):
                idx = int(expected_val)
            else:
                raise ValueError("numeric noul expected answer must be 0 or 1, got %r" % (expected_val,))
        elif isinstance(expected_val, str):
            lower = expected_val.strip().lower()
            if lower in ("true", "1", "yes"):
                idx = 1
            elif lower in ("false", "0", "no"):
                idx = 0
        if idx is None:
            raise ValueError("expected answer %r cannot be parsed as a noul boolean" % (expected_val,))
    else:  # score
        keys = list(crit.keys()) if isinstance(crit, dict) else [str(i) for i in range(len(crit))]
        K = len(keys)
        target = None
        if isinstance(expected_val, str) and expected_val in keys:
            idx = keys.index(expected_val)
            target = [1.0 if i == idx else 0.0 for i in range(K)]
        else:
            try:
                if not isinstance(expected_val, bool):
                    val = float(expected_val)
                    if 0.0 <= val <= float(K - 1):
                        low = int(math.floor(val))
                        if low == K - 1 or val == float(low):
                            target = [1.0 if i == low else 0.0 for i in range(K)]
                        else:
                            frac = val - float(low)
                            target = [0.0] * K
                            target[low] = 1.0 - frac
                            target[low + 1] = frac
                    else:
                        raise ValueError("expected score %r out of bounds [0, %d]" % (expected_val, K - 1))
            except (ValueError, TypeError) as exc:
                if isinstance(exc, ValueError) and "out of bounds" in str(exc):
                    raise
                pass
        if target is None:
            str_val = str(expected_val)
            for i, key in enumerate(keys):
                if str(key) == str_val:
                    target = [1.0 if j == i else 0.0 for j in range(K)]
                    break
        if target is None:
            raise ValueError("expected answer %r not found in score levels %r" % (expected_val, keys))

    if target is None:
        target = [1.0 if i == idx else 0.0 for i in range(k)]
    if label_smoothing > 0.0:
        smooth_val = label_smoothing / k
        target = [(1.0 - label_smoothing) * v + smooth_val for v in target]
    return target


def encode_state(tok, state: Any, max_len: int) -> List[int]:
    """State token ids, capped at `max_len`: no sequence can hold more of the state than that."""
    text = serialize_state(state).replace(tok.mask_token, " ")
    return encode_text(tok, text, add_special_tokens=False)["input_ids"][:max_len]


def make_item(tok, q: Dict[str, Any], target: Sequence[float], state_ids: List[int],
              head_max_len: int, max_len: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """`(item, None)`, or `(None, reason)` when the question cannot be trained on as written.

    `q` is an internal question (`to_internal`). The reasons are `target_mismatch` (the target
    does not have one entry per option), `options_collapsed` (the head budget left two options
    with the same token span, #538) and, when `max_len` is given, `options_beyond_max_len` (the
    head is longer than `max_len`, so `build_sequence` drops the markers of the last options --
    the case `Agent` refuses at inference with "only N of its M option markers fit"). Training on
    any of them would teach the model to tell apart options it cannot see as different, or crash
    the batch on a target longer than its markers.
    """
    k = len(render_options(q))
    if len(target) != k:
        return None, "target_mismatch"
    _ids, markers, stats = build_head(tok, q, head_max_len)
    if len(markers) != k or stats["options_distinct"] < stats["options"]:
        return None, "options_collapsed"
    if max_len is not None and markers and markers[-1] >= max_len:
        return None, "options_beyond_max_len"
    return {"q": q, "state_ids": state_ids, "target": [float(v) for v in target],
            "qtype": QTYPES[q["t"]], "k": k,
            # How many tokens the options themselves render to. `visual_plan` decides whether the
            # block gets a `[VIS_SUMMARY]` row from this number, and the projector that emits the
            # block has to make the same decision -- so it is computed once, here, and carried.
            "option_tokens": (len(_ids) - markers[0]) if markers else 0}, None


def encode_media(specs, encode, keys, workers: int = 0):
    """Run a media encoder over the distinct inputs of a dataset, in parallel when asked.

    `specs` is a list of already-decoded inputs (pixels, waveforms), `encode` the tower callable,
    and `keys` their content keys. Returns ``{key: block}`` -- one block per *distinct* input, which
    is what makes a dataset build cost one tower pass per recording rather than one per question.

    The tower is the expensive half of building a media dataset and it is embarrassingly parallel:
    each input is independent, the tower is frozen, and torch releases the GIL inside its kernels.
    Serial encoding measured 9.2 clips/s on this 56-core machine and eight threads measured 18.9,
    which is why the trainer does not do it one input at a time by default. The thread count is
    split across the workers -- `torch.set_num_threads(cpu // workers)` -- because N workers each
    asking for every core oversubscribes and measured *slower* than serial.

    `workers=0` picks ``min(8, cpu_count)``; `workers=1` keeps the old serial path, which is what a
    caller with its own single-threaded encoder wants.
    """
    unique: Dict[str, Any] = {}
    for spec, key in zip(specs, keys):
        unique.setdefault(key, spec)
    if not unique:
        return {}
    if workers == 1 or len(unique) == 1:
        return {key: encode(spec) for key, spec in unique.items()}

    import torch
    from concurrent.futures import ThreadPoolExecutor

    workers = _resolve_encode_workers(workers)
    previous = torch.get_num_threads()
    torch.set_num_threads(max(1, previous // workers))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            blocks = list(pool.map(encode, unique.values()))
    finally:
        torch.set_num_threads(previous)
    return dict(zip(unique.keys(), blocks))


def _resolve_encode_workers(workers: int) -> int:
    if workers and workers > 0:
        return int(workers)
    return max(1, min(8, os.cpu_count() or 1))


def items_from_rows(tok, rows: Iterable[Dict[str, Any]], max_len: int,
                    head_max_len: int,
                    label_smoothing: float = 0.0,
                    vision=None, encode_visual=None,
                    audio=None, encode_audio=None,
                    encode_workers: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Items from `{state, questions, gold}` or `{state, questions, expected}` rows; returns `(items, skipped)`.

    With `vision` set and `encode_visual` supplied (a callable pixels -> `[n_visual, hidden]`),
    a row whose state carries an image is split the way `Agent._encode_state` splits it: the text
    half is tokenized, the image is encoded once per distinct picture through a `VisualCache`, and
    the block is attached to every item built from that row -- so one image asked five questions
    costs one tower pass, not five.

    `encode_workers` sets how many threads encode the media inputs (0 = auto, 1 = serial); see
    :func:`encode_media`. The result is identical either way -- the same blocks, in the same order --
    so this is a speed knob, not a behaviour one.

    `skipped` counts, by reason, questions that were labelled but could not become an item:
    `empty_text`, `empty_label`, `invalid_question` and `invalid_target` (a `ValueError` from
    `to_internal`, `target_from_gold` or `target_from_expected`), plus the reasons `make_item`
    gives. Questions with neither a `gold` nor an `expected` entry are not counted -- the row
    simply does not label them.
    """
    items, skipped = [], {}
    rows = list(rows)
    # Phase 1: split every state and decode every media input, so the tower passes can run together
    # (phase 2) instead of one at a time in the middle of the row loop.
    prepared = []
    for row in rows:
        state = row.get("state")
        visual_spec = audio_spec = None
        if vision is not None and encode_visual is not None and state is not None:
            state, image = split_state(state)
            if image is not None:
                visual_spec = load_pixels(image, vision.image_size)
                visual_spec = (visual_spec, VisualCache.key(visual_spec))
        if audio is not None and encode_audio is not None and state is not None:
            state, sound = split_audio_state(state)
            if sound is not None:
                waveform = load_waveform(sound, audio.sample_rate)
                audio_spec = (waveform, AudioCache.key(waveform))
        prepared.append({"row": row, "state": state, "visual": visual_spec, "audio": audio_spec})

    blocks: Dict[str, Any] = {}
    if vision is not None and encode_visual is not None:
        specs = [p["visual"][0] for p in prepared if p["visual"] is not None]
        keys = [p["visual"][1] for p in prepared if p["visual"] is not None]
        blocks.update(encode_media(specs, encode_visual, keys, workers=encode_workers))
    if audio is not None and encode_audio is not None:
        specs = [p["audio"][0] for p in prepared if p["audio"] is not None]
        keys = [p["audio"][1] for p in prepared if p["audio"] is not None]
        blocks.update(encode_media(specs, encode_audio, keys, workers=encode_workers))

    # Phase 2: the rows themselves, on already-encoded blocks.
    for item_row in prepared:
        row = item_row["row"]
        state = item_row["state"]
        visual = None
        visual_key = None
        audio_frames = None
        audio_key = None
        text_budget = max_len
        if item_row["visual"] is not None:
            visual_key = item_row["visual"][1]
            visual = blocks[visual_key]
            text_budget = max_len - vision.n_visual_tokens
            if text_budget < 16:
                skipped["image_too_large"] = skipped.get("image_too_large", 0) + 1
                continue
        if item_row["audio"] is not None:
            if visual is not None:
                # One sequence carries one spliced block; a row that names both inputs is a
                # dataset bug, and training it as whichever half was parsed first would report a
                # number for a task the run never measured.
                skipped["both_modalities"] = skipped.get("both_modalities", 0) + 1
                continue
            audio_key = item_row["audio"][1]
            audio_frames = blocks[audio_key]
            text_budget = max_len - audio.n_audio_tokens
            if text_budget < 16:
                skipped["audio_too_large"] = skipped.get("audio_too_large", 0) + 1
                continue
        if state is None or (isinstance(state, str) and not state.strip()):
            # An input on its own is a legal state (spec §6.5): it is the evidence.
            if visual is None and audio_frames is None:
                skipped["empty_text"] = skipped.get("empty_text", 0) + 1
                continue
            state = ""
        state_ids = encode_state(tok, state, text_budget)
        for qid, question in row.get("questions", {}).items():
            gold_q = row.get("gold", {}).get(qid) if isinstance(row.get("gold"), dict) else None
            expected_q = row.get("expected", {}).get(qid) if isinstance(row.get("expected"), dict) else None
            if gold_q is None and expected_q is None:
                continue
            reason = None
            try:
                q = to_internal(qid, question)
            except (ValueError, TypeError, KeyError):
                reason = "invalid_question"
            if reason is None:
                try:
                    if gold_q is not None:
                        target = target_from_gold(q, gold_q)
                    else:
                        if isinstance(expected_q, str) and not expected_q.strip():
                            reason = "empty_label"
                        else:
                            target = target_from_expected(q, expected_q, label_smoothing=label_smoothing)
                except (ValueError, TypeError, KeyError):
                    reason = "invalid_target"
            if reason is None:
                item, reason = make_item(tok, q, target, state_ids, head_max_len, max_len)
            if reason is None:
                if visual is not None:
                    # Patches, projected later inside the training step (see `_visual_encoder_fn`).
                    item["visual_patches"] = visual
                    item["visual_key"] = visual_key
                if audio_frames is not None:
                    # Frames, projected later inside the step (see `_audio_encoder_fn`).
                    item["audio_frames"] = audio_frames
                    item["audio_key"] = audio_key
                items.append(item)
            else:
                skipped[reason] = skipped.get(reason, 0) + 1
    return items, skipped


def attach_projector(encoder, model_dir: str, base_declares_vision: bool) -> str:
    """Restore the base checkpoint's trained projector, or refuse, or say a new one is being made.

    The projector is not part of `model.state_dict()` -- it lives beside the weights (see
    `laya.vision`) -- so `build_vision_encoder` cannot restore it and returns one that is randomly
    initialised. `Agent` loads it explicitly and refuses a checkpoint that ships none; the trainer
    did neither, which meant that starting from a vision checkpoint silently restarted from a
    random projector, and the run's own evaluation table scored that projector instead of the
    checkpoint it had just trained (`VISION_PROGRESS.md`, R8: 0.0333 reported, 0.8728 measured).

    Three outcomes, and the caller prints the returned sentence:

    * the base carries a projector -- load it and continue from it;
    * the base *declares* vision and carries none -- refuse, because training a head against noise
      produces numbers that look like a working checkpoint;
    * the base has no vision block at all -- this run is where the projector is created, which is
      the normal path for a first vision fine-tune.
    """
    if os.path.exists(os.path.join(model_dir, VISION_WEIGHTS_FILE)):
        load_vision_weights(encoder, model_dir)
        return ("continuing from the base checkpoint's projector (%s)" % VISION_WEIGHTS_FILE)
    if base_declares_vision:
        raise FileNotFoundError(
            "%r declares vision support but carries no %r. Refusing to start from a randomly "
            "initialised projector: it would train a head against noise and report numbers that "
            "look like a working checkpoint." % (model_dir, VISION_WEIGHTS_FILE))
    return ("base checkpoint has no projector; this run trains a new one "
            "(n_visual_tokens=%d)" % encoder.config.n_visual_tokens)


def attach_audio_projector(encoder, model_dir: str, base_declares_audio: bool) -> str:
    """Restore the base checkpoint's audio projector, or refuse, or say a new one is being made.

    The same three outcomes as :func:`attach_projector`, for the same reason: the projector lives
    beside the weights rather than in `state_dict()`, so building the encoder returns a randomly
    initialised one. Starting a run from an audio checkpoint without restoring it would train a
    head against noise -- and then report numbers for the projector it just created, not for the
    checkpoint it loaded.
    """
    if os.path.exists(os.path.join(model_dir, AUDIO_WEIGHTS_FILE)):
        load_audio_weights(encoder, model_dir)
        return ("continuing from the base checkpoint's projector (%s)" % AUDIO_WEIGHTS_FILE)
    if base_declares_audio:
        raise FileNotFoundError(
            "%r declares audio support but carries no %r. Refusing to start from a randomly "
            "initialised projector: it would train a head against noise and report numbers that "
            "look like a working checkpoint." % (model_dir, AUDIO_WEIGHTS_FILE))
    return ("base checkpoint has no projector; this run trains a new one "
            "(n_audio_tokens=%d)" % encoder.config.n_audio_tokens)


def eval_items_from_rows(tok, rows: Iterable[Dict[str, Any]], max_len: int, head_max_len: int,
                         vision=None, encode_visual=None, audio=None,
                         encode_audio=None,
                         encode_workers: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """The `--eval` rows, built through the *same* modality path the training rows used.

    An eval set exists to answer "how good is this checkpoint", and for a vision checkpoint the
    answer depends on the picture. Built without `vision=` / `encode_visual=` the items carry no
    visual block, and nothing downstream fails: `_project_visual` finds no patches and returns the
    item unchanged, the batch takes the text-only collate path, and the report scores the state
    text alone. What that mistake cost, measured on CIFAR-100 (`VISION_PROGRESS.md`, R8): the
    trainer printed accuracy 0.0333 for a checkpoint that answers at 0.8728 on the identical
    rows -- a number wrong enough to send several sessions after the wrong hypothesis.

    `label_smoothing=0.0` stays: the eval report is about accuracy, not the training target.
    """
    return items_from_rows(tok, rows, max_len, head_max_len, label_smoothing=0.0,
                           vision=vision, encode_visual=encode_visual,
                           audio=audio, encode_audio=encode_audio,
                           encode_workers=encode_workers)


def rows_from_csv(path: str, text_column: str = "text", label_column: str = "label",
                  question_id: str = "label",
                  instructions: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read a CSV file with one labelled column per row into `{state, questions, expected}` rows.

    Discovers distinct labels in `label_column` (preserving order of appearance) and
    builds a choice question with those labels as a list of criteria to render each option once.
    Uses `utf-8-sig` encoding so UTF-8 BOM headers from Excel are handled transparently.
    """
    import csv

    rows_raw = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("%s is an empty CSV file" % (path,))
        if text_column not in reader.fieldnames:
            raise ValueError("CSV file %s missing text column %r (available: %r)"
                             % (path, text_column, reader.fieldnames))
        if label_column not in reader.fieldnames:
            raise ValueError("CSV file %s missing label column %r (available: %r)"
                             % (path, label_column, reader.fieldnames))
        for r in reader:
            rows_raw.append(r)

    if not rows_raw:
        raise ValueError("%s has no rows" % (path,))

    labels = []
    seen = set()
    for r in rows_raw:
        val = r.get(label_column, "").strip()
        if val and val not in seen:
            seen.add(val)
            labels.append(val)

    if len(labels) < 2:
        raise ValueError("CSV file %s must contain at least 2 distinct labels in %r, found %r"
                         % (path, label_column, labels))

    instr = instructions or "Classify the input into the correct category."
    questions = {
        question_id: {
            "type": "choice",
            "instructions": instr,
            "criteria": labels,
        }
    }

    output_rows = []
    for r in rows_raw:
        text = r.get(text_column, "")
        lbl = r.get(label_column, "").strip()
        output_rows.append({
            "state": text,
            "questions": questions,
            "expected": {question_id: lbl},
        })
    return output_rows


def read_data(path: str, text_column: str = "text", label_column: str = "label",
              question_id: str = "label",
              instructions: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read training/eval rows from either a JSONL file or a CSV file."""
    if path.lower().endswith(".csv"):
        return rows_from_csv(path, text_column=text_column, label_column=label_column,
                             question_id=question_id, instructions=instructions)
    return read_jsonl(path)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def encode_item(tok, item: Dict[str, Any], max_len: int, head_max_len: int,
                option_order: Optional[List[int]] = None, parallel: bool = False,
                vision=None, drop_visual: bool = False,
                max_positions: int = 8192, only_option: Optional[int] = None,
                audio=None, drop_audio: bool = False) -> Dict[str, Any]:
    """The model input for an item, with options in `option_order` and the target to match.

    `build_sequence` puts option `option_order[s]` in slot `s`, so the slot-ordered target is
    `target[option_order[s]]` -- the inverse of what `unpermute_probs` does at inference.

    A vision item carries its encoded `visual` block and an audio item its `audio` block;
    `drop_visual` / `drop_audio` train it as text-only, which is what modality dropout does, and
    the text budget is `max_len - n_*_tokens` so the spliced sequence still fits the checkpoint's
    own `max_len` (the budget inference uses).
    """
    visual = item.get("visual") if vision is not None else None
    audio_frames = item.get("audio") if audio is not None else None
    budget = max_len
    if visual is not None and vision is not None:
        budget = max_len - vision.n_visual_tokens
    elif audio_frames is not None and audio is not None:
        budget = max_len - audio.n_audio_tokens
    ids, markers, *layout = build_sequence(tok, None, item["q"], budget, head_max_len,
                                           option_order=option_order,
                                           state_ids=item["state_ids"], return_layout=parallel,
                                           only_option=only_option)
    target = item["target"]
    if option_order is not None:
        target = [target[i] for i in option_order]
    encoded = {"ids": ids, "markers": markers, "qtype": item["qtype"], "target": target}
    if layout:
        encoded["layout"] = layout[0]
    if visual is not None and vision is not None and not drop_visual:
        # The splice point belongs to the sequence that was actually built. Asking `build_head`
        # for the *full* head while the row holds one option puts the block past the question and
        # into the state text -- a different position from the one `Agent._encode_state` uses, so
        # training and inference would splice the same picture in two different places.
        head_ids, _, _ = build_head(tok, item["q"], head_max_len, option_order=option_order,
                                    only_option=only_option)
        plan = visual_plan(vision, text_len=len(ids), markers=markers, head_len=len(head_ids),
                           max_len=max_len, max_position_embeddings=max_positions,
                           option_tokens=item.get(
                               "option_tokens", (len(head_ids) - markers[0]) if markers else 0))
        encoded["visual"] = visual
        encoded["plan"] = plan
        encoded["markers"] = plan.marker_positions
    if audio_frames is not None and audio is not None and not drop_audio:
        head_ids, _, _ = build_head(tok, item["q"], head_max_len, option_order=option_order,
                                    only_option=only_option)
        plan = audio_plan(audio, text_len=len(ids), markers=markers, head_len=len(head_ids),
                          max_len=max_len, max_position_embeddings=max_positions,
                          option_tokens=item.get(
                              "option_tokens", (len(head_ids) - markers[0]) if markers else 0))
        encoded["audio"] = audio_frames
        encoded["plan"] = plan
        encoded["markers"] = plan.marker_positions
    return encoded


def draw_option_order(item: Dict[str, Any], rng: random.Random,
                      shuffle_types: Sequence[str]) -> Optional[List[int]]:
    """A random permutation for items whose type is in `shuffle_types`, else None (canonical)."""
    if item["q"]["t"] not in shuffle_types or item["k"] < 2:
        return None
    order = list(range(item["k"]))
    rng.shuffle(order)
    return order


def split_calibration(items: Sequence[Any], calib_max: int, calib_frac: float,
                      seed: int) -> Tuple[List[Any], List[Any]]:
    """`(train, calibration)`, with the calibration slice taken before any training.

    Fitting temperatures on items the run has trained on measures the fit, not the calibration
    (`docs/finetune.md`). The split depends only on `seed` and the item count, so every DDP rank
    and every rerun hold out the same items.
    """
    n_calib = min(calib_max, int(len(items) * calib_frac))
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    held = set(order[:n_calib])
    train = [it for i, it in enumerate(items) if i not in held]
    calib = [it for i, it in enumerate(items) if i in held]
    return train, calib


# ----------------------------------------------------------------------------------- losses

def soft_ce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a target distribution, over each row's own options only."""
    logp = torch.log_softmax(logits.masked_fill(~mask, _MASKED), -1)
    return -(target * logp * mask).sum(-1).mean()


def grouped_soft_ce(scores: torch.Tensor, targets: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Cross-entropy across each question's own options, scored one row at a time.

    The state-conditioned layout gives every option its own row, so a row holds one score and a
    per-row softmax is constant (softmax over one number is 1, and the loss is 0 whatever the
    model says). The normalisation has to happen across the *group* instead -- exactly what
    `_decode_answers` does at inference, gathering a question's rows and softmaxing over them. If
    training normalised differently from inference, the checkpoint would be fitted to a quantity
    nobody ever reads.

    `scores` is ``[N]``, `targets` ``[N]`` (each option's weight from the question's target
    distribution) and `groups` ``[N]`` the question each row belongs to.
    """
    if scores.dim() != 1 or targets.dim() != 1 or groups.dim() != 1:
        raise ValueError("grouped_soft_ce takes three 1-D tensors, got %s, %s and %s"
                         % (tuple(scores.shape), tuple(targets.shape), tuple(groups.shape)))
    if not (scores.numel() == targets.numel() == groups.numel()):
        raise ValueError(
            "grouped_soft_ce needs one score, one target weight and one group id per row; got "
            "%d, %d and %d" % (scores.numel(), targets.numel(), groups.numel()))
    total = scores.new_zeros(())
    for group in torch.unique(groups):
        rows = groups == group
        z = scores[rows]
        t = targets[rows]
        total = total + -(torch.log_softmax(z, -1) * t).sum()
    return total / max(1, int(torch.unique(groups).numel()))


def rlcd_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, qtype: torch.Tensor,
              sigma: float, samples: int = 4, w_sph: float = 0.75, w_rps: float = 1.0) -> torch.Tensor:
    """The notebook objective: a policy-gradient term over noisy logits plus soft cross-entropy.

    Zero-mean Gaussian noise (scaled by `sigma`, projected onto each row's options) is added to
    detached logits; each sample is rewarded by `proper_reward` against the target, and the
    normalised advantage weights the Gaussian log-density of the sample.
    """
    k = mask.sum(-1, keepdim=True).float()
    fmask = mask.float()
    eps = torch.randn((samples,) + logits.shape, device=logits.device) * sigma * fmask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, _MASKED), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, fmask, w_sph=w_sph, w_rps=w_rps)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * fmask).sum(-1) / (2 * sigma ** 2)
    return -(adv * logp).mean() + soft_ce_loss(logits, target, mask)


def sigma_at(epoch: int, epochs: int, start: float, end: float) -> float:
    """Exploration noise for `epoch`, annealed linearly from `start` to `end`."""
    return start + (end - start) * (epoch / max(1, epochs - 1))


# ------------------------------------------------------------------------------- checkpoint

def resolve_device(device: Optional[str] = "auto") -> torch.device:
    if device not in (None, "auto"):
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def resolve_checkpoint_dir(model_id_or_path: str, token: Optional[str] = None) -> str:
    """Resolve a local path, model alias ('multilingual', 'english') or Hub repo ID to a local directory."""
    if os.path.isdir(model_id_or_path) and os.path.exists(os.path.join(model_id_or_path, "rl_agent_config.json")):
        return model_id_or_path

    if model_id_or_path.startswith(("/", "./", "../")) or os.path.isabs(model_id_or_path):
        if not os.path.isdir(model_id_or_path):
            raise FileNotFoundError("Local model path not found: %r" % (model_id_or_path,))
        return model_id_or_path

    from .router import resolve_model_spec
    spec = resolve_model_spec(model_id_or_path)
    subfolder = None
    if spec is not None:
        model_id_or_path, subfolder = spec

    from huggingface_hub import snapshot_download
    prefix = f"{subfolder}/" if subfolder else ""
    kw = {
        "token": token or os.environ.get("HF_TOKEN") or None,
        "allow_patterns": [prefix + name for name in (
            "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*",
        )],
    }
    model_dir = snapshot_download(model_id_or_path, **kw)
    if subfolder:
        model_dir = os.path.join(model_dir, subfolder)
    if not os.path.isdir(model_dir) or not os.path.exists(os.path.join(model_dir, "rl_agent_config.json")):
        raise FileNotFoundError("Checkpoint directory not found for %r (resolved to %r)"
                                % (model_id_or_path, model_dir))
    return model_dir


def load_checkpoint(model_dir: str):
    """`(model, tokenizer, cfg)` from a local checkpoint directory or Hub ID, model on CPU in fp32."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from .agent import _fix_tokenizer_config

    model_dir = resolve_checkpoint_dir(model_dir)
    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    return model.float(), tok, cfg


def save_checkpoint(model, tok, cfg: Dict[str, Any], path: str) -> None:
    """Write the layout `laya.load` reads: weights in fp16, encoder config, tokenizer, config."""
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    save_file({k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()},
              os.path.join(path, "model.safetensors"))
    # The projector is not part of `state_dict` (DecisionModel deliberately has no vision
    # parameters, so existing checkpoints keep loading with strict=True), so it is written beside
    # the weights. Skipping this line is silent: every load would rebuild a random projector.
    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is not None:
        from .vision import bundle_tower, save_vision_weights
        save_vision_weights(vision_encoder, path)
        # And the tower itself, so the checkpoint does not need the Hub at inference time.
        bundle_tower(vision_encoder, path)
    audio_encoder = getattr(model, "audio_encoder", None)
    if audio_encoder is not None:
        from .audio import bundle_tower as bundle_audio_tower
        from .audio import save_audio_weights
        save_audio_weights(audio_encoder, path)
        # And the tower itself, so the checkpoint does not need the Hub at inference time.
        bundle_audio_tower(audio_encoder, path)
    model.encoder.config.save_pretrained(os.path.join(path, "encoder"))
    tok.save_pretrained(os.path.join(path, "tokenizer"))
    tmp = os.path.join(path, "rl_agent_config.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, os.path.join(path, "rl_agent_config.json"))


# ------------------------------------------------------------------------------------ train

def optimizer_updates(n_items: int, micro_batch: int, grad_accum: int, epochs: int) -> int:
    """Optimizer steps for `n_items` training items.

    `train_model` calls this, so the count a pre-run diagnostic prints is the number of
    `optimizer.step()` calls the run performs: one per full accumulation window
    (`grad_accum` micro-batches), plus one for the final partial window of every epoch, which
    still steps once even when it holds fewer micro-batches.
    """
    if n_items < 1:
        raise ValueError("n_items must be a positive integer, got %r" % (n_items,))
    steps_per_epoch = math.ceil(n_items / micro_batch)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    return updates_per_epoch * epochs


def update_budget_message(n_train: int, config: TrainConfig) -> str:
    """The pre-run optimizer-update budget line, shared by `finetune` startup and `--dry-run`.

    Says "effective batch up to" because the last window of an epoch is often partial, so not
    every update sees a full `micro_batch x grad_accum` batch.
    """
    updates = optimizer_updates(n_train, config.micro_batch, config.grad_accum, config.epochs)
    return ("optimizer budget: %d train item(s), effective batch up to %d "
            "(micro_batch %d x grad_accum %d), %d optimizer updates over %d epoch(s)"
            % (n_train, config.micro_batch * config.grad_accum,
               config.micro_batch, config.grad_accum, updates, config.epochs))


def _embedding_lookup(model):
    """The encoder's own *pre-norm* token embedding, the space a visual block joins.

    The lookup is bound to a **CPU** copy of the table, deliberately. Batches are assembled on the
    CPU and moved to the device once per step (`_forward`), so a lookup that reached a GPU table
    would either raise on CPU token ids or force a device round trip per row. Gradients do not
    flow through here anyway: `inputs_embeds` bypasses the encoder's embedding layer by design, so
    the table is a constant either way.
    """
    enc = getattr(model, "encoder", None) or getattr(model, "_orig_mod", None)
    if enc is not None and not hasattr(enc, "embeddings"):
        enc = getattr(enc, "encoder", enc)
    table = getattr(getattr(enc, "embeddings", None), "tok_embeddings", None)
    if table is None:
        table = enc.embeddings.word_embeddings
    weight = table.weight.detach().to("cpu")
    return lambda ids: torch.nn.functional.embedding(ids, weight)


def _visual_encoder_fn(encoder):
    """`pixels [3, H, W]` -> the tower's **patch tokens** `[N, tower_dim]`.

    Patches, not the projected block, and that is the whole point. The tower is frozen, so its
    output is a constant per image and caching it is honest. The projector is *trained*, so
    projecting here -- once, before the loop, under `no_grad` -- would leave it out of the graph
    and the checkpoint would ship a projector that had never learned anything. `_project_visual`
    runs it inside the step instead.

    The result is cached on the **CPU** even when the tower runs on a GPU. It is one buffer per
    item for the whole run (196x768 fp16 is 300 KB), so a 150k-item probe would need 45 GB of
    device memory -- more than this machine has -- while the CPU has 251 GB. The per-step cost of
    moving a batch's patches back is one transfer, and `_project_visual` puts them on the device.
    """
    def encode(pixels):
        with torch.no_grad():
            return encoder.encode_patches(pixels.unsqueeze(0))[0].detach().to("cpu")
    return encode


def _project_visual(item, projector, vision=None):
    """Run the trainable projector over a cached patch buffer, inside the graph.

    `vision` is what decides whether the block carries a `[VIS_SUMMARY]` row. The block's width and
    `visual_plan`'s expectation have to agree, and they are computed in two different places: the
    plan from the item's own option-token count, the block here. Passing the same item to both is
    what keeps them equal -- without the flag the projector emits `n_visual_tokens` rows for a head
    the plan sized at `n_visual_tokens + 1`, which is the `plan expects 17 visual tokens, got 16`
    refusal that made every high-cardinality vision run impossible.
    """
    patches = item.get("visual_patches")
    if patches is None:
        return item
    summary = bool(vision is not None and "option_tokens" in item
                   and vision.summary_wanted(item["option_tokens"]))
    device = next(iter(projector.parameters())).device
    projected = dict(item)
    projected["visual"] = projector(patches.unsqueeze(0).to(device), summary=summary)[0]
    return projected


def _audio_encoder_fn(encoder):
    """`waveform [T]` -> the tower's **frame tokens** `[N, tower_dim]`.

    Frames, not the projected block, for the reason `_visual_encoder_fn` gives for patches: the
    tower is frozen, so caching its output is honest, while the projector is trained and therefore
    has to run inside the step (`_project_audio`). The cache is kept on the CPU for the same reason
    too -- a wav2vec-style tower emits ~50 frames per second, which is far more rows per item than
    an image's patch grid.
    """
    def encode(waveform):
        with torch.no_grad():
            return encoder.encode_frames(waveform.unsqueeze(0))[0].detach().to("cpu")
    return encode


def _project_audio(item, projector, audio=None):
    """Run the trainable projector over a cached frame buffer, inside the graph."""
    frames = item.get("audio_frames")
    if frames is None:
        return item
    summary = bool(audio is not None and "option_tokens" in item
                   and audio.summary_wanted(item["option_tokens"]))
    device = next(iter(projector.parameters())).device
    projected = dict(item)
    projected["audio"] = projector(frames.unsqueeze(0).to(device), summary=summary)[0]
    return projected


def collate_training_batch(chunk: Sequence[Dict[str, Any]], pad_id: int, embed_fn=None):
    """Collate encoded items, through the block path when any of them carries one.

    A mixed chunk -- some rows with an input, some without -- is legitimate and is what modality
    dropout produces, so the decision is per chunk rather than per run: if a single row has a
    block, every row of the chunk is spliced and the ones without simply have none. A chunk may
    carry pictures or sounds, never both: one batch is one forward pass with one block per row.
    """
    from .audio import collate_audio
    from .vision import collate_visual

    carries_visual = any("visual" in it for it in chunk)
    carries_audio = any("audio" in it for it in chunk)
    if carries_visual or carries_audio:
        if embed_fn is None:
            raise ValueError("a chunk carrying a block needs `embed_fn` to embed the text half; "
                             "pass the encoder's own token-embedding lookup")
        if carries_visual and carries_audio:
            raise ValueError("this chunk mixes rows with a picture and rows with a sound; one batch "
                             "is one forward pass with one spliced block per row, so split the "
                             "batch by modality")
        if carries_visual:
            return collate_visual(list(chunk), embed_fn, pad_id)
        return collate_audio(list(chunk), embed_fn, pad_id)
    return collate_items([list(chunk)], pad_id)


def _forward(model, batch, device, amp: bool, detach_encoder: bool):
    if "inputs_embeds" in batch:
        # Vision: the visual block is already spliced into the token embeddings (see
        # `collate_training_batch`), so there are no token ids to send.
        args = (None, batch["attention_mask"].to(device), batch["marker_pos"].to(device),
                batch["marker_mask"].to(device), batch["qtype"].to(device))
        kwargs = {"detach_encoder": detach_encoder,
                  "position_ids": batch["position_ids"].to(device),
                  "inputs_embeds": batch["inputs_embeds"].to(device)}
        if "visual_mask" in batch:
            # Without it the readout silently never runs, and the head falls back to the shared
            # linear scorer this checkpoint was configured to replace.
            kwargs["visual_mask"] = batch["visual_mask"].to(device)
        if amp:
            with torch.autocast(device.type, dtype=torch.float16):
                logits, _act = model(*args, **kwargs)
        else:
            logits, _act = model(*args, **kwargs)
        return logits.float()
    args = (batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device))
    kwargs = {"detach_encoder": detach_encoder}
    if "option_ids" in batch:
        kwargs.update(position_ids=batch["position_ids"].to(device), option_ids=batch["option_ids"].to(device))
    if amp:
        with torch.autocast(device.type, dtype=torch.float16):
            logits, _act = model(*args, **kwargs)
    else:
        logits, _act = model(*args, **kwargs)
    return logits.float()


def train_model(model, tok, items: Sequence[Dict[str, Any]], config: TrainConfig, device: torch.device,
                max_len: int, head_max_len: int,
                on_epoch_end: Optional[Callable[[int, float], None]] = None,
                parallel: bool = False, vision=None, embed_fn=None,
                max_positions: int = 8192, state_conditioned: bool = False,
                audio=None, audio_projector=None) -> List[float]:
    """Train `model` in place on `items`; returns the mean loss of each epoch.

    `vision` is the checkpoint's parsed vision config and `embed_fn` its token-embedding lookup.
    Both are needed only when items carry a visual block; a text run passes neither and takes the
    same `collate_items` path it always did.
    """
    config.validate()
    if not items:
        raise ValueError("no training items")
    amp = (device.type == "cuda") if config.amp is None else bool(config.amp)
    checkpointing = amp if config.gradient_checkpointing is None else bool(config.gradient_checkpointing)

    if config.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    elif checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = checkpointing
    model.to(device).train()
    if config.freeze_encoder:
        # A frozen encoder is a fixed feature extractor: dropout in it would only add noise.
        model.encoder.eval()

    # The projector lives beside the model, not in it (DecisionModel carries no vision parameters
    # on purpose, so existing checkpoints keep loading), which means `model.parameters()` does not
    # see it. Without this list it would never be optimised and the checkpoint would ship a
    # projector that had never learned anything.
    # NOT `vision`: that name is the parsed VisionConfig this function was called with, and
    # rebinding it here silently turned every downstream `vision.n_visual_tokens` into an
    # AttributeError on the encoder -- in the real `finetune` path only, because the tests build a
    # model without a `vision_encoder` attribute.
    vision_encoder = getattr(model, "vision_encoder", None)
    projector = None
    if vision_encoder is not None:
        projector = vision_encoder.projector
        projector.to(device)
        for parameter in projector.parameters():
            parameter.requires_grad_(True)
    # The audio projector has the same problem and the same fix: it is not in `state_dict()`, so
    # `model.parameters()` does not see it and it would never be optimised.
    audio_encoder = getattr(model, "audio_encoder", None)
    if audio_encoder is not None:
        audio_projector = audio_encoder.projector
        audio_projector.to(device)
        for parameter in audio_projector.parameters():
            parameter.requires_grad_(True)
    has_block = vision is not None or audio is not None
    groups = [{"params": [p for n, p in model.named_parameters()
                          if not n.startswith("encoder.") and p.requires_grad], "lr": config.head_lr}]
    if projector is not None:
        groups.append({"params": [p for p in projector.parameters() if p.requires_grad],
                       "lr": config.head_lr})
    if audio_projector is not None:
        groups.append({"params": [p for p in audio_projector.parameters() if p.requires_grad],
                       "lr": config.head_lr})
    if not config.freeze_encoder:
        groups.insert(0, {"params": [p for n, p in model.named_parameters()
                                     if n.startswith("encoder.") and p.requires_grad], "lr": config.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(items) / config.micro_batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=optimizer_updates(len(items), config.micro_batch, config.grad_accum, config.epochs),
        eta_min=config.min_lr)
    scaler = torch.amp.GradScaler("cuda") if amp and device.type == "cuda" else None

    torch.manual_seed(config.seed)
    order_rng = random.Random(config.seed)
    params = [p for g in groups for p in g["params"]]
    history = []
    for epoch in range(config.epochs):
        epoch_items = list(items)
        random.Random(config.seed + epoch).shuffle(epoch_items)
        sigma = sigma_at(epoch, config.epochs, config.sigma_start, config.sigma_end)
        total, n_steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(epoch_items), config.micro_batch):
            window = epoch_items[start:start + config.micro_batch]
            # Modality dropout is drawn from the epoch's own RNG, so a run is reproducible and the
            # same item is not dropped on every epoch. Dropping is per *item*: the target stays
            # the true answer, which is what teaches the model that the text alone is sometimes
            # all it gets -- and, on the image-required items, that it is not enough.
            chunk = []
            groups = []          # one entry per row: which question of this window it scores
            for position, it in enumerate(window):
                dropped = bool(has_block and config.modality_dropout
                               and order_rng.random() < config.modality_dropout)
                source = it
                if not dropped:
                    if projector is not None:
                        source = _project_visual(source, projector, vision)
                    if audio_projector is not None:
                        source = _project_audio(source, audio_projector, audio)
                order = draw_option_order(it, order_rng, config.shuffle_options)
                if state_conditioned:
                    # One row per option. The row's own target is the weight that option carries
                    # in the question's target distribution; the normalisation happens across the
                    # group, in `grouped_soft_ce`.
                    for slot in range(it["k"]):
                        encoded = encode_item(tok, source, max_len, head_max_len, order, parallel,
                                              vision=vision, max_positions=max_positions,
                                              drop_visual=dropped, only_option=slot,
                                              audio=audio, drop_audio=dropped)
                        encoded["target"] = [float(it["target"][order[slot] if order else slot])]
                        chunk.append(encoded)
                        groups.append(position)
                    continue
                chunk.append(encode_item(tok, source, max_len, head_max_len, order,
                                         parallel, vision=vision, max_positions=max_positions,
                                         drop_visual=dropped, audio=audio, drop_audio=dropped))
            batch = collate_training_batch(chunk, tok.pad_token_id,
                                           embed_fn=embed_fn if has_block else None)
            # `detach_encoder` severs the gradient at the encoder's output. For a text checkpoint
            # that is free: nothing upstream of the encoder has parameters. For a vision one it is
            # not -- the projector sits *upstream*, spliced into the encoder's input embeddings --
            # so detaching makes it untrainable, and a frozen-encoder vision run would fit the head
            # against a projector that never moved. Keep the graph when there is something to
            # train through it, and pay the activation memory the detach existed to save.
            logits = _forward(model, batch, device, amp,
                              config.freeze_encoder and projector is None and audio_projector is None)
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            if state_conditioned:
                # Every row carries exactly one marker; the question's distribution is the softmax
                # across its rows, which is what `_decode_answers` will compute at inference.
                scores = logits[:, 0]
                group_ids = torch.tensor(groups, device=device)
                loss = grouped_soft_ce(scores, target[:, 0], group_ids)
            elif config.loss == "rlcd":
                loss = rlcd_loss(logits, target, mask, batch["qtype"].to(device), sigma,
                                 config.rl_samples, config.w_sph, config.w_rps)
            else:
                loss = soft_ce_loss(logits, target, mask)
            window_start = (n_steps // config.grad_accum) * config.grad_accum
            window_size = min(config.grad_accum, steps_per_epoch - window_start)
            scaled = loss / window_size
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            n_steps += 1
            if n_steps % config.grad_accum == 0 or start + config.micro_batch >= len(epoch_items):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item()
            if config.log_every and n_steps % config.log_every == 0:
                print("epoch %d/%d step %d loss %.4f" % (epoch + 1, config.epochs, n_steps, loss.item()), flush=True)
        mean = total / max(1, n_steps)
        history.append(mean)
        print("epoch %d/%d mean loss %.4f" % (epoch + 1, config.epochs, mean), flush=True)
        if on_epoch_end is not None:
            on_epoch_end(epoch, mean)
    model.eval()
    return history


@torch.no_grad()
def calibration_records(model, tok, items: Sequence[Dict[str, Any]], device: torch.device,
                        max_len: int, head_max_len: int, batch_size: int = 16, parallel: bool = False,
                        vision=None, embed_fn=None, max_positions: int = 8192, projector=None,
                        state_conditioned: bool = False, audio=None, audio_projector=None):
    """`(qtype, logits, target, k)` records for `fit_temperature_map`, options in canonical order.

    A vision or audio checkpoint has to be *evaluated* through its own path. These records feed both
    the
    before/after accuracy in the training report and the temperature fit, so skipping the block
    here does not fail loudly: it evaluates a text-only model, reports chance-level accuracy for a
    checkpoint that answers its held-out set at 0.62, and fits a temperature to the wrong
    distribution.
    """
    model.eval()
    records = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        if vision is not None and projector is not None:
            chunk = [_project_visual(it, projector, vision) for it in chunk]
        if audio is not None and audio_projector is not None:
            chunk = [_project_audio(it, audio_projector, audio) for it in chunk]
        # A state-conditioned checkpoint reads one row per option, so evaluation has to encode
        # that way too. Building a joint row and slicing `row[:k]` out of a one-marker row does not
        # fail -- it silently scores a sequence the checkpoint never saw, and reports chance-level
        # accuracy for a model that answers its held-out set at 0.91.
        encoded = []
        if state_conditioned:
            for it in chunk:
                for slot in range(it["k"]):
                    row = encode_item(tok, it, max_len, head_max_len, parallel=parallel,
                                      vision=vision, max_positions=max_positions, only_option=slot,
                                      audio=audio)
                    # One marker, so one target weight -- the option's own share of the question's
                    # distribution. The full target belongs to the group, not to the row.
                    row["target"] = [float(it["target"][slot])]
                    encoded.append(row)
        else:
            encoded = [encode_item(tok, it, max_len, head_max_len, parallel=parallel, vision=vision,
                                   max_positions=max_positions, audio=audio) for it in chunk]
        has_block = vision is not None or audio is not None
        batch = collate_training_batch(encoded, tok.pad_token_id,
                                       embed_fn=embed_fn if has_block else None)
        logits = _forward(model, batch, device, amp=False,
                          detach_encoder=not has_block).cpu().numpy()
        cursor = 0
        for it in chunk:
            if state_conditioned:
                # One score per option, gathered across the item's own rows -- the same
                # normalisation `_decode_answers` applies at inference.
                rows = logits[cursor:cursor + it["k"], 0]
                cursor += it["k"]
                records.append((it["qtype"], rows, it["target"], it["k"]))
            else:
                records.append((it["qtype"], logits[cursor, :it["k"]], it["target"], it["k"]))
                cursor += 1
    return records


def calibration_report(records: Sequence[Tuple[int, Any, Any, int]],
                       temperature: Sequence[float]) -> Dict[str, Dict[str, Any]]:
    """Per question type: how many calibration items there were, what was fitted, and what to doubt.

    `records` are the `calibration_records` the temperatures were fitted on, and `temperature`
    the per-type scalars `fit_temperature_map` returned. Each type gets `items`, `temperature`,
    `accuracy` and `mean_confidence` (top probability at the fitted temperature) on the
    calibration items, and a list of `issues`, empty when nothing looks wrong:

    - fewer than `MIN_TYPE_N` items: the temperature was not fitted and stays 1.0;
    - fewer than `CALIB_WARN_N` items: it was fitted, on little evidence;
    - it landed on the `[TEMP_MIN, TEMP_MAX]` clamp;
    - it came back unchanged at 1.0 although it was fitted, which usually means the items gave
      the fit nothing to correct -- what a model already certain and right on them produces.

    A type with no calibration items at all has no issues; it was not in the data.
    """
    import numpy as np

    by_type: Dict[int, List[Tuple[Any, Any]]] = {qt: [] for qt in QTYPE_NAMES}
    for qt, logits, target, _k in records:
        by_type[int(qt)].append((np.asarray(logits, dtype=float), np.asarray(target, dtype=float)))
    report = {}
    for qt, name in QTYPE_NAMES.items():
        pairs, t = by_type[qt], float(temperature[qt])
        entry: Dict[str, Any] = {"items": len(pairs), "temperature": t, "issues": []}
        if pairs:
            right, conf = 0, 0.0
            for logits, target in pairs:
                z = logits / t
                p = np.exp(z - z.max())
                p /= p.sum()
                right += int(p.argmax() == target.argmax())
                conf += float(p.max())
            entry["accuracy"] = right / len(pairs)
            entry["mean_confidence"] = conf / len(pairs)
        n, issues = len(pairs), entry["issues"]
        if 0 < n < MIN_TYPE_N:
            issues.append("not fitted: %d calibration items, fewer than %d, so the temperature stays 1.0"
                          % (n, MIN_TYPE_N))
        elif n >= MIN_TYPE_N:
            if n < CALIB_WARN_N:
                issues.append("fitted on only %d calibration items" % n)
            if t <= TEMP_MIN or t >= TEMP_MAX:
                issues.append("the fitted temperature %.3g is on the [%g, %g] clamp" % (t, TEMP_MIN, TEMP_MAX))
            elif abs(t - 1.0) < 1e-3:
                issues.append("the fit came back unchanged at 1.0 (accuracy %.2f, mean confidence %.3f on the "
                              "calibration items), which usually means they gave it nothing to correct; this "
                              "type's confidences are uncalibrated" % (entry["accuracy"], entry["mean_confidence"]))
        report[name] = entry
    return report


def evaluate_records(records: Sequence[Tuple[int, Any, Any, int]],
                     temperature: Optional[Sequence[float]] = None,
                     temperature_by_options: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Evaluate loss, accuracy, top-1 confidence, ECE and Brier scores on `records`.

    `records` is a sequence of `(qtype, logits, target, k)` tuples as returned by
    `calibration_records`. Predictions and confidences are scaled using `temperature` /
    `temperature_by_options` if provided. Returns overall metrics and per-question-type
    breakdowns.

    `brier` is the multi-class Brier score over the full probability vector:
    `mean(sum((p - target)**2))`. `brier_top1` is the binary Brier score of top-1
    confidence as P(correct): `mean((max(p) - correct)**2)`.
    """
    import numpy as np

    if not records:
        return {
            "items": 0,
            "loss": 0.0,
            "accuracy": 0.0,
            "mean_confidence": 0.0,
            "ece": None,
            "brier": None,
            "brier_top1": None,
            "by_type": {},
        }

    losses: List[float] = []
    corrects: List[bool] = []
    confs: List[float] = []
    briers_mc: List[float] = []
    briers_top1: List[float] = []
    by_type: Dict[int, Dict[str, List[Any]]] = {
        qt: {"losses": [], "corrects": [], "confs": [], "briers_mc": [], "briers_top1": []} for qt in QTYPE_NAMES
    }

    for qt, logits, target, k in records:
        z = np.asarray(logits[:k], dtype=float)
        t = np.asarray(target[:k], dtype=float)
        bucket = temp_bucket(qt, k)
        if temperature_by_options and bucket in temperature_by_options:
            scale = float(temperature_by_options[bucket])
        elif temperature is not None and isinstance(temperature, (list, tuple)) and qt < len(temperature):
            scale = float(temperature[qt])
        elif isinstance(temperature, (int, float)):
            scale = float(temperature)
        else:
            scale = 1.0
        scale = max(scale, 1e-4)
        z_scaled = z / scale
        z_max = np.max(z_scaled)
        exp_z = np.exp(z_scaled - z_max)
        sum_exp_z = np.sum(exp_z)
        p = exp_z / sum_exp_z
        log_p = z_scaled - z_max - np.log(sum_exp_z)
        loss = float(-np.sum(t * log_p))

        pred = int(np.argmax(p))
        gold = int(np.argmax(t))
        correct = bool(pred == gold)
        conf = float(np.max(p))
        b_mc = float(np.sum((p - t) ** 2))
        b_top1 = float((conf - float(correct)) ** 2)

        losses.append(loss)
        corrects.append(correct)
        confs.append(conf)
        briers_mc.append(b_mc)
        briers_top1.append(b_top1)

        if qt in by_type:
            by_type[qt]["losses"].append(loss)
            by_type[qt]["corrects"].append(correct)
            by_type[qt]["confs"].append(conf)
            by_type[qt]["briers_mc"].append(b_mc)
            by_type[qt]["briers_top1"].append(b_top1)

    total_items = len(records)
    acc = float(np.mean(corrects)) if corrects else 0.0
    mean_loss = float(np.mean(losses)) if losses else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0
    ece_val = float(ece_score(np.asarray(confs, dtype=float), np.asarray(corrects, dtype=bool))) if len(corrects) >= 2 else None
    brier_val = float(np.mean(briers_mc)) if briers_mc else None
    brier_top1_val = float(np.mean(briers_top1)) if briers_top1 else None

    by_type_summary = {}
    for qt, name in QTYPE_NAMES.items():
        t_losses = by_type[qt]["losses"]
        t_corrects = by_type[qt]["corrects"]
        t_confs = by_type[qt]["confs"]
        t_briers_mc = by_type[qt]["briers_mc"]
        t_briers_top1 = by_type[qt]["briers_top1"]
        if t_losses:
            t_ece = float(ece_score(np.asarray(t_confs, dtype=float), np.asarray(t_corrects, dtype=bool))) if len(t_corrects) >= 2 else None
            t_brier = float(np.mean(t_briers_mc))
            t_brier_top1 = float(np.mean(t_briers_top1))
            by_type_summary[name] = {
                "items": len(t_losses),
                "loss": round(float(np.mean(t_losses)), 4),
                "accuracy": round(float(np.mean(t_corrects)), 4),
                "mean_confidence": round(float(np.mean(t_confs)), 4),
                "ece": round(t_ece, 4) if t_ece is not None else None,
                "brier": round(t_brier, 4) if t_brier is not None else None,
                "brier_top1": round(t_brier_top1, 4) if t_brier_top1 is not None else None,
            }

    return {
        "items": total_items,
        "loss": round(mean_loss, 4),
        "accuracy": round(acc, 4),
        "mean_confidence": round(mean_conf, 4),
        "ece": round(ece_val, 4) if ece_val is not None else None,
        "brier": round(brier_val, 4) if brier_val is not None else None,
        "brier_top1": round(brier_top1_val, 4) if brier_top1_val is not None else None,
        "by_type": by_type_summary,
    }


def collapse_stats(records: Sequence[Tuple[int, Any, Any, int]]) -> Optional[Dict[str, float]]:
    """Mean within-row logit range, mean soft-CE and mean ln K on records with k >= 2.

    A 1-option row cannot collapse to a prior, so it is skipped. Empty input, or only
    1-option rows, returns None.
    """
    import numpy as np

    ranges, ces, lnks = [], [], []
    for _qt, logits, target, k in records:
        k = int(k)
        if k < 2:
            continue
        z = np.asarray(logits, dtype=float).ravel()[:k]
        t = np.asarray(target, dtype=float).ravel()[:k]
        mass = float(t.sum())
        if mass <= 0.0:
            continue
        t = t / mass
        ranges.append(float(z.max() - z.min()))
        logp = z - np.logaddexp.reduce(z)
        ces.append(float(-(t * logp).sum()))
        lnks.append(math.log(k))
    if not ranges:
        return None
    n = len(ranges)
    return {
        "n": float(n),
        "mean_logit_range": sum(ranges) / n,
        "mean_ce": sum(ces) / n,
        "mean_ln_k": sum(lnks) / n,
    }


def prior_collapse_message(records: Sequence[Tuple[int, Any, Any, int]]) -> Optional[str]:
    """Warning text if calibration logits have collapsed to the class prior, else None.

    The default 4-epoch budget is sized for a few thousand typed decisions. On a few
    hundred or a thousand labelled rows the head can finish with every option at the
    same logit -- chance-level, and silent. #963 measured a mean within-row range of
    0.01 against 0.62 for the untuned checkpoint, and training CE stuck at ln K.

    Fired when the mean within-row logit range is below `COLLAPSE_LOGIT_RANGE`, or when
    mean soft-CE on the same records is within `COLLAPSE_CE_REL` of mean ln K.
    """
    stats = collapse_stats(records)
    if stats is None:
        return None
    reasons = []
    if stats["mean_logit_range"] < COLLAPSE_LOGIT_RANGE:
        reasons.append("mean within-row logit range %.3g" % stats["mean_logit_range"])
    mean_ce, mean_lnk = stats["mean_ce"], stats["mean_ln_k"]
    if mean_lnk > 0.0 and abs(mean_ce - mean_lnk) <= COLLAPSE_CE_REL * mean_lnk:
        reasons.append("calibration cross-entropy %.3f is within %.0f%% of ln K ~ %.3f"
                       % (mean_ce, COLLAPSE_CE_REL * 100.0, mean_lnk))
    if not reasons:
        return None
    return ("laya.train: predictions collapsed to the class prior (%s). "
            "The default epoch budget is often too small on a few hundred or a thousand "
            "labelled rows. Try more epochs, more data, or another seed."
            % "; ".join(reasons))


def finetune(data: str, model_dir: str, output_dir: str, config: Optional[TrainConfig] = None,
             device: Optional[str] = "auto") -> Dict[str, Any]:
    """Preprocess, train, calibrate and save; returns a summary of the run.

    `data` is a JSONL file of `{state, questions, gold}` or `{state, questions, expected}` rows,
    or a CSV file with text and label columns.
    `model_dir` is a local checkpoint directory -- the layout `laya.load` reads. The result in
    `output_dir` loads with `laya.load(output_dir)`.
    """
    config = config or TrainConfig()
    config.validate()
    dev = resolve_device(device)
    model, tok, cfg = load_checkpoint(model_dir)
    # Whether the *base checkpoint* already declares vision, captured before `--vision` can add the
    # block: the two cases want different answers when the weights file is missing.
    base_declares_vision = parse_vision(cfg) is not None
    base_declares_audio = parse_audio(cfg) is not None
    if config.audio is not None:
        # Same rule as vision: the block has to be in the config before anything is saved, so it
        # travels with the weights and `laya.load` rebuilds the encoder from it.
        cfg = dict(cfg, audio=dict(config.audio))
        print("audio: training with a %r block" % (config.audio,), flush=True)
    if config.vision is not None:
        # Declaring vision support is what turns a text checkpoint into a vision one. It has to
        # happen before `parse_vision` below and before any checkpoint is saved, so the block
        # travels with the weights.
        cfg = dict(cfg, vision=dict(config.vision))
        print("vision: training with a %r block" % (config.vision,), flush=True)
    # The base checkpoint is scored on eval_items before train_model runs, and calibration_records
    # sends its inputs to `dev`, so the weights have to be there first (CUDA and MPS crash otherwise).
    model.to(dev)
    max_len = config.max_len or cfg.get("max_len", 512)
    head_max_len = config.head_max_len or cfg.get("head_max_len", 192)
    if config.option_layout is not None:
        cfg = dict(cfg, option_layout=config.option_layout)
    parallel = uses_parallel_layout(cfg)
    if parallel and (parse_vision(cfg) is not None or parse_audio(cfg) is not None):
        raise ValueError("a checkpoint with a spliced block cannot train on the parallel option "
                         "layout: its masks are built from token ids and there are none once the "
                         "block is spliced in")

    # Vision. A checkpoint with no "vision" block parses to None and every branch below is
    # skipped, so a text fine-tune is bit-for-bit what it was -- including its objective, which
    # stays `rlcd` while a vision run starts from `soft-ce` (see `TrainConfig.resolve_loss`).
    vision = parse_vision(cfg)
    audio = parse_audio(cfg)
    max_positions = int(getattr(model.encoder.config, "max_position_embeddings", 8192))
    embed_fn = None
    encode_visual = None
    projector = None
    if vision is not None:
        encoder = build_vision_encoder(vision, hidden_size=model.encoder.config.hidden_size,
                                       tower_dir=local_tower_dir(model_dir))
        model.vision_encoder = encoder
        projector = encoder.projector
        embed_fn = _embedding_lookup(model)
        # The tower has to be on `dev` *before* the items are built, because building them is
        # what encodes every image (`_visual_encoder_fn`). `VisionEncoder` is not an `nn.Module`,
        # so the `model.to(dev)` above cannot reach it, and nothing else moves the tower: without
        # this line a GPU run encodes the whole dataset on the CPU and merely looks like a slow
        # GPU run. Measured here at 2 images/s, versus hundreds once the tower is on the card.
        # The projector is not part of `model.state_dict()` (it lives beside the weights, see
        # `laya.vision`), so building the encoder does NOT restore it: `PooledProjector` comes
        # back randomly initialised. `Agent` loads it explicitly and refuses a checkpoint that
        # ships none; the trainer used to do neither, so a run continued from a vision checkpoint
        # silently restarted from a random projector -- and its own evaluation table scored that
        # random projector rather than the checkpoint it had just trained.
        print("vision: %s" % (attach_projector(encoder, model_dir, base_declares_vision),), flush=True)
        encoder.to(dev)
        encode_visual = _visual_encoder_fn(encoder)
        print("vision checkpoint: tower %s, %d visual tokens, modality dropout %g"
              % (vision.tower, vision.n_visual_tokens, config.modality_dropout), flush=True)
    audio_projector = None
    encode_audio = None
    if audio is not None:
        audio_encoder = build_audio_encoder(audio, hidden_size=model.encoder.config.hidden_size,
                                            tower_dir=local_audio_tower_dir(model_dir))
        model.audio_encoder = audio_encoder
        audio_projector = audio_encoder.projector
        if embed_fn is None:
            embed_fn = _embedding_lookup(model)
        # The same two facts as the vision path: the tower has to be on `dev` before the items are
        # built (building them is what encodes every recording), and the projector is not restored
        # by `build_audio_encoder`, so it is attached explicitly or the run starts from noise.
        print("audio: %s" % (attach_audio_projector(audio_encoder, model_dir,
                                                    base_declares_audio),), flush=True)
        audio_encoder.to(dev)
        encode_audio = _audio_encoder_fn(audio_encoder)
        print("audio checkpoint: tower %s, %d audio tokens, modality dropout %g"
              % (audio.tower, audio.n_audio_tokens, config.modality_dropout), flush=True)
    config.loss = config.resolve_loss(vision is not None, audio is not None)

    rows = read_data(data, text_column=config.text_column, label_column=config.label_column,
                     question_id=config.question_id, instructions=config.instructions)
    items, skipped = items_from_rows(tok, rows, max_len, head_max_len,
                                     label_smoothing=config.label_smoothing,
                                     vision=vision, encode_visual=encode_visual,
                                     audio=audio, encode_audio=encode_audio,
                                     encode_workers=config.encode_workers)
    if not items:
        raise ValueError("%s produced no training items (skipped: %r)" % (data, skipped))
    train_items, calib_items = split_calibration(items, config.calib_max, config.calib_frac, config.calib_seed)

    eval_skipped = {}
    eval_mode = None
    is_held_out = False
    eval_source = None
    eval_items = []
    eval_note = None

    if config.eval_data:
        eval_source = config.eval_data
        eval_rows = read_data(config.eval_data, text_column=config.text_column, label_column=config.label_column,
                              question_id=config.question_id, instructions=config.instructions)
        eval_items, eval_skipped = eval_items_from_rows(tok, eval_rows, max_len, head_max_len,
                                                     vision=vision, encode_visual=encode_visual,
                                                     audio=audio, encode_audio=encode_audio,
                                                     encode_workers=config.encode_workers)

        if not eval_items:
            eval_mode = None
            is_held_out = False
            eval_note = ("Evaluation file %s yielded 0 usable items (skipped: %r); "
                         "no evaluation performed." % (config.eval_data, eval_skipped))
            warnings.warn("laya.train: %s" % (eval_note,), RuntimeWarning, stacklevel=2)
            print("eval items 0 from %s (skipped %r)" % (config.eval_data, eval_skipped), flush=True)
        else:
            # Build input signatures from normalized, usable items:
            # (state token ids, normalized question schema).
            # Checked separately from targets to detect shared inputs even when labels differ (#967).
            def input_sig(it):
                # A media row's text is not its input: the picture or the recording is. Every CIFAR
                # row shares the state text "a photograph" and every ESC-50 row shares "a recording
                # of an environmental sound", so a text-only signature calls two different inputs
                # the same row and reports a held-out eval set as overlapping its training data.
                # The content keys are what make the signature the input rather than the wrapper.
                return (tuple(it["state_ids"]), json.dumps(it["q"], sort_keys=True),
                        it.get("visual_key"), it.get("audio_key"))

            train_input_sigs = {input_sig(it) for it in train_items}
            calib_input_sigs = {input_sig(it) for it in calib_items}

            train_overlap_count = sum(1 for it in eval_items if input_sig(it) in train_input_sigs)
            calib_overlap_count = sum(1 for it in eval_items if input_sig(it) in calib_input_sigs)
            total_overlap_count = sum(
                1 for it in eval_items
                if input_sig(it) in train_input_sigs or input_sig(it) in calib_input_sigs
            )

            if total_overlap_count > 0:
                eval_mode = "overlapping_eval"
                is_held_out = False
                if train_overlap_count > 0 and calib_overlap_count > 0:
                    eval_note = (
                        "Evaluation set contains inputs that overlap training data (%d) and calibration data (%d) "
                        "(total %d/%d items); metrics do not reflect strictly independent held-out evaluation."
                        % (train_overlap_count, calib_overlap_count, total_overlap_count, len(eval_items))
                    )
                elif train_overlap_count > 0:
                    eval_note = (
                        "Evaluation set contains rows that overlap training data (%d/%d items); "
                        "metrics do not reflect strictly independent held-out evaluation."
                        % (train_overlap_count, len(eval_items))
                    )
                else:
                    eval_note = (
                        "Evaluation set contains inputs that overlap calibration data (%d/%d items); "
                        "metrics do not reflect strictly independent held-out evaluation."
                        % (calib_overlap_count, len(eval_items))
                    )
                print("eval items %d from %s (skipped %r, %d overlapping training/calibration items)"
                      % (len(eval_items), config.eval_data, eval_skipped, total_overlap_count), flush=True)
            else:
                eval_mode = "held_out"
                is_held_out = True
                eval_note = "Metrics reflect independent held-out evaluation."
                print("eval items %d from %s (skipped %r, 0 overlapping training/calibration items)"
                      % (len(eval_items), config.eval_data, eval_skipped), flush=True)
    elif calib_items:
        eval_items = calib_items
        eval_source = "calibration_slice"
        eval_mode = "in_sample_calibration"
        is_held_out = False
        eval_note = ("Metrics reflect in-sample calibration fit, not independent held-out evaluation. "
                     "Provide --eval for held-out validation.")
    else:
        eval_items = []
        eval_source = None
        eval_mode = None
        is_held_out = False
        eval_note = "No evaluation performed (0 calibration items and no --eval set was provided)."
        warnings.warn("laya.train: dataset has 0 calibration items and no --eval set was provided; "
                      "skipping evaluation to avoid reporting training fit as generalization.",
                      RuntimeWarning, stacklevel=2)

    print("train items %d, calibration items %d, eval items %d%s, skipped %r, device %s"
          % (len(train_items), len(calib_items), len(eval_items),
             (" (" + str(eval_source) + ")") if eval_source else "", skipped, dev), flush=True)
    print(update_budget_message(len(train_items), config), flush=True)

    # Evaluate base checkpoint on eval items prior to training
    before_eval = None
    if eval_items:
        base_records = calibration_records(model, tok, eval_items, dev, max_len, head_max_len,
                                           parallel=parallel, vision=vision, embed_fn=embed_fn,
                                           projector=projector, max_positions=max_positions,
                                           state_conditioned=uses_state_conditioned_layout(cfg),
                                           audio=audio, audio_projector=audio_projector)
        base_temp = cfg.get("temperature")
        base_temp_by_options = cfg.get("temperature_by_options")
        before_eval = evaluate_records(base_records, base_temp, base_temp_by_options)

    def checkpoint_latest(epoch, _loss):
        # `checkpoint_every=0` turns the per-epoch recovery checkpoint off; the final checkpoint is
        # written after training either way. A media run writes ~1.2 GB here (the encoder plus the
        # bundled tower), which on a slow filesystem costs more than the epoch it is protecting.
        if not config.checkpoint_every:
            return
        if (epoch + 1) % config.checkpoint_every:
            return
        save_checkpoint(model, tok, dict(cfg, max_len=max_len, head_max_len=head_max_len),
                        os.path.join(output_dir, "checkpoint_latest"))

    history = train_model(model, tok, train_items, config, dev, max_len, head_max_len,
                          on_epoch_end=checkpoint_latest, parallel=parallel,
                          vision=vision, embed_fn=embed_fn, max_positions=max_positions,
                          state_conditioned=uses_state_conditioned_layout(cfg),
                          audio=audio, audio_projector=audio_projector)

    records = calibration_records(model, tok, calib_items, dev, max_len, head_max_len,
                                  parallel=parallel, vision=vision, embed_fn=embed_fn,
                                  projector=projector, max_positions=max_positions,
                                  state_conditioned=uses_state_conditioned_layout(cfg),
                                  audio=audio, audio_projector=audio_projector)
    collapse_msg = prior_collapse_message(records)
    if collapse_msg:
        # Same channel as the weak-calibration notes: a checkpoint that predicts the
        # class prior for every input otherwise looks like a finished, unlearnable task.
        warnings.warn(collapse_msg, RuntimeWarning, stacklevel=2)
    fitted = fit_temperature_map(records)
    calibration = calibration_report(records, fitted["temperature"])
    for name, entry in calibration.items():
        for issue in entry["issues"]:
            warnings.warn("laya.train: %s calibration: %s" % (name, issue), RuntimeWarning, stacklevel=2)

    try:
        abstention_thresholds = fit_abstention_thresholds(
            records,
            temperature=fitted["temperature"],
            temperature_by_options=fitted["temperature_by_options"],
            target_error=config.target_error,
            min_bucket_n=config.min_abstain_n,
        )
    except Exception as e:
        warnings.warn("laya.train: failed to fit abstention thresholds: %s" % (e,), RuntimeWarning, stacklevel=2)
        abstention_thresholds = {}

    # Evaluate fine-tuned checkpoint at fitted temperatures
    after_eval = None
    comparison = None
    if eval_items:
        after_records = calibration_records(model, tok, eval_items, dev, max_len, head_max_len,
                                            parallel=parallel, vision=vision, embed_fn=embed_fn,
                                            projector=projector, max_positions=max_positions,
                                            state_conditioned=uses_state_conditioned_layout(cfg),
                                            audio=audio, audio_projector=audio_projector)
        after_eval = evaluate_records(after_records, fitted["temperature"], fitted["temperature_by_options"])
        comparison = {
            "delta_accuracy": round(after_eval["accuracy"] - before_eval["accuracy"], 4),
            "delta_loss": round(after_eval["loss"] - before_eval["loss"], 4),
            "delta_ece": round(after_eval["ece"] - before_eval["ece"], 4) if (after_eval["ece"] is not None and before_eval["ece"] is not None) else None,
            "delta_brier": round(after_eval["brier"] - before_eval["brier"], 4) if (after_eval["brier"] is not None and before_eval["brier"] is not None) else None,
            "delta_brier_top1": round(after_eval["brier_top1"] - before_eval["brier_top1"], 4) if (after_eval["brier_top1"] is not None and before_eval["brier_top1"] is not None) else None,
            "delta_mean_confidence": round(after_eval["mean_confidence"] - before_eval["mean_confidence"], 4),
        }
        if eval_mode == "held_out":
            print("\n=== Evaluation (before vs after fine-tuning on %d held-out items from %s) ==="
                  % (len(eval_items), eval_source), flush=True)
        elif eval_mode == "overlapping_eval":
            print("\n=== Evaluation (before vs after fine-tuning on %d items from %s; overlaps training data) ==="
                  % (len(eval_items), eval_source), flush=True)
        else:
            print("\n=== Calibration Evidence (in-sample calibration slice, %d items; not held-out) ==="
                  % (len(eval_items)), flush=True)
        print("%-22s %-14s %-14s %-14s" % ("Metric", "Before", "After", "Delta"), flush=True)
        print("-" * 65, flush=True)
        for metric, key in [("Accuracy", "accuracy"), ("Loss", "loss"), ("ECE", "ece"),
                            ("Brier (multi-class)", "brier"), ("Brier (top-1)", "brier_top1"),
                            ("Mean Confidence", "mean_confidence")]:
            b_val = before_eval.get(key)
            a_val = after_eval.get(key)
            d_val = comparison.get("delta_" + key)
            b_str = "%.4f" % b_val if b_val is not None else "N/A"
            a_str = "%.4f" % a_val if a_val is not None else "N/A"
            d_str = ("%+.4f" % d_val) if d_val is not None else "N/A"
            print("%-22s %-14s %-14s %-14s" % (metric, b_str, a_str, d_str), flush=True)
        print("-" * 65 + "\n", flush=True)

    if eval_note is None:
        if eval_mode == "held_out":
            eval_note = "Metrics reflect independent held-out evaluation."
        elif eval_mode == "overlapping_eval":
            eval_note = ("Evaluation set contains rows that overlap training data; "
                         "metrics do not reflect strictly independent held-out evaluation.")
        elif eval_mode == "in_sample_calibration":
            eval_note = ("Metrics reflect in-sample calibration fit, not independent held-out evaluation. "
                         "Provide --eval for held-out validation.")
        else:
            eval_note = "No evaluation performed (0 calibration items and no --eval set was provided)."

    train_report = {
        "eval_source": eval_source,
        "eval_mode": eval_mode,
        "is_held_out": is_held_out,
        "eval_items": len(eval_items),
        "note": eval_note,
        "before": before_eval,
        "after": after_eval,
        "comparison": comparison,
        "calibration": calibration,
        "abstention_thresholds": abstention_thresholds,
        "training": asdict(config),
    }

    out_cfg = dict(cfg, max_len=max_len, head_max_len=head_max_len, fine_tuned=True,
                   temperature=fitted["temperature"])
    out_cfg.pop("temperature_by_options", None)
    if fitted["temperature_by_options"]:
        out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
    out_cfg["training"] = dict(out_cfg.get("training") or {}, laya_train=asdict(config),
                               laya_train_calibration=calibration,
                               train_report=train_report)
    if abstention_thresholds:
        out_cfg["training"]["abstention_thresholds"] = abstention_thresholds
    save_checkpoint(model, tok, out_cfg, output_dir)

    checkpoint_latest_dir = os.path.join(output_dir, "checkpoint_latest")
    if os.path.isdir(checkpoint_latest_dir):
        # Synchronize final calibrated config so checkpoint_latest matches the final artifact
        tmp_cfg = os.path.join(checkpoint_latest_dir, "rl_agent_config.json.tmp")
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(out_cfg, f, indent=2)
        os.replace(tmp_cfg, os.path.join(checkpoint_latest_dir, "rl_agent_config.json"))

    # Save the questions schema alongside the checkpoint for inference reuse
    sample_questions = {}
    for r in rows:
        qs = r.get("questions")
        if isinstance(qs, dict):
            for qid, qdef in qs.items():
                if qid not in sample_questions:
                    sample_questions[qid] = qdef
                elif sample_questions[qid] != qdef:
                    warnings.warn("Conflicting schema detected for question %r across rows; "
                                  "keeping first seen definition." % (qid,))
    if sample_questions:
        for d in (output_dir, checkpoint_latest_dir):
            if os.path.isdir(d):
                tmp_path = os.path.join(d, "questions.json.tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(sample_questions, f, indent=2)
                os.replace(tmp_path, os.path.join(d, "questions.json"))

    # Save train_report.json beside questions.json and checkpoint
    for d in (output_dir, checkpoint_latest_dir):
        if os.path.isdir(d):
            tmp_report = os.path.join(d, "train_report.json.tmp")
            with open(tmp_report, "w", encoding="utf-8") as f:
                json.dump(train_report, f, indent=2)
            os.replace(tmp_report, os.path.join(d, "train_report.json"))

    return {
        "train_items": len(train_items),
        "calibration_items": len(calib_items),
        "eval_items": len(eval_items),
        "skipped": skipped,
        "epoch_loss": history,
        "temperature": fitted["temperature"],
        "temperature_by_options": fitted["temperature_by_options"],
        "n_by_bucket": fitted["n_by_bucket"],
        "calibration": calibration,
        "abstention_thresholds": abstention_thresholds,
        "train_report": train_report,
        "output_dir": output_dir,
    }


def dry_run(data: str, model_dir: str, config: Optional[TrainConfig] = None) -> Dict[str, Any]:
    """Inspect dataset and build items without loading model weights; returns dataset stats."""
    from transformers import AutoTokenizer
    from .agent import _fix_tokenizer_config

    config = config or TrainConfig()
    config.validate()
    resolved_dir = resolve_checkpoint_dir(model_dir)
    _fix_tokenizer_config(resolved_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(resolved_dir, "tokenizer"))
    with open(os.path.join(resolved_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    max_len = config.max_len or cfg.get("max_len", 512)
    head_max_len = config.head_max_len or cfg.get("head_max_len", 192)

    rows = read_data(data, text_column=config.text_column, label_column=config.label_column,
                     question_id=config.question_id, instructions=config.instructions)
    items, skipped = items_from_rows(tok, rows, max_len, head_max_len,
                                     label_smoothing=config.label_smoothing)
    # The training budget is set by the post-calibration item count, so split the same way
    # finetune does rather than report on the raw valid-item count.
    train_items, calib_items = split_calibration(items, config.calib_max, config.calib_frac, config.calib_seed)
    summary = {
        "data": data,
        "rows_read": len(rows),
        "valid_items": len(items),
        "train_items": len(train_items),
        "calibration_items": len(calib_items),
        "effective_batch": config.micro_batch * config.grad_accum,
        "optimizer_updates": optimizer_updates(len(train_items), config.micro_batch,
                                               config.grad_accum, config.epochs),
        "skipped": skipped,
        "max_len": max_len,
        "head_max_len": head_max_len,
    }
    if config.eval_data:
        eval_rows = read_data(config.eval_data, text_column=config.text_column, label_column=config.label_column,
                              question_id=config.question_id, instructions=config.instructions)
        eval_items, eval_skipped = items_from_rows(tok, eval_rows, max_len, head_max_len, label_smoothing=0.0)
        summary["eval_data"] = config.eval_data
        summary["eval_rows_read"] = len(eval_rows)
        summary["eval_valid_items"] = len(eval_items)
        summary["eval_skipped"] = eval_skipped
        print("dry-run: eval %d rows read, %d valid items, skipped: %r"
              % (len(eval_rows), len(eval_items), eval_skipped), flush=True)

    print("dry-run: %d rows read, %d valid items, skipped: %r"
          % (len(rows), len(items), skipped), flush=True)
    print("dry-run: " + update_budget_message(len(train_items), config), flush=True)
    return summary
