"""Shared helpers for the Laya examples.

Each example is a standalone script:

    .venv/bin/python examples/02_hello_world_decision.py

Run it from anywhere -- paths resolve from this file, and Python puts the examples/ directory
on `sys.path` automatically, so `from _common import ...` just works.

Checkpoints come from `models/` when a local copy is there, and from the Hub bundle
(`convaiinnovations/laya`, with a subfolder per checkpoint) otherwise, so the examples run after a
plain `pip install laya`. Loading one takes a few seconds -- plus the first download when there is
no local copy -- and the first MPS call pays Metal kernel compilation, so anything that reports
latency warms up first.

A checkpoint that reads a picture or a recording is different: the bundle carries none, because one
is *trained* (`laya-train --vision` / `--audio`). Those examples find one at `LAYA_VISION_MODEL` /
`LAYA_AUDIO_MODEL`, then `models/laya-vision` and `models/laya-audio`, and `media_agent` prints the
command that would make one and exits 0 when there is none -- so `run_all.sh` stays green on a
machine that has never trained a media checkpoint.

This module also imports `laya` for you -- after setting `USE_TF=0`, because a stray TensorFlow
install can deadlock model loading. Examples therefore do `from _common import laya, ...` rather
than importing it themselves first, which would defeat the guard.
"""
import os
import statistics
import sys
import time

os.environ.setdefault("USE_TF", "0")          # torch-only: a stray TF install can deadlock loading
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)          # repository root; <root>/laya/email.py must not shadow stdlib email

import laya  # noqa: E402
from laya.router import Router  # noqa: E402

BUNDLE_REPO = "convaiinnovations/laya"
_SUBFOLDER = {
    "english": None,                 # the bundle's root holds the English checkpoint
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}

LOCAL_MODELS = {
    "english": os.path.join(ROOT, "models", "laya"),
    "multilingual": os.path.join(ROOT, "models", "laya-multilingual"),
    "typed-decisions": os.path.join(ROOT, "models", "laya-typed-decisions"),
}


def has_local(name="english"):
    """True when a local copy of this checkpoint is in `models/`."""
    return os.path.exists(os.path.join(LOCAL_MODELS[name], "model.safetensors"))


def checkpoint(name="english"):
    """A Router model spec: the local directory when there is one, else the Hub repo + subfolder."""
    if has_local(name):
        return LOCAL_MODELS[name]
    return (BUNDLE_REPO, _SUBFOLDER[name])


# What `Router(models=...)` takes: absolute local paths in a checkout that has `models/`,
# Hub specs otherwise.
MODELS = {name: checkpoint(name) for name in LOCAL_MODELS}

# ---------------------------------------------------------------- sample inputs
STATE_EN = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today "
            "or we will cancel our plan.",
}
STATE_HI = {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}
STATE_DE = {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung."}

# the quickstart question set: every primitive, one forward pass
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts",
                     "other": "everything else"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}


def banner(number, title, blurb=""):
    """Print the header every example starts with."""
    line = "=" * 74
    print("\n%s\n  Example %s -- %s\n%s" % (line, number, title, line))
    if blurb:
        for para in blurb.strip().split("\n"):
            print("  " + para.strip())
        print()


def load(name="english", device=None):
    """Load one checkpoint: from `models/` when it is there, else from the Hub.

    `device=None` lets Laya pick: CUDA, then MPS, then CPU.
    """
    if has_local(name):
        return laya.load(LOCAL_MODELS[name], device=device)
    return laya.load(BUNDLE_REPO, subfolder=_SUBFOLDER[name], device=device)


def router(device=None, **kwargs):
    """A Router over the local checkpoints when they are there, else over the Hub bundle.

    A `models=` argument is *added* to the three published checkpoints rather than replacing them,
    so an example can register a checkpoint of its own -- the media ones in examples 46-53 -- and
    still route text to `english` and `multilingual`.
    """
    models = dict(MODELS)
    models.update(kwargs.pop("models", None) or {})
    return Router(models=models, device=device, **kwargs)


def timed(fn, repeat=3):
    """(result, median_ms) for a call, after one untimed warm-up."""
    fn()
    samples = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return result, statistics.median(samples)


def top(probabilities, n=3):
    """The n highest-probability labels, as a printable string."""
    items = sorted(probabilities.items(), key=lambda kv: -kv[1])[:n]
    return "  ".join("%s=%.3f" % (k, v) for k, v in items)


def _answer_conf(a):
    """The gate number as text, or `n/a` for a payload that predates `answer_confidence`."""
    value = a.get("answer_confidence")
    return "%.3f" % value if isinstance(value, (int, float)) else "n/a"


def describe(answers, indent="   "):
    """One readable line per question, whatever primitive it used.

    Both confidence fields are printed: `conf` is `confidence`, the entropy value, and
    `answer_conf` is `answer_confidence`, max(p) -- the number to gate on (see the
    README's threshold section and example 18). Printing only the first made every
    example that uses this helper show a number the README warns against carrying a
    threshold over from.
    """
    for qid, a in answers.items():
        if a["type"] == "choice":
            best = max(a["probabilities"].items(), key=lambda kv: kv[1])
            detail = "%s (p=%.3f, conf=%.3f, answer_conf=%s)" % (
                best[0], best[1], a["confidence"], _answer_conf(a))
            if len(a["probabilities"]) > 2:
                detail += "   next: %s" % top({k: v for k, v in a["probabilities"].items()
                                               if k != best[0]}, 2)
        elif a["type"] == "score":
            levels = len(a["probabilities"]) - 1
            detail = "%.2f / %d (conf=%.3f, answer_conf=%s)" % (
                a["score"], levels, a["confidence"], _answer_conf(a))
        else:
            detail = "%.3f (%s)  conf=%.3f, answer_conf=%s" % (
                a["noul"], "true" if a["noul"] > 0.5 else "false",
                a["confidence"], _answer_conf(a))
        print("%s%-18s %-7s %s" % (indent, qid, a["type"], detail))


def heading(text):
    print("\n   -- %s --" % text)


def device_line(agent):
    print("   device: %s   (torch %s)" % (agent.device, __import__("torch").__version__))


# ---------------------------------------------------------------- media: a picture or a sound
#
# The Hub bundle is three *text* checkpoints. A checkpoint learns to read a picture or a recording
# when somebody trains it -- `laya-train --vision` / `--audio`, see docs/finetune.md -- so a media
# example either works on what needs no weights at all (the state shapes, the transport limits, the
# layout arithmetic, all pure Python) or opens a media checkpoint the reader has, and says how to
# make one when there is none. The split is deliberate: a reader who has never trained a checkpoint
# should still be able to run most of this path.

MEDIA = ("vision", "audio")

MEDIA_ENV = {"vision": "LAYA_VISION_MODEL", "audio": "LAYA_AUDIO_MODEL"}
MEDIA_LOCAL = {
    "vision": os.path.join(ROOT, "models", "laya-vision"),
    "audio": os.path.join(ROOT, "models", "laya-audio"),
}
# The benchmark checkpoints this repository's own research produced, when a checkout still has them
# (they are gitignored): the CIFAR-20 state-conditioned run and the FSDD run behind
# docs/vision-benchmark-results.md and docs/audio-benchmark-results.md. They are here so the
# examples that need a media checkpoint run in the checkout that measured the numbers.
MEDIA_BENCH = {
    "vision": os.path.join(ROOT, ".vision-cifar100", "ckpt20_sc_2k_10ep"),
    "audio": os.path.join(ROOT, ".audio-bench", "checkpoints", "fsdd-sc"),
}
FSDD_DIR = os.path.join(ROOT, ".audio-bench", "data", "free-spoken-digit-dataset-master",
                        "recordings")
# The sentence the FSDD benchmark rows carry beside their recording. It is part of the state, so
# a checkpoint trained with it is being asked a different question when it is reworded --
# example 46 measures that.
DIGIT_TEXT = "a recording of someone saying a digit"
SAMPLE_IMAGE = os.path.join(ROOT, "assets", "laya_vs_jev.png")


def checkpoint_config(path):
    """A checkpoint directory's config, whichever file the trainer wrote it to."""
    for name in ("rl_agent_config.json", "config.json"):
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as handle:
                return __import__("json").load(handle)
    return {}


def media_model(kind):
    """A media checkpoint for `kind` ('vision' or 'audio'), or None when there is none.

    The documented environment variable, when it is set, *is* the answer: a caller who names a
    checkpoint gets that one or gets nothing, not some other directory silently. Unset, the search
    is `models/` and then the benchmark checkpoints a research checkout still holds. A directory
    only counts when its config actually carries the block -- pointing `LAYA_VISION_MODEL` at a text
    checkpoint says "that is not a vision checkpoint" here, rather than failing later inside a
    forward pass.
    """
    named = os.environ.get(MEDIA_ENV[kind])
    candidates = [named] if named else [MEDIA_LOCAL[kind], MEDIA_BENCH[kind]]
    for candidate in candidates:
        if not candidate or not os.path.isdir(candidate):
            continue
        cfg = checkpoint_config(candidate)
        if cfg.get(kind) and cfg[kind].get("enabled") is not False:
            return candidate
    return None


TRAIN = {
    "vision": ("laya-train --data items.jsonl --base english --out ./my-vision-checkpoint \\\n"
               "    --vision '{\"n_visual_tokens\": 16}' --option-layout state_conditioned"),
    "audio": ("laya-train --data items.jsonl --base english --out ./my-audio-checkpoint \\\n"
              "    --audio '{\"n_audio_tokens\": 32}' --option-layout state_conditioned"),
}


def media_agent(kind, number):
    """Open a media checkpoint for one example, or print the skip and exit 0.

    Returns `(agent, questions, path)`. `questions` comes from the checkpoint's own
    `questions.json`: a checkpoint answers the questions it was trained on, and the trainer writes
    them beside the weights for exactly this reason.
    """
    path = media_model(kind)
    if path is None:
        print("\n   SKIP -- no %s checkpoint on this machine, so there is nothing to answer with." % kind)
        print("   No published checkpoint reads a %s; one is trained:" % (
            "picture" if kind == "vision" else "recording"))
        for line in TRAIN[kind].split("\n"):
            print("       " + line)
        print("   then point %s at the output directory (docs/finetune.md)." % MEDIA_ENV[kind])
        raise SystemExit(0)
    import json

    print("   checkpoint: %s" % path)
    questions_path = os.path.join(path, "questions.json")
    if os.path.exists(questions_path):
        with open(questions_path, encoding="utf-8") as handle:
            questions = json.load(handle)
    else:
        questions = QUESTIONS
    return laya.load(path), questions, path


def tokenizer(name="english"):
    """The checkpoint's tokenizer, without loading its weights.

    The layout examples need `[MASK]` positions and a head length, and those come from the
    tokenizer alone -- `laya.common.build_head(tok, q, head_max_len)` is documented as the way to
    measure a question before paying for a forward pass. Local copy first, then the Hub bundle.
    """
    from transformers import AutoTokenizer

    local = os.path.join(LOCAL_MODELS["english"], "tokenizer")
    if os.path.isdir(local):
        return AutoTokenizer.from_pretrained(local)
    return AutoTokenizer.from_pretrained(BUNDLE_REPO, subfolder="tokenizer")


def head_for(tok, question, head_max_len=192):
    """`build_head`'s `(ids, markers, stats)` for a question in the public shape.

    `build_head` reads the short-key internal shape (`t` / `ins` / `crit`); the three-line mapping
    below is the one docs/reference/helpers.md shows for measuring a question by hand, and doing it
    here keeps it identical in every layout example.
    """
    from laya.common import build_head

    q = {"t": question["type"], "ins": question["instructions"], "crit": question.get("criteria")}
    return build_head(tok, q, head_max_len)


def wav_file(path, seconds=0.5, rate=16000, freq=440.0, channels=1):
    """Write a PCM WAV with the standard library and return its path.

    Nothing here is a fixture: the bytes are a real container that the decoder, the header reader
    and the transport limits all read the same way they read a microphone recording. A tone is used
    because a decision about *what the sound is* needs a trained checkpoint, while a decision about
    the *container* does not.
    """
    import array
    import math
    import wave

    samples = array.array("h", (int(12000 * math.sin(2 * math.pi * freq * i / rate))
                                for i in range(int(seconds * rate))))
    if channels > 1:
        wide = array.array("h", bytes(0))
        for sample in samples:
            wide.extend([sample] * channels)
        samples = wide
    with wave.open(path, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


def png_file(path, width=64, height=64):
    """Write a PNG with `zlib` and `struct` and return its path.

    Three chunks -- IHDR, IDAT, IEND -- because that is the whole format for an RGB image, and
    writing it by hand is what makes the transport limits legible: the header states the dimensions
    before a single pixel is decoded, which is exactly what `laya.vision.check_image_bytes` checks.
    """
    import struct
    import zlib

    raw = bytearray()
    for y in range(height):
        raw.append(0)                                   # filter type 0, "none", per scanline
        for x in range(width):
            raw += bytes(((x * 4) % 256, (y * 4) % 256, 128))

    def chunk(tag, payload):
        body = tag + payload
        return (struct.pack(">I", len(payload)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)   # 8-bit, colour type 2 (RGB)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))
    return path


def sample_clip(directory):
    """A recording to run the sound examples on: a real one if this checkout has FSDD, else a tone.

    Returns `(path, description)`. FSDD's filename states the label (`0_jackson_0.wav` is a spoken
    zero) and the recordings are 8 kHz mono, which is the rate the default tower was trained on --
    so the example prints a correct answer rather than the mechanics alone.
    """
    if os.path.isdir(FSDD_DIR):
        names = sorted(name for name in os.listdir(FSDD_DIR) if name.endswith(".wav"))
        if names:
            name = names[0]
            return os.path.join(FSDD_DIR, name), "FSDD %s (label from the filename: %s)" % (
                name, name.split("_")[0])
    return wav_file(os.path.join(directory, "tone.wav")), "a 440 Hz tone written by this example"


def labelled_clips(count=24):
    """`[(path, label)]` of real labelled recordings, or `[]` when this checkout has no FSDD.

    The label is the leading digit of the filename, which is how FSDD ships. The gate examples need
    ground truth, and inventing it from a tone would measure nothing.
    """
    if not os.path.isdir(FSDD_DIR):
        return []
    names = sorted(name for name in os.listdir(FSDD_DIR) if name.endswith(".wav"))
    per_digit = {}
    for name in names:
        per_digit.setdefault(name.split("_")[0], []).append(name)
    out = []
    for digit in sorted(per_digit):
        for name in per_digit[digit][:max(1, count // max(1, len(per_digit)))]:
            out.append((os.path.join(FSDD_DIR, name), digit))
    return out[:count]
