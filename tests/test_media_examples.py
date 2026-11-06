"""The media learning path (examples 42-53) has to stay true to itself.

Four things can drift silently, and each of them costs a reader rather than this suite:

* an example exists but the learning path never mentions it, so nobody finds it;
* the path promises N scripts and the folder holds a different N;
* `_common.media_model` hands a *text* checkpoint to a media example, which then fails inside a
  forward pass instead of saying "that is not a vision checkpoint";
* the synthetic containers the no-checkpoint examples write stop being the containers the engine's
  own readers accept -- the whole premise of 42-44 and 48-49 is that a hand-written WAV and PNG are
  the same bytes a microphone and a camera produce.

Everything here runs without a checkpoint and without the Hub, so it belongs in CI. The layout
claims are pinned against `visual_plan`/`audio_plan` themselves with a stub tokenizer, which is the
same way `tests/test_vision.py` and `tests/test_audio.py` size a plan.

Run directly: `python tests/test_media_examples.py`
"""
import json
import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

PASS, FAIL, SKIPPED = [], [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, ok, detail=""):
    if ok:
        PASS.append(name)
    else:
        FAIL.append("%s: %s" % (name, detail))


class _Skip(Exception):
    pass


def _examples():
    """`[(number, basename)]` for every numbered example, in order."""
    found = []
    for name in sorted(os.listdir(os.path.join(REPO, "examples"))):
        match = re.match(r"^(\d\d)_[a-z0-9_]+\.py$", name)
        if match:
            found.append((int(match.group(1)), name))
    return found


def _read(*parts):
    with open(os.path.join(REPO, *parts), encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------------------------
# The path is complete: numbering, and both READMEs naming every page
# ---------------------------------------------------------------------------------------------

def test_every_example_number_appears_exactly_once():
    numbers = [number for number, _ in _examples()]
    check("the numbering has no gaps", numbers, list(range(1, len(numbers) + 1)))


def test_the_media_stage_is_contiguous_and_last():
    """42-53 are the media stage: a gap or an interleaved page would break the stage ranges."""
    numbers = [number for number, _ in _examples()]
    tail = [n for n in numbers if n >= 42]
    check("stage 9 and 10 are one run at the end", tail, list(range(42, 42 + len(tail))))
    check_true("stage 9 opens at 42", tail and tail[0] == 42, tail[:3])


def test_the_examples_readme_names_every_example():
    page = _read("examples", "README.md")
    missing = [name for _, name in _examples() if name not in page]
    check("every example file is named in examples/README.md", missing, [])


def test_the_examples_readme_names_the_media_stages():
    page = _read("examples", "README.md")
    check_true("stage 9 is titled", "Stage 9 — a decision about a sound (42-47)" in page,
               "no '## Stage 9 — a decision about a sound (42-47)' heading")
    check_true("stage 10 is titled", "Stage 10 — a decision about a picture (48-53)" in page,
               "no '## Stage 10 — a decision about a picture (48-53)' heading")
    check_true("the path says how many scripts it has", "53 runnable scripts" in page,
               "examples/README.md must state the script count in the same words")
    media_numbers = tuple(str(n) for n in range(42, 54))
    rows = [line for line in page.splitlines()
            if line.startswith("|") and any("| %s |" % n in line for n in media_numbers)]
    check_true("the concept index covers every media page", len(rows) >= 12, rows[:2])


def test_the_readme_agrees_on_the_count():
    count = len(_examples())
    root = _read("README.md")
    check_true("README.md states the same number of examples",
               "%d runnable examples" % count in root or "%d-script learning path" % count in root,
               "README.md does not say %d" % count)
    check_true("README.md's run_all line agrees",
               "all %d examples" % count in root, "README.md's run_all line is stale")
    check_true("README.md's stage table covers the media stages",
               "| 9. `42`-`47` |" in root and "| 10. `48`-`53` |" in root,
               "README.md's stage table has no media rows")


def test_run_all_picks_the_media_examples_up():
    script = _read("examples", "run_all.sh")
    check_true("run_all.sh globs any two-digit example",
               "[0-9][0-9]_*.py" in script, "run_all.sh no longer matches examples/4x and 5x")


# ---------------------------------------------------------------------------------------------
# `media_model` refuses a checkpoint that cannot read the modality
# ---------------------------------------------------------------------------------------------

def test_media_model_ignores_a_text_checkpoint():
    """The failure it prevents: a media example opening `models/laya` and dying inside a forward."""
    sys.path.insert(0, os.path.join(REPO, "examples"))
    import _common

    with tempfile.TemporaryDirectory() as tmp:
        text_ckpt = os.path.join(tmp, "text")
        os.makedirs(text_ckpt)
        with open(os.path.join(text_ckpt, "rl_agent_config.json"), "w", encoding="utf-8") as handle:
            json.dump({"max_len": 512}, handle)

        saved = dict(os.environ)
        # Both names point at the text checkpoint: a named checkpoint is the answer, so neither
        # modality may fall back to some other directory on this machine.
        for kind in ("vision", "audio"):
            os.environ[_common.MEDIA_ENV[kind]] = text_ckpt
        try:
            check("a text checkpoint is not a vision checkpoint",
                  _common.media_model("vision"), None)
            check("a text checkpoint is not an audio checkpoint",
                  _common.media_model("audio"), None)

            media_ckpt = os.path.join(tmp, "media")
            os.makedirs(media_ckpt)
            with open(os.path.join(media_ckpt, "rl_agent_config.json"), "w", encoding="utf-8") as handle:
                json.dump({"vision": {"n_visual_tokens": 16}}, handle)
            os.environ[_common.MEDIA_ENV["vision"]] = media_ckpt
            check("a checkpoint that declares the block is found",
                  _common.media_model("vision"), media_ckpt)
            check("the audio name still resolves to nothing",
                  _common.media_model("audio"), None)

            with open(os.path.join(media_ckpt, "rl_agent_config.json"), "w", encoding="utf-8") as handle:
                json.dump({"vision": {"enabled": False}}, handle)
            check("an explicitly disabled block is not found",
                  _common.media_model("vision"), None)
        finally:
            os.environ.clear()
            os.environ.update(saved)


# ---------------------------------------------------------------------------------------------
# The synthetic containers are real containers
# ---------------------------------------------------------------------------------------------

def test_the_hand_written_wav_is_the_wav_the_engine_reads():
    sys.path.insert(0, os.path.join(REPO, "examples"))
    import _common
    from laya.audio import audio_info, check_audio_bytes, decode_waveform

    with tempfile.TemporaryDirectory() as tmp:
        path = _common.wav_file(os.path.join(tmp, "tone.wav"), seconds=0.25, rate=16000, freq=440.0)
        data = open(path, "rb").read()
        info = check_audio_bytes(data)
        check("the header states the rate written", info["sample_rate"], 16000)
        check("the header states one channel", info["channels"], 1)
        check("the header states the duration", round(info["duration_seconds"], 3), 0.25)
        waveform, rate = decode_waveform(data)
        check("the payload decodes to the frame count written", rate, 16000)
        check("one channel of 0.25 s", tuple(waveform.shape), (1, 4000))

        stereo = _common.wav_file(os.path.join(tmp, "stereo.wav"), seconds=0.1, rate=44100,
                                  channels=2)
        check("a stereo file is measured as stereo", audio_info(open(stereo, "rb").read())["channels"], 2)


def test_the_hand_written_png_is_the_png_the_engine_reads():
    sys.path.insert(0, os.path.join(REPO, "examples"))
    import _common
    from laya.vision import check_image_bytes, image_info, load_pixels

    with tempfile.TemporaryDirectory() as tmp:
        path = _common.png_file(os.path.join(tmp, "picture.png"), width=48, height=32)
        data = open(path, "rb").read()
        info = check_image_bytes(data)
        check("the header states the format written", info["format"], "png")
        check("the header states the width written", info["width"], 48)
        check("the header states the height written", info["height"], 32)
        check("image_info reads the header without a decoder", image_info(data)["width"], 48)
        pixels = load_pixels(path, image_size=224)
        check("the decoder produces the tensor the tower takes", tuple(pixels.shape), (3, 224, 224))
        check_true("the pixels are normalised into [0, 1]",
                   float(pixels.min()) >= 0.0 and float(pixels.max()) <= 1.0,
                   (float(pixels.min()), float(pixels.max())))


# ---------------------------------------------------------------------------------------------
# The layout claims the two block pages print
# ---------------------------------------------------------------------------------------------

class _FakeTok:
    """Enough of a tokenizer for `build_head`: one id per word, and the four special ids."""
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 0, 1, 4, 2
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (len(word) % 90) for word in text.split() if word]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _plan_inputs(option_count, head_max_len=192):
    from laya.common import build_head

    tok = _FakeTok()
    question = {"t": "choice", "ins": "pick one",
                "crit": {"option %d" % i: "description %d" % i for i in range(option_count)}}
    ids, markers, _stats = build_head(tok, question, head_max_len)
    return ids, markers


def test_the_block_is_spliced_after_the_head_and_the_markers_never_move():
    from laya.audio import audio_plan, parse_audio
    from laya.vision import parse_vision, visual_plan

    ids, markers = _plan_inputs(4)
    option_tokens = len(ids) - markers[0]
    for name, cfg, plan in (
            ("vision", parse_vision({"vision": {"n_visual_tokens": 16}}), visual_plan),
            ("audio", parse_audio({"audio": {"n_audio_tokens": 32}}), audio_plan)):
        planned = plan(cfg, text_len=len(ids) + 30, markers=markers, head_len=len(ids),
                        max_len=512, option_tokens=option_tokens)
        check("%s: splice at the end of the head" % name, planned.splice, len(ids))
        check("%s: the markers are untouched" % name, list(planned.marker_positions), list(markers))
        check("%s: the block is charged to the sequence" % name, planned.length, len(ids) + 30 + planned.n_block)


def test_the_summary_row_appears_only_past_the_threshold():
    from laya.audio import audio_plan, parse_audio
    from laya.vision import parse_vision, visual_plan

    wide_ids, wide_markers = _plan_inputs(40, head_max_len=512)
    wide_tokens = len(wide_ids) - wide_markers[0]
    for name, cfg, plan, width in (
            ("vision", parse_vision({"vision": {"n_visual_tokens": 16}}), visual_plan, 16),
            ("audio", parse_audio({"audio": {"n_audio_tokens": 32}}), audio_plan, 32)):
        check_true("%s: the wide head crosses the threshold" % name,
                   wide_tokens > cfg.summary_for_options_over, (wide_tokens, cfg.summary_for_options_over))
        wide = plan(cfg, text_len=len(wide_ids) + 10, markers=wide_markers,
                    head_len=len(wide_ids), max_len=1024, option_tokens=wide_tokens)
        check("%s: the summary row is what the block gained" % name, wide.n_block, width + 1)
        check("%s: the plan says so" % name, wide.summary, True)


def test_the_splice_puts_the_block_at_exactly_the_planned_index():
    import torch

    from laya.vision import assemble_inputs_embeds, parse_vision, visual_plan

    ids, markers = _plan_inputs(4)
    text_len = len(ids) + 20
    cfg = parse_vision({"vision": {"n_visual_tokens": 8}})
    plan = visual_plan(cfg, text_len=text_len, markers=markers, head_len=len(ids),
                       max_len=512, option_tokens=len(ids) - markers[0])
    text = torch.zeros(1, text_len, 4)
    block = torch.ones(1, plan.n_visual, 4)
    merged = assemble_inputs_embeds(text, block, plan)
    check("the merge is text plus block", tuple(merged.shape), (1, text_len + plan.n_visual, 4))
    column = merged[0, :, 0].tolist()
    check("the block starts at `splice`", column.index(1.0), plan.splice)
    check("every visual row is there", column.count(1.0), plan.n_visual)
    check("no text row was lost", column.count(0.0), text_len)


def test_a_state_carries_one_block_and_the_other_key_becomes_text():
    """Example 53's claim: the split takes one input, the other key stays as data."""
    from laya.audio import split_state as split_sound
    from laya.vision import split_state as split_picture

    both = {"image_path": "a.png", "audio_path": "b.wav", "text": "both"}
    text_half, image_spec = split_picture(both)
    check("the picture is taken", image_spec, "a.png")
    check("the sound key stays in the text half", "audio_path" in text_half, True)
    text_half, sound_spec = split_sound({"audio_path": "b.wav", "text": "only a sound"})
    check("the sound is taken", sound_spec, "b.wav")
    check("with nothing beside it, the text half is the sentence", text_half, "only a sound")


# ---------------------------------------------------------------------------------------------
# The pages that need no checkpoint must not need one
# ---------------------------------------------------------------------------------------------

WEIGHTLESS = (
    "42_sound_in_the_state.py",
    "43_decoding_and_sample_rate.py",
    "44_sound_transport_limits.py",
    "45_sound_block_layout.py",
    "48_image_in_the_state.py",
    "49_image_transport_limits.py",
    "50_vision_block_layout.py",
    "53_routing_media.py",
)
NEEDS_MEDIA = (
    "46_sound_decision.py",
    "47_sound_necessity_gate.py",
    "51_vision_decision.py",
    "52_vision_gates.py",
)


def test_the_weightless_pages_load_no_checkpoint():
    """42-45, 48-50 and 53 are the part of the path a reader with no media checkpoint can run.

    A `Router` is fine here: constructing one loads nothing, and `route()` answers from the state's
    keys, which is the point of example 53.
    """
    offenders = []
    for name in WEIGHTLESS:
        source = _read("examples", name)
        for needle in ("media_agent(", ".predict(", "laya.load(", "= load("):
            if needle in source:
                offenders.append("%s uses %s" % (name, needle.strip()))
    check("the weightless pages never open a checkpoint", offenders, [])


def test_the_checkpoint_pages_say_how_to_make_one():
    missing = []
    for name in NEEDS_MEDIA:
        source = _read("examples", name)
        if "media_agent(" not in source:
            missing.append("%s does not call media_agent" % name)
        if "laya-train" not in _read("examples", "_common.py"):
            missing.append("_common.py has no laya-train command to print")
    check("every media page routes its checkpoint through media_agent", missing, [])


def test_media_agent_exits_zero_with_a_command_when_there_is_none():
    """A skip has to be a pass for `run_all.sh`, and has to say what would make it run."""
    source = _read("examples", "_common.py")
    check_true("the skip prints the training command", "TRAIN[kind]" in source, "no TRAIN hint")
    check_true("the skip exits 0", "raise SystemExit(0)" in source, "a skip that fails the sweep")


# ---------------------------------------------------------------------------------------------

TESTS = [
    test_every_example_number_appears_exactly_once,
    test_the_media_stage_is_contiguous_and_last,
    test_the_examples_readme_names_every_example,
    test_the_examples_readme_names_the_media_stages,
    test_the_readme_agrees_on_the_count,
    test_run_all_picks_the_media_examples_up,
    test_media_model_ignores_a_text_checkpoint,
    test_the_hand_written_wav_is_the_wav_the_engine_reads,
    test_the_hand_written_png_is_the_png_the_engine_reads,
    test_the_block_is_spliced_after_the_head_and_the_markers_never_move,
    test_the_summary_row_appears_only_past_the_threshold,
    test_the_splice_puts_the_block_at_exactly_the_planned_index,
    test_a_state_carries_one_block_and_the_other_key_becomes_text,
    test_the_weightless_pages_load_no_checkpoint,
    test_the_checkpoint_pages_say_how_to_make_one,
    test_media_agent_exits_zero_with_a_command_when_there_is_none,
]

if __name__ == "__main__":
    for fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            FAIL.append("%s: %s" % (fn.__name__, exc))
        except _Skip:
            pass

    print("\n%d passed, %d failed, %d skipped" % (len(PASS), len(FAIL), len(SKIPPED)))
    for failure in FAIL:
        print("  FAIL " + failure)
    for skipped in SKIPPED:
        print("  SKIP " + skipped)
    sys.exit(1 if FAIL else 0)
