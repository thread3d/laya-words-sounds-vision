"""Example 53 -- routing a picture or a sound to the checkpoint that can read it.

A deployment that serves text, pictures and recordings has to pick a checkpoint per request, and the
state's own shape is the strongest evidence about which one can answer at all. This page shows the
rule, what happens when no media checkpoint is registered, how to turn the rule off, and why one
state carries one block.

Runs with or without media checkpoints -- the routing rule is the subject, and it is pure Python.
"""
import atexit
import os
import shutil
import tempfile

from _common import (MEDIA_ENV, banner, heading, media_model, png_file, router, wav_file)

banner("53", "Routing media to the checkpoint that reads it", """
    A text checkpoint cannot see pixels or hear samples, whatever language the text beside them is
    in. So the router's media rules sit above every language rule: an image-bearing state goes to
    the checkpoint registered as the vision one, a sound-bearing state to the audio one, and only
    then does script or language detection get a say.

    The rules are opt-in in the way that matters -- they fire only when such a checkpoint is
    registered, so a Router that has none routes exactly as it always did. That is why this page
    runs whether or not you have trained one.
    """)

workdir = tempfile.mkdtemp(prefix="laya_routing_")
atexit.register(shutil.rmtree, workdir, ignore_errors=True)
picture = png_file(os.path.join(workdir, "picture.png"), width=64, height=64)
clip = wav_file(os.path.join(workdir, "clip.wav"), seconds=0.4, rate=16000)

QUESTIONS = {
    "kind": {"type": "choice", "instructions": "What is this?",
             "criteria": {"a photograph": "pixels", "a recording": "samples",
                          "a document": "text"}},
}

heading("one state, one block")
from laya.audio import split_state as split_sound
from laya.vision import split_state as split_picture

both = {"image_path": picture, "audio_path": clip, "text": "a photograph of a dog barking at a door"}
text_half, image_spec = split_picture(both)
print("   state            : %s" % ", ".join("%s=..." % key for key in both))
print("   picture split    : image %s, text half still holds %s"
      % ("yes" if image_spec else "no", sorted(k for k in text_half if k != "text")))
sound_half, sound_spec = split_sound({"audio_path": clip, "text": "a recording"})
print("   sound split      : sound %s, text half %r" % (sound_spec, sound_half))
print("   A state carries one spliced block, so a checkpoint that declares *both* modalities")
print("   refuses this state by name:")
print("       \"a state may carry a picture or a sound, not both: this one has both. Send them as")
print("       two requests, or put the one the decision is about in the state and the other in")
print("       the text.\"")
print("   With a single-modality checkpoint the extra key is not refused -- it stays in the text")
print("   half and is serialized as data. That is why the rule below routes by the state's shape")
print("   rather than relying on the model to notice.")

heading("the router, with media checkpoints registered when they exist")
media = {kind: media_model(kind) for kind in ("vision", "audio")}
registered = {kind: path for kind, path in media.items() if path}
for kind, path in media.items():
    print("   %-6s checkpoint : %s" % (kind, path or "none (%s unset)" % MEDIA_ENV[kind]))
r = router(models=registered) if registered else router()
print("   Router(models=%s), vision_model=%r, audio_model=%r"
      % (sorted(registered) or "{}", r.vision_model, r.audio_model))

STATES = [
    ("an image state", {"image_path": picture, "text": "a photograph"}),
    ("a sound state", {"audio_path": clip, "text": "a recording"}),
    ("a text state", {"body": "The invoice was charged twice. Please refund the duplicate."}),
    ("German text", {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung."}),
]
for label, state in STATES:
    decision = r.route(state, QUESTIONS)
    print("   %-14s -> %-10s %s" % (label, decision.model, decision.reason))

heading("turning the rule off")
plain = router(models=registered, vision_model=None, audio_model=None) if registered else None
if plain is not None:
    decision = plain.route({"image_path": picture, "text": "a photograph"}, QUESTIONS)
    print("   vision_model=None, audio_model=None -> %s (%s)" % (decision.model, decision.reason))
    print("   A deployment that wants media states routed by their *text* -- because it resolves")
    print("   the pixels itself, or because the same checkpoint is supposed to answer -- turns the")
    print("   rules off rather than unregistering the checkpoints.")
else:
    print("   `Router(vision_model=None, audio_model=None)` -- with no media checkpoint registered")
    print("   the rules cannot fire in the first place, which is the same outcome by another route.")

heading("what does not change")
print("   `model=` and `task=` still win: they name a checkpoint, and a media rule names which")
print("   checkpoint *could* answer. `lang=` and `lang_guess=` name a *language*, so they never")
print("   override the image or sound rule -- there is no point routing an image-bearing state to")
print("   the German text model. To force that, say `model=\"multilingual\"` explicitly.")
print("   Everything else is example 20-24: script, detected language, the default checkpoint, and")
print("   the `routing` block every answer carries.")

print("""
   The shape of a state is the cheapest routing signal there is -- `has_image` and `has_audio`
   inspect a dictionary's keys and nothing else -- and it is also the most reliable, because it
   answers a question about capability rather than about content. A text checkpoint cannot see a
   picture at any confidence; it simply cannot see one.

   That is the end of the media path. From here the rest is the same engine: hooks and caching
   (hooks/index.md), the HTTP API and its per-request limits (http-api.md), and the evaluation
   harness that runs these gates in CI (evals.md).
   """)
