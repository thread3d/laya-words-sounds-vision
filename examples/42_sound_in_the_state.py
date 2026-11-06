"""Example 42 -- a state can carry a sound.

Writes a PCM WAV with the standard library, puts it in a state, and takes the state apart again
with `laya.audio.split_state`: the four accepted keys, what happens to the text beside them, and
why a recording with no text at all is a legal state. No checkpoint is loaded.
"""
import atexit
import os
import shutil
import tempfile

from _common import banner, heading, wav_file

banner("42", "A sound in the state", """
    A decision about a sound is a state that carries a recording, the way a decision about a
    document is a state that carries text. The engine never transcribes it: a frozen audio
    tower turns the waveform into rows, and the decision head learns which qualities of those
    rows the options are asking about. That is what keeps the call one forward pass.

    This page is the state contract, and it needs no checkpoint. Everything here is the same
    whether the answer comes from a checkpoint you trained or from one that does not exist
    yet -- which is why the sound path can be learned before anything is trained.
    """)

workdir = tempfile.mkdtemp(prefix="laya_sound_")
atexit.register(shutil.rmtree, workdir, ignore_errors=True)
clip = wav_file(os.path.join(workdir, "tone.wav"), seconds=0.5, rate=16000, freq=440.0)
size = os.path.getsize(clip)

heading("the recording, written by this example")
print("   %s" % clip)
print("   %d bytes: 0.5 s of 440 Hz at 16 kHz, mono, 16-bit -- the rate the default tower" % size)
print("   (`facebook/wav2vec2-base`) was trained on, so nothing has to be resampled.")

# ---------------------------------------------------------------- the four accepted keys
from laya.audio import AUDIO_KEYS, split_state

heading("the four keys a state may use")
print("   %s" % ", ".join("`%s`" % key for key in AUDIO_KEYS))
print("   `audio_path`   a path on this machine -- the model reads it in-process")
print("   `audio`        the container bytes, or the base64 of them")
print("   `audio_base64` the same thing, named for what it is")
print("   `audio_waveform` samples you already decoded -- skips the codec and the container")
print("   `audio_url` is *not* in that list: a URL is something the caller resolves, because a")
print("   self-hosted engine that fetches whatever URL a request names is an egress proxy pointed")
print("   at the operator's own network (example 44).")

# ---------------------------------------------------------------- the text half
heading("split_state: the recording, and everything beside it")
state = {"audio_path": clip, "text": "is this a steady tone or a spoken word?",
         "vendor": "acme", "campaign": "spring"}
text_half, spec = split_state(state)
print("   state      : %s" % ", ".join("%s=..." % key for key in state))
print("   text half  : %r" % (text_half,))
print("   sound spec : %r" % (spec,))
print("   `vendor` and `campaign` stay in the text half: a key the sound path does not own is")
print("   still data the model should see. Only the recording's own key and `text` move out.")

heading("a recording with no text at all is a legal state")
only_sound, spec = split_state({"audio_path": clip})
print("   %r -> text half %r, sound spec %r" % ({"audio_path": clip}, only_sound, spec))
print("   the text half is empty on purpose: the recording *is* the evidence, and asking a")
print("   question about it does not need a sentence pretending to describe it.")

heading("the same recording, named three ways")
import base64
import torch

with open(clip, "rb") as handle:
    raw = handle.read()
as_bytes = split_state({"audio": raw})[0]
as_base64 = split_state({"audio_base64": base64.b64encode(raw).decode("ascii")})[0]
as_samples = split_state({"audio_waveform": torch.zeros(8000)})[0]
print("   `audio` bytes       -> text half %r" % (as_bytes,))
print("   `audio_base64`      -> text half %r" % (as_base64,))
print("   `audio_waveform`    -> text half %r" % (as_samples,))
print("   All three hand back an empty text half, because the *state* is the recording. What")
print("   differs is where the samples come from: the last one has already been decoded by you,")
print("   so `laya.audio.load_waveform` returns it without touching a decoder at all.")

print("""
   The state says what the evidence is; the question says what to do with it. A checkpoint
   trained with `--audio` reads the recording and ignores the four keys' spelling -- path,
   bytes, base64 and waveform all arrive at the tower as samples. Example 43 is what those
   samples look like once the container is stripped off, example 44 is what the engine refuses
   before it decodes anything, and example 46 is the decision itself.
   """)

print("   temp files: %s" % workdir)
