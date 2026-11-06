"""Example 43 -- from a container to samples: channels, rate and gain.

Decodes one recording and watches each thing the engine does to it on the way to the tower: the
stereo downmix, the resample to the tower's rate, the peak normalisation. All four are the reason a
decision about a sound does not depend on how it was recorded. No checkpoint is loaded.
"""
import atexit
import os
import shutil
import tempfile

from _common import banner, heading, wav_file

banner("43", "From a container to samples", """
    A checkpoint states the sample rate its tower was trained on -- 16 000 Hz for the default
    `facebook/wav2vec2-base`, the rate Wav2Vec2's convolutional front end expects. A tower fed
    the wrong rate does not fail: it hears the sound *stretched*, because its strides are fixed
    and the number of frames it produces is a function of sample count.

    A phone recording is 44 100 or 48 000 Hz and usually stereo, so the decoder has four jobs
    before the tower sees anything: unpack the container, downmix to mono, resample to the
    tower's rate, and normalise the peak. This example does each one in order and prints the
    shape at every step.
    """)

workdir = tempfile.mkdtemp(prefix="laya_decode_")
atexit.register(shutil.rmtree, workdir, ignore_errors=True)
clip = wav_file(os.path.join(workdir, "phone.wav"), seconds=1.0, rate=44100, freq=440.0,
                channels=2)

from laya.audio import audio_info, decode_waveform, load_waveform, resample

# ---------------------------------------------------------------- 1. the header
heading("1. the header, read without decoding anything")
info = audio_info(open(clip, "rb").read())
for key in ("format", "sample_rate", "channels", "bits", "duration_seconds"):
    print("   %-16s %s" % (key, info[key]))
print("   Every one of those comes from the first 44 bytes. The same call is what enforces the")
print("   byte and duration limits in example 44, before a decoder is allowed to run.")

# ---------------------------------------------------------------- 2. decode
heading("2. decode: [channels, frames]")
samples, rate = decode_waveform(open(clip, "rb").read())
print("   decoded      : shape %s at %d Hz, dtype %s" % (tuple(samples.shape), rate, samples.dtype))
print("   so 1.0 s of stereo audio is %d frames on each of %d channels."
      % (samples.shape[1], samples.shape[0]))

# ---------------------------------------------------------------- 3. downmix
heading("3. downmix to mono")
mono = samples.mean(dim=0, keepdim=True)
print("   mean of the channels : shape %s" % (tuple(mono.shape),))
print("   Mono is not a convenience. A decision about a sound has to be the same decision")
print("   whatever channel layout the caller recorded in, so the sample count -- and therefore")
print("   the number of frames the tower sees -- must not depend on it.")

# ---------------------------------------------------------------- 4. resample
heading("4. resample to the tower's rate")
at_16k = resample(mono, rate, 16000)
print("   %d frames at %d Hz -> %d frames at 16000 Hz" % (mono.shape[1], rate, at_16k.shape[1]))
naive = mono.shape[1] / 16000.0
print("   If the tower were handed the 44 100 samples and told nothing, it would read them as")
print("   16 000 Hz audio: %.2f s of sound stretched out of %.2f s -- the same tone, a third of" % (naive, 1.0))
print("   its real pitch. That is why `sample_rate` is part of the checkpoint's config and not")
print("   something inferred from the shape.")

# ---------------------------------------------------------------- 5. normalise
heading("5. normalise: gain is not the signal")
quiet = load_waveform({"waveform": at_16k * 0.1}, sample_rate=16000)
loud = load_waveform({"waveform": at_16k}, sample_rate=16000)
print("   the same tone 20 dB quieter, after `load_waveform`:")
print("   peak quiet/samples %.3f, peak loud/samples %.3f, largest difference %.3f"
      % (float(quiet.abs().max()), float(loud.abs().max()),
         float((quiet - loud).abs().max())))
raw = load_waveform({"waveform": at_16k * 0.1}, sample_rate=16000, normalise=False)
print("   with `normalise=False`: peak %.3f -- the level survives."
      % float(raw.abs().max()))
print("   Default is `normalise=True`, because a bark recorded 20 dB quieter must not change the")
print("   answer. Set it to False when the level *is* the signal -- \"is this too loud for a")
print("   quiet carriage?\" -- and carry the loudness in the text instead.")

# ---------------------------------------------------------------- 6. the whole pipeline
heading("6. all of it in one call")
waveform = load_waveform(clip, sample_rate=16000)
print("   load_waveform(path, sample_rate=16000) -> shape %s, peak %.2f"
      % (tuple(waveform.shape), float(waveform.abs().max())))
print("   One-dimensional, in [-1, 1], at the tower's rate: exactly what the encoder takes.")

print("""
   Four steps, one function, and each of them exists because skipping it changes the answer
   rather than the error. The pipeline is also why `load_waveform` accepts a `{"waveform":
   samples}` mapping: a caller with its own decoder (a microphone, an audio pipeline, a video
   frame's soundtrack) skips the container and the codec to hand over samples at a rate it is
   making a claim about.
   """)

print("   temp files: %s" % workdir)
