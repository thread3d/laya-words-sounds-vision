"""Example 44 -- what the engine refuses before it decodes anything.

A recording costs two things and only one of them is bytes: an hour of 48 kHz stereo is 690 MB of
samples inside a 12 MB compressed file. Both are answered from the container header, before any
decoder runs. This example trips each limit and prints the refusal, and shows why a URL is refused
outright. No checkpoint is loaded.
"""
import atexit
import os
import shutil
import tempfile

from _common import banner, heading, wav_file

banner("44", "Two costs, one header", """
    An upload that fits in a request body is not the same as a recording that fits in memory.
    48 kHz stereo, 16-bit, is 192 000 bytes per second: an hour of it is 691 200 000 bytes of
    samples, and a compressed copy of it fits inside the 12 MB a request body is allowed to
    carry. So "the
    upload was small" is the wrong question, and the sample *count* is the right one.

    The engine answers both from the header, before a decoder is allowed to run, which is also
    why the limits hold on an install without the optional audio extra: parsing a header needs
    no codec.
    """)

workdir = tempfile.mkdtemp(prefix="laya_limits_")
atexit.register(shutil.rmtree, workdir, ignore_errors=True)

from laya.audio import (MAX_AUDIO_BYTES, MAX_AUDIO_SECONDS, AudioRefused, audio_info,
                        check_audio_bytes, read_audio_bytes)

heading("the two limits")
print("   MAX_AUDIO_BYTES   %d bytes (%.0f MB) -- the same request-body budget images get"
      % (MAX_AUDIO_BYTES, MAX_AUDIO_BYTES / 1024 / 1024))
print("   MAX_AUDIO_SECONDS %g seconds" % MAX_AUDIO_SECONDS)
hour = 3600 * 48000 * 2 * 2
print("   an hour of 48 kHz stereo, 16-bit: %d bytes of samples (%.0f MiB), and a 12 MB request"
      % (hour, hour / 1024 / 1024))
print("   body can carry the compressed form of it. What has to fit in memory is the samples.")

# ---------------------------------------------------------------- the byte limit
heading("1. the byte limit")
small = wav_file(os.path.join(workdir, "ok.wav"), seconds=0.1, rate=16000)
header = open(small, "rb").read()
oversized = header[:44] + b"\x00" * (MAX_AUDIO_BYTES + 1 - 44)
print("   a 0.1 s clip       : %d bytes -> %s" % (len(header), audio_info(header)["format"]))
print("   one byte over the limit (%d bytes): refused before the header is even parsed:"
      % len(oversized))
try:
    check_audio_bytes(oversized)
except AudioRefused as exc:
    print("       %s: %s" % (exc.reason, exc.detail))
print("   A caller with its own limit passes it: `check_audio_bytes(data, max_bytes=...)`.")

# ---------------------------------------------------------------- the duration limit
heading("2. the duration limit -- the one bytes cannot see")
long_clip = wav_file(os.path.join(workdir, "long.wav"), seconds=130, rate=8000)
long_bytes = os.path.getsize(long_clip)
info = audio_info(open(long_clip, "rb").read())
print("   a 130 s, 8 kHz mono clip: %d bytes (%.1f MB), %g seconds of samples"
      % (long_bytes, long_bytes / 1024 / 1024, info["duration_seconds"]))
print("   It is %.0f%% of the byte budget and still refused:" % (100.0 * long_bytes / MAX_AUDIO_BYTES))
try:
    check_audio_bytes(open(long_clip, "rb").read())
except AudioRefused as exc:
    print("       %s: %s" % (exc.reason, exc.detail))
print("   `max_seconds=0` turns the duration check off for a deployment that trusts its own")
print("   inputs; the byte limit still applies. A container that states no duration at all is")
print("   refused while the cap is in force -- \"I could not measure it\" is not \"it is short\".")

# ---------------------------------------------------------------- the URL refusal
heading("3. a URL is refused, and that is a security control")
for spec in ("http://169.254.169.254/latest/meta-data/",
             "http://localhost:9200/_cat/indices"):
    try:
        read_audio_bytes(spec)
    except AudioRefused as exc:
        print("   %-42s -> %s" % (spec, exc.reason))
print("   A self-hosted engine that fetches whatever URL a request names is a server-side request")
print("   forgery primitive pointed at the operator's own network: the cloud metadata endpoint")
print("   and the local search cluster are one request away. Turn it on with")
print("   `LAYA_ALLOW_AUDIO_URLS=1` (the image path has `allow_url=True`) and put your own egress")
print("   policy in front, or resolve the URL yourself and pass the bytes.")

# ---------------------------------------------------------------- what is left
heading("4. what the limits do not cover")
print("   They bound the *input*. They do not bound the work a checkpoint does on what it")
print("   accepts: a 120-second clip is 120 s of frames through the tower, which is why the")
print("   duration cap exists at all. If a deployment serves long recordings, cut them into")
print("   windows and ask a question per window -- that is a decision engine's normal shape, and")
print("   `laya.audio.load_waveform(..., max_seconds=...)` is how one window is cut.")

print("""
   Every refusal on this page is a typed exception with a `.reason` and a `.detail`, so a
   server can map them to a status code without parsing a message: `audio_too_large`,
   `audio_too_long`, `audio_unmeasurable`, `audio_unsupported`, `audio_invalid`. The HTTP server
   in this repository does exactly that, which is why the limits live beside the decoder rather
   than in the middleware in front of it.
   """)

print("   temp files: %s" % workdir)
