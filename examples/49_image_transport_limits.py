"""Example 49 -- what the engine refuses before it decodes an image.

A 40-byte PNG can declare 30 000 x 30 000 pixels, and decoding it first is a decompression bomb.
Every limit on this page is answered from the image header, before Pillow is imported. No checkpoint
is loaded.
"""
import atexit
import os
import shutil
import struct
import tempfile

from _common import SAMPLE_IMAGE, banner, heading, png_file

banner("49", "Two limits and a header", """
    An image costs bytes to carry and pixels to decode, and the two are wildly different
    numbers: a 12 MB upload can be a 1.5 GB bitmap. So the engine asks the *header* both
    questions -- width, height, format -- and refuses before any decoder is loaded, which is also
    why the limits hold on an install without the optional Pillow dependency.

    A decompression bomb is the extreme case: PNG compresses runs of one colour, so a 30 000 x
    30 000 single-colour image is a few hundred bytes on the wire and 2.7 GB of RGB samples when
    decoded. Nothing about the upload says so. The header does.
    """)

workdir = tempfile.mkdtemp(prefix="laya_image_limits_")
atexit.register(shutil.rmtree, workdir, ignore_errors=True)

from laya.vision import (MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, ImageRefused, check_image_bytes,
                         image_info, read_image_bytes)

heading("the two limits")
print("   MAX_IMAGE_BYTES  %d bytes (%.0f MiB)" % (MAX_IMAGE_BYTES, MAX_IMAGE_BYTES / 1024 / 1024))
print("   MAX_IMAGE_PIXELS %d pixels (%.1f megapixels)"
      % (MAX_IMAGE_PIXELS, MAX_IMAGE_PIXELS / 1e6))
print("   A 30 000 x 30 000 picture is %d pixels: %.0fx the pixel budget, and 2.7 GB decoded as"
      % (30000 * 30000, 1.0 * 30000 * 30000 / MAX_IMAGE_PIXELS))
print("   RGB (30 000 x 30 000 x 3 bytes).")

heading("1. the header answers both questions, before decoding")
small = png_file(os.path.join(workdir, "small.png"), width=64, height=64)
info = image_info(open(small, "rb").read())
print("   a 64x64 PNG : %d bytes -> %s %dx%d"
      % (os.path.getsize(small), info["format"], info["width"], info["height"]))
if os.path.exists(SAMPLE_IMAGE):
    real = image_info(open(SAMPLE_IMAGE, "rb").read())
    print("   %-11s : %d bytes -> %s %dx%d"
          % (os.path.basename(SAMPLE_IMAGE), os.path.getsize(SAMPLE_IMAGE),
             real["format"], real["width"], real["height"]))
print("   PNG, JPEG, GIF and WebP are all read this way; `image_info` returns None for anything")
print("   else rather than guessing, so a caller decides whether an unidentified header is a")
print("   refusal or merely unmeasured.")

heading("2. the decompression bomb")
bomb = bytearray(open(small, "rb").read())
bomb[16:20] = struct.pack(">I", 30000)          # IHDR width
bomb[20:24] = struct.pack(">I", 30000)          # IHDR height
print("   the same %d-byte file with two header fields changed: %s %dx%d"
      % (len(bomb), image_info(bytes(bomb))["format"], image_info(bytes(bomb))["width"],
         image_info(bytes(bomb))["height"]))
try:
    check_image_bytes(bytes(bomb))
except ImageRefused as exc:
    print("       %s: %s" % (exc.reason, exc.detail))
print("   Refused without a decoder running, which is the whole point: by the time Pillow has")
print("   allocated the buffer, the memory is already gone.")

heading("3. the byte limit")
oversized = open(small, "rb").read()
oversized = oversized + b"\x00" * (MAX_IMAGE_BYTES + 1 - len(oversized))
print("   one byte over the limit (%d bytes):" % len(oversized))
try:
    check_image_bytes(oversized)
except ImageRefused as exc:
    print("       %s: %s" % (exc.reason, exc.detail))
print("   `check_image_bytes(data, max_bytes=..., max_pixels=...)` for a deployment with its own")
print("   numbers; the HTTP server reads them from `LAYA_MAX_IMAGE_BYTES` and")
print("   `LAYA_MAX_IMAGE_PIXELS`, and the same check runs on every request.")

heading("4. an unidentified header")
try:
    check_image_bytes(b"not an image at all, just some bytes")
except ImageRefused as exc:
    print("   %-26s -> %s: %s" % ("b'not an image at all...'", exc.reason, exc.detail))
print("   An image with no measurable size is refused only while a pixel limit is in force --")
print("   \"I could not measure it\" is not the same as \"it is small\".")

heading("5. a URL is refused, and that is a security control")
for spec in ("http://169.254.169.254/latest/meta-data/",
             "http://localhost:9200/_cat/indices"):
    try:
        read_image_bytes(spec)
    except ImageRefused as exc:
        print("   %-42s -> %s" % (spec, exc.reason))
print("   A self-hosted decision engine that fetches whatever URL a request names is a")
print("   server-side request forgery primitive pointed at the operator's own network. Opt in")
print("   with `read_image_bytes(..., allow_url=True)` and put your own egress policy in front,")
print("   or resolve the URL yourself and pass the bytes.")

print("""
   Every refusal is a typed exception with a `.reason` and a `.detail` -- `image_too_large`,
   `image_unsupported`, `image_unmeasurable`, `image_invalid` -- so a server maps them to status
   codes without parsing prose. The limits live beside the decoder rather than in the middleware
   in front of it, because the middleware is exactly the layer that does not know what a PNG
   header means.
   """)
