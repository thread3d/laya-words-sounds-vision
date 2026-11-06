"""Example 48 -- a state can carry a picture.

Writes a PNG byte by byte with `zlib` and `struct`, puts it in a state, and takes the state apart
again with `laya.vision.split_state`: the accepted keys, the pixels the encoder actually receives,
and why a picture with no text at all is a legal state. No checkpoint is loaded.
"""
import atexit
import os
import shutil
import tempfile

from _common import SAMPLE_IMAGE, banner, heading, png_file

banner("48", "A picture in the state", """
    A decision about a picture is a state that carries one. The picture is not captioned, not
    OCRed and not described by a language model on the way in: a frozen vision tower turns it
    into patches, a projector turns those into rows, and the decision head learns which of the
    options the rows support. One forward pass, same as text.

    This page is the state contract and the decode. It needs no checkpoint, so the picture path
    can be learned before anything is trained for it.
    """)

workdir = tempfile.mkdtemp(prefix="laya_vision_")
atexit.register(shutil.rmtree, workdir, ignore_errors=True)
picture = png_file(os.path.join(workdir, "picture.png"), width=64, height=64)

heading("the image, written by this example")
print("   %s" % picture)
print("   %d bytes: 64x64 RGB, three PNG chunks (IHDR, IDAT, IEND) and nothing else."
      % os.path.getsize(picture))
print("   Writing it by hand is what makes the limits in example 49 legible: the IHDR chunk")
print("   states the dimensions in 8 bytes, so a decoder can be protected before it runs.")

from laya.vision import IMAGE_KEYS, has_image, image_info, load_pixels, split_state

heading("the three keys a state may use")
print("   %s" % ", ".join("`%s`" % key for key in IMAGE_KEYS))
print("   `image_path`   a path on this machine -- read in-process, never uploaded")
print("   `image`        the bytes, a base64 string, or a mapping with a `pixels` tensor")
print("   `image_base64` the same thing, named for what it is")
print("   `image_url` is accepted for *routing* and nothing else: a URL is something the caller")
print("   resolves, because a self-hosted engine that fetches what a request names is an egress")
print("   proxy pointed at the operator's own network (example 49).")

heading("has_image: the routing question, answered cheaply")
print("   has_image({'image_path': ...})  -> %s" % has_image({"image_path": picture}))
print("   has_image({'body': ...})        -> %s" % has_image({"body": "just text"}))
print("   The router asks this on every request and it inspects a dictionary's keys, nothing else")
print("   -- no decoding, no checkpoint (example 53 is the routing rule itself).")

heading("split_state: the picture, and everything beside it")
state = {"image_path": picture, "text": "what is in this picture?", "vendor": "acme"}
text_half, spec = split_state(state)
print("   state      : %s" % ", ".join("%s=..." % key for key in state))
print("   text half  : %r" % (text_half,))
print("   image spec : %r" % (spec,))
print("   `vendor` stays in the text half: a key the vision path does not own is still data the")
print("   model should see. Only the picture's own key and `text` move out.")

heading("a picture with no text at all is a legal state")
only_picture, spec = split_state({"image_path": picture})
print("   %r -> text half %r" % ({"image_path": picture}, only_picture))
print("   The picture is the evidence. A sentence beside it is optional, and when the decision is")
print("   \"which of these is shown?\" it is usually just noise the head has to ignore.")

heading("the same picture, named three ways")
import base64

with open(picture, "rb") as handle:
    raw = handle.read()
print("   `image_path`   -> spec %r" % (split_state({"image_path": picture})[1],))
print("   `image` bytes  -> spec %s (%d bytes)" % (type(split_state({"image": raw})[1]).__name__,
                                                   len(split_state({"image": raw})[1])))
as_base64 = split_state({"image_base64": base64.b64encode(raw).decode("ascii")})[1]
print("   `image_base64` -> %d characters, decoded at the other end" % len(as_base64))

heading("what the encoder actually receives")
pixels = load_pixels(picture, image_size=224)
print("   load_pixels(path)      -> shape %s, dtype %s, range [%.2f, %.2f]"
      % (tuple(pixels.shape), pixels.dtype, float(pixels.min()), float(pixels.max())))
print("   Resized and centre-fitted to the tower's 224x224, three channels, values in [0, 1]:")
print("   whatever the caller sent -- 64x64 here, a 4000x3000 phone photo there -- arrives as")
print("   the one tensor shape the tower was trained on.")
tensor = pixels.unsqueeze(0)
print("   `{\"image\": {\"pixels\": tensor}}` bypasses the decoder the same way the sound path's")
print("   `{\"waveform\": ...}` does -- here it comes back as %s."
      % (tuple(load_pixels({"pixels": tensor}, image_size=224).shape),))

heading("a real file from this repository")
if os.path.exists(SAMPLE_IMAGE):
    real = load_pixels(SAMPLE_IMAGE, image_size=224)
    print("   %s" % os.path.basename(SAMPLE_IMAGE))
    print("   %d bytes on disk -> %s tensor. Nothing about the format matters past this point."
          % (os.path.getsize(SAMPLE_IMAGE), tuple(real.shape)))
    info = image_info(open(SAMPLE_IMAGE, "rb").read())
    print("   header says: %s %dx%d" % (info["format"], info["width"], info["height"]))

print("""
   The state says what the evidence is; the question says what to do with it. A checkpoint
   trained with `--vision` reads the picture and ignores how the caller spelled the key -- path,
   bytes, base64 and a pre-decoded tensor all arrive as pixels. Example 49 is what the engine
   refuses before it decodes, example 50 is the block layout, and example 51 is the decision.

   Pillow is the one optional dependency on this path (`pip install laya[vision]`), imported only
   when a decoder is actually needed -- which is why `pixels` exists: a caller with its own
   decoder never installs it.
   """)
