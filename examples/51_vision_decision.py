"""Example 51 -- a decision about a picture.

Loads a checkpoint trained with `--vision`, asks it about a picture, and then asks the same question
with the picture removed. The gap between the two answers is the mechanism the gates in example 52
measure over a whole dataset -- and the last section is what a *text* checkpoint does with the same
state, which is nothing, silently.

Needs a vision checkpoint; says how to make one and exits 0 when there is none.
"""
import atexit
import os
import shutil
import tempfile
import time

from _common import SAMPLE_IMAGE, banner, describe, heading, load, media_agent, png_file, timed

banner("51", "A decision about a picture", """
    A checkpoint trained with `--vision` answers a question about pixels in one forward pass, in
    the same three answer shapes as every other Laya decision. Nothing is captioned, OCRed or
    described on the way in: the tower turns the picture into patches, the projector turns those
    into rows spliced after the option markers, and the head scores the options against them.

    The second half of this page is the part that matters operationally. A vision checkpoint that
    learned nothing answers just as confidently as one that learned something, so the check is to
    take the picture away and see whether the answer survives.
    """)

agent, questions, path = media_agent("vision", "51")
workdir = tempfile.mkdtemp(prefix="laya_picture_")
atexit.register(shutil.rmtree, workdir, ignore_errors=True)

heading("what is being decided")
print("   checkpoint : %s" % path)
print("   picture    : %s (%d bytes)" % (SAMPLE_IMAGE, os.path.getsize(SAMPLE_IMAGE)))
print("   questions  : %s" % ", ".join("`%s` (%s)" % (qid, q["type"])
                                      for qid, q in questions.items()))
state = {"image_path": SAMPLE_IMAGE, "text": "a photograph"}

t0 = time.perf_counter()
cold = agent.predict(state, questions)
cold_ms = (time.perf_counter() - t0) * 1000
heading("the answer")
describe(cold["answers"])
print("\n   first media call: %.0f ms -- it includes building and loading the vision tower"
      % cold_ms)
_, warm_ms = timed(lambda: agent.predict(state, questions))
print("   repeat call     : %.0f ms (the block is cached by a hash of the decoded pixels)" % warm_ms)
print("   The cache removes the *tower* pass, not the head. This checkpoint uses the")
print("   `state_conditioned` layout, so a 20-class question is 20 rows, each carrying the block --")
print("   the price the layout pays for never changing its answer when the options are reordered")
print("   (example 52 measures that). On a single-row joint head the tower dominates instead, and")
print("   the cache is what makes a repeat cheap.")

heading("any picture, at the tower's input size")
other = png_file(os.path.join(workdir, "generated.png"), width=96, height=96)
out = agent.predict({"image_path": other}, questions)
print("   a %d x %d picture written by this example -> %s"
      % (96, 96, max(out["answers"][next(iter(questions))]["probabilities"].items(),
                     key=lambda kv: kv[1])[0]))
print("   The state has no text at all here, and that is legal: the picture is the evidence. Every")
print("   input arrives at the tower as the same 224 x 224 x 3 tensor, whatever it was on disk.")

heading("the same state with the picture removed")
ablation = agent.predict({"text": "a photograph"}, questions)
describe(ablation["answers"])
qid = next(iter(questions))
with_picture = cold["answers"][qid]
without = ablation["answers"][qid]
if with_picture["type"] == "choice":
    a = max(with_picture["probabilities"].items(), key=lambda kv: kv[1])
    b = max(without["probabilities"].items(), key=lambda kv: kv[1])
    moved = 0.5 * sum(abs(with_picture["probabilities"].get(label, 0.0)
                          - without["probabilities"].get(label, 0.0))
                      for label in set(with_picture["probabilities"]) | set(without["probabilities"]))
    print("\n   with the picture    : %-22s p=%.3f" % (a[0], a[1]))
    print("   picture removed     : %-22s p=%.3f" % (b[0], b[1]))
    print("   total variation distance between the two distributions: %.3f" % moved)
    print("   A checkpoint that had learned nothing would print two confident answers here with a")
    print("   distance near 0. What this single row cannot tell you is the *accuracy* on either")
    print("   side -- that needs a labelled set, and it is example 52's job.")

heading("what a text checkpoint does with the same state")
text_agent = load("english")
plain = text_agent.predict(state, questions)
describe(plain["answers"])
print("   It does not refuse, and it does not warn. `image_path` is not one of its keys, so the")
print("   picture is serialized into the state as text -- a path string -- and the checkpoint")
print("   answers about it at whatever confidence it can. That is the single most useful thing to")
print("   know before deploying media: a checkpoint that cannot see cannot say so. Route by the")
print("   state's own shape (example 53) rather than by hoping the model notices.")

print("""
   The measured run behind the shipped recipe, on 20-way CIFAR-100 with the
   `state_conditioned` layout: 0.8130 held out, gate 1 +0.760 (removing the picture drops the
   accuracy by that much), and 0.0 % of answers move when the options are reordered. The
   `sequential` layout on the same data reaches 0.7822, +0.730, and 28.3 % -- a better answer
   *and* an unstable one is worse than a stable one, which is why the layout that ships is the
   one that costs more rows per item.

   Example 52 runs both gates.
   """)
