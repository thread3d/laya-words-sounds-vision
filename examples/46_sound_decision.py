"""Example 46 -- a decision about a sound.

Loads a checkpoint trained with `--audio`, asks it questions about a recording, and shows the three
things a deployment needs to know: the first media call is the expensive one, the block is built
once per recording and shared by every question about it, and the text beside a recording is part
of the state, so rewording it moves the answer.

Needs a media checkpoint; says how to make one and exits 0 when there is none.
"""
import atexit
import shutil
import tempfile
import time

from _common import (DIGIT_TEXT, banner, describe, heading, media_agent, sample_clip, timed)

banner("46", "A decision about a sound", """
    A checkpoint trained with `--audio` reads a recording the way a text checkpoint reads a
    document: one forward pass, typed answers, calibrated confidence. There is no transcript
    anywhere in the path -- the tower sees the waveform, and the decision head has learned which
    qualities of it the options are asking about.

    Two consequences a deployment cares about. The first is cost: the first media call builds and
    loads the tower, and after that a recording is encoded once and cached by the hash of its
    samples. The second is honesty: a sound-only answer cannot be checked by reading it, so the
    gate example (47) is what tells you the recording is doing the work.
    """)

agent, questions, path = media_agent("audio", "46")
workdir = tempfile.mkdtemp(prefix="laya_decision_")
atexit.register(shutil.rmtree, workdir, ignore_errors=True)
clip, description = sample_clip(workdir)

heading("what is being decided")
print("   checkpoint : %s" % path)
print("   recording  : %s" % description)
print("   questions  : %s" % ", ".join("`%s` (%s)" % (qid, q["type"])
                                      for qid, q in questions.items()))
state = {"audio_path": clip, "text": DIGIT_TEXT}

t0 = time.perf_counter()
cold = agent.predict(state, questions)
cold_ms = (time.perf_counter() - t0) * 1000
heading("the answer")
describe(cold["answers"])
print("\n   first media call: %.0f ms -- it includes building and loading the audio tower"
      % cold_ms)

heading("the same recording, asked again")
_, warm_ms = timed(lambda: agent.predict(state, questions))
print("   first call  : %.0f ms (tower load + one pass)" % cold_ms)
print("   repeat call : %.0f ms (the block is cached by a hash of the samples)" % warm_ms)
print("   So a server warms the path once before it takes traffic. Two *different* recordings in")
print("   one call each pay for their own block; the same recording never does.")

heading("several questions, one call")
audio_q = dict(questions)
digit = next(iter(audio_q))
if audio_q[digit]["type"] == "choice":
    audio_q["is_human_voice"] = {"type": "noul",
                                 "instructions": "Is this a human voice rather than noise?"}
_, ms_one = timed(lambda: agent.predict(state, questions))
_, ms_two = timed(lambda: agent.predict(state, audio_q))
print("   one question  : %.0f ms" % ms_one)
print("   two questions : %.0f ms" % ms_two)
print("   Both are a single batched forward pass: a call is a batch of rows, one row per question,")
print("   and the recording's block is built once and shared by every row. The second question")
print("   therefore costs a row of compute, not a second pass over the tower.")
describe(agent.predict(state, audio_q)["answers"])

heading("the text beside the recording is part of the state")
print("   The same recording, three wordings. This checkpoint was trained with the phrase")
print("   %r, and moving away from it moves the answer:" % DIGIT_TEXT)
for text in (DIGIT_TEXT, "a recording of a single spoken digit", ""):
    answer = agent.predict({"audio_path": clip, "text": text}, questions)["answers"][digit]
    best = max(answer["probabilities"].items(), key=lambda kv: kv[1])
    conf = answer.get("answer_confidence")
    print("   text=%-40r -> %-8s (p=%.3f, answer_conf=%s)"
          % (text, best[0], best[1],
             "%.3f" % conf if isinstance(conf, (int, float)) else "n/a"))
print("   That is not a defect: the state is the recording *and* the sentence, and a head trained")
print("   on one pairing has not seen another. It is the sound version of example 12 -- wording")
print("   matters -- and the practical rule is to keep the phrase a checkpoint was trained with,")
print("   or train it on the phrasings you intend to serve.")

heading("what the call reports")
usage = cold["usage"]
print("   state_tokens      : %d  (the text half only; the block is charged to `max_len` instead)"
      % usage["state_tokens"])
print("   truncated         : %s" % usage["truncated"])
print("   input_tokens      : %d for %d question(s)" % (usage["input_tokens"], len(questions)))
print("   See example 45 for the block's own size and where it lands in the sequence.")

print("""
   The published numbers for this recipe on public data: FSDD, 10 spoken digits, 3 000 training
   clips, 15 epochs, frozen encoder -> 0.7550 held out, with the recording removed the accuracy
   falls by 0.655 (gate 1), and 69 ms median per decision on a CPU. A frozen linear probe on the
   same features scores 0.7950, so the decision head lands within four points of the best thing
   that can be done with those features at all -- what an audio checkpoint buys is not a better
   embedding but a *decision*: typed options, calibrated confidence, one pass.

   The next example is the gate: it removes the recording and asks whether the answer was ever
   coming from it.
   """)
