"""Example 47 -- the gate that says the checkpoint listened.

Answers a small labelled set of recordings twice: as given, and with the recording stripped out of
the state. The accuracy that survives the ablation is the accuracy of a text model, so the gap
between the two is how much of the decision the sound is carrying. This is gate 1, and it is the
one that catches a "sound checkpoint" that is a text model with an expensive tower bolted to it.

Needs a media checkpoint and a folder of labelled recordings; says how to make one and exits 0
otherwise.
"""
import os

from _common import (DIGIT_TEXT, FSDD_DIR, banner, heading, labelled_clips, media_agent,
                     timed)

banner("47", "The gate that says it listened", """
    A checkpoint trained on recordings can pass every other check while ignoring them. Fusion
    that learns nothing, a projector that cancels itself out, a tower that was never in the
    gradient path -- each of those produces a model that answers plausibly from the *text* beside
    the recording, and no accuracy number on a well-labelled set would reveal it, because the
    text is often enough.

    The measurement that catches it is subtraction. Ask the same questions with the recording
    present and with it removed; the accuracy that survives is what the text alone bought, and the
    difference is what the sound bought. `laya-evals ablate --modality audio --min-gain` is this
    page as a command.
    """)

agent, questions, path = media_agent("audio", "47")
rows = labelled_clips(count=30)
if not rows:
    print("\n   SKIP -- no labelled recordings to measure against.")
    print("   This checkout has no FSDD copy (%s)." % FSDD_DIR)
    print("   Point a dataset at your own clips -- one state per row, `expected` per question --")
    print("   and run the same gate from the command line:")
    print("       laya-evals ablate eval.jsonl --model %s --modality audio --min-gain 0.35" % path)
    raise SystemExit(0)

from laya.evals import Dataset, Example
from laya.evals_audio import audio_ablation

heading("the set being asked about")
print("   checkpoint : %s" % path)
print("   rows       : %d recordings, labelled by filename (%s)" % (len(rows), os.path.basename(
    os.path.dirname(rows[0][0]))))
print("   question   : %s" % ", ".join("%s (%s)" % (qid, q["type"]) for qid, q in questions.items()))

examples = [Example(state={"audio_path": clip, "text": DIGIT_TEXT},
                    questions=questions, expected={qid: label for qid in questions},
                    tags=(label,))
            for clip, label in rows]
dataset = Dataset(examples)

heading("one row, both ways")
one = dataset.examples[0]
with_audio = agent.predict(one.state, one.questions)["answers"]
without = agent.predict({"text": one.state["text"]}, one.questions)["answers"]
qid = next(iter(one.questions))
first, second = with_audio[qid], without[qid]


def headline(answer):
    if answer["type"] == "choice":
        best = max(answer["probabilities"].items(), key=lambda kv: kv[1])
        return "%s (p=%.3f)" % best
    if answer["type"] == "score":
        return "%.2f" % answer["score"]
    return "%.3f" % answer["noul"]


print("   expected %-6s | with the recording: %-18s | recording removed: %s"
      % (one.expected[qid], headline(first), headline(second)))

heading("the gate, over the whole set")
report, ms = timed(lambda: audio_ablation(agent, dataset, min_gain=0.35), repeat=1)
with_accuracy = report["per_tag"]["all"]["with_audio"]
without_accuracy = report["per_tag"]["all"]["without_audio"]
print("   rows with a recording : %d" % report["rows_with_audio"])
print("   rows text-only        : %d  (excluded -- there is nothing to remove)" % report["rows_text_only"])
print("   accuracy, as given    : %.4f" % with_accuracy)
print("   accuracy, ablated     : %.4f" % without_accuracy)
print("   gain (gate 1)         : %+.4f   threshold +0.35 -> %s"
      % (report["gain"], "PASS" if report["passed"] else "FAIL"))
print("   gate took %.0f s over %d rows x 2 passes" % (ms / 1000.0, len(examples)))

heading("the companion check")
print("   text-only rows unchanged : %s (%.1f%% of them moved)"
      % (report["text_only_unchanged"], 100.0 * report["text_only_change_rate"]))
print("   Removing the recording from a row that has none must change nothing. If a row moves")
print("   anyway, the pass itself is not deterministic and every number above is noise -- which")
print("   is why the report carries it beside the gate rather than in a footnote.")

print("""
   Read the two accuracies, not just the gap. A checkpoint whose ablated accuracy is high was
   never reading the recording; a checkpoint whose ablated accuracy is *chance* is fine, even if
   its headline accuracy is unimpressive -- that is what a decision about a sound looks like when
   the sound is the only evidence there is.

   One caveat about the sample this page uses: 30 clips from the benchmark checkpoint's own
   speaker set score near 1.00 here, which is optimistic -- the gate's *gain* is the number to
   read, not the headline accuracy. The published held-out run is FSDD at 15 epochs with a frozen
   encoder: 0.7550 held out and a gate-1 gain of +0.655, so removing the recording takes it from
   three-quarters right to near chance. The same command works for a picture with
   `laya-evals ablate --modality image`; example 52 runs both gates for vision.
   """)
