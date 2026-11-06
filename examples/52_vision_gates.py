"""Example 52 -- the two gates a vision checkpoint has to pass.

Gate 1 asks whether the picture is doing the work: remove it and the accuracy must fall. Gate 2b asks
whether the answer is stable: reorder the options and the chosen label must not move. This page runs
gate 2b over the images in this repository (it needs no labels) and shows gate 1's mechanism on a
single row, with the command for the labelled version.

Needs a vision checkpoint; says how to make one and exits 0 when there is none.
"""
import glob
import os

from _common import SAMPLE_IMAGE, banner, heading, media_agent

banner("52", "The two gates", """
    Two failures look like success, and accuracy alone cannot separate them.

    The first is decoration: a checkpoint whose fusion learned nothing answers from the text
    beside the picture, and on a set where the text hints at the answer it scores well without
    ever reading the pixels. Gate 1 measures it by subtraction -- the picture is removed and the
    accuracy has to fall.

    The second is instability: with the joint (`sequential`) layout all options sit in one row,
    so rewriting the list rewrites every option's position, and the same picture with the same
    options in a different order can come back with a different label. Gate 2b permutes the
    options and counts how often the answer moves. A caller who reorders a list -- a UI sorted by
    price, a set that came out of a database -- is doing exactly that on every request.
    """)

agent, questions, path = media_agent("vision", "52")

pictures = sorted(glob.glob(os.path.join(os.path.dirname(SAMPLE_IMAGE), "*.png")))
pictures = [p for p in pictures if os.path.getsize(p) < 2 * 1024 * 1024][:6]

from laya.evals import Dataset, Example

QUESTION = {
    "type": "choice",
    "instructions": "Which coarse class is shown in this image?",
    "criteria": {"flowers": "blooms, petals, stems", "fish": "aquatic animals with fins",
                 "vehicles 1": "cars, trucks, road traffic",
                 "household furniture": "chairs, tables, cupboards"},
}
dataset = Dataset([Example(state={"image_path": picture, "text": "an image"},
                           questions={"class": QUESTION}, expected={},
                           tags=(os.path.basename(picture),))
                   for picture in pictures])
print("\n   %d images from %s" % (len(dataset.examples), os.path.dirname(SAMPLE_IMAGE)))
print("   question : %s (%d options)" % (QUESTION["instructions"], len(QUESTION["criteria"])))

heading("gate 2b -- option-order invariance")
from laya.evals_vision import option_order_flip_rate

print("   %-6s %-10s %-8s %s" % ("seed", "compared", "flips", "flip rate"))
for seed in (0, 1, 2):
    report = option_order_flip_rate(agent, dataset, seed=seed, max_flip_rate=0.02)
    print("   %-6d %-10d %-8d %.1f%%   -> %s"
          % (seed, report["questions_compared"], report["flips"],
             100.0 * report["flip_rate"], "PASS" if report["passed"] else "FAIL"))
first = option_order_flip_rate(agent, dataset, seed=0, max_flip_rate=0.02)
print("   %d comparisons over %d rows and 3 permutations. The measured run behind the numbers\n   below is in docs/vision-benchmark-results.md."
      % (3 * first["questions_compared"], len(dataset.examples)))
print("   `laya-evals flip eval.jsonl --model <checkpoint> --max-flip-rate 0.02` is this gate as a")
print("   command, and it is the one to put in CI: a permutation is free to run and impossible to")
print("   argue with.")

heading("gate 1 -- picture necessity, on one row")
qid = next(iter(dataset.examples[0].questions))
with_picture = agent.predict(dataset.examples[0].state, {"class": QUESTION})["answers"][qid]
without = agent.predict({"text": "an image"}, {"class": QUESTION})["answers"][qid]
best = lambda a: max(a["probabilities"].items(), key=lambda kv: kv[1])  # noqa: E731
print("   with the picture : %-22s p=%.3f" % best(with_picture))
print("   picture removed  : %-22s p=%.3f" % best(without))
print("   These two numbers are on a chart image, not on the photographs this checkpoint was")
print("   trained for, so read the movement between them rather than either label: the same")
print("   question came back with a different distribution once the picture left, which is all one")
print("   row can show. The gate itself is the *accuracy* drop over a labelled set, which is what")
print("   `image_ablation` computes:")
print("   labelled set, which is what `image_ablation` computes:")
eval_file = os.environ.get("LAYA_VISION_EVAL")
if eval_file and os.path.exists(eval_file):
    from laya.evals import Dataset as _Dataset
    from laya.evals_vision import image_ablation

    labelled = _Dataset.from_jsonl(eval_file)
    report = image_ablation(agent, labelled, min_gain=0.35)
    print("   %s: %d rows, with picture %.4f, ablated %.4f, gain %+.4f -> %s"
          % (os.path.basename(eval_file), report["rows_with_image"],
             report["per_tag"]["all"]["with_image"], report["per_tag"]["all"]["without_image"],
             report["gain"], "PASS" if report["passed"] else "FAIL"))
else:
    print("       laya-evals ablate eval.jsonl --model %s --min-gain 0.35" % path)
    print("       (set LAYA_VISION_EVAL to a labelled jsonl and this example runs it here too)")

heading("why the layout decides gate 2b")
print("   `sequential`   : all options in one row. Reordering rewrites every option's position,")
print("                    and the encoder attends by position -- 28.3 % of answers moved on the")
print("                    measured CIFAR-20 run, at 0.7822 held out.")
print("   `state_conditioned`: one option per row, compared in the softmax. Reordering only")
print("                    reorders the rows -- 0.0 % moved, at 0.8130 held out.")
print("   Order invariance is therefore a property of the checkpoint's layout, not of the data it")
print("   was trained on. Gate 2b is how you find out which one you have.")

print("""
   Run both gates before a media checkpoint reaches a user-facing path, and read the ablated
   accuracy rather than only the gap. A checkpoint that scores 0.80 with the picture and 0.75
   without it is a text model with an expensive tower; one that scores 0.55 and 0.10 is a
   decision engine reading pixels, and 0.55 may well be the honest ceiling for the task.

   The published run for the shipped recipe: 0.8130 held out, gate 1 +0.760, gate 2b 0.0 %. The
   sound path has the same two gates -- example 47 runs the first, and `laya-evals ablate
   --modality audio` is its command line.
   """)
