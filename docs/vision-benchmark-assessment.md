# How good is the vision system, against public benchmarks?

Written 2026-11-03, branch `vision` (`laya/`, HEAD `7abbe3a3`, package v0.4.45).
Companion to the session ledger (not in this tree). Every number below is either
(a) quoted from a primary source fetched today, (b) taken from the ledger's recorded
measurement, or (c) measured fresh in this session and marked **[measured here]**.

> **Superseded — read `vision-benchmark-results.md` first.** §4 asked for a real-data run; it was
> done, on CIFAR-100, end to end. Two rounds of it changed the answer to the question in this
> document's title:
>
> * **20 options, joint layout:** Laya's head is the best arm on the ladder — **87.28 %**, ECE
>   0.018, gate 1 gain +0.819 — but **fails gate 2b at 16.9 % flips**, so it is not shippable.
> * **20 options, `state_conditioned`** (the layout the spec says ships): **81.30 %**, gate 1
>   **+0.760** and gate 2b **0.0 %** — the first artifact in this project that passes **both**
>   gates on a public dataset. The cost is epochs, not accuracy ceiling
>   (`vision-benchmark-results.md` §4.5).
> * **100 options:** the head collapsed to the uniform prior (1.02 %) in four configurations,
>   because `[VIS_SUMMARY]` was documented and never implemented. That is **fixed in v0.4.50**,
>   including the marker arithmetic that put 0 of 100 markers on a `[MASK]`; the 100-way accuracy
>   has not been re-measured since.
>
> So the verdict below is obsolete where it says "unrankable": there is now a measured,
> gate-passing result at 20 options. It still stands for anything above ~30 options until that
> re-run exists.

---

## 0. The short answer

At the time of writing there was **no public benchmark result for Laya's vision system**,
because the object it builds — a typed multiple-choice decision head over an image, one forward
pass, no generation — is not the object any public benchmark scores. There is one now: CIFAR-100
as a 20-option `choice` question, with the three controls that make the number mean something
(`vision-benchmark-results.md`). What follows is how the question decomposes, and which of its
claims a public benchmark can check at all:

| layer | externally comparable? | verdict |
|---|---|---|
| SigLIP2-B/16 tower (92.9 M) | **yes**, fully | competent 2025 encoder, ~7 pts off its own family's best, inherited not earned |
| projector + marker-scoring head (0.79 M) | **no** — no published benchmark exists | design supported by one analogue; Laya's own numbers are synthetic |
| the whole system on its own eval | **yes, and it loses** | a **12-parameter** pixel average scores **1.0000** where the system scores 0.2530 **[measured here]** |

The headline finding is not "the model is weak". It is stronger and more useful:
**the only benchmark this system has cannot distinguish it from a 12-number hack**,
so none of its published-looking accuracies (0.99, 1.00) are evidence of vision quality —
and its one honest generalisation measurement (0.2530) is 75 points *below* that hack.

---

## 1. Layer 1 — the tower. This part *is* on the public leaderboards.

Laya ships `google/siglip2-base-patch16-224` frozen. Its numbers are Google's, published
in SigLIP 2 Table 1 ([arXiv 2502.14786](https://arxiv.org/html/2502.14786v1), cross-checked
row by row in `.survey/notes-siglip.md`). The B/16 @ 224 row Laya uses:

| encoder | INet-val 0-shot | INet-v2 | INet-ReaL | ObjNet | COCO T→I |
|---|---|---|---|---|---|
| CLIP B/16 | 68.3 | — | — | — | 33.1 |
| OpenCLIP B/16 | 70.2 | — | — | — | — |
| MetaCLIP B/16 | 72.4 | — | — | — | — |
| EVA-CLIP B/16 | 74.7 | — | — | — | — |
| DFN B/16 | 76.2 | — | — | — | — |
| **SigLIP2 B/16 @224 (Laya's Tier-L tower)** | **78.2** | 71.4 | 84.8 | 73.6 | 52.1 |
| SigLIP2 L/16 @256 | 82.5 | 76.8 | 87.3 | 83.0 | 54.7 |
| SigLIP2 so400m/14 @384 | 84.1 | 78.7 | 88.1 | 86.0 | 55.8 |
| SigLIP2 g/16 @384 | **85.0** | 79.8 | 88.5 | 88.0 | 56.1 |

**Reading it honestly.** At 92.9 M parameters this is the strongest encoder of its size
class in that table — it beats every B/16 contrastive encoder listed, by 2.0 pts over the
next best (DFN 76.2) and by 9.9 over CLIP B/16. It is **6.8 pts below** the same family's
best (g/16 @384, 8.4× the parameters) and **5.9 pts below** so400m/14 @384, which the spec
makes Tier M.

Two qualifiers that matter for the product:

* These are **zero-shot text-image alignment** numbers. They say the embedding is good;
  they say nothing about whether a *marker-scoring decision head* can use it (§2).
* The spec itself (§5, "what the evidence does not cover") records that **no benchmark
  measures masked-option scoring over image+text**, and that **no calibration evidence
  exists for image-conditioned decision heads**. That is still true; nothing in this
  assessment changes it.

For a decision-shaped yardstick the closest published analogue remains the frozen
DINOv3 ViT-L/16 + small-head study the spec quotes (§5.1): linear-on-CLS **0.435**
precision@1 versus **0.696** for attention over the patch tokens. That is a +26-point
argument for Laya's design family, measured in a completely different domain
(medieval illuminations), and it is the strongest external support the head has.

---

## 2. Layer 2 — the system on its own benchmark, versus a trivial baseline

The only quantitative evaluation the vision system has is a synthetic task generated by
`e2e/make_dataset.py`: 64×64 PNGs, one of four flat colours on a noisy background with a
distractor stripe, and deliberately uninformative text ("an item from the intake queue"),
so the answer can only come from the pixels. Four-way choice, chance = 0.25.

**[measured here]** I ran the most trivial classifier that could possibly work on the exact
held-out sets the gates were measured on: per-image mean RGB, class centroids of the
training split, nearest centroid. No encoder, no text, no learning — **12 parameters**.

| eval set | system accuracy | mean-RGB centroid (12 params) |
|---|---|---|
| `e2e/probe_eval1k.jsonl` (1000 items, GPU probe) | **0.2530** (chance) | **1.0000** |
| `e2e/big_eval/dataset.jsonl` (320 items, joint layout) | **0.99375** | **1.0000** |
| `e2e/evaldata/dataset.jsonl` (80 items) | 0.906 (`ckpt_sc4`, R4) | **1.0000** |

This is the decisive result, and it cuts in both directions:

* **The eval has no headroom.** The task is linearly separable in raw pixel space. A model
  scoring 1.00 on it has demonstrated that the plumbing works, not that it can see. Every
  vision accuracy recorded in the vision ledger (0.906, 0.99375, 1.0000) is therefore
  **not evidence of vision capability**, and should not be quoted as if it were.
* **The one number that is not trivially reachable is badly wrong.** The 30 k-image GPU
  probe — the largest, most general run — scored **0.2530**, i.e. chance, on a task a
  12-parameter colour average solves perfectly.

### 2.1 Calibration: the axis the product is built on

From `e2e/run_program.log` (the Stage-1 GPU fine-tune, 30 k images, 3 epochs, train loss 0.0000):

| metric | before | after |
|---|---|---|
| accuracy | 0.2360 | **0.2530** |
| ECE *(lower better)* | 0.3168 | **0.7469** |
| mean confidence | 0.5528 | **0.9999** |
| multi-class Brier | 1.0257 | 1.4938 |

The system ended at chance accuracy with **confidence 0.9999**. For scale, this repo's own
text checkpoints report ECE **0.213 raw → 0.081 after temperature fitting** (`README.md`
§ calibration). The vision checkpoint is **9× the text path's fitted ECE** and **3.5× its
raw ECE**, while claiming certainty. Since calibrated confidence is objective (1) of the
spec and the product's whole differentiator, this is not a cosmetic defect — a caller
thresholding on `min_confidence` would act on every single one of those chance answers.

The training report's own warning names the cause: *"the fit came back unchanged at 1.0 …
which usually means they gave it nothing to correct; this type's confidences are
uncalibrated."*

---

## 3. What *is* measured and defensible

Three properties the system has, that published numbers do not give it but that do hold:

1. **Option-order invariance: 0.0 %.** On `e2e/ckpt_sc6/checkpoint_latest` the
   state-conditioned layout flips on **0.0 %** of permutations, against the **8–10 %**
   published for joint option scoring (the spec's "See the Options" evidence). The GPU
   artifact measured 6.0 %, the joint layout 5 %. The 0.0 % is a real architectural win —
   it is the layout that ships.
2. **Both gates passed on one artifact.** `ckpt_sc6`: ablation gain **0.75** (gate 1 PASS,
   threshold 0.35) and flip rate **0.0 %** (gate 2b PASS). That is the system doing what
   the spec asked: the image is necessary to the decision, and option order does not
   move it. Caveat: on an eval set where 12 parameters score 1.00 (§2).
3. **Cost and shape.** 394.8 M text encoder + 92.9 M tower + 0.79 M projector ≈ **488 M
   parameters**, one forward pass, no decoder. Measured on this machine's `gfx1030` card:
   tower **11.6 ms**, text encoder 31.4 ms @192 tok ⇒ **~43 ms/decision**, batched, p99
   predictable. A 2.2–3 B VLM answering the same question decodes for hundreds of ms to
   seconds. That is the entire design bet, and it is intact.

For context on the generative side (fetched today from the model cards): Qwen2.5-VL-3B
scores MMMU 53.1, MMMU-Pro 31.6, DocVQA 93.9, MathVista 62.3; SmolVLM2-2.2B scores MMMU 42,
DocVQA 80.0, ChartQA 68.8. Those models *are* on the leaderboards and would beat this
system on any real visual-semantic task by tens of points — at 4–6× the parameters and
without the determinism, the abstention contract, or the one-pass cost. The spec's §1.3
declares that comparison a non-goal, and this assessment agrees: it is not the axis on
which this system can be called good or bad.

The abstention literature is worth noting for the same reason: `UNK-VQA`, `VISREAS` and
`TUBench` all exist because MLLMs *cannot* reliably decline an unanswerable visual query —
which is exactly the failure Laya's calibrated-confidence design is aimed at, and exactly
the property its own vision checkpoint currently violates at ECE 0.747.

---

## 4. What a comparable number would require

Nothing about the code blocks this; the missing piece is data and an adapter.

1. **A real-data fine-tune.** The 150 k-item probe the spec schedules has never run here
   (~21 h at the measured 2 images/s unless the device is used properly; the GPU path is
   now fixed in v0.4.45, so this is a scheduling problem, not a defect).
2. **A public benchmark in Laya's own answer shape** — choice questions over an image.
   A four-option public task is directly analogous to the synthetic one and would put a
   number beside SigLIP2's 78.2 and beside a tiny CNN. ImageNet-100 as a 100-option
   `laya.predict_tournament` run is the cheapest credible version.
3. **Both gates on that set** — ablation gain and flip rate are the two numbers no public
   multimodal leaderboard reports, and they are the ones that would make this system
   interesting to anyone else.

Until (1)–(3), the only defensible public statement is: *the perception layer is a
well-chosen, well-benchmarked 92.9 M SigLIP2 encoder; the decision layer that sits on top
of it has no public benchmark, one synthetic evaluation, and no evidence of generalisation
beyond its training distribution.*

---

## 5. Bottom line

*(Revised twice: after the CIFAR-100 run, and again after the layout comparison and the
`[VIS_SUMMARY]` fix. Numbers from `vision-benchmark-results.md`.)*

* **As a vision model at low cardinality, it is good and measured, and the layout decides whether
  it is shippable.** The joint layout reaches 87.28 % on 20-way CIFAR-100 held out — above a frozen
  linear probe (84.44 %) and the tower's own zero-shot head (57.50 %) — but flips 16.9 % of its
  answers when the option list is reordered. `state_conditioned`, the layout the spec says ships,
  reaches **81.30 %** with gate 1 at **+0.760** and gate 2b at **0.0 %**: both gates pass, which no
  earlier checkpoint in this project managed on real data.
* **The price is epochs, not accuracy ceiling**: a `state_conditioned` item is `k` rows, so it
  needs roughly three times the epochs to reach the same accuracy and costs about five times as
  much to score. Whether it closes the remaining ~6-point gap to the 10 000-item joint run with more
  data is untested.
* **As a vision model at high cardinality, it does not work yet — and the reason is now fixed but
  not re-measured.** The head collapsed to the uniform prior (1.02 % against a 1 % chance) in four
  configurations because `[VIS_SUMMARY]` was never implemented, and because the plan shifted every
  marker by one for an insertion that never happens (0 of 100 markers landed on a `[MASK]`). Both
  are fixed in v0.4.50; a 100-option checkpoint now trains, and its accuracy has not been taken.
* **As an architecture**, the latency story is unchanged and confirmed: one deterministic forward
  pass, ~43 ms, with the head's sequence length — not the tower — dominating the cost.

Quote the 20-option numbers with the layout and the epoch budget attached: 87.28 % is the joint
layout at 10 000 items, 81.30 % is `state_conditioned` at 2 000 items and 10 epochs, and neither is
a general vision number. Above ~30 options there is still nothing to quote.

