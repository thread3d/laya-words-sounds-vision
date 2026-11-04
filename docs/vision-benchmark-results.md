# Vision benchmark on real data — first measured numbers

Branch `vision` (`laya/`, HEAD `7abbe3a3`, v0.4.45), sessions 2026-11-03. Companion to
the vision ledger (the ledger) and `vision-benchmark-assessment.md` (the "compared to public
benchmarks" analysis, which this document turns from argument into measurement).

Every number below is **measured in this session on this machine's GPUs**, on a real public
benchmark, scored on a held-out split no arm ever trained on.

---

## 0. Headline

CIFAR-100 as a typed multiple-choice decision — one `choice` question per image, the answer carried
by the pixels only. Held-out: 5 000 rows (100-way) / 5 000 rows (20-way).

| arm | 100-way accuracy | 20-way accuracy |
|---|---|---|
| mean-RGB centroid (300 parameters, no learning) | 4.68 % | — |
| SigLIP2 zero-shot (Google's own head) | 73.64 % | 57.50 % |
| linear probe on the frozen embedding | 79.54 % | 84.44 % |
| **Laya's vision head** (27.3 M trained) | **1.02 %** (chance 1 %) | **87.28 %** |

Three results, in order of how much they should change the plan:

1. **At 20 options Laya wins outright** — 87.3 % against the linear probe's 84.4 % and zero-shot's
   57.5 %, **with better calibration than the linear probe** (ECE 0.018 vs 0.129) and a real
   risk–coverage curve (96.2 % accuracy on the 80 % it answers). Gate 1 passes by a mile: 87.3 %
   with the image, 5.4 % without — chance — gain **+0.819** (the gate needs 0.35).
2. **The joint layout fails gate 2b on real data: 16.9 % option-order flips**, against the 8-10 %
   the published literature reports and against 0.00 % for both classifier controls (which are
   order-invariant by construction). The spec says a joint layout at that flip rate does not ship
   (§4.1, §9). §4.5 then measures the layout that does ship: `state_conditioned` reaches
   **0.0 % flips with 0.813 held-out accuracy**, passing both gates.
3. **At 100 options the head does not learn at all** — it collapses to the uniform prior in four
   independent runs, including one with a trainable text encoder. The information is present in the
   same frozen features (a 76 900-parameter linear layer gets 79.5 % out of them); the decision head
   cannot reach it. §4 identifies the mechanism and the missing piece of code.

---

**Parameter counts, measured from the artifact** (not assumed): frozen ModernBERT-large encoder
394 781 696, frozen SigLIP2 tower 92 884 224, trained head 27 300 614 (2-layer marker head
25 192 448 + scorer 1 052 673 + act head 263 938 + type embeddings 3 072 + projector 788 480).

## 1. What was run

**Data.** `uoft-cs/cifar100` from HuggingFace (parquet; the PNG bytes are written through unchanged,
so every arm sees identical pixels and `laya.vision.load_pixels` does the 224 resize). Train split:
100 images × 100 classes = **10 000 items**. Held-out: 50 × 100 = **5 000 items**. Each item is one
`choice` question whose criteria are the class names (underscores rendered as spaces), an
uninformative state text (`"a photograph"`), and the label under `expected`. The 20-way variant
reuses the same images and the same selection with the **parquet's own `coarse_label` column** —
not a hand-typed mapping — giving 10 000 / 5 000 items over 20 classes.

**Harness** (all in `.vision-cifar100/`, nothing in the package changed):

| file | role |
|---|---|
| `cifar_prep.py` / `make_coarse.py` | parquet → PNGs + Laya JSONL (100-way / 20-way) |
| `controls.py`, `controls_calib.py`, `controls20.py` | zero-shot, linear probe, centroid; calibration and risk-coverage |
| `train.sh` | training through `laya-train`: `--dataset 20\|100`, `--layout`, `--visual-tokens`, `--train-rows`, `--seed` |
| `eval_cifar.py` | accuracy, ECE, risk–coverage, gate 1, gate 2b, from the saved artifact |
| `gpuenv.sh` | `LD_LIBRARY_PATH` for `libcaffe2_nvrtc.so`, `HSA_OVERRIDE_GFX_VERSION` unset |

**Environment.** All three of this machine's blockers are the ledger's, all still present, all
worked around: `/dev/kfd` invisible inside `workspace-write` (GPU work ran unsandboxed), `torch/lib`
missing from `LD_LIBRARY_PATH`, and MIOpen's `Conv2d` segfaulting — `research/benchmarks/vision_cifar100/tower_patch.py` reroutes
SigLIP2's patch-embed conv through `unfold` + matmul, numerically identical to 1.1e-06.

**Controls, defined precisely.**
* *zero-shot*: SigLIP2's published protocol — the image embedding against text embeddings of
  `a photo of a {class}.` (7 templates at 100-way, 5 at 20-way, averaged, L2-normalised), multiplied
  by the checkpoint's own `logit_scale` (112.7) and softmaxed over the option set. Without that
  scale the probabilities are meaningless for calibration (they sum to ~0.01); with it, they are
  the model's actual confidences.
* *linear probe*: multinomial logistic regression on the frozen, L2-normalised image embedding,
  trained on the same 10 000 training images.
* *centroid*: per-image mean RGB (16×16), class centroids from the training split, nearest centroid.

---

## 2. The control ladder

Identical 5 000 held-out rows for every arm.

### 100-way (fine CIFAR-100)

| arm | trained params | accuracy | mean conf | ECE | acc @ 70 % coverage |
|---|---|---|---|---|---|
| mean-RGB centroid | 300 | 4.68 % | — | — | — |
| SigLIP2 zero-shot | 0 | **73.64 %** | 0.673 | **0.063** | 89.17 % |
| linear probe (frozen features) | 76 900 | **79.54 %** | 0.953 | 0.155 | 93.51 % |
| **Laya head** (16 visual tokens, frozen encoder) | 27 300 614 | **1.02 %** | 0.011 | 0.0008 | 1.43 % |

### 20-way (coarse CIFAR-100)

| arm | trained params | accuracy | mean conf | ECE | acc @ 80 % / 50 % coverage |
|---|---|---|---|---|---|
| SigLIP2 zero-shot | 0 | 57.50 % | 0.577 | **0.013** | 65.73 % / 80.08 % |
| linear probe (frozen features) | 15 380 | 84.44 % | 0.973 | 0.129 | 93.57 % / 98.68 % |
| **Laya head** (16 visual tokens, frozen encoder) | 27 300 614 | **87.28 %** | 0.889 | **0.018** | **96.23 % / 99.36 %** |

Two facts here outlive whatever happens to Laya:

* **SigLIP2's zero-shot head is calibrated out of the box** — ECE 0.013-0.063 with no fitting. Any
  calibration claim Laya makes is against that, not against nothing.
* **The classifier controls are order-invariant by construction** (0.0000 flip rate, measured):
  they score each label independently and take an argmax. Laya's gate 2b is a floor it must reach,
  not a bar it beats; the 8-10 % figures it is compared against belong to methods that score options
  jointly.

---

## 3. Laya's head

### 3.1 The 20-way checkpoint: competitive, calibrated, and not yet shippable

`ckpt_cifar20_frozen` — 10 000 items, 4 epochs, frozen tower *and* frozen text encoder, 16 visual
tokens, joint (`sequential`) option layout, ~18 min on one card.

| metric | value |
|---|---|
| accuracy (5 000 held out) | **87.28 %** |
| mean confidence | 0.889 |
| ECE (15 bins) | **0.0184** |
| acc @ 90 / 80 / 70 / 50 / 30 % coverage | 92.56 / 96.23 / 98.03 / 99.36 / 99.87 % |
| **gate 1 — image ablation** | with **87.30 %**, without **5.40 %**, **gain +0.819** → threshold 0.35, **PASS** |
| **gate 2b — option-order flips** | **16.9 %** (169 / 1 000) → published 8-10 %, **FAIL** |

Read together: the image is doing essentially all of the work (without it the model is at chance,
5.4 % against 5 %), the head beats a linear probe on the same frozen features, and its confidences
are trustworthy enough to drive a coverage curve. **And it is unstable under reordering** — a
calibrated probability that moves when the caller reorders a list is the exact failure the spec
says must block a release.

The synthetic runs in the vision ledger reported a 0.0 % flip rate. That was the
`state_conditioned` layout on a four-colour toy task. This checkpoint is the joint layout on real
images, and it lands where the literature does — near the published 8-10 %, above it.

### 3.2 The 100-way collapse

| run | visual tokens | text encoder | training loss | outcome |
|---|---|---|---|---|
| `ckpt_cifar_frozen` | 16 | frozen | 4.6039 (`ln 100 = 4.605`) | uniform prior; **1.02 %** held out |
| `ckpt_cifar_frozen64` | 64 | frozen | 4.5953 at step 800 | no progress |
| `ckpt_cifar_trainable` | 16 | trained | 4.6064 | uniform prior |

The collapse is exact and immediate: the loss sits on `ln 100` from the first hundred steps and
never moves. It is not a budget problem — the 20-way run reaches 0.36 with the same recipe, data and
step count — and it is not "the text encoder is frozen", because training the encoder changes
nothing.

**The gates fire correctly on this collapse** (500 eval rows): gate 1 gain **+0.0000** (1.00 % with
the image, 1.00 % without — the image contributes exactly nothing) and gate 2b **75.2 %** (a uniform
scorer's argmax is arbitrary, so permuting the list moves it). Both refused a model answering at
chance, which is what they exist to do.

---

## 4. Mechanism: the `[VIS_SUMMARY]` token was not implemented, and its marker math was wrong

*(Both fixed in v0.4.50 — the section is kept as the measurement that found them, and it says at the end what changed.)*

The spec anticipated this failure and prescribes the fix. `docs/vision-decision-system.md` §4.1:

> **[VIS_SUMMARY]** — one token, the pooled global visual vector, placed immediately after the
> block *and* (for high-cardinality questions) once more in front of the first marker. With
> `head_max_len=192` and 20 options the first marker can otherwise sit ~180 tokens from the splice
> point, outside the 128-token local window. One extra token buys global reach in all 28 layers
> instead of 10.

The compliance table in the same document (line 1640) lists it as **implemented**. It is not.
Three reproduced facts:

**(1) Nothing produces the token.** `PooledProjector.forward` emits exactly `n_visual_tokens`
(`adaptive_avg_pool1d` → linear → type vector). `visual_plan` promises `n_visual_tokens + 1`
whenever the head exceeds `summary_for_options_over = 96` rendered option tokens — which is every
head with more than ~30 options. The result is a hard failure before a single step runs:

```
laya-train: error: plan expects 17 visual tokens, got 16
```

This is why every run above passes `"summary_for_options_over": 0`: it is currently the only way to
build a high-cardinality vision item. **The one code path written for high-cardinality vision
decisions has never executed.**

**(2) Even with the token supplied, the markers would read the wrong tokens.** `visual_plan` shifts
every marker by +1 (`marker_positions = [m + 1 for m in markers]`) and moves the splice to
`head_len + 1`, while `_cat` inserts the block *at* the splice point — after every marker. The shift
compensates for an insertion that never happens. Measured on the real tokenizer, 100-option question:

```
head_len=311 markers[0]=10 option_tokens=301 distinct=100
  summary=True  n_visual=17 splice=312 marker0= 11 -> 0/100 markers land on a [MASK]
  summary=False n_visual=16 splice=311 marker0= 10 -> 100/100 markers land on a [MASK]
```

`tests/test_vision.py::test_a_high_cardinality_head_keeps_every_marker_on_its_mask_token` pins the
plan's numbers (`marker_positions == [4, 6]`, `splice == 8`) but never checks that those positions
still index `[MASK]` tokens in an assembled sequence — which is why the suite does not see it.

**(3) The geometry matches.** Marker distance to the visual block, computed with the real
tokenizer for the configurations actually run:

| configuration | head length | markers | markers within 128 tokens of the block |
|---|---|---|---|
| 20 options, 16 visual | 90 | 20 | **20 / 20** |
| 20 options, 64 visual | 90 | 20 | 20 / 20 |
| 100 options, 16 visual | 314 | 100 | **42 / 100** |
| 100 options, 64 visual | 314 | 100 | 42 / 100 |

At 100 options, **58 of the 100 option markers sit outside the 128-token local window** of the
visual block, so in 18 of ModernBERT-large's 28 layers they cannot attend to the image directly.
At 20 options every marker is inside it — and the 20-way run is the one that learns. The correlation
is real; it is not yet a proof, because one 20-way variant with 64 visual tokens also failed
(§5), so the geometry is necessary-looking but not sufficient-explaining.

**Fixed since this was written (v0.4.50).** `PooledProjector.forward(patches, summary=True)`
appends one globally-pooled, projected row; `VisionEncoder.encode(pixels, summary=...)` forwards
it; `visual_plan` sizes the block for it; and the trainer, the agent and the frozen-patch path all
decide from the same number — the item's own `option_tokens`, computed once in `make_item` — so the
plan and the projector can no longer disagree. The markers no longer shift:
`marker_positions = markers` and `splice = head_len`, which is the only placement consistent with
`_cat`.

**Measured on the real tokenizer, 100 options:** all 100 markers index a `[MASK]` token (0 of 100
before), `plan.n_visual == n_visual_tokens + 1`, and `laya-train` builds and trains a 100-option
vision checkpoint instead of refusing with `plan expects 17 visual tokens, got 16`. What this
changes for the 100-way **accuracy** is not yet measured — the four collapses in §3.2 were all run
with `summary_for_options_over: 0`, i.e. without the token, so the geometry they blamed is the
geometry this fixes and the number has to be re-taken.

The spec's second placement (a copy of the token in front of the first marker) is still not
implemented: it needs a second splice point in `assemble_inputs_embeds`, which changes the layout
of every vision checkpoint. It is worth doing only if the re-run shows the single token is not
enough.

### 4.1 A third defect, found while scoring: mixed batches crash

`Agent.predict_batch` raises when one batch mixes a state that carries an image with one that does
not:

```python
agent.predict_batch([{"image_path": p}, {"text": "hello"}], question, batch_size=2)
# KeyError: 'visual'   (laya/vision.py:792, collate_visual)
```

All-image batches work and all-text batches work; only the mixture raises. That matters beyond the
harness: gate 1 is *by construction* a comparison of image and text-only states, so a batched
ablation cannot run at all, and `collate_visual`'s own contract says the opposite ("the ones without
simply have none"). Every gate number in this document was therefore measured one state per forward
pass, which is why the gate passes cost more than the 5 000-row accuracy pass.

---

## 4.5 The `state_conditioned` layout on the same data, and the first artifact that passes both gates

Every table above is the joint (`sequential`) layout -- the one that scores all options in one
sequence, and the one that flips 16.9 % of its answers when a caller reorders a list. The layout
the spec says ships is `state_conditioned`: each option is scored in its own pass, so permuting the
list cannot move the answer by construction. Until this run it had only ever been measured on the
four-colour synthetic task, where its 0.0 % flip rate cost nothing because the accuracy was at
chance.

Three arms, identical in everything but the layout and the epoch budget. Same 2 000-item
class-balanced stride of the 20-way training split (1 800 items after the calibration slice), same
seed 77, same frozen tower and frozen text encoder, scored on the full 5 000-row held-out split
with `eval_cifar.py`.

| arm | layout | items | epochs | held-out acc | ECE | gate 1 gain | **gate 2b flips** |
|---|---|---|---|---|---|---|---|
| `ckpt20_joint_2k` | sequential | 2 000 | 3 | 0.7822 | 0.0181 | +0.730 | **28.3 %** |
| `ckpt20_sc_2k` | state_conditioned | 2 000 | 3 | 0.3696 | 0.0224 | +0.305 | **0.0 %** |
| `ckpt20_sc_2k_10ep` | state_conditioned | 2 000 | 10 | **0.8130** | 0.0229 | **+0.760** | **0.0 %** |
| *`ckpt_cifar20_frozen`* (reference) | sequential | 10 000 | 4 | 0.8728 | 0.0184 | +0.819 | 16.9 % |

**What this settles: order invariance is real on real data.** 0 of 1 000 permutations moved the
answer for the state-conditioned arm, against 283 for the joint arm on identical rows, identical
budget and identical seed, and against 169 for the full-data joint checkpoint. That is gate 2b
passing by the letter of the spec rather than by the arithmetic of a toy task.

**`ckpt20_sc_2k_10ep` is the first vision checkpoint in this repository to pass both gates on a
public dataset**: gate 1 at +0.760 (the image is doing the work; 5 % without it is chance) and gate
2b at 0.000 with ECE 0.023. It trains in ~19 minutes and scores in ~30 (a state-conditioned item is
`k` rows, so scoring costs ~5x a joint arm).

**What it costs: sample efficiency.** In the joint layout one row sees all twenty options and the
model compares them in that row; in `state_conditioned` a row sees one option and the comparison
happens in the softmax across rows. At three epochs the arm is badly undertrained -- 0.370 against
the joint arm's 0.782 on the same items -- and it takes ten epochs (~19 min) to overtake it (0.813
against 0.782) while staying ~6 points under the 10 000-item joint checkpoint. So the honest
summary is: **the layout buys order invariance and pays for it in epochs, not in accuracy
ceiling** -- whether the 0.87 of the large joint run is reachable with more data is untested here.

Two caveats, recorded rather than hidden:

* the 20-way numbers above sit on 2 000 training items; the 0.8728 reference used 10 000. Only the
  first two rows are a data-matched comparison.
* runs are single-seed and, at this scale, not reproducible: of four identical joint-layout runs
  started this session, three collapsed to the class prior (mean loss stuck at `ln 20`) and one
  learned (`VISION_PROGRESS.md` records the same instability). The flip rate is not a marginal
  effect and did not depend on which run learned, but an accuracy comparison between two single
  runs carries that risk explicitly.

---

## 5. Variance, stated plainly

Runs are **single-seed** except where noted. One result did not reproduce:

* `20 options × 64 visual tokens`, seed 31: collapsed to `ln 20` (2.997) over four epochs.
* The same configuration at seed 77 is **not** on the prior: epoch means 1.185 → 0.575 → 0.426 →
  **0.312**, against a flat 2.997 (`ln 20 = 2.996`) at every step of the seed-31 run.

So the 64-token failure is best read as a seed artefact rather than an effect of the visual budget,
and "more visual tokens is fine" stands as the working assumption. Every *100-way* run collapsed
identically across three configurations, so that one is not seed variance.

Two other things this session could not settle, both recorded rather than papered over:

* **`Agent.predict_batch` on image states measured slower than one-at-a-time** in one configuration
  (1.25 s/item against 57 ms) and faster in another (28 ms/item against 57 ms). Worth its own look;
  not investigated here.
* The trainer's own console report disagreed with the artifact again — for `ckpt_cifar20_frozen` it
  printed **accuracy 0.0333** where the artifact measures **0.8728**. **Fixed since (v0.4.47).**
  It was two defects, both silent: the `--eval` rows were built without the image (so the table
  scored the state text alone, and chance is exactly what those rows score), and `finetune` never
  loaded the base checkpoint's projector, so `laya-train --base <vision checkpoint>` restarted from
  a random one and scored *that*. With both fixed, the "Before" row for a checkpoint that measures
  0.7822 reads **0.7900**. Every number in this document was measured before that fix and comes
  from `eval_cifar.py` against the saved checkpoint, never from the report.

---

## 6. Cost

One `gfx1030` (Radeon Pro W6800-class, 32 GiB), fp32, micro-batch 8 × grad-accum 2:

| run | wall clock |
|---|---|
| 10 000 items × 3 epochs, 100 options | ~34 min |
| 10 000 items × 4 epochs, 20 options | ~18 min |
| item construction (10 000 images encoded + 1 M option tokenisations) | ~3 min of either |
| scoring 5 000 held-out rows + both gates | ~5 min |

The frozen-patch cache costs ~0.6 MB per item (~6 GB per process at 10 000 items). **The head's
sequence length, not the tower, dominates a vision decision's cost** — the spec's §3.5 arithmetic,
confirmed the first time the path was run above four options.

---

## 7. What this changes in the earlier assessment

`vision-benchmark-assessment.md` said the system was "unrankable publicly" and that its only numbers
were synthetic. Both are now replaced by measurement:

1. **The tower is not the limitation.** Frozen SigLIP2 base/16 plus a linear layer reaches 79.5 %
   (100-way) and 84.4 % (20-way) on real photographs of a public benchmark.
2. **The head is competitive at low cardinality** — 87.3 % and ECE 0.018 at 20 options, beating both
   controls — **and fails its own order-invariance gate there.**
3. **The head does not work at high cardinality**, with the mechanism identified in code and the
   mitigation unimplemented. Until `[VIS_SUMMARY]` exists, no Laya vision checkpoint can answer a
   question with more than ~30 options at all — it raises before training.
4. **Calibration is not yet a Laya win**: the tower's own zero-shot head is already well calibrated
   (ECE 0.013-0.063), so fitted temperatures have to beat that.

---

## 8. Reproduce

The harness lives in `research/benchmarks/vision_cifar100/`, and its README carries the full
command list. In short:

```bash
BENCH=research/benchmarks/vision_cifar100
DATA=$PWD/.vision-cifar100            # gitignored: data, weights, checkpoints

# 1. data (HuggingFace mirror; the .toronto.edu tarball measured ~60 KB/s from here)
mkdir -p "$DATA/data"
curl -sSLo "$DATA/data/train.parquet" \
  https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet
curl -sSLo "$DATA/data/test.parquet" \
  https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet

# 2. items, both cardinalities
python3 "$BENCH/cifar_prep.py"      # 100-way jsonl + PNGs
python3 "$BENCH/make_coarse.py"     # 20-way jsonl, parquet coarse labels

# 3. the frozen tower's own heads
bash -c "source $BENCH/gpuenv.sh && python3 $BENCH/controls.py"   # embeddings + ladder (GPU)
python3 "$BENCH/controls_calib.py"                                # 100-way calibration (CPU)
python3 "$BENCH/controls20.py"                                    # 20-way calibration (CPU)

# 4. Laya, frozen tower and frozen text encoder
PYTHON=python3 bash "$BENCH/train.sh" "$DATA/ckpt_cifar_frozen"   3 8 --dataset 100 --freeze-encoder
PYTHON=python3 bash "$BENCH/train.sh" "$DATA/ckpt_cifar20_frozen" 4 8 --dataset 20  --freeze-encoder

# 4b. the layout comparison in §4.5 -- one flag, and ~10 epochs because an item is k rows
PYTHON=python3 bash "$BENCH/train.sh" "$DATA/ckpt20_sc_2k_10ep" 10 8 --dataset 20 \
    --layout state_conditioned --train-rows 2000 --freeze-encoder

# 5. score the artifact, with both gates
python3 "$BENCH/eval_cifar.py" "$DATA/ckpt_cifar20_frozen" --data eval20.jsonl --tag _20
python3 "$BENCH/eval_cifar.py" "$DATA/ckpt_cifar_frozen"   --tag _100
```

GPU work must run unsandboxed: a container that binds a minimal `/dev` hides `/dev/kfd`, and
`torch.cuda.device_count()` then reports 0 while the cards are perfectly healthy.
