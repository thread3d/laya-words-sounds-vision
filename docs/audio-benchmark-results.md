# Audio benchmark results — ESC-50, measured here

What the sound path scores on a public benchmark, next to the controls and the published systems
that answer a comparable question. Dataset selection, the published table and the
like-for-like rules are in [audio-benchmark-assessment.md](audio-benchmark-assessment.md); the
harness is `research/benchmarks/audio_esc50/`.

Everything below was measured on this machine (56 CPU cores, no GPU) with the command shown, and
the JSON artifacts are beside the rows in `$AUDIO_BENCH_DATA` (gitignored — they are ~760 MB of
cached frames and checkpoints; the numbers are reproduced here and by the commands).

## 1. The controls, on frozen wav2vec2-base features

One tower pass per clip (frozen, no AudioSet pre-training, no fine-tuning of the tower), features
mean-pooled over the 249 frames of a 5-second clip, then two heads: nearest class mean (no
learning) and ridge regression (a linear probe, penalty chosen by inner CV on the training fold
only). Published protocol: 5-fold cross-validation, each fold scored once.

```bash
python3 research/benchmarks/audio_esc50/probe.py --folds 1 2 3 4 5 --workers 10 --threads 5
```

ESC-50, 50 classes, 1 600 train / 400 held out per fold:

| arm | mean ± std over 5 folds | per fold |
|---|---|---|
| chance | 0.0200 | — |
| centroid (no learning) | 0.2975 ± 0.0159 | .312 .278 .302 .315 .280 |
| **ridge (linear probe)** | **0.5265 ± 0.0368** | .580 .487 .540 .542 .482 |
| ridge on mean+std features | 0.4695 ± 0.0220 | .492 .460 .458 .497 .440 |

ESC-10, 10 classes, 320 train / 80 held out per fold:

| arm | mean ± std over 5 folds |
|---|---|
| chance | 0.1000 |
| centroid | 0.5875 ± 0.0345 |
| **ridge (linear probe)** | 0.7125 ± 0.0209 |
| ridge on mean+std features | **0.7300 ± 0.0292** |

The Free Spoken Digit Dataset — 3 000 half-second recordings of the digits zero to nine, six
speakers, split **by speaker** (1 000 recordings held out) so the number measures digit recognition
rather than voice recognition. FSDD clips are 8 kHz, so every one of them is *upsampled* to the
tower's 16 kHz:

```bash
python3 research/benchmarks/audio_esc50/prep_fsdd.py
python3 research/benchmarks/audio_esc50/probe.py --dataset fsdd
```

| arm | speaker-disjoint hold-out (1 000 clips) |
|---|---|
| chance | 0.1000 |
| centroid | 0.5330 |
| **ridge (linear probe)** | **0.7950** |

FSDD is the fast tier: 16 MB and half-second clips, so prep, tower pass and probe finish in a
couple of minutes. That is what makes the harness testable end to end in one sitting — and it is how
the "every eval row overlaps training" defect (see the [changelog](https://github.com/thread3d/laya-words-sounds-vision/blob/main/CHANGELOG.md))
was caught before it reached the slow arm.

Two facts fall out of that table, and they set the bar for the engine:

* **Adding the temporal standard deviation hurts on the 50-way task and helps on the 10-way one.**
  The 50-way probe does better when it averages the whole clip; the 10-way one gains a little from
  knowing how much the features vary. The engine gets the block, not a pooled vector, so it is not
  limited to either.
* **A frozen speech-trained encoder probes at 52.7 % on ESC-50.** That is the number the engine has
  to beat to be worth its extra machinery, and it is squarely in the published band for frozen
  speech encoders (Whisper 52.8 %, Dasheng 54.8 %) — well below the fine-tuned AudioSet models
  (AST 95.6 %, BEATs 98.1 %), which is the comparison the assessment page argues is not
  like-for-like.

## 2. Where that sits against the field

| system | ESC-50 | protocol |
|---|---|---|
| SVM on MFCC | 39.6 % | hand features |
| Whisper | 52.8 % | frozen representation probe |
| Dasheng | 54.8 % | frozen, linear probe |
| **Laya's tower, linear probe (measured here)** | **52.7 %** | frozen, 5-fold |
| Piczak ConvNet | 64.5 % | from scratch |
| BEATs iter3 | 72.7 % | frozen, linear probing |
| Human participants | 81.3 % | no training |
| BYOL-A | 83.2 % | linear evaluation, frozen |
| AST-S (ImageNet only) | 88.7 ± 0.7 % | fine-tuned |
| PANNs CNN14 | 94.7 % | AudioSet pre-trained, fine-tuned |
| AST-P (ImageNet + AudioSet) | 95.6 ± 0.4 % | fine-tuned |
| BEATs iter3+ | 98.1 % | AudioSet-supervised, fine-tuned |

The Laya row here is the *tower's* feature quality, not the engine's answer — it exists so the
engine's number can be read as "better or worse than a linear probe on the same features", which is
the only comparison that isolates what the decision head adds.

## 3. The engine arms

`run_engine_arms.sh` trains two Laya checkpoints through `laya-train --audio` on the same rows and
scores each with `eval_gate.py`:

* `esc10-sc` — 10 options, `state_conditioned` (one row per option), 320 clips per epoch, both
  gates: it is the arm where option-order invariance can be measured cheaply, and the layout the
  [vision work](vision-benchmark-results.md) found passes both gates.
* `esc50-seq` — 50 options, the joint head, 1 600 clips per epoch: the arm with more data, scored on
  accuracy and gate 1.

The bar each has to clear is written down above rather than chosen afterwards: **> 52.7 %** for the
engine to beat a linear probe on its own frozen features at 50-way, and gate 1 (the accuracy lost
when the recording is removed) has to be large for the answer to be about the sound at all.

### The first arm, and what it taught

The first engine arm was the ESC-10 one, trained the way the fine-tuning guide trains a text
checkpoint: `laya-train --audio --freeze-encoder`, 8 epochs, its defaults (`head_lr=1e-4`, effective
batch 64 → 40 optimizer updates per epoch on 288 items).

| ESC-10, fold 1, 80 held-out clips | accuracy | gate 1 gain | gate 2b flips |
|---|---|---|---|
| before fine-tuning | 0.0875 | — | — |
| after 8 epochs, default recipe (320 updates) | **0.1000** | **0.00** | 0.0000 |
| after 4 epochs, `--grad-accum 1 --head-lr 1e-3` (144 updates) | **0.1625** | — | — |

Chance is 0.10 and the linear probe on the same tower's features is 0.71. The model learned to be
under-confident (ECE 0.188 → 0.006, mean confidence 0.276 → 0.106) and nothing else, and gate 1 is
exactly zero: the answer does not depend on the recording at all.

That result has two possible explanations that point in opposite directions, and the difference
matters:

* **a broken path** — the audio block never reaches the head, or its gradient never flows — in which
  case the whole sound capability is decorative; or
* **an under-trained head** — 288 clips, a 1024-wide two-layer head, and 300-odd optimizer updates
  is simply not enough learning, in which case the path is fine and the *recipe* is the finding.

`research/benchmarks/audio_esc50/synth.py` separates them in two minutes. It writes 64 one-second
recordings — a 440 Hz tone against an 880 Hz one, the same state text for both — and trains the real
path on them:

| synthetic two-class tone task | loss | held-out accuracy |
|---|---|---|
| epoch 1 | 0.023 | — |
| epoch 2 | 0.000 | — |
| after 4 epochs (32 updates) | 0.000 | **1.0000** (from 0.5000) |

So the block carries learnable signal and gradients reach it: the path is sound, and the ESC-10
number is an optimization-and-data finding, not a wiring bug. The same experiment is now a test —
`test_the_audio_block_can_learn_a_task_end_to_end` in `tests/test_train_audio.py` — because a suite
that only asserted "a weight moved" would have passed through both explanations.

### The second arm: more data, same wall

FSDD gives 2 000 training recordings rather than 288, and the same recipe as the second ESC-10 row
(`--grad-accum 1 --head-lr 1e-3`), so the comparison isolates the data:

| FSDD, 1 000 speaker-disjoint held-out clips | accuracy | optimizer updates |
|---|---|---|
| linear probe on the same frozen features | **0.7950** | closed form |
| 3 epochs, `--grad-accum 1 --head-lr 1e-3` | 0.1490 | 750 |
| **15 epochs, `--grad-accum 1`, default `head_lr`** | **0.7550** | 3 750 |

A `state_conditioned` arm on the same public data was attempted and **cost too much for a CPU**:
601 items × 10 rows = 6 010 rows per epoch, and measured **~11 minutes per epoch** — about 2.2 hours
for the 12-epoch run the recipe wants. The reason is worth knowing rather than treating as bad luck:
with the encoder frozen, a *text* step skips the backward pass, but a media step cannot — the
projector sits upstream of the encoder, so its gradient has to traverse all 28 frozen layers. A
profile without the encoder attached measured 2.4 minutes per epoch; in the real run it is 11. That
is the price of the projector, and it is why every media arm here is quoted with its wall clock.

So the head is slow, not unable: 3 750 updates take it from chance to **75.5 %**, within 4 points of the
linear probe on the same frozen features. That is the engine's first real public number.

```bash
./research/benchmarks/audio_esc50/train.sh --arm fsdd-seq --epochs 15 -- --grad-accum 1
python3 research/benchmarks/audio_esc50/eval_gate.py \
  --checkpoint .audio-bench/checkpoints/fsdd-seq --rows ../fsdd/fsdd_eval.jsonl --batch-size 8
```

### The gates on that checkpoint

The two gates are the reason the accuracy above means anything. Both were run on the same 1 000
held-out recordings:

| gate | measured | verdict (`--min-gain 0.35`, `--max-flip-rate 0.02`) |
|---|---|---|
| accuracy | 0.7550 | — |
| accuracy with the recording removed | **0.1000** (chance) | — |
| **gate 1 — sound necessity** | **+0.655** | **passes** |
| **gate 2b — option-order invariance** | **27.4 % of answers flip** | **fails** |

### The layout the product ships, measured on a learning model

Gate 2b above fails because the arm uses the **joint** layout. The layout that ships is
`state_conditioned`, and its order invariance was measured on a control that learns, cheaply:
`synth.py --classes 20 --layout state_conditioned` — 20 options, tones, 58 training rows, 8 epochs
(64 optimizer updates).

| 20-option control, `state_conditioned` | measured | verdict |
|---|---|---|
| accuracy | **1.0000** (from 0.0000) | — |
| accuracy with the recording removed | 0.0938 | — |
| **gate 1 — sound necessity** | **+0.906** | **passes** |
| **gate 2b — option-order invariance** | **0.0 % of answers flip** | **passes** |
| median decision latency | 498 ms | 20 rows per decision |

So the same path, with the layout the product ships, passes both gates — on a model that is
demonstrably learning. The joint layout's 27.4 % on real data and this 0.0 % are the two halves of
the option-order story, and they agree with the vision program's 28.3 % / 0.0 % on CIFAR-20.

Gate 1 is the headline the shipped checkpoint could not produce before: remove the audio and the
model falls from 75.5 % to chance, so the answer is about the recording and not about the text around
it. Gate 2b is the failure the [vision program](vision-benchmark-results.md) already measured at
28.3 % on CIFAR-20 — this arm uses the **joint** (`sequential`) layout, whose options are scored
against each other, so a reordered list moves the answer. The fix is the same one vision shipped:
`--option-layout state_conditioned`, one row per option, which scores 0.0 % flips by construction and
costs `k` rows per item instead of one (ten times the training cost here, which is why the gate-2b=0
measurement on a *learning* audio model is left for a GPU run — the chance-level state_conditioned
arm already measured 0.0 %).

### The 50-way arm, and a high-cardinality control

The literature-comparable arm — 50 options, the joint head, 1 600 training clips per epoch — was run
to the same wall-clock budget a CPU allows in one sitting: **2 epochs, 360 optimizer updates**.

| ESC-50, fold 1, 400 held-out clips | accuracy | note |
|---|---|---|
| linear probe on the same frozen features | **0.5265** | closed form |
| engine, 2 epochs (360 updates) | 0.0200 | chance is 0.0200 |

It did not learn, and the honest reading is the same one the FSDD arm gives from the other side: at
this budget the head is not trained. The 50-way engine number needs the FSDD arm's 15 epochs at
least, which is 8x the wall clock and a GPU run here. Reported rather than omitted, because a
2-epoch arm at chance is exactly what an untuned benchmark looks like.

A 50-option head also crosses `summary_for_options_over`: the audio block carries its summary row,
which is the high-cardinality path the vision program once shipped broken. So the summary row got a
control of its own, `synth.py --classes 100` (120 rows, ten tones among 100 options):

| 100-option synthetic task | accuracy |
|---|---|
| before | 0.1000 |
| after 6 epochs (84 updates) | **0.3000** |
| chance | 0.0100 |

The summary row learns. That is the check that the vision program's trap — a plan reserving a row
the projector never emitted, so high-cardinality heads could not even build — is not present on the
audio path.

### Latency

The same checkpoint, on this CPU, answering a 10-option question about a one-second recording:

| measurement | value |
|---|---|
| median decision | **69 ms** |
| mean | 69 ms |
| p95 | 111 ms |

That is the whole path — decode, the tower pass when the clip is new, the splice, the encoder — for a
decision that depends on the audio (gate 1 = +0.655 above). The clip is one second and the
`sequential` layout scores one row; a `state_conditioned` decision costs ten rows.

Two throughput facts shaped the recipes, both measured here rather than assumed (56 CPU cores, no
GPU, `ModernBERT-large` encoder):

| row shape | rows/s at `--micro-batch 8` | at `64` |
|---|---|---|
| 10 options (~80 tokens incl. the audio block) | 12.7 | **25.4** |
| 50 options (~195 tokens) | **5.9** | 5.3 |

Batching doubles the throughput of short rows and does nothing for long ones: a 50-option head is
memory-bandwidth bound, so the only lever there is fewer epochs or a GPU. That is why the FSDD arms
run at `--micro-batch 16` and the 50-way arm is quoted in hours.

`state_conditioned` costs ten rows per item but each row is *cheaper* — measured 42.3 rows/s against
12.2 for the joint layout on the same data, because a row carries one option instead of ten markers.

And the largest single cost of a media run turned out not to be the model at all: `laya-train` wrote
a recovery checkpoint after every epoch, and a media checkpoint is ~1.2 GB (the encoder *plus* the
bundled tower). Measured: **~6 minutes per write against 2.4 minutes of training per epoch** — the
crash protection cost more than the epoch it protects. `--checkpoint-every 0` keeps the final
checkpoint and drops the per-epoch ones, and the FSDD arms here run with it.

The learning rate, not only the budget, was wrong for this regime, and it went the counter-intuitive
way: raising `head_lr` from the default 1e-4 to 1e-3 made the loss *rise* (epoch means 2.146 then
2.217), while the default rate descended on the same data (1.800 after one epoch). A 1024-wide head
over a frozen encoder on a few thousand rows has a narrow stable band, and the shipped default is
inside it. That is worth knowing before anyone tunes a small media fine-tune: more updates, not a
bigger step.

Three epochs on 2 000 recordings is still not enough, and the trainer says so itself: it emitted its
class-prior collapse warning ("mean within-row logit range 0.00876; calibration cross-entropy 2.302
is within 2% of ln K ~ 2.303 … try more epochs, more data, or another seed"). That warning is the
product doing exactly what it was built to do, and it is the reason the 20-epoch run exists: the
question is whether the head is *slow* or *unable*.

### The harness itself is checked

A benchmark number is worth what its measurement is worth, so the two knobs that changed the harness
are verified against the unbatched path on real ESC-50 rows, not only on synthetic ones:

| measurement | single-row gate | batched gate (`--batch-size 6`) |
|---|---|---|
| accuracy (12 held-out rows, a smoke checkpoint) | 0.0833 | **0.0833** |
| gate 1 gain | 0.0833 | **0.0833** |
| mean decision latency | 571 ms | 498 ms |

Same answers, same arithmetic, both passes batched identically. The latency gain is modest there
because a `state_conditioned` decision is ten rows, so the batch is spent on one question.
