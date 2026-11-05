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
| 20 epochs, same recipe | *measured below* | 5 000 |

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
