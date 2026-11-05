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

The numbers these produce are added to this page by that run. The bar each has to clear is written
down above rather than chosen afterwards: **> 52.7 %** for the engine to beat a linear probe on its
own frozen features at 50-way, and gate 1 (the accuracy lost when the recording is removed) has to
be large for the answer to be about the sound at all.
