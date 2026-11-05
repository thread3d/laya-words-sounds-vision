# Audio benchmarks — the landscape, and what a Laya number can honestly be compared to

The [sound decision system](audio-decision-system.md) shipped without a public number, and said so.
This page is the ground work for fixing that: which labelled audio sets exist, which published
systems answer a comparable question, and what the difference between them and this engine is. The
harness and its measured output are in
[`research/benchmarks/audio_esc50/`](https://github.com/thread3d/laya-words-sounds-vision/tree/main/research/benchmarks/audio_esc50),
and its results are written up in
[audio-benchmark-results.md](audio-benchmark-results.md).

## 1. What question is being answered

Laya does not classify. It answers a `choice` question whose options are rendered as **text**, and
its answer is a distribution over those options produced by `[MASK]` markers attending to a spliced
audio block. Scoring it on a public audio set therefore asks two things at once:

* does the frozen tower's content reach the decision head at all (the [gate 1](evals.md#media-gates-does-it-actually-look-at-the-picture-or-listen-to-the-recording)
  question, measured as the accuracy lost when the recording is removed), and
* how accurate is the answer, against systems that were built to classify instead.

The second number is only interpretable next to systems measured the same way, which is what §3 is
for.

## 2. Candidate labelled sets

| dataset | what it is | size | licence | used here? |
|---|---|---|---|---|
| [ESC-50](https://github.com/karolpiczak/ESC-50) | 2 000 five-second environmental recordings, 50 classes, 40 per class, pre-arranged into 5 folds so one source recording never straddles a fold | 616 MB zip, 44.1 kHz mono WAV | CC BY-NC 3.0 | **yes** — the standard small-data benchmark, with a human baseline and a decade of published numbers |
| ESC-10 | the curated 10-class subset inside ESC-50 (400 clips) | inside ESC-50 | same | **yes** — small enough to run the `state_conditioned` layout (one row per option) on a CPU inside a session |
| [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) | 3 000 spoken digits, 10 classes, 6 speakers, 8 kHz | ~10 MB | CC BY-SA 4.0 | planned fast tier: a decision task a CPU can train end to end in minutes |
| [Speech Commands v2](https://arxiv.org/abs/1804.03209) | 105 829 one-second utterances, 35 words | 2.4 GB | CC BY 4.0 | not yet — the published numbers (AST 98.1 %, PANNs 97.1 %) need a GPU to reach |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | 8 732 urban clips, 10 classes | ~6 GB | CC BY-NC | no — the download needs a form and an email |
| [FSD50K](https://zenodo.org/record/4060432) | 51 197 clips, 200 classes | tens of GB | CC BY 4.0 (clips vary) | no — too large for this machine's budget |
| [AudioSet](https://research.google.com/audioset/) | 2 M+ YouTube clips, 527 classes | TB-scale | YouTube ToS | no — not a local dataset, and the download is the experiment |
| [CREMA-D](https://github.com/CheyenneCao/CREMA-D), [RAVDESS](https://zenodo.org/record/1188976) | acted emotional speech, 6-8 classes | 7 GB / 200 MB | CC BY-NC / CC BY-NC-SA | a natural *decision* task (is this escalation?) but big; not in this round |

ESC-50 is the right first target: it is small enough to run all five folds, its licence permits
research use, every comparable system reports on it with the same 5-fold protocol, and it includes a
measured **human** baseline (81.3 %), which is a far more meaningful bar for a decision engine than
the fine-tuned SOTA.

## 3. Systems that do comparable work

ESC-50, 5-fold cross-validation. The **protocol** column is the part that decides whether a number
is comparable to ours.

| system | accuracy | protocol | source |
|---|---|---|---|
| Human participants | 81.3 % | no training, per-clip judgement | [Piczak 2015](https://github.com/karolpiczak/ESC-50) |
| SVM on MFCC | 39.6 % | hand features + classifier | leaderboard ([sota2](https://www.sota2.com/research/sota/audio-classification-on-esc-50)) |
| Piczak ConvNet | 64.5 % | trained from scratch, no external data | [Piczak 2015](https://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf) |
| L3-Net | 79.3 % | SoundNet audio+video pre-training | leaderboard |
| SSAST | 82.2 % | AudioSet-20k pre-training, fine-tuned | leaderboard |
| Wav2Vec 2.0 (95 M) | 82.3 % | *teacher model*, as reported by a later paper | leaderboard |
| AVTS | 82.3 % | audio+video features, linear classification | leaderboard |
| BYOL-A (5.3 M) | 83.2 % | **linear evaluation on frozen features** | leaderboard |
| BEATs iter3 | 72.7 % | **frozen encoder, linear probing** | leaderboard |
| BEATs (multi-head probing) | 83.2 % | probing, not fine-tuning | leaderboard |
| Dasheng | 54.8 % | **frozen, linear probing** | leaderboard |
| Whisper | 52.8 % | **frozen representation probe** | leaderboard |
| CLAP | 82.6 % | **zero-shot**, text prompts | leaderboard |
| PANNs (CNN14) | 94.7 % | AudioSet pre-training, fine-tuned | [AST paper §3.2](https://arxiv.org/abs/2104.01778) |
| AST-S (ImageNet only) | 88.7 ± 0.7 % | spectrogram transformer, fine-tuned | [AST §3.2](https://arxiv.org/abs/2104.01778) |
| AST-P (ImageNet + AudioSet) | 95.6 ± 0.4 % | spectrogram transformer, fine-tuned | [AST §3.2](https://arxiv.org/abs/2104.01778) |
| BEATs iter3+ | 98.1 % | AudioSet-supervised pre-training + fine-tuning | [BEATs](https://arxiv.org/abs/2212.09058) |

Two lessons from that table, both of which shape the harness:

1. **Frozen is a different question from fine-tuned.** The same model family spans 52.8 % (Whisper
   frozen) to 98.1 % (BEATs fine-tuned with AudioSet supervision). Laya's tower is frozen by design —
   the trainable part is the projector and the head — so the rows marked *frozen probe* are the only
   published numbers that are like-for-like, and the fine-tuned rows are reported as context.
2. **Probing a general audio model with a linear head lands around 73-83 %** when the encoder was
   pre-trained on AudioSet (BYOL-A, BEATs), and around 53-55 % for speech-trained encoders probed
   with a plain linear classifier (Whisper, Dasheng). A wav2vec2-base tower — speech pre-trained,
   frozen — belongs in the second family, which is why its measured probe is compared to *those*
   rows and not to AST.

Human accuracy (81.3 %) sits in the same band as those frozen probes. That is the honest context
for any number this engine produces: beating a linear probe on the same features is the claim worth
making; matching a fine-tuned AudioSet model is not a claim a frozen tower can support.

## 4. What the harness measures

`research/benchmarks/audio_esc50/` builds ESC-50 (and its ESC-10 subset) into Laya's own shape — a
`choice` question over the class names, the recording in `state`, nothing in the text half that could
name the answer — and measures four arms on the identical rows:

| arm | what it is | why it is there |
|---|---|---|
| `chance` | 1 / classes | the floor |
| `centroid` | nearest class mean of the frozen embedding | a head with no learning at all |
| `ridge` | linear probe on the frozen embedding, penalty chosen by inner CV on the training fold only | the standard "trained head on a frozen tower" number, and the one the published probing rows report |
| **Laya** | the engine: frozen tower → projector → block spliced after the option head, trained with `laya-train --audio` | the thing being measured |

Then the two gates from [the evaluation harness](evals.md#media-gates-does-it-actually-look-at-the-picture-or-listen-to-the-recording):
`ablate --modality audio` (does the answer depend on the recording?) and `flip` (does reordering the
options change it?). A Laya number without those two gates would repeat exactly the mistake the
vision program made and then had to fix.

## 5. Cost, on this machine

CPU only (56 cores, no GPU), measured on the first pass:

| step | measured cost |
|---|---|
| decode one 5 s clip (`laya.audio.load_waveform`, 44.1 kHz → mono 16 kHz) | 0.04 s |
| frozen wav2vec2-base over 5 s | ~1.7 s single-threaded-ish; ~2.4 clips/s with 10 workers × 5 threads |
| all 2 000 clips, one tower pass (cached to disk) | ~10-15 min |
| a Laya training row (ModernBERT-large, ~190 tokens incl. 33-token audio block) | ~0.2-0.5 s per row |

That budget is what fixed the shapes: the probe runs all five folds on the full 2 000 clips, and the
engine's cross-validated `state_conditioned` run uses ESC-10 (10 options → 10 rows per clip), where
both gates can be measured inside a session.
