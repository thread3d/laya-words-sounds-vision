# Fusing a vision encoder into a bidirectional, masked-token-scoring decision encoder

**Research report · compiled 2026-11-02 · for the Laya / typed-decision-model (TDM) setting**

Target system under discussion: a ModernBERT-large–class bidirectional encoder (1024 hidden, 28 layers,
8192 positions) whose *head is masked-token scoring inside the sequence*:

```
[CLS] <question-type instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state_text [SEP]
```

Each option is preceded by a `[MASK]`; an option's score is read from the model's prediction of that
option's own tokens. There is no decoder, no autoregressive sampling at inference, and the output must
be a calibrated distribution over a caller-supplied option set (choice / ordinal / noul), plus a
confidence usable for abstention or escalation. Training objectives in play: soft cross-entropy against
a target distribution, and a GRPO-style sampled-logit term scored by a strictly proper scoring rule
(in this project's lineage called *RLCD*).

**How to read the evidence.** Every non-trivial claim is tagged:

- **[Established]** — replicated across multiple papers and settings, or a directly measured result in a
  primary source.
- **[Indirect]** — measured in a nearby setting (usually a decoder-LLM VLM); plausible, not proven, for a
  bidirectional encoder.
- **[Inference]** — my reasoning from architecture config / first principles. Treat as a hypothesis to test,
  not a finding.

**Currency caveat.** The TDM line (Jev and its relatives: Visual Jev, Intern-Decision, RSJEV, UniJev,
OmniMed-Jev, See-the-Options/Iris, and the TypedBench/DecisionRobust/FlipJev evaluation papers) is
September–October 2026 preprints, largely from one ecosystem and not peer reviewed. It is *the* closest
prior art to this system, so it is used heavily below — but its numbers should be read as early evidence,
not settled fact. The classical VLP literature (2019–2024) is peer reviewed and is used for the
architectural lessons.

---

## 0. Verdict summary

| Question | Answer this report supports | Confidence |
|---|---|---|
| Fusion topology for masked-option scoring | **Single-sequence early fusion into the existing bidirectional encoder**, with the option readout staying in-sequence. Add *one* small co-attention/conditioning block rather than only concatenating, if the budget allows. | High that early fusion works; Medium that concat-only is enough |
| Does cross-attention beat naive concat? | In the one controlled ablation that exists (METER), **yes**: cross-attention > merged self-attention, and encoder-only > encoder-decoder. This is the strongest argument against pure concatenation. | [Established] for METER's setting |
| Should options see each other? | **No.** Joint candidate encoding produces 8–10 % answer flips under option reordering; independent/state-conditioned encoding gives exactly 0 % flips. | [Established] (2026, single lab) |
| Score tokens or pooled vectors? | **Tokens.** A pooled candidate encoding mapped visually distinct candidates to cosine 0.999 of each other — pooling destroys exactly the discriminability a decision head needs. | [Established] (2026, single lab) |
| Visual token budget | **64–196 tokens** is enough for coarse classification; **≥ 576–1 024** for OCR/text-in-image; plan a confidence-gated cascade rather than one fixed budget. | High |
| Position ids | Give visual tokens their own explicit position block (2D/grid-aware RoPE, or learned 2D emb) — do **not** let them inherit sequential text positions silently. | [Established] for VLMs; [Inference] for ModernBERT |
| Will the text encoder learn to ignore the image? | **Yes, by default** — that is the documented norm (VALUE, Frank et al.), and the `[MASK]`-before-option design makes the shortcut *architecturally attractive*. Plan for it from step 1. | [Established] |
| Calibration | Decision-interface readout + post-hoc temperature scaling + bin-reweighting (CORA) is the evidence-backed stack; a separate "evidence-sufficiency" head is documented **not** to beat plain confidence gating. | Medium-high |
| Injection via text-in-image | Real channel, but the *decision* threat is different in kind: typed decision models resisted 10 240/10 240 conventional injection attempts and were still flipped 43 % by a short optimised suffix (90–100 % cross-task transfer). | [Established] (2026, single lab) |

---

## 1. Precedents: what was actually built, and what it cost

### 1.1 The classical vision-and-language encoders

| Model | Image input | Fusion topology | Readout / objective | Documented cost or outcome |
|---|---|---|---|---|
| **ViLBERT** ([1908.02265](https://arxiv.org/abs/1908.02265)) | Faster R-CNN regions (≥ 10/region, 100 regions) | **Two-stream + co-attention**: separate text and visual streams, "co-attentional transformer layers" that exchange keys/values; 6 layers per stream + 6 co-attn | MLM (text), masked region modelling (visual), NSP-style alignment; downstream heads per task | Roughly **2.5× BERT-base parameters** (two full streams plus co-attention blocks ≈ 280 M); region extraction alone cost more FLOPs than the multimodal interaction (the complaint ViLT later formalised). Pretrained on Conceptual Captions only (3.3 M pairs). |
| **LXMERT** ([1908.07490](https://arxiv.org/abs/1908.07490)) | Object-relation features (Faster R-CNN, 36 regions) | **Three encoders**: object-relation (5), language (9), **cross-modality (5)** — cross-attention both directions per layer | MLM, masked object prediction (feature regression + label classification), cross-modality matching, image-QA | VQA modelled as **classification over 3 129 answers** — i.e. a closed-set probability readout, an early precedent for typed output. NLVR2 54 % → 76 %. |
| **VisualBERT** ([1908.03557](https://arxiv.org/abs/1908.03557)) | Regions + position/segment embeddings | **Single-stream (early) fusion**: region embeddings concatenated with word embeddings, then ordinary self-attention does implicit alignment | MLM + "visually-grounded" objectives | Simplest possible topology; competitive on VQA/VCR/NLVR2/Flickr30K, and analysis showed implicit grounding emerges without supervision. |
| **UNITER** ([1909.11740](https://arxiv.org/abs/1909.11740)) | Regions (Faster R-CNN) | **Single-stream, deep early fusion** (12 layers); image = region tokens with 7-D position embeddings | MLM + **MRM (3 variants)** + ITM + WRA (optimal transport) | Key methodological finding: **conditional masking** (mask one modality while observing the other fully) beats joint random masking; full task combination best (ablation over 4 tasks). SOTA on 6 task families. |
| **ViLT** ([2102.03334](https://arxiv.org/abs/2102.03334)) | **Raw patches**, linear projection — no conv, no region supervision | Single-stream early fusion | MLM, ITM; whole-word masking + image augmentation to stop trivial solutions | "Up to tens of times faster" than region-based VLP; competitive accuracy. Established that *feature extraction, not fusion, was the bottleneck*. |
| **ALBEF** ([2107.07651](https://arxiv.org/abs/2107.07651)) | ViT patches (unfrozen ViT-B/16) | Two encoders **aligned contrastively first**, then fused by cross-attention | ITM with **hard negatives mined from contrastive similarity**, MLM, momentum distillation | The "align before fuse" lesson: unaligned visual and word tokens make fusion hard to learn. This is the canonical fix for the fusion-learns-nothing failure. |
| **FLAVA** ([2112.04482](https://arxiv.org/abs/2112.04482)) | ViT patches | **Single multimodal encoder (12 layers)** that concatenates projected image patches with text embeddings (plus a multimodal codebook for masked multimodal modelling), alongside unimodal encoders | Contrastive (unimodal + global), **MLM on text, MIM on image patches**, masked multimodal modelling | One model, 35 tasks. Evidence that a *single bidirectional fusion encoder* is competitive across classification-style tasks when the objectives are right. |
| **METER — "An Empirical Study of Training End-to-End Vision-and-Language Transformers"** ([2111.02387](https://arxiv.org/abs/2111.02387)) | CLIP-ViT / Swin patches | Systematic dissection of **merged attention (concat+self-attn) vs co-attention (cross-attn)** and encoder-only vs encoder-decoder | Supervised VLP (MLM+MIM+ITM+…), same recipe for all axes | Findings, quotable: (i) **"the inclusion of cross-attention benefits multimodal fusion … better downstream performance than using self-attention alone"**; (ii) **"under a fair comparison setup, the encoder-only VLP model performs better than the encoder-decoder model"**; (iii) the vision transformer matters more than the language transformer; (iv) higher fine-tuning image resolution gives large gains when resolution/patch ratio is low. 77.64 % VQAv2 test-std with 4 M images. |
| **M³P** ([2006.02635](https://arxiv.org/abs/2006.02635)) | Regions | Single-stream multilingual fusion, with **Multimodal Code-switched Training** | MLM + MRM + ITM + code-switching | Precedent for a *non-English-centred* multimodal encoder; relevant if the decision model must be multilingual (Laya is). |
| **LAVENDER** ([2206.07160](https://arxiv.org/abs/2206.07160)) | Video-text (but same lesson for images) | Single multimodal encoder | **Masked language modelling as the single unified interface** for pre-training *and* all downstream tasks, including multiple-choice VQA | The most direct precedent for the design under discussion: a bidirectional encoder whose *only* head is MLM, answering discriminative tasks by masked-token scoring, competitive on 14 benchmarks. No decoder, no task-specific heads. |

### 1.2 The frozen-encoder "bridges" (relevant only as feature extractors / cost comparisons)

| Model | Topology | What it proves | Cost |
|---|---|---|---|
| **BLIP** ([2201.12086](https://arxiv.org/abs/2201.12086)) | Multimodal mixture of encoder-decoder (unimodal encoders + image-grounded text encoder/decoder), CapFilt bootstrapping | One vision-text pair can serve both understanding (encoder branch) and generation (decoder branch); the encoder branch is a bidirectional fusion encoder | Trains a full ViT + text transformer |
| **BLIP-2** ([2301.12597](https://arxiv.org/abs/2301.12597)) | **Q-Former**: 32 learnable queries compress the frozen ViT into the frozen LLM; two stages (representation → generative) | A 32-token bottleneck plus staged alignment buys large gains: +8.7 % zero-shot VQAv2 over Flamingo-80B with **54× fewer trainable params** | 188 M trainable on top of two frozen models |
| **Flamingo** ([2204.14198](https://arxiv.org/abs/2204.14198)) | **Perceiver Resampler** (fixed 64 visual tokens/image) + **gated cross-attention dense layers** interleaved into a frozen LM | Interleaved visual+text sequences and in-context few-shot; close-ended MCQ VQA explicitly evaluated | 80 B params; cross-attn insertion into every few LM blocks |
| **MiniGPT-4** ([2304.10592](https://arxiv.org/abs/2304.10592)) | One **linear projection** between frozen ViT and frozen Vicuna | The floor of alignment cost: one linear layer + short alignment stage already yields useful behaviour | ~1 projection matrix |
| **DeepStack** ([2406.04334](https://arxiv.org/abs/2406.04334)) | Re-injects visual features at **several early transformer layers** rather than only the input | Visual information injected deep rather than only at the embedding layer improves grounding at low token cost | Trivial params, changes the sequence plumbing |
| **Qwen2-VL M-RoPE** ([2409.12191](https://arxiv.org/abs/2409.12191)) | Vision tokens positioned by **multimodal rotary position embedding** with separate temporal/height/width components | The reference solution for "how do I place image tokens in a RoPE model" at any resolution | — |

**Cost summary that matters for this project [Established]:** the historical cost driver was *visual
feature extraction*, not fusion (ViLT's central argument), then *sequence length* once patches replaced
regions (MM1, Prismatic). With a 8192-position bidirectional encoder, 256–512 visual tokens is a 3–6 %
sequence-length increase — cheap. The real costs are (a) the vision tower forward pass, (b) the *loss of
locality* that extra tokens impose on local-attention layers (§2.4), and (c) the engineering of a custom
attention mask if you want option isolation (§2.3).

### 1.3 2024–2026 "multimodal encoder" and embedding lines

- **MetaCLIP / MetaCLIP-style curation** ([2309.16671](https://arxiv.org/abs/2309.16671)): data curation,
  not architecture, explains CLIP-class quality. Implication: for a decision head, the *text-side* label
  semantics the vision tower was aligned to is a data problem before it is a fusion problem.
- **Perception Encoder (PE)** ([2504.13181](https://arxiv.org/abs/2504.13181)): contrastive-only scales to
  excellent general embeddings, but **"the best visual embeddings are not at the output of the network"** —
  intermediate layers carry them; PE recovers them by *language alignment* (aligning features to an LLM)
  and *spatial alignment* (distillation). Two transferable findings: (i) tap intermediate layers, not the
  final CLS; (ii) for their video aggregation, **average pooling beat attention-based compression**, and
  **"we did not find scaling the text encoder to be beneficial"** for the cross-modal objective.
- **GME** ([2412.16855](https://arxiv.org/abs/2412.16855)), **VLM2Vec** ([2410.05160](https://arxiv.org/abs/2410.05160)),
  **xVLM2Vec** ([2503.09313](https://arxiv.org/abs/2503.09313)), **LamRA** ([2412.01720](https://arxiv.org/abs/2412.01720)):
  the "turn a VLM into an embedding/judge model" family. All read out a **single pooled vector**
  (last-token or EOS hidden state) trained contrastively. Useful as *retrieval/reranking* baselines, but
  they are the wrong readout for a decision head — see the pooling result in §2.3, and note that
  VLM2Vec-style models need an instruction-conditioned vector, i.e. the task lives in the text, not in a
  calibrated distribution.
- **VL-JEPA** ([2512.10942](https://arxiv.org/abs/2512.10942)): predicts *continuous target embeddings*
  instead of tokens, and explicitly reports "discriminative VQA without any architecture modification",
  plus open-vocabulary classification. The best current evidence that a non-generative, embedding-space
  vision-language objective is competitive with token-space training (50 % fewer trainable params in a
  controlled comparison).
- **InternVL-X** ([2503.21307](https://arxiv.org/abs/2503.21307)) and the token-compression line
  (§3.2) for the image→token plumbing.

**Assessment [Indirect].** None of these is a bidirectional encoder + in-sequence masked-token decision
head; they are either contrastive dual encoders (one pooled vector) or decoder LLMs. There is, as of this
writing, **no peer-reviewed precedent for exactly the target architecture**. The two families closest to
it are LAVENDER (MLM-as-universal-interface, 2022) and the 2026 typed-decision models below.

### 1.4 The closest prior art: multimodal typed decision models (Sept–Oct 2026)

This is the literature the target system actually belongs to, and it answers several of the
questions below directly.

| Paper | Design | Findings that bear on this project |
|---|---|---|
| **Visual Jev** ([2609.25845](https://arxiv.org/abs/2609.25845)) | Qwen3-VL-4B backbone; **frozen vision tower, LoRA on the language tower**; image+context encoded once into a shared prefix KV cache; each question's suffix run as a batch row; **196 visual tokens**; probabilities read from the LM head over candidate tokens | Answer-supervised post-training 0.706 → 0.761 macro; 8.9× faster than serial execution (3.4× vs. an already-batched baseline that recomputes the prefix); a **matched typed head gave no consistent accuracy advantage over the LM-head readout**; the appendix reports a **negative result for an evidence-sufficiency head** (see §4.4); scaling 4B→8B only +0.018. |
| **See the Options / Iris** ([2610.33702](https://arxiv.org/abs/2610.33702)) | Visual *candidates*; compares three candidate-encoding designs on one backbone and dataset | **Joint encoding** (candidates see each other) = 90.6/89.5/91.6 % but **8–10 % answer flips under candidate reordering**; **independent** per-candidate encoding = 78.4/72.1/74.8 % with **exactly zero** flips; **state-conditioned** (candidate sees the request but never another candidate) = **92.1/87.0/90.2 % at ⅓ the latency** with calibrated null probability AUROC 0.93–0.97. Pooled candidates that see each other buy nothing: "the encoder maps visually distinct elements to vectors within **cosine 0.999** of each other". Independent encoding *falls short* when candidates must be compared to a query image (MMEB-V2 reranking). |
| **Intern-Decision** ([2610.22131](https://arxiv.org/abs/2610.22131)) | 0.8 B / 2 B / 4 B Qwen3.5 backbones; **vision tower and projector frozen**; "autoregressive masked language modelling": one **decision placeholder per field inside a JSON skeleton**, all fields scored in a **single causal forward pass** via the native LM head | Directly the same *idea* as `[MASK]`-per-field scoring, in a decoder. 90.02 % mean vs. 88.74 % for Jev; **44.16 ms** mean latency for a 289-token, 3-field request (RTX 4090); **temperature scaling** on a calibration split: multiclass Brier 0.628 → 0.550, ECE 0.213 → 0.089; confidence-gated handoff to a stronger model 73.9 % → 87.4 % on JevBench-Hard by routing 42/111 items. |
| **RSJEV** ([2610.08539](https://arxiv.org/abs/2610.08539)) | "OnePass Decider": candidate-conditioned multimodal discriminative decision, **no autoregressive decoding**, 0.8 B | Beats CNN/Transformer/Mamba/CLIP/MLLM baselines on 3 remote-sensing benchmarks at far lower inference cost. Existence proof for "one pass, no generation, image + candidate set → distribution". |
| **OmniMed-Jev** ([2610.00381](https://arxiv.org/abs/2610.00381)) | Medical decisions as `Choice`/`Noul`/`Score` over runtime candidate sets; interface-controlled comparison vs. a generative baseline on the **same backbone, data and schedule** | Decision distribution vs. token-probability: **calibration error cut by up to an order of magnitude**, reliability error by up to 2×, with comparable point accuracy (generative wins only on counting). The authors themselves flag the confound: the comparison cannot separate the interface from the training differences. |
| **UniJev** ([2610.25784](https://arxiv.org/abs/2610.25784)) | Trimodal (text/image/video) System-One models with a shared scoring head; SFT vs. RLCD separated | SFT **improves accuracy over backbone readouts in all three modalities**; **RLCD "changes accuracy little and has inconsistent effects on visual calibration"**; a supervised confidence-bin objective (**CORA**) lowers image/video ECE at 2 B and 9 B while preserving accuracy, without RL. |
| **Beyond CLIP** ([2610.41199](https://arxiv.org/abs/2610.41199)) | Few-shot industrial inspection with Jev / Clef-Flash vs. CLIP, 2 reference images per class | Decision models 0.75–1.00 / 0.71–0.99 vs. **CLIP 0.67–0.84**; text-description refinement helps in 6/8 combinations; a calibration step reduces Brier in 13/15 configurations. |
| **CLEF on breast ultrasound** ([2610.13675](https://arxiv.org/abs/2610.13675)) | Same interfaces on 2 818 BUS images | Heavy prompt sensitivity; best 74.85/66.28/51.63 % and *all* configurations below a majority-class predictor on BUS-BRA. Sobering: multimodal decision models are not uniformly good, and class-specific evaluation is required. |
| **Typed decision models: early evidence audit** ([2609.32160](https://arxiv.org/abs/2609.32160)), **TypedBench** ([2610.11392](https://arxiv.org/abs/2610.11392)), **SBERT2S1** ([2610.02486](https://arxiv.org/abs/2610.02486)), **OpenJev-RLCD** ([2609.38850](https://arxiv.org/abs/2609.38850)) | Reviews/benchmarks/recipes for TDMs | See §4.3 and §5.5 — these contain the only published numbers on the exact training objectives in use here. |

---

## 2. Early fusion vs. cross-attention adapters vs. late fusion

### 2.1 The three families, priced

| Topology | Canonical papers | Params added | Where the image lives | Fits a masked-token option head? |
|---|---|---|---|---|
| **Early / single-stream fusion** (visual tokens concatenated into the text sequence, one bidirectional stack) | VisualBERT, UNITER, ViLT, FLAVA, LAVENDER | Projector only (≈ 0.5–2 M) | Inside the same sequence, same layers, same attention | **Yes, natively** — the `[MASK]` readout already sits in that sequence |
| **Co-attention / cross-attention adapters** (separate or lightly-shared streams, cross-attn blocks; optionally a resampler bottleneck) | ViLBERT, LXMERT, ALBEF, METER's co-attention, Flamingo, BLIP-2 Q-Former | 10–100 M (cross-attn layers, 32–64 query resampler) | Side stream + inserted blocks | **Yes, if the last block still writes into the text stream** — the readout need not move |
| **Late fusion** (independent encoders, combine scores) | CLIP, Iris/See-the-Options independent encoding, VLM2Vec-style pooled embeddings | 0 in the encoder; a projection + score combiner | Outside the sequence | Only for whole-input scoring, not per-option masked tokens — **unless** each option is separately encoded (Iris) |

### 2.2 Which topology is known to *work* for scoring options via masked tokens

**[Established] Early fusion into one bidirectional stack works, and MLM is a sufficient interface.**
UNITER, VisualBERT, ViLT and FLAVA all train masked-token objectives over a sequence holding both
modalities, and LAVENDER shows the extreme case: a multimodal encoder whose *only* head is MLM, applied
to multiple-choice VQA, competitive on 14 benchmarks. There is no evidence that masked-token scoring
breaks under early fusion.

**[Established] Cross-attention adds something that self-attention merging does not.** METER's
controlled ablation (same data, same recipe, one axis at a time) reports cross-attention fusion beating
self-attention-only fusion, and encoder-only beating encoder-decoder. This is the single strongest
published argument that a *pure* "concatenate patches and hope" design is leaving accuracy on the table.
The mechanism is the one ALBEF names: unaligned visual and word tokens make fusion targets hard to learn,
so an explicit query/key interface helps.

**[Established, one lab] Per-candidate independence beats candidate interaction for invariance, and state-conditioning
is the best of both.** See-the-Options measured it: joint encoding flips 8–10 % of answers on reordering;
independent encoding flips 0 % but loses accuracy; the state-conditioned design (the candidate sees the
request/state but never other candidates) recovers the accuracy (92.1/87.0/90.2 %) *and* keeps exact order
invariance *and* runs at ⅓ latency via a forked cache, with a well-ranked null probability.

**[Established, one lab] Pooling kills discriminability.** Same paper: visually distinct candidates were mapped to
vectors within cosine 0.999 after pooling; the gains attributed to "candidate readout" designs were really
gains from *scoring without pooling*. Consequence for this project: the existing masked-token readout is an
asset — do not "simplify" it into a pooled-embedding classifier when adding vision.

**Known to fail, or to cost more than it is worth:**

- **[Established] Region-feature pipelines** (Faster R-CNN + spatial embeddings) are the expensive,
  expressive-power-capped option (ViLT). Do not reintroduce them.
- **[Established] Bottleneck resamplers lose detail.** TokenPacker ([2407.02392](https://arxiv.org/abs/2407.02392))
  motivates itself exactly on this: resamplers/abstractors "fail to capture finer details and undermine
  visual reasoning"; its coarse-to-fine alternative compresses 75–89 % while matching or beating them.
  Q-Former-style 32-query bottlenecks are fine for captioning/VQA, questionable for text-in-image.
- **[Indirect] Full cross-attention everywhere is expensive to retrofit** into an alternating
  local/global FlashAttention encoder (new kernels, new mask plumbing) — Flamingo's solution was gated
  cross-attention *dense layers* at intervals, which is the cheaper variant to consider.

### 2.3 The structural problem specific to `[MASK]`-before-option scoring

**[Inference, but grounded in the order-invariance result of §2.2]** Your sequence puts *all* options in one
bidirectional sequence, so when you score option *i*, options *j ≠ i* are fully visible. Three consequences:

1. **Candidate leakage / non-comparability.** Option *i*'s probability is conditioned on the identity of its
   competitors, so scores are not comparable across requests with different option sets, and a model can
   learn "compare A to B" instead of "compare A to the image". See-the-Options' 8–10 % flip rate is the
   visible symptom of exactly this in a joint-encoding design.
2. **Text-only shortcut amplification.** The scoring target is the option's own token string. Many option
   strings are *predictable from the question/state text alone* (yes/no, "approve"/"deny", or a label
   mentioned in the document). So plain masked-token CE trains the model to maximise
   `P(option string | all text)` — the image contributes nothing to the gradient unless the text is
   deliberately uninformative. **This is the mechanism by which unimodal collapse will happen here, and it
   is built into the readout, not just into the data** (§5.3 for the evidence and for fixes).
3. **Order sensitivity.** Reordering options changes the answer unless trained and measured explicitly.

Mitigations, cheapest first (all [Inference] until measured):

- **Randomise option order in every training example and report a *flip rate*** as a first-class metric.
- **Mask the competitors' attention**: a per-option 2D attention mask that blocks option↔option attention
  (each option may attend to instructions, visual tokens, and state; not to siblings). Implementable as a
  segmented mask if you use a kernel that accepts arbitrary masks; with FlashAttention + local/global
  alternation this needs care and is the main engineering cost.
- **Or score options in separate passes** with a shared prefix cache (Visual Jev's "shared visual context"
  trick; Iris's forked cache) — this gives exact independence and lets you cache the image, at the price of
  N passes over a short suffix rather than 1 pass over a long sequence.
- **Fixed slots**: render each option into fixed letter/position slots so its position id does not depend on
  which option it is (the "letter slots" idiom used by OneJev in the TDM literature).

### 2.4 Position ids and attention masks for interleaved visual tokens in a ModernBERT-class encoder

Facts about the encoder, from the published config
([`answerdotai/ModernBERT-large/config.json`](https://huggingface.co/answerdotai/ModernBERT-large/blob/main/config.json),
architecture paper [2412.13663](https://arxiv.org/abs/2412.13663)):

| Property | Value | Consequence for visual tokens |
|---|---|---|
| `hidden_size` / layers / heads | 1024 / 28 / 16 | 1 024-dim visual tokens need no extra projection width |
| `position_embedding_type` | RoPE, **two** thetas: `global_rope_theta=160000`, `local_rope_theta=10000` | Global and local layers encode distance differently — visual tokens' *distance* to text is seen under two different scalings. (The `position_embedding_type: "absolute"` field in config is legacy and not what the layers do.) |
| `global_attn_every_n_layers` = 3 | ~1 in 3 layers attends globally; the rest are local with `local_attention = 128` | **In local layers, a token sees only ±~63 positions.** Visual tokens appended at the far end of the sequence are invisible to a `[MASK]` position in local layers and reachable only through the ~9–10 global layers. Information *does* flow, but with fewer hops and less bandwidth. Interleaving or adjacency matters here in a way it does not in a dense-attention ViT. |
| `max_position_embeddings` = 8192 | 8 192 positions | See the budget arithmetic in §2.5. |

Practical rules:

1. **[Established for VLMs, Inference here] Give visual tokens their own position space.** The reference
   solutions are M-RoPE's separate temporal/height/width axes (Qwen2-VL, [2409.12191](https://arxiv.org/abs/2409.12191))
   and 2D RoPE for ViTs ([2403.13298](https://arxiv.org/abs/2403.13298)). For ModernBERT this means either
   (a) reusing the 1-D RoPE with a **reserved contiguous position block** per image plus a *type embedding*
   (0 = text, 1 = image) — the cheap VisualBERT/UNITER-style solution — or (b) extending the rotary
   computation to a 2-D/3-D grid for the image span only (a modelling change in the attention kernel).
   Silently letting image tokens consume sequential text positions compresses the text's own positional
   signal and makes "where in the image" and "where in the sentence" the same axis. **[Inference]**
2. **[Established] Type/segment embeddings are the standard cheap disambiguator** (VisualBERT, UNITER,
   FLAVA): without them, the encoder cannot tell a word from a patch at the same position.
3. **[Inference] Placement interacts with local attention.** Two viable layouts:
   *contiguous block* (image tokens grouped, e.g. right after the instructions or after `state_text`) —
   good for locality, bad for reach; or *interleaved* with `[MASK]`/option groups — better for local
   layers, but fragments the image's own spatial adjacency unless the 2-D position structure is preserved.
   Measure both; do not assume.
4. **[Established] Do not make the vision tower's output the fusion input blindly.** DeepStack
   ([2406.04334](https://arxiv.org/abs/2406.04334)) shows re-injecting visual features into *several early
   layers* beats injecting once at the embedding layer. In a bidirectional encoder this is cheap: sum the
   projected visual features into the visual positions at layers `1, 3, 5` (a residual add), rather than
   only at layer 0.
5. **[Established] Mask semantics.** In UNITER, *conditional* masking (mask one modality, observe the other
   fully) beat joint random masking. If you add any visual-side objective (masked-region/feature
   regression), keep it conditional on full text and vice versa.

### 2.5 A concrete sequence layout (inference, to be ablated)

```
[CLS] <qtype> <instructions…> [SEP]
      [VIS] v_1 … v_K [/VIS]                       # K = 196 (see §3)
      [MASK] opt0_tokens [SEP?]                     # each option preceded by [MASK]
      [MASK] opt1_tokens …
[SEP] state_text …
```

with: type embedding = 1 on `v_*` and on `[VIS]`/`[/VIS]`; a reserved position block `[P_vis, P_vis+K)` for
visual tokens (text keeps its own monotone positions); visual features also residual-added at layers 1/3/5;
option↔option attention optionally masked (§2.3).

**Budget arithmetic (8192 total).** Instructions + question type: 100–300. Options: `n_opt × (1 + len(opt))`,
typically 4–16 options × 5–30 tokens = 100–500. State text: up to ~4–6 k. Visual: 196. Everything fits
comfortably; the binding constraint is *not* the context window but (a) latency, and (b) local-attention
reach (§2.4). If state text is long, put the visual block **after** `state_text` and before the options, or
duplicate a single `[VIS_SUMMARY]` token adjacent to each `[MASK]` — the latter is [Inference] but directly
motivated by the 128-token local window.

---

## 3. Token budget engineering

### 3.1 What the measured numbers say

| Source | Setting | Result |
|---|---|---|
| **MM1** ([2403.09611](https://arxiv.org/abs/2403.09611)) | 64 vs. 144 image tokens × 224/336 px; avg-pool / attention-pool / C-Abstractor | "**Number of visual tokens and image resolution matter most, while the type of VL connector has little effect**" — zero/few-shot rises with both; at 336 px/144 tokens all three connectors are within noise; at low token counts average-pooling equals C-Abstractor, at higher counts the convolutional abstractor wins. Frozen image encoder better at low resolution (+2.2), unfreezing better at 1344 px (+2.9). |
| **Prismatic VLMs** ([2402.07865](https://arxiv.org/abs/2402.07865)) | Design-space ablation (backbone, projector depth, resize strategy, single-stage vs. staged) | A **2-layer MLP projector** is enough; an explicit projector-pretraining stage is **unnecessary**; resize strategy and which frozen features you fuse (DINOv2 + SigLIP) move the needle more than projector depth. |
| **Perception Encoder** ([2504.13181](https://arxiv.org/abs/2504.13181)) | Video/image aggregation & alignment | **Average pooling beat attention-based compression** for aggregation; scaling the *text* encoder did not help the cross-modal objective; the useful features are in intermediate layers. |
| **Token merging / pruning** | ToMe ([2210.09461](https://arxiv.org/abs/2210.09461)), PruMerge ([2403.15388](https://arxiv.org/abs/2403.15388)), TokenPacker ([2407.02392](https://arxiv.org/abs/2407.02392)) | ToMe: 2× ViT throughput at 0.2–0.3 % accuracy drop, training-free. PruMerge: 14× visual-token compression on LLaVA-1.5 with comparable VQA. TokenPacker: 75–89 % compression "comparable or better" — by *keeping* high-resolution region cues rather than dropping them. |
| **Visual Jev** ([2609.25845](https://arxiv.org/abs/2609.25845)) | Production-ish decision serving | **196 visual tokens** (a 14×14 grid) for GQA-choice / SNLI-VE-style forced choice. |
| **Intern-Decision** ([2610.22131](https://arxiv.org/abs/2610.22131)) | 0.8–4 B decision models | Fixed 289-token request (text + image reference) at 44 ms; visual budget not the bottleneck at these sizes. |
| **Confidence under visual token pruning** ([2604.12035](https://arxiv.org/abs/2604.12035)) | Pruning + calibration study across selectors and MLLMs | When the *queried object* loses all its tokens: accuracy 59 % → **17 %**, but **confidence stays at the unpruned level**. Most-attended-token selectors produce confident errors that **temperature scaling cannot re-rank**. Selectors that avoid redundant tokens stay near unpruned calibration. Used in reverse as a risk-controlled cascade with conformal risk control: ~⅓ of the prefill tokens while bounding deviation from the unpruned model. |

### 3.2 Compression strategies, ranked for this use case

1. **2-D adaptive pooling to a fixed grid** (MM1's 144-token setting; Visual Jev's 196) — simple,
   predictable, keeps spatial layout for 2-D RoPE. **Default for classification-style decisions.**
2. **Pixel-shuffle / patch-merging at the ViT output** (InternVL-style 2×2 shuffle into the width) — same
   family as (1), implemented in the tower; good when you want resolution × tokens as two dials.
3. **Coarse-to-fine projectors** (TokenPacker) — the right choice when the task is OCR-ish: keeps
   high-resolution region cues as keys/values instead of discarding them.
4. **Attention-based resampling** (Perceiver Resampler, Q-Former) — compact (32–64 tokens) and excellent
   for generative bridges; **loses fine text** ([Indirect] TokenPacker's motivation; PE's pooling result is
   the counter-datapoint for aggregation).
5. **Token merging / pruning at inference** (ToMe, PruMerge, FastV-style selectors) — free speed for a fixed
   tower, but the pruning-confidence study above is a direct warning: pruning removes *evidence* without
   moving confidence, which is precisely the worst failure for a calibrated decision system.
6. **Learned compression with a relevance objective** ([EvoComp](https://arxiv.org/abs/2604.17087), 2604.17087)
   — promising, immature.

### 3.3 Classification vs. OCR-heavy: how few tokens actually suffice

- **[Established] Coarse classification saturates early.** 64–196 tokens carry enough signal for scene/label
  choice tasks (MM1: performance rises with tokens but 64 is usable; Visual Jev uses 196 and matches
  much larger budgets in a matched-scaling comparison; RSJEV does one-pass scene classification at 0.8 B).
  CLIP-style zero-shot uses a *single* pooled vector for the same job — with materially lower accuracy on
  fine-grained classes, which is exactly what the decision-model line buys back.
- **[Established] Text-in-image is a different regime.** Resolution and token count dominate: MM1 finds both
  matter and the connector does not; METER finds large fine-tuning-resolution gains when the
  resolution/patch ratio is low; PE reports 94.6 DocVQA / 80.9 InfographicVQA only at high resolution;
  and the DeepSeek-OCR critique ([2601.03714](https://arxiv.org/abs/2601.03714)) shows the failure mode
  from the other direction: **strip the language prior and OCR accuracy collapses from ~90 % to ~20 %**,
  and **lower visual token counts correlate with greater reliance on priors** — i.e. a token-starved
  model "reads" by hallucinating plausible text. For a decision system, that is a calibration
  catastrophe disguised as high accuracy.
- **[Inference] Practical policy.** Budget **one cheap pass** (64–196 tokens) for the common case, and
  escalate the *same request* to a high-resolution pass (576–1 024 tokens, tiles or coarse-to-fine
  projector) when the decision's confidence is low **or** when the question type is declared OCR/text-in-image
  by the caller. This mirrors the conformal-risk-controlled cascade of [2604.12035](https://arxiv.org/abs/2604.12035)
  (~⅓ tokens while bounding deviation from the unpruned model) and Visual Jev's shared-prefix cheapness.
  Do **not** adopt aggressive token pruning as a default: it is the documented recipe for confident errors.

---

## 4. Calibration and abstention in multimodal decision systems

### 4.1 The baseline facts

- **[Established] Post-hoc temperature scaling** on a held-out split is the workhorse (Guo et al.,
  [1706.04599](https://arxiv.org/abs/1706.04599)) and it is what the TDM line actually uses: Intern-Decision
  reports multiclass Brier 0.628 → 0.550 and ECE 0.213 → 0.089 on a 96-case pilot; the Beyond-CLIP
  inspection study reduces Brier in 13/15 configurations; SBERT2S1 finds that "after temperature scaling,
  no objective is clearly better calibrated than cross-entropy" ([2610.02486](https://arxiv.org/abs/2610.02486)).
- **[Established] Zero-shot CLIP is miscalibrated**, and a *single* learned temperature transfers across
  datasets and prompts for a given CLIP model ([2303.12748](https://arxiv.org/abs/2303.12748)).
- **[Established] Modern architectures are better but not calibrated under shift** (Minderer et al.,
  [2106.07998](https://arxiv.org/abs/2106.07998)).
- **[Established] MLLMs are miscalibrated, and fine-tuning/multimodal training does not reliably change
  that**; they tend to answer rather than abstain ([2412.14660](https://arxiv.org/abs/2412.14660)).
  Verbalised confidence is a poor substitute ([2405.02917](https://arxiv.org/abs/2405.02917)).
- **[Established] A decision interface changes what the probability *means*.** OmniMed-Jev's framing is the
  cleanest statement: a generative model's probability is "a probability about the next token rather than
  about the decision"; making the decision distribution the output cut calibration error by up to an order of
  magnitude and reliability error by up to 2× on the same backbone/data/schedule ([2610.00381](https://arxiv.org/abs/2610.00381)).
  Caveat from the same paper: the comparison cannot separate interface from the training differences it
  implies.

### 4.2 Does fusion degrade calibration?

Three mechanisms, all documented:

1. **[Established] Missing/degraded modality wrecks calibration, and the damage is not just information
   loss.** Removing vision from a VLM causes "large drops in accuracy and **severe miscalibration**", and
   even when a text description preserves the key content, confidence becomes unreliable
   ([2605.12517](https://arxiv.org/abs/2605.12517)). The proposed fix (a lightweight cross-attention module
   predicting *imagined* latent visual embeddings from text) improves both accuracy and ECE — interesting in
   its own right as a "text-only fallback" design.
2. **[Established] More modalities can be worse than text alone.** In biomedical decision making, "text-only
   reasoning consistently outperforms vision-only or vision–text settings, with multimodal inputs often
   performing worse than text alone", and the authors attribute it to weak *grounded* visual understanding
   ([2512.13747](https://arxiv.org/abs/2512.13747)). A fusion that is not forced to use the image produces
   exactly the pattern: added variance, no added signal, worse calibration.
3. **[Established] Pruning/compression breaks confidence silently** ([2604.12035](https://arxiv.org/abs/2604.12035)),
   and **temperature scaling cannot fix it** because the ranking itself is wrong. Any token-budget reduction
   must be validated on calibration, not only accuracy.

What *helps*, with evidence: post-hoc temperature scaling; **confidence-bin reweighting** (CORA:
group predictions by confidence and push each prediction's confidence toward the empirical accuracy of its
bin — lowers image/video ECE at 2 B and 9 B *without* RL, [2610.25784](https://arxiv.org/abs/2610.25784));
modality-wise calibration splits (fit separate temperatures per modality-availability regime —
[Inference] from the missing-modality findings); and conformal methods for the *decision*, not for the
accuracy (§4.4).

### 4.3 Abstention: what works, and what is documented not to

- **Conformal prediction** gives distribution-free coverage for sets/scores
  ([2107.07511](https://arxiv.org/abs/2107.07511)), including image classifiers with adaptive sets
  ([2009.14193](https://arxiv.org/abs/2009.14193)); the NLP survey is [2405.01976](https://arxiv.org/abs/2405.01976);
  modality-fusion variants exist ([2608.07183](https://arxiv.org/abs/2608.07183)). The right use in this
  system is **risk control on the decision** (choose a confidence threshold so the abstain-rate/error
  trade-off is guaranteed at a stated risk level), which is exactly what [2604.12035](https://arxiv.org/abs/2604.12035)
  does for token budgets.
- **[Established negative result] A separate "evidence-sufficiency" head is not a better gate than the
  decision's own confidence.** Visual Jev trained an answerability head and compared it on the deployment
  mix (intact, degraded, matched controls): the plain decision-CE baseline ordered the stream *better*
  — AURC 0.0362 vs. 0.0456 for the sufficiency variant, confidence AUROC 0.839 vs. 0.813 — and in no case
  did gating close the gap. Their explanation is the useful part: a model that abstains whenever evidence is
  damaged abstains on items it would have answered correctly anyway ("even with the evidence region
  destroyed, the backbone is still right on roughly three quarters of them, from context and from prior").
  **Gate on calibrated decision confidence, not on a second opinion about the input.** [Established, single lab]
- **[Established] Fallback/`noul` options are unreliable in their own right.** The Paired Fallback Robustness
  test ([2610.19856](https://arxiv.org/abs/2610.19856)) finds that correct recognition of a category does not
  transfer to choosing the right fallback, errors persist even when the correct category is supplied, and
  *logically equivalent phrasings of the same fallback rule* produce substantial disagreement. If `noul` is
  the abstention channel, it must be trained and tested as its own skill, with paraphrase-consistency checks.
- **[Established] Confidence quality ≠ accuracy.** In a 0.15 B–9 B sweep of System-One models, "accuracy,
  calibration and sensitivity … are distinct: models of similar accuracy differ widely in AUROC"
  ([2610.09683](https://arxiv.org/abs/2610.09683)). Report AUROC/AURC, not just ECE.
- **[Established] Underconfidence is a real failure mode for TDMs.** TypedBench finds the hosted model
  "systematically underconfident" and that, under asymmetric costs, **using its probabilities can be worse
  than taking its top answer** ([2610.11392](https://arxiv.org/abs/2610.11392)) — the mirror image of the
  usual LLM overconfidence. Measure both directions.
- **[Established] Confidence-gated escalation works** when the gate's ranking is good: routing the least
  confident 30 % of decisions produced gains proportional to the actor's AUROC (rank correlation 0.87), and
  Intern-Decision's handoff lifted 73.9 % → 87.4 % on hard items.

### 4.4 A calibration/abstention recipe for this system

1. Train with the **decision distribution as the output** (already the design) — this is the best-documented
   single lever for making the number a probability *of the decision*.
2. **Temperature scaling** on a per-modality-availability split (image present / image absent / degraded).
3. **CORA-style confidence-bin reweighting** as a second, supervised correction; validate that it preserves
   accuracy.
4. **Conformal risk control** on the abstention threshold, with the risk defined on the *decision* (e.g.
   wrong-and-acted-on), evaluated by risk–coverage curves on a deployment mix that includes degraded imagery.
5. **Never** gate on a separate "is the image sufficient?" classifier (§4.3).
6. Report: Brier (multiclass), ECE with finite-sample noise floor, AURC, AUROC, risk–coverage, flip rate
   under option shuffling, and accuracy under image ablation.

---

## 5. Training recipe

### 5.1 Staged alignment, then fusion, vs. joint from the start

| Evidence | What it supports |
|---|---|
| **ALBEF** ([2107.07651](https://arxiv.org/abs/2107.07651)): contrastive alignment *before* fusion, and hard negatives mined from the contrastive similarity for ITM | **[Established]** unaligned visual/word tokens make fusion objectives hard; align first |
| **BLIP-2** ([2301.12597](https://arxiv.org/abs/2301.12597)): stage 1 representation learning with frozen ViT and frozen LLM, stage 2 generative bootstrapping | **[Established]** two-stage beats joint at a fraction of the trainable parameters |
| **Prismatic** ([2402.07865](https://arxiv.org/abs/2402.07865)): single-stage pipeline; an explicit projector-pretraining stage is unnecessary; frozen-vs-unfrozen depends on resolution | **[Established]** a separate alignment *stage* is not magic; stage boundaries matter less than resolution/data/backbone choices |
| **Perception Encoder** ([2504.13181](https://arxiv.org/abs/2504.13181)): "language alignment" (features → LLM) is what moves embeddings to the end of the network | **[Established]** aligning to the *consumer* of the features is the productive objective — here the consumer is the ModernBERT stack with its masked-token head |
| **MM1** ([2403.09611](https://arxiv.org/abs/2403.09611)): caption + interleaved + **text-only** data mix matters most | **[Established]** text-only data is not optional — it protects language quality while vision is added |
| **Visual Jev** ([2609.25845](https://arxiv.org/abs/2609.25845)): frozen vision tower + LoRA on the language tower + answer supervision, 3 000 updates, ~45 min on one RTX 5090, 196 visual tokens | **[Established]** a small, cheap, single-stage recipe is sufficient for decision quality in a 4 B-class model |

**Recommended for this project [Inference, from the above]:** three short stages, each with an exit gate —
(1) **projector warm-up**: train only the visual projector + type/position embeddings, with the ModernBERT
weights frozen, on image-caption/retrieval-style objectives *plus* the decision objective where the text is
deliberately uninformative (this forces image use early — see §5.3); gate on "image-ablated accuracy drops";
(2) **joint decision training**: unfreeze the top *k* encoder layers + LoRA, train decision CE + alignment on
the real option sets; gate on "accuracy with image − accuracy without image > 0 on the *image-required*
slice"; (3) **calibration**: freeze, fit temperature + CORA; gate on ECE/AURC.

### 5.2 Parameter-efficient options

- **[Established] LoRA on the language tower with a frozen vision tower is enough** for decision quality
  (Visual Jev). **[Established]** Intern-Decision freezes both the vision tower *and* the projector and
  fine-tunes only the language backbone.
- **[Established] Frozen vs. unfrozen image encoder is resolution-dependent** (MM1: frozen better at low
  resolution, unfrozen better at 1344 px). So decide with the token/resolution budget, not by principle.
- **[Established] A 2-layer MLP projector is sufficient** (Prismatic); adapters/resamplers need to earn their
  keep. If you want cross-attention, add it as *gated adapter layers* at a few depths (Flamingo-style)
  rather than rebuilding the stack.
- **[Established] Do not scale the text encoder for the alignment objective** (PE's finding); spend that
  budget on resolution/tokens.

### 5.3 Modality collapse / unimodal bias: the evidence, and why this architecture invites it

- **[Established] Pretrained V+L models attend to text far more than images**, with a minority of heads
  carrying cross-modal interaction (VALUE probing, [2005.07310](https://arxiv.org/abs/2005.07310)).
- **[Established] The asymmetry is measurable by ablation**: models are much more damaged by ablating text
  than by ablating vision, i.e. "not symmetrically cross-modal"
  (Frank, Bugliarello, Elliott, [2109.04448](https://arxiv.org/abs/2109.04448)).
- **[Established] Contrastive objectives let models pass benchmarks without composition/order information**
  (ARO, [2210.01936](https://arxiv.org/abs/2210.01936)) — the fix there is **composition-aware hard negatives**.
- **[Established] The two modalities live in different cones** at init, and contrastive learning preserves a
  gap ([2203.02053](https://arxiv.org/abs/2203.02053)) — a reason to stage/project rather than assume
  concatenation "just works".
- **[Established] The vision tower often contains the information the decision stage fails to use**: linear
  probes solve tasks the full model fails, and VLMs fail at trivial spatial questions while near-perfect when
  shapes are separated ([2407.06581](https://arxiv.org/abs/2407.06581)). The bottleneck is the interface, not
  the features.
- **[Established, 2024–2026]** attention-level diagnosis and rebalancing: modality bias measured through
  attention ([2508.02419](https://arxiv.org/abs/2508.02419)), gradient-based modality rebalancing
  ([2503.13834](https://arxiv.org/abs/2503.13834)), and the blunt clinical finding that text-only > vision or
  vision+text ([2512.13747](https://arxiv.org/abs/2512.13747)).
- **Why `[MASK]`-before-option makes it worse [Inference]:** the training target is the option's own token
  string; whenever that string is predictable from the text (yes/no, a label named in the state text, a
  canonical ordinal word), CE is minimised with zero contribution from the image. The image becomes
  *decoration*, and the model is simultaneously being trained to score options and to imitate the text
  prior. This predicts the exact pathology reported for VQA-style models — and predicts that
  **image-ablated accuracy will be high**, which is the metric to watch.

### 5.4 Techniques that have evidence behind them

| Technique | Evidence | Notes for this architecture |
|---|---|---|
| **Modality dropout** (randomly drop the image tokens during training) | Standard in multimodal fusion; used for robustness to missing modalities; the missing-modality calibration work shows the text-only regime needs its own calibration ([2605.12517](https://arxiv.org/abs/2605.12517)) | Cheap and useful, but on its own it *teaches* the text-only solution. Pair it with image-required examples (below) so dropout is a robustness measure, not the easiest path. |
| **Masked-region / masked-image modelling (conditional)** | UNITER's MRM variants + conditional masking ([1909.11740](https://arxiv.org/abs/1909.11740)); M³P ([2006.02635](https://arxiv.org/abs/2006.02635)); FLAVA's MIM ([2112.04482](https://arxiv.org/abs/2112.04482)) | Forces the encoder to *represent* visual content, independent of the decision loss. In a bidirectional encoder this is nearly free: add a lightweight visual-feature-regression head on a random subset of visual positions. |
| **Contrastive alignment before/alongside fusion** | ALBEF ([2107.07651](https://arxiv.org/abs/2107.07651)); PE language alignment ([2504.13181](https://arxiv.org/abs/2504.13181)) | Aligns the projector cheaply; also gives you a retrieval baseline for free. |
| **Hard negatives that require the image** | ALBEF ITM hard negatives; ARO composition-aware negatives ([2210.01936](https://arxiv.org/abs/2210.01936)); **Visual Jev's grey-image baseline** and the "image-required" slice design | Strongest single fix: construct option sets whose correct answer *cannot* be inferred from text. Measure accuracy on this slice separately; if it is at chance, fusion is not working regardless of aggregate accuracy. |
| **Text-only data in the mix** | MM1 data lesson ([2403.09611](https://arxiv.org/abs/2403.09611)) | Preserves language quality; does not by itself fix collapse. |
| **Vokenization-style grounded supervision** | [2010.06775](https://arxiv.org/abs/2010.06775) (pre-LLM, still the clearest formulation of "make the visual signal a token-level target") | Conceptually the same trick as masked-token scoring with a vision-conditioned target; useful if you want an auxiliary objective on the `[MASK]` slots themselves. |
| **Attention/gradient rebalancing** | [2508.02419](https://arxiv.org/abs/2508.02419), [2503.13834](https://arxiv.org/abs/2503.13834) | Diagnose with attention maps before adding machinery. |
| **Confidence-bin calibration** | CORA ([2610.25784](https://arxiv.org/abs/2610.25784)) | Fixes *confidence*, not collapse; do not confuse the two. |

### 5.5 Soft cross-entropy vs. a proper-scoring RL term (the "RLCD" question)

This is the part of the design with the most directly relevant published numbers, all 2026 and all
single-ecosystem:

- **The RLCD idea, as implemented in this lineage** ([2609.38850](https://arxiv.org/abs/2609.38850)): sample a
  rationale, read *the answer distribution the model commits to afterwards*, and score that sampled
  distribution with a **strictly proper scoring rule**. Two theoretical points worth importing: (i) a
  variance identity shows that scoring the *mixture* of several samples rewards rationales for disagreeing,
  and **RLVR is exactly the mixture objective without its diversity term**; (ii) the per-rationale objective
  has a calibrated optimum, but naive optimisation either switches reasoning off or is drowned by
  high-variance policy gradients. Recommended order: **calibrate, then reinforce**.
  (Note the name collision with the 2023 "RLCD: Reinforcement Learning from Contrastive Distillation"
  ([2307.12950](https://arxiv.org/abs/2307.12950)), which is unrelated.)
- **Direct comparison against cross-entropy** ([2610.02486](https://arxiv.org/abs/2610.02486)): "the released
  RLCD recipe of open System One models **trails cross-entropy by 2.5–3.0 points**"; the deficit "stems
  mainly from its reward normalisation, which inflates the noisy score-function term 3.6–15-fold"; an
  **unbiased leave-one-out estimator recovers most of the gap**; and after temperature scaling **no objective
  is clearly better calibrated than cross-entropy**.
- **Trimodal evidence** ([2610.25784](https://arxiv.org/abs/2610.25784)): SFT improves accuracy in all three
  modalities; **RLCD "changes accuracy little and has inconsistent effects on visual calibration across
  recipes"**. Their conclusion — add a separate supervised confidence objective (CORA) rather than relying on
  RL for calibration — is the practical recommendation.
- **GRPO** itself ([2402.03300](https://arxiv.org/abs/2402.03300)) was designed for verifiable-reward
  reasoning; the TDM literature's consistent finding is that verifiable-reward RL tends toward
  **overconfidence** ([2609.38850](https://arxiv.org/abs/2609.38850)). If a sampled-logit term is kept, it
  should be judged on **Brier/log-score and calibration**, never on accuracy alone, and it should be
  compared against a plain-CE control under identical temperature scaling.

**[Inference] Recommendation:** treat soft-CE as the primary learner; keep any sampled-logit proper-scoring
term as a **small-weight auxiliary** with an unbiased (leave-one-out) estimator and explicit variance
control; use a supervised confidence objective (CORA-style) for calibration; and make "no objective beats CE
after temperature scaling" the null hypothesis your experiments must reject.

### 5.6 Collapse-detection gates (put these in the training loop)

1. **Image-ablated accuracy** on held-out data: if it is within noise of full-image accuracy on the
   *image-required* slice, the fusion is not being used. (This is VALUE/Frank's ablation protocol, applied
   to a decision head.)
2. **Grey-image and matched-control accuracy** (Visual Jev's design) — a floor, not a target.
3. **Attention mass on visual positions** per layer/head; a collapse shows up as near-zero mass in most
   heads (VALUE; [2508.02419](https://arxiv.org/abs/2508.02419)).
4. **Option-order flip rate** (§2.3).
5. **ECE/AURC per modality regime** (image / degraded / absent).

---

## 6. Adversarial robustness: text inside images, and what a non-generative decision system should do

### 6.1 What has been demonstrated against VLMs

| Attack | Mechanism | Reported effect |
|---|---|---|
| **FigStep** ([2311.05608](https://arxiv.org/abs/2311.05608)) | Harmful instructions **rendered as typographic images** — black-box jailbreak, no text prompt | 82.5 % average attack success on six open-source LVLMs; the authors attribute it to **safety alignment being absent for visual embeddings** |
| **Abusing images and sounds for indirect instruction injection** ([2307.10490](https://arxiv.org/abs/2307.10490)) | Adversarial perturbation blended into an image; when the user asks about the image, the perturbation steers the output | Proof of concept on LLaVA and PandaGPT |
| **Image Hijacks** ([2309.00236](https://arxiv.org/abs/2309.00236)) | Adversarial *images* control generative models at runtime (output hijack, jailbreak, prompt leakage) | Established the runtime-control framing |
| **Image-based Prompt Injection (IPI), black-box** ([2603.03637](https://arxiv.org/abs/2603.03637)) | Segmentation-based region choice + adaptive font scaling + background-aware rendering to hide instructions in natural images | Up to 64 % attack success under stealth constraints against GPT-4-turbo |
| **Covert visual prompt injection** ([2603.29418](https://arxiv.org/abs/2603.29418)) | Imperceptible perturbation + bounded text overlay, optimised to match text-rendered targets at coarse and fine granularity | Beats prior methods against closed-source MLLMs |
| **Typographic injection against agents** ([2510.04257](https://arxiv.org/abs/2510.04257)), **goal hijacking via visual prompt injection** ([2408.03554](https://arxiv.org/abs/2408.03554)), **adversarial images hijacking web agents** ([2610.09240](https://arxiv.org/abs/2610.09240)) | Agentic settings where the screenshot is the untrusted input | The realistic threat model for a decision model that takes screenshots |

### 6.2 What is different when nothing is generated

**[Established, single lab] The attack surface shrinks and changes shape.** FlipJev
([2610.38895](https://arxiv.org/abs/2610.38895)) red-teamed a typed decision model and found:

- **"Jev resists common prompt injections: no strategy changes a correct answer in 10 240 paired tests."**
  With a finite output space and no free-text channel, classic injection (get the model to say X) has little
  to grab onto.
- **But a short black-box suffix flips confident answers within minutes**: one suffix per dataset flips
  **43 %** of unseen documents, and a suffix generated on one task flips **90–100 %** of samples on *other*
  tasks. Universal, transferable adversarial suffixes are the real threat.

**[Established] Robustness to perturbation ≠ robustness to injection.** DecisionRobust
([2610.19118](https://arxiv.org/abs/2610.19118)) evaluates 30 text and 17 vision configurations of
JEV-style decision models: image perturbations transfer weakly across models even with multiple surrogates,
relative robustness varies by task and attack type, and **"label-consistency checks substantially reduce
measured attack success across text and image settings, highlighting the importance of validating apparent
failures."** That last clause is a warning about *benchmark validity* as much as about models: many
apparent successes are the model producing a valid alternative label.

### 6.3 The "render untrusted text as an image" defence is a trap for *your* architecture

Render-Before-Reading ([2609.36121](https://arxiv.org/abs/2609.36121)) documents a systematic **modality
asymmetry**: MLLMs are *more* likely to follow adversarial instructions when they arrive as **text** than
when the same instruction is delivered as an **image** — attributed to text-centric instruction tuning. Their
training-free defence (render all untrusted payloads as typographic images before the model sees them) cuts
attack success across ten models and two benchmarks, *and* they show that benign fine-tuning on
image-rendered instructions erodes the asymmetry.

For a system that *fuses a vision encoder into the decision encoder*, this is a serious wrinkle:
**rendering untrusted text as an image still delivers the instruction into the model — through the exact
channel you just made more powerful.** The defence relies on the payload being *harder to obey* in the image
channel; a model trained to read text out of images for OCR-type decisions is precisely the model that
weakens that assumption. **[Inference, but a direct consequence of the two results.]** Keep rendered-text
handling and instruction following on separate, explicitly-trained pathways, and never let the image channel
be treated as instruction-bearing.

### 6.4 What a non-generative decision system should actually do

1. **Capability separation, not persuasion.** CaMeL ([2503.18813](https://arxiv.org/abs/2503.18813)) is the
   right mental model: extract control flow from the *trusted* request; untrusted data can influence the
   *values* a decision returns but never the *program flow*. In this architecture: **the option set, the
   question type and the instruction template come from the caller (trusted); the image and the state text
   are data.** Enforce this structurally — the image must be unable to introduce options or change the
   question type, and the parser must reject any output that is not one of the supplied option ids.
2. **Provenance markers instead of prompts.** Spotlighting ([2403.14720](https://arxiv.org/abs/2403.14720))
   reduced indirect-injection success from > 50 % to < 2 % with minimal utility loss, by making provenance a
   continuous signal in the input. The architectural analogue here: separate **type embeddings** for
   trusted instructions / options vs. untrusted state text vs. visual tokens, plus delimiter tokens, so the
   distinction is a learned feature of every layer rather than an instruction the model may ignore.
3. **Treat "text found inside the image" as a distinct, declared sub-task.** If the decision genuinely needs
   OCR (e.g. "does the banner say success?"), the correct move is a *typed* question whose option set is the
   extracted string vs. distractors, evaluated with an OCR-specific high-resolution pass — not free-form
   reading that can carry instructions. This also matches the token-budget evidence in §3.3.
4. **Label-consistency and output-space validation** ([2610.19118](https://arxiv.org/abs/2610.19118)):
   verify that the emitted distribution is over the supplied options, that paraphrases of the same question
   agree, and that the decision is invariant to option order. Report these as robustness metrics.
5. **Assume optimised suffixes exist.** Since a few query-black-box suffixes flip 43 % / transfer 90–100 %
   across tasks, robustness must be measured with **adaptive attacks trained against your model**, not with
   prompt-injection suites. Budget for input-side defences (randomised resizing/padding, compression,
   ensembling over transforms) and, more importantly, for **abstention**: an attacked input tends to be
   confidently wrong, so pair every decision with the calibration stack of §4 and escalate when two
   preprocessing variants disagree.
6. **Red-team the decision, not the text.** The failure to fear is not "the model says something nasty", it
   is "the model returns `approve` with p = 0.98 on an adversarial document". Your evaluation should
   therefore include: confidence–correctness curves *under attack*, flip rate under paraphrase, and
   disagreement rate across input transforms.

---

## 7. Established vs. speculative: one-table summary

| Claim | Status |
|---|---|
| Masked-token scoring over a fused bidirectional sequence can solve multiple-choice / VQA-style decisions | **Established** (UNITER, ViLT, FLAVA, **LAVENDER** as the purest case) |
| Cross-attention fusion > merged self-attention fusion, and encoder-only > encoder-decoder, all else equal | **Established** in METER's controlled setting; **indirect** elsewhere |
| Candidate interaction breaks order invariance; state-conditioned independent candidate encoding is best | **Established**, single lab (See the Options) |
| Pooled embeddings destroy candidate discriminability (cosine 0.999) | **Established**, single lab |
| Visual token count and resolution matter more than connector architecture | **Established** (MM1); corroborated by Prismatic/PE |
| 64–196 visual tokens suffice for coarse classification; OCR needs high resolution and far more tokens | **Established** (MM1, PE, DeepSeek-OCR critique) |
| Aggressive token pruning yields confident errors that temperature scaling cannot fix | **Established** (2604.12035) |
| A decision interface improves calibration relative to token probabilities | **Established with a stated confound** (OmniMed-Jev) |
| Temperature scaling helps; confidence-bin reweighting helps further; RL (RLCD) is inconsistent for visual calibration | **Established** (Intern-Decision, SBERT2S1, UniJev) |
| A separate evidence-sufficiency head is a worse abstention gate than decision confidence | **Established**, single lab (Visual Jev appendix) |
| Fallback/`noul` rules are unreliable and paraphrase-sensitive | **Established**, single lab (2610.19856) |
| Pretrained V+L models are text-dominant; the bias is measurable by modality ablation | **Established** (VALUE, Frank et al., ARO) |
| Typed decision models resist conventional prompt injection but are flipped by optimised universal suffixes | **Established**, single lab (FlipJev) |
| A single fused ModernBERT + ViT encoder can be trained with a small staged recipe at low cost | **Indirect** (Visual Jev's LoRA recipe at 4 B; no bidirectional-encoder instance published) |
| Image tokens' 1-D RoPE positions should be a reserved block with type embeddings; 2-D RoPE may be better | **Inference** from Qwen2-VL / 2-D RoPE ViT / VisualBERT-style type embeddings |
| ModernBERT's 128-token local window changes optimal visual-token placement | **Inference** from config (no published study) |
| Pure concatenation is enough (no cross-attention block) | **Unknown / likely not** — METER argues otherwise; untested for this head |
| Masking option↔option attention improves calibration/order invariance | **Inference** — untested, but directly motivated by the joint-encoding flip result |

---

## 8. Suggested first experiments (cheap → expensive)

1. **Ablation harness before any training change.** Image-ablated accuracy, grey-image accuracy, and
   attention mass on visual positions for an existing text-only model with visual tokens injected at zero
   weight. This gives the collapse baseline and the eval plumbing (§5.6).
2. **Topology A/B at fixed budget**: (A) pure early fusion (concat + type emb + reserved position block),
   (B) A + a single gated cross-attention block after layer 2 (Flamingo-style), (C) A + DeepStack re-injection
   at layers 1/3/5. 196 visual tokens everywhere. Measure accuracy on the image-required slice, flip rate,
   ECE.
3. **Option-interaction test**: joint option scoring vs. per-option scoring with a shared prefix cache vs.
   option↔option masked attention. Measure flip rate and accuracy on distractor-heavy sets.
4. **Token-budget sweep** at 64 / 144 / 196 / 576 / 1 024 on a classification suite and on a text-in-image
   suite; check whether the pruning/OCR evidence reproduces; then implement the confidence-gated cascade
   (§3.3).
5. **Calibration sweep** after (1)–(4): temperature, temperature-per-modality-regime, + CORA +
   conformal risk control on the abstain threshold; report Brier, ECE (with noise floor), AURC, risk–coverage.
6. **Objective A/B**: soft-CE vs. soft-CE + proper-scoring sampled-logit term (small weight, LOO estimator),
   under identical post-hoc temperature scaling. Pre-register the prediction from [2610.02486](https://arxiv.org/abs/2610.02486)
   that CE wins or ties after calibration.
7. **Robustness pass**: DecisionRobust-style label-consistency checks + a FlipJev-style query-based suffix
   attack against your own model, measured as decision-flip rate at high confidence.

---

## 9. References

Primary sources, all arXiv abstract pages unless noted.

**Classical vision-and-language encoders**
- ViLBERT — https://arxiv.org/abs/1908.02265
- LXMERT — https://arxiv.org/abs/1908.07490
- VisualBERT — https://arxiv.org/abs/1908.03557
- UNITER — https://arxiv.org/abs/1909.11740
- ViLT — https://arxiv.org/abs/2102.03334
- ALBEF — https://arxiv.org/abs/2107.07651
- FLAVA — https://arxiv.org/abs/2112.04482
- METER (empirical study of end-to-end V+L transformers) — https://arxiv.org/abs/2111.02387
- M³P — https://arxiv.org/abs/2006.02635
- LAVENDER (MLM as the universal interface) — https://arxiv.org/abs/2206.07160

**Frozen-encoder bridges and modality plumbing**
- BLIP — https://arxiv.org/abs/2201.12086
- BLIP-2 (Q-Former) — https://arxiv.org/abs/2301.12597
- Flamingo (Perceiver Resampler, gated cross-attention) — https://arxiv.org/abs/2204.14198
- MiniGPT-4 (single linear projection) — https://arxiv.org/abs/2304.10592
- DeepStack — https://arxiv.org/abs/2406.04334
- Qwen2-VL (M-RoPE) — https://arxiv.org/abs/2409.12191
- Rotary Position Embedding for Vision Transformer — https://arxiv.org/abs/2403.13298

**Encoders, embeddings, compression**
- MetaCLIP — https://arxiv.org/abs/2309.16671
- Perception Encoder — https://arxiv.org/abs/2504.13181
- VLM2Vec — https://arxiv.org/abs/2410.05160 · xVLM2Vec — https://arxiv.org/abs/2503.09313
- GME — https://arxiv.org/abs/2412.16855 · LamRA — https://arxiv.org/abs/2412.01720
- InternVL-X — https://arxiv.org/abs/2503.21307
- VL-JEPA — https://arxiv.org/abs/2512.10942
- Token Merging — https://arxiv.org/abs/2210.09461 · LLaVA-PruMerge — https://arxiv.org/abs/2403.15388 · TokenPacker — https://arxiv.org/abs/2407.02392
- MM1 — https://arxiv.org/abs/2403.09611 · Prismatic VLMs — https://arxiv.org/abs/2402.07865
- Confidence under visual token pruning — https://arxiv.org/abs/2604.12035
- Visual merit or linguistic crutch? (DeepSeek-OCR critique) — https://arxiv.org/abs/2601.03714

**Modality bias / collapse**
- VALUE: Behind the Scene — https://arxiv.org/abs/2005.07310
- Vision-and-Language or Vision-for-Language? — https://arxiv.org/abs/2109.04448
- Mind the Gap (modality gap) — https://arxiv.org/abs/2203.02053
- ARO (bag-of-words behaviour) — https://arxiv.org/abs/2210.01936
- Vision language models are blind — https://arxiv.org/abs/2407.06581
- Modality bias via attention lens — https://arxiv.org/abs/2508.02419 · See-Saw modality balance — https://arxiv.org/abs/2503.13834
- Why Text Prevails (vision may undermine medical decisions) — https://arxiv.org/abs/2512.13747
- Vokenization — https://arxiv.org/abs/2010.06775

**Calibration, uncertainty, abstention**
- On Calibration of Modern Neural Networks — https://arxiv.org/abs/1706.04599
- Revisiting the Calibration of Modern Neural Networks — https://arxiv.org/abs/2106.07998
- Enabling calibration in zero-shot inference of large VLMs (CLIP) — https://arxiv.org/abs/2303.12748
- Calibration and performance of MLLMs — https://arxiv.org/abs/2412.14660
- Bridging the missing-modality gap (text-only calibration of VLMs) — https://arxiv.org/abs/2605.12517
- Overconfidence is key (verbalized uncertainty) — https://arxiv.org/abs/2405.02917
- Conformal prediction: gentle introduction — https://arxiv.org/abs/2107.07511 · image sets — https://arxiv.org/abs/2009.14193 · NLP survey — https://arxiv.org/abs/2405.01976 · conformal fusion under missing modalities — https://arxiv.org/abs/2608.07183
- Uncertainty-aware decision making in MLLMs (survey) — https://arxiv.org/abs/2608.17084
- Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA 2007 — https://doi.org/10.1198/016214506000001437

**Vision + typed decision models (2026)**
- Visual Jev — https://arxiv.org/abs/2609.25845
- See the Options (Iris / candidate memory) — https://arxiv.org/abs/2610.33702
- Intern-Decision — https://arxiv.org/abs/2610.22131
- RSJEV — https://arxiv.org/abs/2610.08539
- OmniMed-Jev — https://arxiv.org/abs/2610.00381
- UniJev (trimodal, CORA) — https://arxiv.org/abs/2610.25784
- Beyond CLIP (few-shot, calibrated defect detection) — https://arxiv.org/abs/2610.41199
- CLEF on breast ultrasound — https://arxiv.org/abs/2610.13675
- Typed Decision Models: early evidence audit — https://arxiv.org/abs/2609.32160
- TypedBench — https://arxiv.org/abs/2610.11392
- SBERT2S1 (retrieval → typed decisions; RLCD vs CE) — https://arxiv.org/abs/2610.02486
- OpenJev-RLCD (calibrate, then reinforce) — https://arxiv.org/abs/2609.38850
- Lost in the Fallback (fallback reliability) — https://arxiv.org/abs/2610.19856
- System Switch (confidence-gated escalation) — https://arxiv.org/abs/2610.09683
- DecisionRobust — https://arxiv.org/abs/2610.19118
- FlipJev (universal suffixes) — https://arxiv.org/abs/2610.38895

**Adversarial / injection**
- FigStep — https://arxiv.org/abs/2311.05608
- Abusing images and sounds for indirect instruction injection — https://arxiv.org/abs/2307.10490
- Image Hijacks — https://arxiv.org/abs/2309.00236
- Image-based Prompt Injection (black-box) — https://arxiv.org/abs/2603.03637
- Covert visual prompt injection — https://arxiv.org/abs/2603.29418
- AgentTypo — https://arxiv.org/abs/2510.04257 · goal hijacking via visual prompt injection — https://arxiv.org/abs/2408.03554 · adversarial images hijack web agents — https://arxiv.org/abs/2610.09240
- Spotlighting — https://arxiv.org/abs/2403.14720
- CaMeL: Defeating Prompt Injections by Design — https://arxiv.org/abs/2503.18813
- Render Before Reading (visual rendering as a defense) — https://arxiv.org/abs/2609.36121

**Architecture references**
- ModernBERT — https://arxiv.org/abs/2412.13663 · config (RoPE thetas, local window, global every 3rd layer) — https://huggingface.co/answerdotai/ModernBERT-large/blob/main/config.json
- DeepSeekMath (GRPO) — https://arxiv.org/abs/2402.03300 · RLCD (contrastive distillation, unrelated) — https://arxiv.org/abs/2307.12950
