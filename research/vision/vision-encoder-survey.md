# Vision encoder survey for a non-generative, 33 ms, bidirectional-fusion decision system

**Target system.** A bidirectional text encoder (ModernBERT-large shape: 1024 hidden, 28 layers, 16 heads,
FFN 2624, vocab 50368, 8192 positions, local attention window 128, every 3rd layer global — verified from
`answerdotai/ModernBERT-large/config.json`) plus a masked-option scoring head, one forward pass, no
generation, ~33 ms per decision on a T4, calibrated confidence.

**Date of evidence collection:** 2026-11-02. Every parameter count below was *measured*, not quoted.

---

## 0. How these numbers were obtained (and what that rules out)

| Fact class | Method | Why it is trustworthy |
|---|---|---|
| Parameter counts (backbone / text / total) | HTTP **range-read of each checkpoint's `model.safetensors` header** (first 8 bytes = header length, then the JSON header), summing `prod(shape)` per tensor and grouping by module prefix; all shards summed | Exact integers, not paper roundings. Paper "0.09B" ≠ 93,673,984 params. Cross-checked against `timm`'s own measured `infer_gmacs`. |
| Embedding width, patch size, depth, native token grid | The same tensor shapes (`patch_embedding.weight[width,3,p,p]`, `position_embedding[N,width]`, max block index) | Shape-derived, not config-derived — several families ship empty or misleading configs (PE has *no* `config.json`; SigLIP2 NaFlex keeps `max_num_patches` in the *preprocessor* config) |
| Licence / gating | HF API (`license:` tag, `gated` field) **plus** the actual `LICENSE*` file where one exists, plus the licence text where the tag says `other` | Tags and reality diverge (DINOv3 is `gated: manual` + custom licence; PE is Apache-2.0 but ships *no* LICENSE file) |
| Accuracy / retrieval / dense numbers | The papers' own arXiv HTML/PDF (all IDs fetched and checked; several task-brief IDs were wrong and are corrected below) | Table-cell level provenance; disagreements between papers recorded rather than smoothed |
| Latency | **Measured T4 data for transformer vision encoders is very thin but not empty**: two independent public sources exist (DINOv2 ViT-B/S + DPT at batch 1 / TRT fp16; CLIP ViT-L/14 at batch 8), and they calibrate a stable efficiency constant (~25–28 % of the 65 TFLOPS fp16 peak). Nothing exists for SigLIP*, DINOv3, PE, InternViT or Qwen on T4. Three evidence tiers are used and always labelled: **(measured-T4)**, **(measured-3090)**, **(estimate)** | See §8. Anything marked *(estimate)* is arithmetic on cited inputs, not a measurement. |

**Tooling limitation, stated up front.** The harness `web_search` tool timed out on every call in this
environment, and Bing/DuckDuckGo returned bot challenges or unrelated pages. All evidence therefore comes
from first-party HTTP fetches: `huggingface.co` (API + raw files + ranged safetensors reads), `arxiv.org`
(HTML full text of the papers), `raw.githubusercontent.com`, `ai.meta.com`. **Third-party benchmark blog
posts, ONNX-Runtime-style leaderboards and vendor latency tables are therefore absent, and questions that
only such sources could answer are marked UNVERIFIED rather than guessed.** One exception was found and
used: `timm`'s published inference-benchmark CSVs.

**Corrections to the brief's premises** (all verified against primary sources):

| Brief said | Actually |
|---|---|
| RADIO = arXiv 2312.03015 | 2312.03015 is *PartSLIP++*. RADIO is **2312.06709** (AM-RADIO) |
| EVA-02 = arXiv 2303.15389 | That is *EVA-CLIP*. EVA-02 is **2303.11331** |
| `nvidia/RADIOv2.5-L` | No such repo (HTTP 401). RADIOv2.5 is a *paper* (2412.07679); its weights ship as **`C-RADIOv2-*`** |
| "Meta PE has a text encoder PE-Lang" | **PE-Lang is a vision tower**, not a text encoder: PE-Core truncated to layer 23/47, `pool_type="none"`, aligned to a frozen Llama-3 8B |
| "DINOv3 saturated claims" | The DINOv3 paper never claims losses or scaling saturate. It claims **classification keeps improving while segmentation degrades past ~200k iterations**, and that *Gram anchoring* is what unblocks scale |
| "SigLIP2 NaFlex = unbounded tokens" | Bounded by `max_num_patches` (default **256**), and trained only at {128, 256, 576, 784, 1024} |
| "`transformers` implements EVA" | It does not — EVA/EVA-02 are **not in `transformers` at all** (verified by directory listing; note `docs/.../model_doc/eva` returns HTTP 200 for any string, so status-probing gives a false positive) |

---

## 1. Headline findings

1. **The vision encoder is not the bottleneck — the visual tokens you inject into the 28-layer bidirectional
   encoder are.** One token costs **267.9 MMAC** of ModernBERT-large (28 × 9.57 MMAC/token/layer, arithmetic
   from the config). SigLIP2-B/16@224 produces 196 patch tokens → **52.5 GMAC** of text-side compute to read
   them, versus **17.4 GMAC** for the entire ViT image pass. Cutting 196 tokens to 64 pooled tokens saves more
   time than downgrading a ViT-L to a ViT-B *and* running it at 224 instead of 384.
2. **The 33 ms budget is reachable at batch 1 in fp16 — and only with a B-size backbone.** Measured T4
   evidence puts a ViT-class encoder at **~25–28 % of the 65 TFLOPS fp16 peak** (two datapoints in-band plus a
   third below it, §8.1), which prices the recommended configuration at **~8 ms** and a So400m/14@384 configuration at **~48 ms**.
   Batch-1 weight-read floors (300 GB/s datasheet) are 0.62 ms for a 93 M B/16, **2.03 ms for any 300 M L/14**,
   2.85 ms for So400m, 4.34 ms for C-RADIOv4-H, 12.53 ms for PE-Core-G. **INT8 is not the escape hatch**: the one
   published T4 ViT INT8 attempt ran **2× slower than fp16** (§8.1), and fp32 (8.1 TFLOPS peak) is out of the
   question.
3. **A pooled/global vector is the single most decision-relevant thing a vision encoder can emit, and the
   evidence is quantitative.** RADIO's own ablation: switching a student from a CLS-token summary to an
   average-pooled summary moves ADE20K linear-probe from **49.01 → 38.36 mIoU** while k-NN barely moves
   (83.91 → 83.83) — i.e. *the summary head carries the linearly-separable semantics*. PE-Core's frozen
   DAVIS J&F collapses **56.8 → 42.8 (−14.0)** between its best intermediate layer and its last layer.
   Where you read a frozen encoder matters as much as which encoder you pick.
4. **Licence reality is the opposite of the folk wisdom in several places.** SigLIP/SigLIP2 (**Apache-2.0,
   ungated**) and *all* `facebook/PE-Core`, `PE-Spatial`, `PE-Lang` (**Apache-2.0, ungated, no LICENSE file**)
   are the clean ones. DINOv3 is `gated: manual` under a custom **DINOv3 License** (commercial use *is*
   permitted, but requires accepting the gate, carrying the agreement on redistribution, and displaying
   "Built with DINOv3"). MetaCLIP 2 is **CC-BY-NC-4.0** (non-commercial). RADIO **v1 is NSCLv1
   non-commercial**, while **C-RADIOv2/v3/v4 are NVIDIA Open Model License (commercial OK)**. InternViT is
   MIT. Qwen2.5-VL is Apache-2.0 (tag only — no LICENSE file in the repo).
5. **ONNX readiness is better than expected, with three concrete traps.** Verified graph-level (artifacts
   downloaded and `graph.input`/`opset_import` parsed, not README-trusted): SigLIP2 fixed-res **16** exports,
   SigLIP (18), CLIP (8), **DINOv2 (20, opset 14, `ai.onnx`-only — the cleanest in the set)**, **DINOv3 (14,
   opset 20, RoPE expressed as `Range`/`Slice`/`MatMul`, H/W genuinely symbolic)**, **SigLIP2-NaFlex (patch
   axis dynamic, but `com.microsoft` fused ops)**, **Qwen2.5-VL vision encoder (fully dynamic resolution,
   heavy `com.microsoft` dependency)**, PE-Core (2). Traps: **`PE-Core-B16-224-ONNX` has batch hard-frozen to
   2** (`pixel_values[2,3,224,224]`); **InternViT has no public ONNX at all**; RADIO/C-RADIO/DFN/EVA-02 are
   community-only. And **OpenCLIP/DFN have no `transformers` `open_clip` module at all**.
6. **Distilled/small variants are widely available and are the right default at this latency.** SigLIP2-B/16
   (92.9 M), PE-Core-T/16 (6.1 M) / S/16 (23.8 M) / B/16 (93.7 M), DINOv3 ViT-S/16 (~21 M) and S+/B, DINOv2
   ViT-S/14 (22.1 M), C-RADIOv4-SO400M (431 M), AiMv2-L-distilled, MobileCLIP2-S0…L. The PE-Core T (6.1 M,
   **IN-1k 0-shot 62.1**) and S (23.8 M, **72.7**) towers are the smallest published members of a
   fully-benchmarked family here — S/16 at 23.8 M is within 3.7 of DINOv2 ViT-S/14's *linear-probe* 81.1 while
   being a zero-shot classifier, and it costs 577 tokens at 384 px.
7. **No published benchmark measures what this project actually does.** Every accuracy number in §6 is
   zero-shot classification, retrieval, linear probing or VQA. Nothing published scores *masked options* over
   an image + bidirectional text; §12 spells out the gap between "best encoder" and "best encoder for a
   frozen decision head".

---

## 2. Inventory: exact parameter counts, widths, patches, depths

All counts from safetensors headers (2026-11-02). "vision" = the vision tower including its own pooling head;
"total" = what the checkpoint file contains. Where a repo ships no safetensors (PE `.pt`-only, OpenAI CLIP
`.bin`-only) the timm-mirror safetensors were measured instead and the repo is marked.

### 2.1 SigLIP 2 and SigLIP 1 (all sizes, fixed + NaFlex)

| HF id | vision params | text params | total | width | patch | depth | patch tokens @ native |
|---|---|---|---|---|---|---|---|
| `google/siglip2-base-patch16-224` | 92,884,224 | 282,303,744 | 375,187,970 | 768 | 16 | 12 | 196 |
| `google/siglip2-base-patch16-256` | 92,930,304 | 282,303,744 | 375,234,050 | 768 | 16 | 12 | 256 |
| `google/siglip2-base-patch16-384` | 93,176,064 | 282,303,744 | 375,479,810 | 768 | 16 | 12 | 576 |
| `google/siglip2-base-patch16-512` | 93,520,128 | 282,303,744 | 375,823,874 | 768 | 16 | 12 | 1024 |
| `google/siglip2-base-patch32-256` | 94,552,320 | 282,303,744 | 376,856,066 | 768 | 32 | 12 | 64 |
| `google/siglip2-base-patch16-naflex` | 92,930,304 | 282,303,744 | 375,234,050 | 768 | 16 | 12 | ≤256 (`max_num_patches`) |
| `google/siglip2-large-patch16-256` | 315,956,224 | 565,570,560 | 881,526,786 | 1024 | 16 | 24 | 256 |
| `google/siglip2-large-patch16-384` | 316,283,904 | 565,570,560 | 881,854,466 | 1024 | 16 | 24 | 576 |
| `google/siglip2-large-patch16-512` | 316,742,656 | 565,570,560 | 882,313,218 | 1024 | 16 | 24 | 1024 |
| `google/siglip2-so400m-patch14-224` | 427,680,704 | 707,782,896 | 1,135,463,602 | 1152 | 14 | 27 | 256 |
| `google/siglip2-so400m-patch14-384` | 428,225,600 | 707,782,896 | 1,136,008,498 | 1152 | 14 | 27 | 729 |
| `google/siglip2-so400m-patch16-256` | 427,888,064 | 707,782,896 | 1,135,670,962 | 1152 | 16 | 27 | 256 |
| `google/siglip2-so400m-patch16-384` | 428,256,704 | 707,782,896 | 1,136,039,602 | 1152 | 16 | 27 | 576 |
| `google/siglip2-so400m-patch16-512` | 428,772,800 | 707,782,896 | 1,136,555,698 | 1152 | 16 | 27 | 1024 |
| `google/siglip2-so400m-patch16-naflex` | 427,888,064 | 707,782,896 | 1,135,670,962 | 1152 | 16 | 27 | ≤256 |
| `google/siglip2-giant-opt-patch16-384` | **1,163,659,776** | 708,225,648 | 1,871,885,426 | 1536 | 16 | 27 | 576 |
| `google/siglip-base-patch16-224` (v1) | 92,884,224 | 110,271,744 | 203,155,970 | 768 | 16 | 12 | 196 |
| `google/siglip-large-patch16-384` (v1) | 316,283,904 | 336,194,560 | 652,478,466 | 1024 | 16 | 24 | 576 |
| `google/siglip-so400m-patch14-384` (v1) | 428,225,600 | 449,734,896 | 877,960,498 | 1152 | 14 | 27 | 729 |

Notes that matter operationally:
- **SigLIP2-B/16's vision tower is byte-identical in shape to SigLIP-1's** (92,884,224) — SigLIP2 is a
  weights/tokenizer swap, which is unusual and useful for A/B testing.
- **No CLS token anywhere in SigLIP.** Both towers pool with a **MAP head** (multi-head attention pooling:
  learned `probe[1,1,d]` → MHA over patch tokens → residual LayerNorm → MLP → take token 0). The vision
  `head` is 7,087,104 params for B/16 and lives inside `vision_model.*`. **Its width is the tower width**
  (768 / 1024 / 1152 / 1536) — there is *no* dedicated cross-modal projection tensor like CLIP's
  `visual_projection`, because for B/L/So the text tower has the same width, so none is needed.
- The **g-opt** vision tower does carry a MAP head too (probe `[1,1,1536]`), but its width (1536) differs from
  its paired text tower (1152) and **no projection tensor is shipped** — so g-opt cannot be used as a working
  CLIP without you adding the 1536→1152 alignment. Its vision tower is **1.164 B** params, not the "1B" of the
  abstract.
- Consequence for `transformers`: `get_image_features()` returns a **pooled** vector, `last_hidden_state`
  (or the vision-model output) gives patch tokens. You get both from the same forward pass.
- No text decoder is released. The paper is explicit that the LocCa captioning decoder "is not part of the
  model release". What ships is vision encoder + text *encoder* + logit scale/bias.

### 2.2 CLIP / OpenCLIP / DFN

| HF id | vision params | text params | total | width | patch | depth | patch tokens |
|---|---|---|---|---|---|---|---|
| `openai/clip-vit-base-patch32` (via `timm/vit_base_patch32_clip_224.openai`) | 87,849,216 | 37,828,608 *(timm text side)* | 151,277,313 | 768 | 32 | 12 | 49 |
| `openai/clip-vit-large-patch14` | 303,180,033 | 123,060,557 | 427,616,847 | 1024 | 14 | 24 | 256 |
| `timm/vit_large_patch14_clip_224.openai` | 303,966,208 | 85,054,464 *(text head trimmed)* | 427,616,513 | 1024 | 14 | 24 | 256 |
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 87,456,050 | 63,166,029 | 151,277,440 | 768 | 32 | 12 | 49 |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 630,766,337 | 352,984,141 | 986,109,775 | 1280 | 14 | 32 | 256 |
| `apple/DFN5B-CLIP-ViT-H-14-378` | *(no safetensors — `.bin` only)* | — | — | 1280 | 14 | 32 | 729 |

CLIP has an explicit `visual_projection`/`text_projection` pair (1.38 M for L/14) plus `logit_scale`; the
CLS token is at index 0 of `last_hidden_state` and `pooler_output` is the projected CLS. DFN/DFN5B licence is
`apple-amlr` (Apple ML Research licence) — **not** Apache/MIT, and it ships no safetensors, which is what
makes it awkward for an ONNX pipeline.

### 2.3 DINOv2 / DINOv3 (no text tower at all)

| HF id | params | width | patch | depth | patch tokens @224 | CLS | registers |
|---|---|---|---|---|---|---|---|
| `facebook/dinov2-small` | 22,056,576 | 384 | 14 | 12 | 256 | 1 | 0 |
| `facebook/dinov2-base` | 86,580,480 | 768 | 14 | 12 | 256 | 1 | 0 |
| `facebook/dinov2-large` | 304,368,640 | 1024 | 14 | 24 | 256 | 1 | 0 |
| `facebook/dinov2-giant` | 1,136,480,768 | 1536 | 14 | 40 | 256 | 1 | 0 |
| `facebook/dinov2-with-registers-base` | 86,583,552 | 768 | 14 | 12 | 256 | 1 | 4 |
| `facebook/dinov2-with-registers-large` | 304,372,736 | 1024 | 14 | 24 | 256 | 1 | 4 |
| `facebook/dinov2-with-registers-giant` | 1,136,486,912 | 1536 | 14 | 40 | 256 | 1 | 4 |
| `facebook/dinov3-vits16-pretrain-lvd1689m` | ~21 M *(paper/card; gated — header not fetchable)* | 384 | 16 | 12 | 196 | 1 | 4 |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | ~86 M | 768 | 16 | 12 | 196 | 1 | 4 |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | ~300 M | 1024 | 16 | 24 | 196 | 1 | 4 |
| `facebook/dinov3-vith16plus-pretrain-lvd1689m` | ~840 M | 1280 | 16 | 32 | 196 | 1 | 4 |
| `facebook/dinov3-vit7b16-pretrain-lvd1689m` | ~6.7 B | 4096 | 16 | 40 | 196 | 1 | 4 |
| `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` | ~29 M | — | — | — | — | — | — |

DINOv2's stored position table is **1370 entries** (37×37+1 = 518 px native), so 518 is the resolution the
checkpoint was built for and 224 is a resize; HF interpolates. DINOv3 config (read from the ONNX mirror,
since `facebook/dinov3-*` is gated): `rope_theta=100.0`, `num_register_tokens=4`, `patch_size=16`,
`model_type=dinov3_vit`, gated MLP **off** for S.

### 2.4 Meta Perception Encoder (Core / Spatial / Lang)

All PE checkpoints ship **only `.pt` files** (1.79 GB for B/16, 9.68 GB for G/14) plus an **empty 0-byte
`config.yaml`**; the config lives in `core/vision_encoder/config.py`. Params below are measured from the
`facebook/*_timm` mirrors, which are the same towers.

| HF id | backbone params | ViT width | attn-pool out (CLIP dim) | patch | depth | native res | tokens |
|---|---|---|---|---|---|---|---|
| `facebook/PE-Core-T16-384` | 6,141,056 | 192 | 512 | 16 | 12 | 384 | 576+CLS |
| `facebook/PE-Core-S16-384` | 23,783,168 | 384 | 512 | 16 | 12 | 384 | 576+CLS |
| `facebook/PE-Core-B16-224` | 93,673,984 | 768 | 1024 | 16 | 12 | 224 | 196+CLS |
| `facebook/PE-Core-L14-336` | 317,152,256 | 1024 | 1024 | 14 | 24 | 336 | 576+CLS |
| `facebook/PE-Core-G14-448` | 1.88 B | 1280 | 1280 | 14 | 50 | 448 | 1024 (no CLS) |
| `facebook/PE-Spatial-T16-512` | ~6 M | 192 | — (`output_dim=None`) | 16 | 12 | 512 | 1024+CLS |
| `facebook/PE-Spatial-S16-512` | 21,983,232 | 384 | — | 16 | 12 | 512 | 1024+CLS |
| `facebook/PE-Spatial-B16-512` | 86,433,792 | 768 | — | 16 | 12 | 512 | 1024+CLS |
| `facebook/PE-Spatial-L14-448` | ~317 M | 1024 | — | 14 | 24 | 448 | 1024+CLS |
| `facebook/PE-Spatial-G14-448` | ~1.88 B | 1280 | — | 14 | 50 | 448 | 1024 |
| `facebook/PE-Lang-L14-448` | ~1.7 B (23 of 24 layers) | 1024 | none | 14 | 23 | 448 | 1024+CLS |
| `facebook/PE-Lang-G14-448` | — (47 of 50 layers) | 1280 | none | 14 | 47 | 448 | 1024 |

Key structural facts: **every PE-Core has an 8-head attention-pool block**; B and L *additionally* carry a CLS
token; **G has no CLS**; `PE-Spatial`/`PE-Lang` set `pool_type="none"`, so they emit patch tokens only.
There is **no `PE-Video` checkpoint** — `encode_video()` mean-pools 8 frames through the same weights; the
actual temporal successor is **PE-AV** (`facebook/pe-av-*`, Apache-2.0, released 2025-12-16) plus
`pe-a-frame-*`. There is **no separate PE-Spatial paper** — it is §5 of arXiv 2504.13181.

### 2.5 InternViT, Qwen2.5-VL, RADIO/C-RADIO, EVA-02, AiMv2

| HF id | vision params | width | patch | depth | native / scheme | tokens |
|---|---|---|---|---|---|---|
| `OpenGVLab/InternViT-300M-448px-V2_5` | 304,012,288 | 1024 | 14 | 24 | 448 tile | 1024/tile → 256 after 2×2 unshuffle |
| `OpenGVLab/InternViT-6B-448px-V2_5` | **5,536,496,000** (d45 — "6B" is a stale name; 224px and V1-0 are d48) | 3200 | 14 | 45 | 448 tile | as above |
| `Qwen/Qwen2.5-VL-7B-Instruct` (visual tower) | 676,550,144 | 1280 | 14 | 32 | native dynamic | `(H/28)·(W/28)`; default cap 16,384 |
| `Qwen/Qwen2.5-VL-3B-Instruct` (visual tower) | 668,684,288 | 1280 | 14 | 32 | native dynamic | same cap |
| `nvidia/RADIO-B` | 98,233,353 | 768 | 16 | 12 | any, 16 k pos table | N |
| `nvidia/RADIO-L` | 319,881,225 | 1024 | 16 | 24 | any | N |
| `nvidia/RADIO-H` | 651,642,889 | 1280 | 16 | 32 | any | N |
| `nvidia/C-RADIOv2-B/L/H` | 98.2 M / 319.9 M / 651.6 M | 768/1024/1280 | 16 | 12/24/32 | any | N |
| `nvidia/C-RADIOv3-B/L/H` | 98,254,857 / 319,934,473 / 651,642,889 | 768/1024/1280 | 16 | 12/24/32 | any | N |
| `nvidia/C-RADIOv4-SO400M` | 431,237,240 | 1152 | 16 | 27 | ≤2048×2028 in 16 px steps | 128×128 max grid |
| `nvidia/C-RADIOv4-H` | 651,645,448 | 1280 | 16 | 32 | same | same |
| `timm/eva02_base_patch16_clip_224.merged2b` | 86,263,040 | 768 | 16 | 12 | 224 | 196+CLS |
| `timm/eva02_large_patch14_clip_336.merged2b` | 304,432,832 | 1024 | 14 | 24 | 336 | 576+CLS |
| `timm/eva02_enormous_patch14_clip_224.laion2b` | 4,350,556,928 | 1792 | 14 | 64 | 224 | 257 (incl. CLS) |
| `apple/aimv2-large-patch14-224/336/448/native` | 309.2–310.0 M | 1024 | 14 | 24 | native variant exists | 256/576/1024 |
| `apple/aimv2-1B-patch14-448` | 1,236,534,528 | 2048 | 14 | 24 | 448 | 1024 |
| `apple/aimv2-huge-patch14-448` | 682,031,104 | 1536 | 14 | 24 | 448 | 1024 |

InternViT and Qwen2.5-VL are *decoder-backbone* towers: neither is evaluated standalone anywhere in their own
papers (§7.3, §6.3 of the notes), and both are expensive at batch 1 (see §8).

---

## 3. Token counts, and who can do variable/native resolution

`V` = variable/native aspect ratio supported. "bounded?" is the operational question: can token count blow up?

| Family | fixed resolutions | patch tokens at those resolutions | V? | token count bounded? |
|---|---|---|---|---|
| SigLIP2 B/16 | 224 / 256 / 384 / 512 | 196 / 256 / 576 / 1024 | ✗ (NaFlex variant: ✓) | fixed: yes by construction |
| SigLIP2 B/32-256 | 256 | **64** | ✗ | yes |
| SigLIP2 L/16 | 256 / 384 / 512 | 256 / 576 / 1024 | ✗ | yes |
| SigLIP2 So400m/14 | 224 / 384 | 256 / 729 | ✗ | yes |
| SigLIP2 So400m/16 | 256 / 384 / 512 | 256 / 576 / 1024 | ✗ | yes |
| SigLIP2 g-opt/16 | 384 | 576 | ✗ | yes |
| **SigLIP2 NaFlex** (B/16, So/16 only) | any, aspect-preserving, snapped to patch multiples | **≤ `max_num_patches` = 256** | ✓ | **yes** — masked attention above the real length; trained at {128,256,576,784,1024} |
| SigLIP 1 | 224 / 256 / 384 / 512 | 196 / 256 / 576 / 1024 | ✗ | yes |
| CLIP | 224 (B/32, B/16, L/14), 336 (L/14-336) | 49 / 196 / 256 / 576 | ✗ | yes |
| DINOv2 | any (interpolated pos-emb) | 224→256, 448→1024, 518→1369 | ✓ (interp) | no hard cap; 1370 pos table native |
| DINOv3 | any (RoPE, no pos table) | 224→196, 512→1024, 768→2304, 4096→65536 | ✓ | **not architecturally bounded** |
| PE-Core | 224 / 336 / 384 / 448 | 196 / 576 / 576 / 1024 (+1 CLS) | ✓ in code (RoPE2D `update_grid` + bilinear abs-pos interp, **no token cap**) | no cap |
| PE-Spatial | 512 / 448 | 1024 (+1 CLS) | ✓ | no cap |
| InternViT | 448 tiles ×1..40 (+thumbnail) | `(n_tiles+1)×256` → 3328 default, 10496 max | ✓ (tiles) | **yes, by `n_max`** (1–40) |
| Qwen2.5-VL | native dynamic | `(H/28)·(W/28)`; `min_pixels`=3136 → 4 tokens, `max_pixels`=12,845,056 → **16,384 tokens** | ✓ | **yes, by pixel budget** (verified: 12845056/784 = 16,384 exactly) |
| RADIO / C-RADIO | any, 16-px granularity | N (16,384-entry pos table = 128×128) | ✓ | effectively 16,384 |
| EVA-02 | 224 / 336 (+448 variants) | 196 / 576 | ✗ | yes |
| AiMv2 | 224 / 336 / 448 / native | 256 / 576 / 1024 | ✓ (native ckpt) | no cap in native |

**What NaFlex actually buys, in numbers** (Table 7 + §3.1.1, verified): the paper's own summary is that "the
NaFlex variant outperforms the standard variant on the majority of these retrieval benchmarks, in particular for
small sequence lengths (and hence resolutions) which tend to suffer more from aspect ratio distortion", while
"on benchmarks predominantly based on natural images, the standard B-sized variant outperforms NaFlex". The
concrete B/16 trade at sequence length 256: **ImageNet 0-shot 79.1 → 78.5** (−0.6), while the OCR/document/screen
retrieval pairs rise — e.g. **HierText 6.1/9.8 → 7.4/12.9**, SciCap 17.1/14.2 → 19.7/17.1, Screen2Words
12.9/22.9 → 14.8/26.6 (T→I / I→T R@1). So NaFlex is worth choosing **if the images are documents, screens or
panoramas**; for square natural images it is a small loss. Two further caveats from the same source: NaFlex
"interpolates fairly well between training resolutions, but **does not extrapolate well**", and self-distillation
+ masked prediction are **disabled** for the NaFlex checkpoints, so they are not trained with the full SigLIP 2
recipe.

For this project the relevant reading is blunt: **the useful token budgets are 64, 196/256 and 576**, and
everything above 256 costs more in the text encoder than in the ViT (§8). NaFlex's `max_num_patches=256`
default happens to be exactly the right budget; Qwen's 16,384 default is 4,000× over budget and would need
`max_pixels` cut to ~200 k pixels (≈256 tokens) to be usable at all.

---

## 4. Embedding width, projection layer, and does it hand you a global vector?

| Family | vision width | projection already present? | global vector out of the box? | what it is |
|---|---|---|---|---|
| SigLIP2 | 768 / 1024 / 1152 / 1536 | **no dedicated cross-modal projection** — the MAP head *is* the pooling head and emits the tower width (which equals the text width for B/L/So) | **yes** | MAP attention-pool over patch tokens; `get_image_features()`; patch tokens from the same forward |
| SigLIP 1 | 768 / 1024 / 1152 | yes, same MAP head | yes | attention pool |
| CLIP | 768 / 1024 / 1280 | **yes** — explicit `visual_projection` | **yes** | projected CLS |
| DINOv2 | 384 / 768 / 1024 / 1536 | ✗ (none) | **yes** | `pooler_output` = LayerNorm'd CLS; patch tokens at `[1:]` |
| DINOv3 | 384 … 4096 | ✗ | **yes** | `cls_tokens` (backbone) / `last_hidden_state[:,0]`; **4 registers must be dropped, not averaged** |
| PE-Core | 192 / 384 / 768 / 1024 / 1280 | **yes** — `output_dim` (512/1024/1280) | **yes (two of them)** | 8-head attention pools **plus** a CLS on T/S/B/L |
| PE-Spatial / PE-Lang | as Core | ✗ (`output_dim=None`) | ✗ by design | patch tokens only |
| InternViT | 1024 / 3200 | ✗ | yes (CLS at index 0) | plain CLS |
| Qwen2.5-VL | 1280 → 3584 after merger | merger into LM width | ✗ | no CLS, no pooler |
| RADIO / C-RADIO | 768 / 1024 / 1280 / 1152 | **yes** — summary + spatial adaptors | **yes, explicitly** | `summary` `(B,C)` **and** `spatial_features` `(B,N,C)` in one call |
| EVA-02 | 768 / 1024 / 1792 | CLIP head in the `-CLIP` variants | yes (CLS) | CLS |
| AiMv2 | 1024 / 1536 / 2048 | ✗ | via attention pooling | — |

**RADIO/C-RADIO is the only family whose API is explicitly "global vector + patch tokens"** and whose card
states the intended use: "`summary` is similar to the `cls_token` … meant to represent the general concept of
the entire image … `spatial_features` … suitable for dense tasks". Everything else needs you to take
`pooled_output` and `last_hidden_state` separately, which is fine but undocumented as a joint interface.

Widths that map cleanly onto the target 1024-hidden text encoder with a *single* linear layer: **DINOv2-L
(1024→1024, identity-shaped), CLIP-L/14 (1024), PE-Core-L/B (`output_dim=1024`), RADIO-L (1024)**.
SigLIP2-B (768→1024) and PE-Core-B (768→1024) need a real projection; SigLIP2-So400m (1152→1024) needs a
down-projection. None of this is expensive — it is one matrix.

---

## 5. Licence and gating (commercial usability, as of 2026-11-02)

| Family / repo | HF `license:` | `gated` | commercial use | caveats |
|---|---|---|---|---|
| `google/siglip2-*` (all 16) | **apache-2.0** | **false** | **yes** | none. big_vision code Apache-2.0, "other materials" CC-BY-4.0 |
| `google/siglip-*` (v1) | apache-2.0 | false | yes | card written by HF, not Google |
| `openai/clip-*` | *(no tag on the older repos)* | false | MIT in practice (open_clip re-releases are `mit`) | no safetensors |
| `laion/CLIP-*` | mit | false | yes | — |
| `apple/DFN5B-CLIP-*` | **apple-amlr** | false | **restricted** — Apple ML Research licence, not OSI | no safetensors |
| `facebook/dinov2-*` | **apache-2.0** | false | **yes** | — |
| `facebook/dinov3-*` | **other** (`dinov3-license`) | **manual** | **yes, conditionally** | must accept the gate; redistributions must carry the agreement; must display **"Built with DINOv3"**; Meta may amend the terms; no non-commercial clause exists in the text |
| `facebook/PE-Core-*`, `PE-Spatial-*`, `PE-Lang-*` | **apache-2.0** | **false** | **yes** | **no LICENSE file in any model repo** (404) — the licence is tag + card + `LICENSE.PE` in GitHub. Only **PLM** (Perception-LM) is FAIR-Noncommercial (`LICENSE.PLM`) |
| `facebook/pe-av-*`, `pe-a-frame-*` | apache-2.0 | false | yes | — |
| `facebook/metaclip-2-*`, `metaclip-*` | **cc-by-nc-4.0** | false | **NO** | non-commercial |
| `OpenGVLab/InternViT-*` | **mit** | false | yes | — |
| `Qwen/Qwen2.5-VL-*` | apache-2.0 | false | yes | **no LICENSE file** in the repo — tag is the only normative statement |
| `nvidia/RADIO-*` (v1) | other (`nvidia-source-code-license`) | false | **NO** | NSCLv1 §3.3: may only be used non-commercially |
| `nvidia/C-RADIOv2/v3/v4-*` | other → **NVIDIA Open Model License** (June 2024) | false | **yes** | card says "ready for commercial/non-commercial use" |
| `nvidia/C-RADIOv4-*` | same | false | yes | released **2026-01-27**; tech report arXiv 2601.17237 |
| `apple/aimv2-*`, `apple/MobileCLIP2-*` | **apple-amlr** | false | restricted | — |
| EVA-02 (timm mirrors) | mit | false | yes | no `transformers` implementation |

Two operational notes: (a) **gating is a deployment risk, not just a paperwork step** — a `gated: manual`
repo cannot be pulled by an unauthenticated CI job, which is why the DINOv3 entries in this survey have
header-derived params for DINOv2 but paper-quoted params for DINOv3; (b) **licence ≠ training-data licence** —
none of these releases ships its pretraining images under the weight licence.

---

## 6. Published quality numbers

Retrieval is **Recall@1**, and *neither* SigLIP paper states whether COCO is the 1k or 5k split — the numbers
below are as-published, and the split is UNVERIFIED (do not compare them to a 5k-split number from elsewhere).

### 6.1 SigLIP 2 vs SigLIP 1 (paper Table 1)

| ViT | Res | Tokens | Model | IN-1k 0-shot | IN-v2 | ObjectNet | IN 10-shot | COCO T→I | COCO I→T | Flickr T→I | Flickr I→T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B/32 | 256 | 64 | SigLIP 2 | 74.0 | 66.9 | 66.1 | 66.6 | 47.2 | 63.7 | 75.5 | 89.3 |
| B/16 | 224 | 196 | SigLIP | 76.2 | 69.5 | 70.7 | 69.9 | 47.2 | 64.5 | 77.9 | 89.6 |
| **B/16** | **224** | **196** | **SigLIP 2** | **78.2** | 71.4 | 73.6 | 72.1 | 52.1 | 68.9 | 80.7 | 93.0 |
| B/16 | 256 | 256 | SigLIP 2 | 79.1 | 72.5 | 74.5 | 73.1 | 53.2 | 69.7 | 81.7 | 94.4 |
| B/16 | 384 | 576 | SigLIP 2 | 80.6 | 73.8 | 77.1 | 74.7 | 54.6 | 71.4 | 83.8 | 94.9 |
| B/16 | 512 | 1024 | SigLIP 2 | 81.2 | 74.5 | 77.8 | 75.2 | 55.2 | 71.2 | 84.5 | 95.5 |
| L/16 | 384 | 576 | SigLIP 2 | 83.1 | 77.4 | 84.4 | 79.5 | 55.3 | 71.4 | 85.0 | 95.2 |
| So/14 | 384 | 729 | SigLIP 2 | 84.1 | 78.7 | 86.0 | 80.4 | 55.8 | 71.7 | 85.7 | 94.9 |
| So/16 | 512 | 1024 | SigLIP 2 | 84.3 | 79.1 | 86.2 | 80.5 | 56.0 | 71.3 | 85.5 | 95.4 |
| g/16 | 384 | 576 | SigLIP 2 | **85.0** | 79.8 | 88.0 | 82.5 | 56.1 | 72.8 | 86.0 | 95.4 |

Reference rows in the same table: CLIP B/16 68.3, OpenCLIP B/16 70.2, MetaCLIP B/16 72.4, EVA-CLIP B/16 74.7,
DFN B/16 76.2, CLIP L/14 75.5, MetaCLIP L/14 79.2, DFN (L) 82.2, OpenCLIP G/14 80.1.

**SigLIP2 B/16 at 224 is +2.0 IN-1k and +4.9 COCO-T→I over SigLIP 1 at the same token count** — the
SigLIP2 recipe, not the size, is what bought the improvement.

### 6.2 Meta PE-Core (card, primary source)

| Model | Res | Params | IN-1k | IN-v2 | IN-A | ObjectNet | COCO-T2I | K400 | VTT-T2I |
|---|---|---|---|---|---|---|---|---|---|
| PE-Core-T/16 | 384 | 6.1 M | 62.1 | 54.7 | 21.1 | 43.9 | 33.0 | 41.5 | 28.8 |
| PE-Core-S/16 | 384 | 23.8 M | 72.7 | 65.0 | 49.5 | 60.0 | 42.6 | 55.0 | 39.3 |
| PE-Core-B/16 | 224 | 93.7 M | **78.4** | 71.7 | 62.4 | 71.9 | 50.9 | 65.6 | 47.6 |
| PE-Core-L/14 | 336 | 317.2 M | 83.5 | 77.9 | 89.0 | 84.7 | 57.1 | 73.4 | 50.3 |
| PE-Core-G/14 | 448 | 1.88 B | 85.4 | 80.2 | 92.6 | 88.2 | 58.1 | 76.9 | 51.2 |

*(Source: `apps/pe/README.md` for T/S, model card for B/L/G; both fetched. The README's `S/16` width of "394"
is a typo — `config.py` says 384, matching the measured `patch_embedding.weight[384,3,16,16]`.)*

**The most relevant table in this entire survey for a frozen decision head** is PE's own encoder-probing
comparison (paper Table 8), because it is the only place where a contrastive model's *frozen* features are put
next to DINOv2/RADIO/AiMv2/InternVL/EVA on the same probe battery:

| Encoder | Params | Res | Pretrain data | IN-1k **k-NN** | IN-1k **linear** | IN-1k **attention** |
|---|---|---|---|---|---|---|
| DINOv2-g | 1.1 B | 224 | 145 M | 83.5 | 86.5 | 87.2 |
| RADIOv2.5-g | 1.1 B | 518 | — | 85.3 | — | — |
| AIMv2 3B | 2.7 B | 448 | 7.2 B | — | — | 89.5 |
| InternVL-C (6B) | 5.5 B | 224 | 5 B | — | 88.2 | — |
| EVA-18B | 17.5 B | 224 | 2 B | — | 88.9 | — |
| **PE-Core-G** | **1.9 B** | 448 | 5.4 B | **86.8** | **89.5** | **89.8** |

A 1.9 B contrastive model **matches a 17.5 B EVA-18B on frozen ImageNet linear probe and beats DINOv2-g by
3.0 (linear) / 2.6 (attention)**. This is the strongest published counter-example to "contrastive encoders lose
to SSL encoders for frozen classification" — with the caveat that it is measured at G scale, and PE publishes no
equivalent probe table for B/L.

Head-to-head at the same 224 px / ~197-token operating point, PE-Core-B16-224 vs SigLIP2-B/16@224: PE edges the
aggregate (**IN-1k 78.4 vs 78.2**, IN-v2 71.7 vs 71.4); SigLIP2 edges the robustness and retrieval columns
(**ObjectNet 73.6 vs 71.9**, COCO T→I 52.1 vs 50.9). Neither gap is decision-relevant. The PE family's
distinguishing property is not its top-1 but its *internal* features (§7.2), plus the second global read-out.

### 6.3 DINOv2 / DINOv3 (frozen probes; no text retrieval)

| Model | IN-1k linear | IN-1k k-NN | ADE20K lin. mIoU | VOC lin. | NYUv2 depth RMSE ↓ |
|---|---|---|---|---|---|
| DINOv2 ViT-S/14 | 81.1 | 79.0 | 44.3 | 81.1 | 0.449 |
| DINOv2 ViT-B/14 | 84.5 | 82.1 | 47.3 | 82.5 | 0.399 |
| DINOv2 ViT-L/14 | 86.3 | 83.5 | 47.7 | 82.1 | 0.384 |
| DINOv2 ViT-g/14 | 86.5 | 83.5 | 49.0 | 83.0 | 0.344 |
| DINOv2 ViT-g/14 **frozen + ViT-Adapter + Mask2Former** | — | — | **60.2** (66 % of weights frozen) | — | — |
| DINOv3 ViT-7B | **88.4** | — | 55.9 (linear) / **62.6–63.0** (frozen encoder + decoder, TTA) | — | 0.309 |

DINOv3 7B, same source: IN-R 91.1, ObjectNet 79.0, Cityscapes 81.1 linear, **COCO detection with a frozen
backbone 66.1 mAP**, DAVIS J&F 71.1 → 83.3. The 60.2 mIoU and the 66.1 mAP are the two cleanest public
demonstrations that a *frozen* SSL backbone plus a small trained decoder is competitive on real tasks.

DINOv3 at matched token counts (its Table 14, IN-ReaL / ADE20k / NYU↓):

| Size | DINOv2 | DINOv3 | SigLIP 2 |
|---|---|---|---|
| S | 87.3 / 45.5 / 0.446 | 87.0 / **47.0** / **0.403** | — |
| B | 89.0 / 48.4 / 0.416 | 89.3 / **51.8** / **0.373** | 89.3 / 41.6 / 0.512 |
| L | 89.7 / 48.8 / 0.394 | **90.2 / 54.9 / 0.352** | 90.1 / 43.6 / 0.484 |

This is the cleanest head-to-head in the literature for *dense vs semantic* trade-off: **SigLIP 2 wins
semantic (IN-ReaL tie, IN-R 89.2 vs 88.1), DINOv3 wins dense by ~11 mIoU and 0.13 depth RMSE.** Note there is
**no COCO/Flickr retrieval number for any DINO SSL backbone** — DINO has no text tower. A text-aligned ViT-L
("dino.txt") reports R@1 I→T 63.7 / T→I 45.6, well below SigLIP2.

### 6.4 InternViT, RADIO, EVA-02

| Encoder | IN-1k | ADE20K (linear probe, frozen) | VOC | Source |
|---|---|---|---|---|
| InternViT-300M-V2.5 | **frozen-probe scores *declined* across versions**: 82.5 → 78.5 avg; ADE20K seg 47.2 → 39.4 | ↓ | — | InternVL papers; card says V2.5 is "better suited for building MLLMs than traditional computer vision tasks" |
| OpenCLIP ViT-H/14 (teacher) | 77.19 (0-shot) / 81.10 (k-NN) | 40.04 | 68.03 | AM-RADIO Table 7 |
| MetaCLIP ViT-H/14 (teacher) | 80.51 / 82.12 | 35.39 | 62.62 | ibid. |
| DINOv2 ViT-L/14 (teacher) | n/a / 83.44 | 46.55 | 82.07 | ibid. |
| **RADIO ViT-H/14** | **82.73 / 85.29** | **50.32** | **85.00** | ibid. |
| EVA-02 L/14-336 (supervised IN-1k) | 89.7 (EVA-02-E, 4.35 B) | — | — | EVA-02 paper |

**RADIO's frozen summary beats all three of its own teachers on k-NN, ADE20K linear probe and VOC linear
probe.** For a frozen-feature decision system that is the most directly relevant published result in this
whole survey. InternViT is the counter-example: its authors report that its frozen-feature quality *fell* as
it was optimised for MLLM integration.

### 6.5 The single best cross-family table available: C-RADIOv4's report (Table 1)

The C-RADIOv4 tech report (arXiv 2601.17237, 24 Jan 2026) is the only paper that probes **SigLIP2, DINOv3,
RADIOv2.5, C-RADIOv3 and C-RADIOv4 in one frozen-feature battery**, so it is the best available answer to
"which encoder's frozen features are actually best", even though it is written by the family's authors.

| Model | Variant | Params | Zero-shot | k-NN | ADE20K lin. | VOC lin. | NAVI | SPair |
|---|---|---|---|---|---|---|---|---|
| SigLIP2 | SO400M | 412 M | **83.88** | 85.76 | 44.0 | 48.8 | 38.7 | — |
| SigLIP2 | g | 1,164 M | **84.75** | 86.39 | 42.7 | 72.7 | 49.4 | 42.6 |
| DINOv3 | H+ | 841 M | — | 85.77 | 54.8 | 63.3 | 56.3 | — |
| DINOv3 | 7B | 6,716 M | — | 85.42 | 55.9 | 86.6 | 64.4 | 58.7 |
| RADIOv2.5 | H | 631 M | 82.51 | 85.81 | 51.58 | 85.97 | 60.89 | 56.24 |
| C-RADIOv3 | H | 631 M | 82.65 | 86.23 | 52.75 | 86.41 | 62.10 | 58.54 |
| **C-RADIOv4** | **SO400M** | **412 M** | 82.01 | 85.76 | **55.14** | **87.22** | 62.44 | 60.01 |
| **C-RADIOv4** | **H** | **631 M** | 83.09 | 86.59 | **55.20** | **87.24** | 63.44 | 60.57 |

Three readings that matter for a decision system:

1. **Semantic vs dense is a clean trade and it is large.** SigLIP2 wins zero-shot (83.88 / 84.75) and loses
   dense by 10–11 mIoU (ADE20K 44.0 vs 55.1). If your options are *attribute/category* questions, that gap is
   worth nothing; if they are *spatial/relational*, it is worth everything.
2. **A 412 M fusion student matches a 6.7 B teacher on frozen dense probes** — C-RADIOv4-SO400M reaches ADE20K
   55.14 / VOC 87.22 / SPair 60.01 against DINOv3-7B's 55.9 / 86.6 / 58.7, i.e. **16× fewer parameters for
   equal-or-better frozen dense features**. Its own resolution sweep (Table 2) is nearly identical too:
   ADE20K 55.20 → 57.02 → 57.72 at 512/1024/1536 px vs DINOv3-7B's 55.9 → 57.3† → 57.8†.
3. **Distillation from the right teachers is what does it**: C-RADIOv4's teachers are SigLIP2, DINOv3 **and
   SAM3**, and v4 adds a mask-imitation objective. So it is not "a compromise between semantic and dense" —
   it inherits both.

Caveats: author-run benchmark, the numbers are at unstated resolutions, and "probe3d" columns (NAVI/SPair)
are 3D/semantic-correspondence tasks far from a masked-option head.

### 6.6 The closest published analogue to *this* system: frozen DINOv3 + small discriminative head

If you want one paper to design against, it is arXiv 2610.38462 (Oct 2026): 9,735 images, 2,333 concepts,
extreme multi-label annotation — **a frozen DINOv3 ViT-L/16 with 196 patch tokens, and three small heads
trained on top**. That is structurally the system in question (frozen extractor, non-generative head, decision
output), so its ablation table (their Table 2) transfers better than any VLM benchmark. prec@1 / MRR:

| Head | Loss | Backbone | prec@1 | MRR |
|---|---|---|---|---|
| Flat linear on **CLS** | BCE | frozen | 0.435 | 0.588 |
| Flat linear on CLS + **LoRA r=16** | BCE | LoRA | 0.413 | 0.584 |
| Flat linear on CLS | **ASL** | frozen | 0.565 | 0.697 |
| Flat linear on CLS + LoRA r=16 | ASL | LoRA | **0.500** | 0.612 |
| CAT (hierarchical cross-attention over patches) | hier-CE | frozen | 0.565 | 0.678 |
| CAT | hier-ASL | frozen | 0.609 | 0.701 |
| **ML-Decoder** (100 learned group queries over 196 patches) | BCE | frozen | **0.696** | 0.764 |
| ML-Decoder | ASL | frozen | **0.696** | 0.801 |
| ML-Decoder **+ CLS in the patch context** (197 tokens) | ASL | frozen | **0.696** | **0.806** |

Four transferable results, in descending order of practical value:

1. **Attention over patch tokens beats a projection of the pooled vector: 0.696 vs 0.565 prec@1 (+0.131),
   +0.104 MRR.** The head that wins is 100 learned queries cross-attending over the 196 patch tokens — no
   global vector involved.
2. **Adding the CLS token to that context buys +0.005 MRR and 0.000 prec@1.** So once the head already attends
   over patches, **the pooled vector is near-redundant** — measurable, but not worth architectural effort. This
   is the most direct published answer to "global vector or patch tokens?" for a decision head.
3. **LoRA on the frozen backbone *hurts*: 0.565 → 0.500 under ASL, 0.435 → 0.413 under BCE.** The authors'
   reading: at this corpus scale "the frozen DINOv3 representations are inherently more robust than any
   low-rank adaptation … regardless of which projections are adapted". Direct support for keeping the tower
   frozen in a small-head system.
4. **Fixing the loss was worth as much as fixing the architecture** — BCE → Asymmetric Loss moved the flat head
   +0.130 prec@1 (0.435 → 0.565), while the head change moved it +0.131. With thousands of near-mutually-exclusive
   option labels, the loss is the cheapest lever in the system.

Caveats: single domain (medieval manuscript iconography), single backbone (DINOv3 ViT-L/16), one seed per
configuration, and the concepts are visual-object labels rather than options-in-text. But the *architecture
shape* matches, which no VLM benchmark does.

---

## 7. What "dense" and "spatial" actually buy, per family

### 7.1 SigLIP 2's dense evidence (only So400m, only 224/384)

Frozen-representation probes (linear or DPT decoder) — paper Table 2:

| Encoder | PASCAL mIoU | ADE20k mIoU | NYUv2 RMSE | NAVI RMSE | NYUv2 normals |
|---|---|---|---|---|---|
| CLIP L/14@224 | 74.5 | 39.0 | 0.553 | 0.073 | 24.3 |
| SigLIP So/14@224 | 72.0 | 37.6 | 0.576 | 0.083 | 25.9 |
| **SigLIP 2 So/14@224** | **77.1** | **41.8** | **0.493** | **0.067** | 24.9 |
| SigLIP So/14@384 | 73.8 | 40.8 | 0.563 | 0.069 | 24.1 |
| **SigLIP 2 So/14@384** | **78.1** | **45.4** | **0.466** | **0.064** | **23.0** |

Also: a **frozen** SigLIP2-B/16@256 + from-scratch 6-layer transformer decoder gets **83.76 Acc@0.5 on
RefCOCO val** (SigLIP 1 B/16: 64.05) — a 20-point jump on a spatial task with the backbone frozen. There is
**no dense number for B/16, L/16, or NaFlex anywhere in the paper** — a survey table putting SigLIP2-B next to
DINOv2-B on ADE20K has no primary source.

### 7.2 PE's "the best embeddings are not at the output"

The PE paper's claim (arXiv 2504.13181, "Perception Encoder: The best visual embeddings are not at the output
of the network") is narrower than its title: the features *are present in intermediate layers* ("match or
exceed the leading models") and the model "fails to output them" at the last layer. The evidence:

| Frozen read-out | Best layer | Last layer |
|---|---|---|
| PE-Core-G DAVIS J&F | **56.8** (layer 32 of 50) | **42.8 (−14.0)** |
| PE-Spatial-G DAVIS J&F | 61.5 | 61.5 (layer 50/50, no collapse) |
| PE-Core-G NYU depth RMSE | — | **0.249** (beats DINOv2-g's 0.279 even at its *last* layer) |
| PE-Core-G ADE20K linear | — | 41.5 → **49.3** once self-distilled + SAM-2.1-mask-distilled into PE-Spatial-G |

This has a direct consequence for a decision head: **if you use PE-Core, probe 2–3 intermediate layers, not
just the last**, and if you need spatial fidelity out of the box, use **PE-Spatial** (which fixes the collapse:
frozen ADE20K linear probe 27.6 / 37.5 / 44.4 / 48.1 / 49.3 for T/S/B/L/G, zero-shot DAVIS J&F 55.0 / 57.5 /
58.9 / 60.6 / 61.5). The word "modality gap" does **not** appear in the paper — its claim is an *output*
problem, not a representational one.

### 7.3 InternViT / Qwen: dense-backbone encoders with no standalone story

InternViT-300M-V2.5 is MIT, 304 M, patch 14, 448-px tiles, and its token budget is genuinely bounded
(`(n_tiles+1)×256`, `n_max` 1–40 → 256…10,496). But its own papers report frozen-probe *regression* across
versions, and Qwen2.5-VL's tower has **zero** standalone evaluations — no linear probe, no ImageNet number
anywhere in the Qwen2.5-VL paper. For a decision system, a tower that exists only as an LM adapter is an
unproven extractor.

---

## 8. Latency: measured, derivable, and unknown

Three evidence tiers, labelled everywhere: **(M-T4)** measured on a T4 and cited, **(M-3090)** measured on an
RTX 3090 and cited, **(est)** my arithmetic on cited inputs.

### 8.1 Measured T4 datapoints — the ones that exist

| Model (as benchmarked) | Input | Batch | Precision / runtime | Latency | Effective TFLOP/s |
|---|---|---|---|---|---|
| DINOv2 ViT-B/14 + DPT (Depth-Anything-V2 **Base**, 97.5 M, 1025.5 GFLOPs) **(M-T4)** | 770×770 | 1 | TensorRT 10 fp16 | **55.40 ms** (18.1 FPS) | 18.5 |
| DINOv2 ViT-S/14 + DPT (DA-V2 **Small**, 24.8 M, 346.9 GFLOPs) **(M-T4)** | 770×770 | 1 | TensorRT 10 fp16 | **20.99 ms** (47.6 FPS) | 16.5 |
| DINOv2 ViT-B/14 + DPT **(M-T4)** | 644×644 | 1 | TensorRT 10 fp16 | **35.10 ms** | — |
| DINOv2 ViT-S/14 + DPT **(M-T4)** | 644×644 | 1 | TensorRT 10 fp16 | **13.62 ms** | — |
| CLIP ViT-L/14 **(M-T4)** | 224×224 | 8 | TensorRT fp16 | **65.63 ms total = 8.20 ms/image** | ~9.8 |
| CLIP ViT-L/14, INT8 (modelopt) **(M-T4)** | 224×224 | 8 | TensorRT INT8 | **131.76 ms — 2× *slower*** | — |
| CLIP ViT-L/14, INT8+FP16 mixed, after fix **(M-T4)** | 224×224 | 8 | TensorRT | **48.08 ms** | — |
| CLIP ViT-L/14 **(M-A6000, for scaling only)** | 224×224 | 1 / 8 | TensorRT fp16 | 3.26 / 16.86 ms | — |

Sources: `github.com/ultralytics/ultralytics/docs/macros/yolo-depth-speed-comparison.md` (harness: "inference-only,
`batch=1`, TensorRT fp16 on a Tesla T4", mean±std after warmup) and
`github.com/NVIDIA/Model-Optimizer/issues/167` (`trtexec --shapes=input:8x3x224x224 --fp16`, Azure NC4as_T4_v3).

Three things this table settles:

- **T4 achieves ~25–28 % of its 65 TFLOPS fp16 peak on ViT workloads**, and the fraction is remarkably stable
  across a 3× model-size change (18.5 vs 16.5 TFLOP/s). At batch 8 the fraction *falls* (~15 %, CLIP ViT-L/14),
  so batch does not rescue efficiency as fast as intuition suggests — A6000 goes 3.26 → 16.86 ms for 8× the
  work (5.2× latency, ~65 % batching efficiency).
- **The calibration constant for estimates below is 16.5–18.5 TFLOP/s at batch 1 fp16/TensorRT.** Everything
  I mark *(est)* divides GFLOPs by that band. It is arithmetic on cited numbers, not a measurement, and it does
  not hold above batch 1.
- **INT8 is not a free win here**: the one published T4 ViT INT8 attempt was **2× slower than fp16** and only
  beat it (48.08 vs 65.63 ms) after explicit quantisation + excluding `Add`. Budget fp16.

### 8.2 Measured anchors from a comparable GPU (RTX 3090, AMP/fp16, `timm` public CSVs) **(M-3090)**

Useful for *relative* ordering, and for cross-checking my FLOP model (3090 peak: 71 TFLOPS fp16 vs T4's 65).

| model | res | img/s (batch 1024 unless noted) | ms/img | GMACs (timm-measured) | my computed GMAC |
|---|---|---|---|---|---|
| `vit_base_patch32_224` | 224 | 5,098.7 | 0.20 | 4.41 | 4.2 |
| `vit_base_patch16_224` | 224 | 1,570.7 | 0.64 | 17.58 | 17.4 |
| `vit_base_patch16_siglip_224` | 224 | 1,434.6 | 0.70 | 17.73 | 17.4 |
| `vit_large_patch14_224` | 224 | 359.9 | 2.78 | 81.08 | 80.5 |
| `eva02_large_patch14_clip_336` | 336 | 113.2 (768) | 8.84 | 191.34 | 190.6 |
| `vit_so400m_patch14_siglip_384` | 384 | 80.8 (512) | 12.4 | 335.40 | 346.5 |
| `vit_base_patch14_dinov2` | 518 | 181.9 (512) | 5.50 | 151.71 | 151.5 |
| `vit_large_patch14_dinov2` | 518 | 58.8 (384) | 17.0 | 507.15 | 508.0 |

Source: `huggingface/pytorch-image-models`, `results/benchmark-infer-amp-nchw-pt240-cu124-rtx3090.csv`.
The FLOP model reproduces timm's measured values to within 3 %.

**Second, independent cross-check — against the DINOv3 paper's own FLOP table** (Fig. 16a, "Inference GFLOPs"
at 256² and 512²): ViT-S (21 M) **12 / 63**, ViT-S+ (29 M) 16 / 79, ViT-B (86 M) **47 / 216**, ViT-L (300 M)
**163 / 721**, ViT-H+ (840 M) 450 / 1903, ViT-7B (6716 M) 3550 / 14515; ConvNeXt-T/S/B/L 5/11/20/38 at 256.
My formula gives ViT-S 12.1, ViT-B 45.9, ViT-L 161 — **within 2.4 % of the paper's own numbers**, on a family I
did not calibrate against. That is why §8.4's table is usable as arithmetic rather than guesswork.

### 8.3 The T4 hard floor: bandwidth (300 GB/s, datasheet)

At batch 1 every weight is read once per image, so `bytes / 300 GB/s` is a floor no kernel trick beats:

| encoder (vision tower) | params | fp16 MB | **fp16 ms floor** | int8 |
|---|---|---|---|---|
| PE-Core-T/16-384 | 6.1 M | 12 | 0.04 | 0.02 |
| DINOv3-S/16 | ~21 M | 42 | 0.14 | 0.07 |
| DINOv2-S/14 | 22.1 M | 44 | 0.15 | 0.07 |
| PE-Core-S/16-384 | 23.8 M | 48 | 0.16 | 0.08 |
| DINOv2-B/14 | 86.6 M | 173 | 0.58 | 0.29 |
| **SigLIP2-B/16** | **92.9 M** | **186** | **0.62** | **0.31** |
| **PE-Core-B/16** | **93.7 M** | **187** | **0.62** | **0.31** |
| RADIO-B | 98.2 M | 196 | 0.65 | 0.33 |
| CLIP-L/14, DINOv2-L/14, EVA-02-L, InternViT-300M | ~304 M | ~608 | **2.03** | 1.01 |
| SigLIP2-L/16 | 316.3 M | 633 | 2.11 | 1.05 |
| SigLIP2-So400m/14 | 428.2 M | 856 | **2.85** | 1.43 |
| C-RADIOv4-SO400M | 431.2 M | 862 | 2.87 | 1.44 |
| C-RADIOv4-H | 651.6 M | 1303 | 4.34 | 2.17 |
| Qwen2.5-VL ViT | 676.6 M | 1350 | 4.50 | 2.25 |
| SigLIP2-g-opt | 1163.7 M | 2327 | 7.76 | 3.88 |
| PE-Core-G/14-448 | 1880 M | 3760 | **12.53** | 6.27 |

**Consequence:** on a T4, an L-size tower cannot go below ~2 ms and So400m cannot go below ~2.85 ms per image.
Those are floors — §8.1 shows DINOv2-ViT-B at 770² actually lands at 55 ms, ~90× its 0.58 ms floor, because
compute and launch overhead dominate.

### 8.4 The number that actually decides the architecture

ModernBERT-large, from its config: `3·1024² (qkv) + 1024² (out) + 2·1024·2624 (FFN) = 9.57 MMAC/token/layer`
→ **267.9 MMAC = 0.536 GFLOP per token across 28 layers** (attention term excluded).

| visual tokens into the text encoder | extra GMAC | extra GFLOP | **(est) ms on T4** @18.5 TFLOP/s |
|---|---|---|---|
| 0 (global vector as one token) | 0.27 | 0.54 | 0.03 |
| 64 (2×2 unshuffle of 256, or B/32-256) | 17.1 | 34.3 | **1.9** |
| 196 (SigLIP2 / PE-Core B/16 @224) | 52.5 | 105.0 | **5.7** |
| 256 | 68.6 | 137.2 | 7.4 |
| 576 (@384) | 154.3 | 308.6 | 16.7 |
| 729 (So400m/14 @384) | 195.3 | 390.6 | 21.1 |
| 1024 | 274.3 | 548.7 | 29.7 |
| 3328 (InternVL default) | 891.6 | 1783.2 | 96.4 |

Full-system estimates for the recommended configuration *(est)*:

| component | GFLOP | ms on T4 |
|---|---|---|
| SigLIP2-B/16 @224 image pass | 34.8 | 1.9 |
| 64 visual tokens in the 28-layer encoder | 34.3 | 1.9 |
| 128 text tokens | 68.6 | 3.7 |
| 20 option tokens | 10.7 | 0.6 |
| projection + scoring head | <1 | 0.1 |
| **total** | **~149** | **~8.2 ms** |

That is **~4× headroom** against 33 ms, which is the right place to be: the headroom absorbs ORT-vs-TRT
regression, batch-1 launch overhead and the gap between my 18.5 TFLOP/s estimate and reality. The same
system with So400m/14@384 and 729 tokens is **~880 GFLOP ≈ 48 ms** *(est)* — over budget before any overhead.
**The vision tower choice moves the total by ~2 ms; the token count moves it by ~20 ms.**

### 8.5 ONNX export status — verified by inspecting the shipped graphs

| Family | artifact | input signature as shipped | opset / ops | verdict |
|---|---|---|---|---|
| **SigLIP2** fixed-res | `onnx-community/siglip2-base-patch16-{224,256,384,512}-ONNX` + patch32/naflex/large/so400m/giant-opt (16 repos) | — | — | ✅ widely used (10,298 downloads on b/16-224) |
| **SigLIP2 NaFlex** | `onnx-community/siglip2-base-patch16-naflex-ONNX` | `pixel_values[b, max_num_patches, 768]`, `pixel_attention_mask`, `spatial_shapes` | **21 + `com.microsoft` 1** | ✅ **exports with a dynamic patch axis** — patch-embed becomes a `MatMul` on pre-flattened patches. ⚠️ needs ORT contrib ops (`MultiHeadAttention`); HF has a known `RuntimeError` when `actual_patches != max_num_patches` ([transformers#37907](https://github.com/huggingface/transformers/issues/37907)) |
| **DINOv2** | `onnx-community/dinov2-small/-base/-large/-with-registers-*` | `pixel_values[batch,3,h,w]` → `[b, floor(h/14)*floor(w/14)+1, 768]` | **14, `ai.onnx` only** | ✅ **cleanest in the survey**; `interpolate_pos_encoding` is in-graph as a `Resize` |
| **DINOv3** (no 7B export; S/S+/B/L/H+/ConvNeXt/SAT-493M only) | `onnx-community/dinov3-*-ONNX` (14) and `-ONNX-MHA` | plain: symbolic `s99/s100/s4`, out `[s99, (s100//16)*(s4//16)+5, 768]` | plain **20**; MHA **21 + `com.microsoft`** | ✅ RoPE exports as `Range`/`Slice`/`MatMul`; **variable resolution genuinely works** (the `+5` = 1 CLS + 4 registers) |
| **PE-Core** | `onnx-community/PE-Core-B16-224-ONNX`, `PE-Core-L14-336-ONNX` | B16-224: **`pixel_values[2,3,224,224]` — batch frozen to 2**; L14-336: `[batch,3,336,336]` | **14, `ai.onnx` only** | ⚠️ portable ops (`Conv/MatMul/Slice/Softmax`) but **no `config.json`/`preprocessor_config.json` shipped**, and B16 needs a re-export for any batch ≠ 2 |
| **Qwen2.5-VL** | `onnx-community/Qwen2.5-VL-3B-Instruct-ONNX` → `vision_encoder.onnx` | `pixel_values[num_patches,1176]`, `image_grid_thw[num_images,3]` | **21 + `com.microsoft`** (`RotaryEmbedding`, `MultiHeadAttention`, `SimplifiedLayerNormalization`, `NonZero`, `TopK`, `ScatterND`) | ✅ exports with fully dynamic resolution, ❌ heavily runtime-specific |
| **CLIP / SigLIP 1** | `onnx-community/clip-vit-base-patch32-ONNX`, `.../siglip-*` | — | — | ✅ |
| **OpenCLIP / DFN5B** | none official; third-party only (all ~0 downloads) | — | — | ❌ **`transformers` has no `open_clip` module** |
| **InternViT** | **none** (only `radna/Triton-InternViT-*` Triton backends) | — | — | ❌ |
| **RADIO / C-RADIO** | community-only (`ZumaInnovations/C-RADIOv4-SO400M-ONNX`, …) | — | — | ⚠️ `transformers` *does* have a `radio` module |
| **EVA-02** | one production artifact: `SmilingWolf/wd-eva02-large-tagger-v3` (fp32 only) | — | — | ⚠️ no `transformers` impl; export from `timm` is on you |
| PE-Spatial / AiMv2 / MobileCLIP2 | none found | — | — | ❌ |

**ONNX Runtime-specific facts worth knowing** (all cited):

- **Exporting with `attn_implementation="sdpa"` *loses* attention fusion.** [onnxruntime#21208](https://github.com/microsoft/onnxruntime/issues/21208)
  reports `{'Attention': 0, 'LayerNormalization': 3, 'SkipLayerNormalization': 23}` for a CLIP-ViT encoder:
  LayerNorm fusions fire, attention fusion does not, and the `eager` path is what ORT's patterns expect.
- **ORT's transformer optimizer does support ViTs, but only BERT-style fusions** — `"vit": (BertOnnxModel, "pytorch", 1)`,
  added in [PR #15289](https://github.com/microsoft/onnxruntime/pull/15289) on the reasoning "ViT … follows BERT
  rather closely". **SigLIP is optimised as `--model_type clip`** ([PR #23528](https://github.com/microsoft/onnxruntime/pull/23528)).
  There is no `--use_dynamic_axes` flag in current `optimizer.py`.
- **Extended fusions only apply to CPU/CUDA/ROCm-assigned nodes** — a TensorRT-partitioned graph gets them only
  where the EP partition allows.
- **All available measured data is TensorRT; ORT publishes no ViT/CLIP benchmark at all** (its 179-page
  performance docs contain no such table). Treat §8.1's numbers as TensorRT numbers.
- **DINOv2's `interpolate_pos_encoding` export bug is fixed — in the merge of 2024-09-05, not before.** The
  original code used Python-side typecasts and `scale_factor` interpolation, raising `TypeError` on export
  ([transformers#26790](https://github.com/huggingface/transformers/issues/26790)); the fix
  ([#33226](https://github.com/huggingface/transformers/pull/33226)) is what "means these vision models can be
  exported to ONNX with dynamic shapes enabled". **Require a `transformers` release after 2024-09-05** if you
  need dynamic H/W axes — older pins silently force you into fixed-resolution exports.

### 8.6 Batching and memory

- Exactly **one** measured T4 batch>1 datapoint exists (ViT-L/14 @224, bs8 = 65.63 ms). No T4 batch-1
  counterpart is published, so a T4 scaling curve cannot be computed from public data.
- Non-T4 illustration: A6000 fp16 ViT-L/14 goes 3.26 ms (bs1) → 16.86 ms (bs8) = 5.2× latency for 8× work.
- **Memory (arithmetic):** ViT-L fp16 weights ≈0.61 GB of the T4's 16 GB. For ViT-L/14@384, batch 32, 577 tokens,
  width 1024, one activation tensor is `32·577·1024·2 B = 37.8 MB`; a handful live per layer plus the residual
  stream puts peak around 1–2 GB — fits, but with little headroom at 768². No measurement establishes an
  actual max batch; **do not claim one.**
- Because batch is symbolic in the shipped SigLIP2/DINOv3 graphs but **frozen at 2 in PE-Core-B16**, re-batching
  may require a re-export.

## 9. Distilled / small variants — the complete menu

| Base family | Small variants available today | smallest usable |
|---|---|---|
| SigLIP2 | B/32-256 (94.6 M, **64 tokens**), B/16 (92.9 M), B/16-224/256/384/512, NaFlex B | B/32-256 for a 64-token budget |
| SigLIP 1 | B/16 (92.9 M) | B/16 |
| PE-Core | **T/16-384 (6.1 M)**, **S/16-384 (23.8 M)**, B/16 (93.7 M); PE-Spatial S (22.0 M), B (86.4 M) | PE-Core-T/16: 6.1 M params, 577 tokens |
| DINOv2 | S/14 (22.1 M), B/14 (86.6 M) | S/14 |
| DINOv3 | **ViT-S/16 (~21 M)**, S+/16, B/16, plus ConvNeXt-T/S/B/L distilled from 7B | ViT-S/16 (gated) |
| CLIP | B/32 (87.8 M, 49 tokens) | B/32 |
| InternViT | 300M only ("6B" is the large one) | 300M |
| Qwen2.5-VL | the tower is identical (676 M) in 3B and 7B — no small variant | none |
| RADIO | C-RADIOv2/v3/v4 B (98 M), v4-SO400M (431 M) | C-RADIOv2-B / v3-B (98 M) |
| EVA-02 | B/16 (86.3 M) | B/16 |
| MobileCLIP2 (Apple, AML licence) | S0/S2/S3/S4/B/L | S0 — but no `transformers` support, custom code |
| AiMv2 | L (309 M) and `large-patch14-224-distilled`, 1B, 3B | L-distilled |

Notable: the **PE-Core-T/16-384 at 6.1 M params with 577 tokens** is the extreme low end, and its token count
(577) makes it *more* expensive end-to-end than a B/16@224 (93.7 M, 197 tokens) once the text encoder is
counted — 154 GMAC vs 52 GMAC. **A small ViT at high resolution is not a latency win in this architecture.**

---

## 10. The four specific questions

### 10.1 Which encoders are best when the head is a masked-option scoring head?

The head here is a scoring function over option spans produced by a frozen-ish bidirectional encoder — the
vision tower is a pure feature extractor and its output must be *read* by text positions. That imposes four
requirements, in priority order:

1. **A pooled vector that is linearly decodable** (so the fusion layer has something low-entropy to start
   from), evidenced by RADIO's CLS→avg-pool ablation (49.01 → 38.36 ADE20K linear probe) and by every
   released family shipping an attention-pool head rather than a mean pool.
2. **Patch tokens with real localisation** for relational/spatial options, evidenced by Cambrian-1's
   CV-Bench (2D relations, counting, depth order) where SSL/dense features close the gap with CLIP features.
3. **Small token budget** — §8.4.
4. **Stable frozen behaviour**, because a decision system wants frozen features plus a small trained head
   (cheap retraining, stable calibration).

Ranking on those four:

| Rank | Encoder | Why |
|---|---|---|
| 1 | **SigLIP2-B/16** (`-224`, or NaFlex with `max_num_patches=128–256`) | MAP attention-pool (attention-pooled global) + 196 patch tokens; Apache-2.0; 16 ONNX exports; the SigLIP family is the **#1 encoder family in Cambrian-1's 20-encoder MLLM benchmark** (SigLIP So400m/14@384 rank 1 overall, SigLIP L/16@384 rank 5); published frozen-backbone spatial evidence (RefCOCO 83.76 with a frozen tower) |
| 2 | **PE-Core-B16-224** | IN-1k 78.4 at 197 tokens; attention-pool **plus** CLS (two independent global read-outs to average or concatenate); Apache-2.0; unbounded resolution in code; and the published "best features are mid-network" result gives you a *principled* reason to tap layers 6/9/12 for a decision head |
| 3 | **C-RADIOv4-SO400M / -H** | the only encoders that explicitly return `summary` **and** `spatial_features`, distilled from SigLIP2-g + DINOv3-7B + SAM3 (2026), commercial licence, arbitrary resolution ≤2048×2028 — but 431–652 M params and ≥2.9/4.3 ms bandwidth floor |
| 4 | **DINOv3 ViT-S/16 or PE-Spatial-B16-512** | if the options are *spatial/relational* (depth order, left/right, counting), the dense evidence favours DINOv3 (L: ADE20K 54.9 vs SigLIP2 43.6; NYU 0.352 vs 0.484) or PE-Spatial (frozen ADE20K linear 44.4 at B/16) over any contrastive-CLIP encoder |
| 5 | SigLIP2-So400m/14@224 (256 tokens) | the accuracy ceiling if you can afford 2.85 ms of bandwidth and a 1152→1024 projection |

**What to avoid for this head:** Qwen2.5-VL and InternViT towers (no standalone evaluation; 1024–16,384 tokens;
export hostility), OpenAI CLIP (superseded on every axis by SigLIP2 at identical token count), MetaCLIP 2
(non-commercial), DFN (AML licence, no safetensors), EVA-02 (no `transformers` implementation, 4.35 B for the
flagship).

### 10.2 Is there published evidence about which encoders transfer best to *decision/classification* rather
than captioning/generation?

Yes, and it is consistent — but it is all *probe*-based, never a masked-option head:

- **Cambrian-1** (arXiv 2406.16860) evaluates >20 encoders through MLLMs, produces a ranking table
  (Table 2): SigLIP So400m/14@384 **#1 overall**, OpenCLIP ConvNeXt-XXL@1024 #2, DFN-CLIP H/14@378 #3, SigLIP
  L/16@384 #5, EVA-CLIP-02 L/14@336 #7 overall but **#2 on vision-centric tasks**. On the SSL-only side
  DINOv2-L/14@518 ranks #1 across the board and #5 vision-centric. Their explicit Finding: "Language-supervised
  models outperform self-supervised or other models, [but] a well-trained self-supervised model like DINOv2 can
  also achieve competitive performance on vision-centric tasks." Its benchmark is VQA-shaped, so it measures
  *readable-by-an-LLM* features, which is close to but not the same as *decodable-by-a-linear-head* features.
- **RADIO/AM-RADIO** (2312.06709) evaluates everything frozen: k-NN ImageNet, zero-shot with the CLIP teacher's
  text tower, ADE20K/VOC linear probes, and frozen-feature LLaVA VQA. RADIO-H beats all three teachers
  (82.73 0-shot / 85.29 k-NN / 50.32 ADE20K / 85.00 VOC) — direct evidence that a distilled frozen extractor
  can beat its own teachers on discriminative tasks.
- **DINOv3** (2508.10104) is a frozen-backbone paper end-to-end: 7B reaches **88.4 IN-1k linear**,
  ADE20K 62.6–63.0 frozen with an adapter, COCO 66.1 mAP, DAVIS-L 83.3 — and it reports that ViT-H+ matches
  the 8× larger ViT-7B, i.e. the frozen-feature scaling curve is still climbing for discriminative tasks.
- **PE-Core** (2504.13181, Table 8) runs the cleanest frozen-classification battery in the literature: PE-Core-G
  (1.9 B) posts **86.8 k-NN / 89.5 linear / 89.8 attention** on ImageNet, vs DINOv2-g (1.1 B) 83.5 / 86.5 / 87.2,
  RADIOv2.5-g 85.3 k-NN, InternVL-C (5.5 B) 88.2 linear and EVA-18B (17.5 B) 88.9 linear. A contrastive encoder
  matching a 9× larger EVA on a *frozen* probe is the single strongest piece of evidence that this family is not
  a bad choice for a decision head. (G-scale only — no B/L probe table is published.)
- **SigLIP 2** reports frozen-backbone localization (RefCOCO +19.7 Acc@0.5 over SigLIP 1 with a frozen tower)
  and frozen dense probes (So400m only).
- **The counter-evidence:** InternViT's own papers report frozen-probe *decline* from V1 to V2.5 (82.5 → 78.5
  avg; ADE20K 47.2 → 39.4) as the tower was tuned for MLLM integration. Optimising an encoder for generation
  can *cost* you its frozen discriminative quality — the most important negative result in this survey for
  this project.

### 10.3 Current consensus (2025–2026): frozen vs fine-tuned, pooling, "align then fuse"

**Frozen vs fine-tuned — the literature genuinely disagrees, and the disagreement is informative:**

| Source | Finding |
|---|---|
| Prismatic VLMs (2402.07865, 2024) | "finetuning the vision backbone **dramatically degrades performance across almost all benchmarks** – especially on localization tasks" (their Fig. 5), in both single- and multi-stage paradigms |
| Cambrian-1 (2406.16860, 2024) | "**Unfreezing the vision encoder is widely beneficial.** Language-supervised models always benefit; SSL models particularly benefit on vision-centric benchmarks" |
| InternVL / LLaVA lineage | freeze for the align stage, unfreeze later; InternViT-6B is trained *for* this and still shows frozen-probe regression |

Reconciliation that both papers' data supports: unfreezing helps when (a) you have enough instruction-style
data to regularise it, and (b) the metric rewards spatial detail; freezing is safer for small heads, small
data, and calibration. **For a 33 ms decision system with a small head, frozen is the right default** — and
the evidence says the frozen penalty shows up mostly on localisation, not on global classification, which is
what a masked-option head needs.

**Pooling before a small fusion layer — the published answer is emphatic, and it is *not* "Perceiver resampler":**

Cambrian-1 Table 4 (verified cell-by-cell; four encoders — CLIP-L/14@336 + SigLIP-So400M@384 +
ConvNeXt-XXL@1024 + DINOv2-L/14@518 — with a Vicuna-1.5-7B, 576 tokens where used):

| Connector | General | Knowledge | OCR & Chart | Vision-Centric |
|---|---|---|---|---|
| Concatenation (no compression) | 67.2 | 48.9 | 50.1 | 52.6 |
| **Resampler (Perceiver/Q-Former style)** | 63.1 | 46.5 | **27.1 (−23.0)** | **42.6 (−10.0)** |
| SVA (no multi-agg) | 68.0 | 49.5 | 55.2 | 52.6 |
| SVA (full, Cambrian's design) | **68.5** | **49.7** | **55.5** | **53.2** |

A vanilla Perceiver resampler **loses 23 points on OCR & Chart and 10 on vision-centric tasks** while losing
only 4.1 on general benchmarks — i.e. it degrades precisely the information a spatial/relational decision needs.
The paper's own explanation: the resampler "lacks spatial inductive biases". And compression is not free even
with a good connector (Table 5, 576 → 36 tokens at fixed encoder): the best method loses ~19 pts on OCR & Chart
while General loses ~2.

Honeybee corroborates at equal token counts, comparing projectors on a normalised average of six spatial tasks:
**C-Abstractor 53.5 > Linear 52.6 > D-Abstractor 52.9 > Resampler 43.9** (144–256 tokens; C-Abstractor is
conv-then-resampler). Its reading: "the resampler, without consideration of local context preservation, shows
poor performance."

**Counter-evidence, recorded honestly:** MM1 (arXiv 2403.09611) tests attention pooling vs average pooling vs a
convolutional mapper on the same tokens and finds no conclusive winner, and Prismatic fuses two encoders by
**channel concatenation per patch** (token count unchanged, width doubled) with a significant gain on
localisation (p = 0.00162). The reconciling reading across all four: **connector *type* is second-order; whether
the connector preserves spatial structure and how many tokens you keep are first-order.**

So the ordering for a *decision* head is: **keep a spatially-structured reduction** (pixel-shuffle / 2×2
unshuffle as InternVL ships it — 1024 → 256 tokens; or conv-then-attention-pool), and **do not** use a vanilla
Perceiver resampler with few latents if the options refer to parts of the image. If the options are purely
global/attribute questions and latency is king, attention-pool to 1–16 tokens and accept the loss.

**"Align then fuse" — the staged recipe is standard, but its necessity is disputed:**

| Stage | What is trained | Evidence |
|---|---|---|
| 1. Align | vision tower frozen; only the projector learns (captioning/contrastive alignment) | LLaVA, BLIP-2. InternVL 2.5 §3.2 calls the stages **Stage 1** (MLP projector only, both towers frozen), **Stage 1.5** (low-LR tower adaptation) and **Stage 2** (joint, unified LR) |
| 2. Fuse | projector + LM on instruction/decision data | InternVL reports the biggest jump here |
| 3. Optional unfreeze | last-k blocks of the tower, **at a reduced LR** | disputed (§ above) — and if you do it, use LoRA or a low LR, not full fine-tuning |
| Counter-finding | Prismatic: "multi-stage training procedures adopted by existing work **can be eliminated without impact on performance**, reducing compute costs by 20-25%" | 2402.07865 |

**Attribution correction:** "align then fuse" is *not* InternVL's wording — a full-text search of
arXiv 2312.14238 (InternVL), 2404.16821 (InternVL 1.5) and 2412.05271 (InternVL 2.5) returns **zero**
occurrences of "align then fuse" and zero of "fusion". The *idea* is real and I recommend it; the slogan is
folklore, so cite the staged description rather than a quotation.

**A convergent free win worth acting on:** two independent labs found the **last layers of a vision tower are
worse** for downstream use and simply dropped them — Meta's PE paper (best intermediate layer beats the last by
−14.0 DAVIS J&F, §7.2) and InternVL 1.5, which "found that the features from the **fourth-to-last layer**
perform best for multimodal tasks, so we directly discarded the weights of the last three layers, reducing
InternViT-6B from 48 layers to 45". For a frozen-tower design this is a cheap experiment: probe the last ~4
blocks before committing to the final one.

For this project the staged recipe still applies, but compressed: stage 1 = train the 1024-dim projection
(+ optional pooled-token adaptor) with the text encoder frozen on image–text alignment; stage 2 = train
projection + text encoder on the decision task; stage 3 = only if the data supports it, unfreeze the last
2–4 ViT blocks, watching calibration.

### 10.4 Encoders that emit a global vector *plus* patch tokens — and which is better for a decision head

- **Both, deliberately:** RADIO/C-RADIO (`summary` `(B,C)` + `spatial_features` `(B,N,C)`, contract stated in
  the card), PE-Core (attention-pool head **and** CLS on T/S/B/L), SigLIP/SigLIP2 (MAP pooled output + patches
  from the same forward), DINOv2/v3 (CLS + patch tokens + 4 registers that must be **dropped, not averaged**).
- **Pooled-only by design:** PE-Spatial, PE-Lang (`pool_type="none"`), Qwen2.5-VL (no CLS, no pooler),
  AiMv2.
- **Patches-only by design:** InternViT at fixed 448 (CLS exists but the pipeline drops it).

Published evidence on *which* is better for a decision head:

| Evidence | Result |
|---|---|
| RADIO (2312.06709, Table 5) | CLS-token summary **49.01** vs average-pooled summary **38.36** on ADE20K **linear probe**; k-NN unmoved (83.91 → 83.83). A learned/structured summary ≫ naive mean-pool — **for a dense task** |
| **A 2026 frozen-ViT pooling study** (arXiv 2606.11606, n≈493 k chest radiographs, 5 frozen ViTs + a ResNet control, three pools from one forward pass) | **patch-mean ≈ CLS**: across the whole panel the CLS-vs-patch-mean gap is **0.000–0.058 AUC**. What *does* separate is **region-restricted** pooling: patch-local minus CLS is **+0.41 … +0.49 AUC** per model, and under isotropic blur all global pools sit at chance (0.500) while patch-local reaches 0.994–1.000. Also: "disease AUC on globally decided tasks ranged **0.642–0.913**" — **for globally-decided labels the CLS probe is fine** |
| Registers (2309.16588, Table 1) | a *single* patch token is ~20 pts worse than CLS on ImageNet (65.8 / 69.0 vs **86.0**), and the high-norm "outlier" patches that *do* carry global info are exactly what inflates a naive mean-pool |
| DINOv3's text aligner (2508.10104) | "A key enhancement … is the **concatenation of the mean-pooled patch embeddings with the output CLS token**" — two transformer layers on a frozen backbone; concat, not either/or |
| SigLIP2 dense protocol | "we concatenate **the output embedding of the MAP head**" where others concatenate CLS to each patch — pooled⊕patch again |
| PE paper | the last layer's pooled output is worse than intermediate patch features for dense/spatial tasks (−14.0 DAVIS J&F) — pooled output is not automatically the best global read-out either |
| Prismatic | channel-concatenating two encoders' patch features beats either alone on localisation |

**Synthesis — and it is subtler than "keep both".** Patch-mean and CLS are *the same information* (0.000–0.058
AUC apart); neither is a better global summary, and neither captures anything a region-restricted pool can
(ΔAUC +0.41…+0.49). So the real distinction is not CLS-vs-mean, it is **global vs. spatially-preserved**. That
gives a clean decision rule for this system:

- **If options are global attributes** ("is this a cat", "is this document signed") → any pooled vector works;
  use the encoder's own attention pool, cheap and adequate. The 0.642–0.913 AUC band for globally-decided labels
  says this is fine.
- **If options refer to parts of the image** ("is the *left* connector attached", "does the *third* row match") →
  no global pool will do it, however sophisticated; you must keep spatial structure in the tokens you feed the
  fusion layer, and you should keep *more* of them (which is where the token-budget analysis of §8.4 bites).
- **Put an attention-over-patches head on it, and treat the pooled vector as a free but marginal extra.**
  The one published head-to-head on this architecture (§6.6) is blunt: 100 learned queries cross-attending over
  196 patches give prec@1 **0.696** vs **0.565** for a linear probe on CLS (+0.131), and then adding the CLS
  token to that patch context buys **+0.005 MRR and 0.000 prec@1**. So: attention over patches is the win, and
  the pooled vector is measurably-but-negligibly additive. Keep it (it costs 0.27 GMAC, the cheapest token in
  the system, and it is what DINOv3's aligner, SigLIP2's dense probes and DINOv2's probe protocol all concat),
  but do not build the design around it.

Then **pixel-shuffle 2×2** (196 → 49–64 tokens) if the budget bites — but note from Cambrian Table 5 that
aggressive compression costs ~19 points on OCR/chart-style tasks while costing ~2 on general ones, so compress
to the budget the *task* needs, not to the smallest number that fits.

---

### 10.5 Calibration on a frozen encoder + small head — the standard recipe partially inverts

The product promise is calibrated confidence, and this is the one place where the literature for *this specific
architecture* says something that would be missed by copying the generic recipe.

| Evidence | Finding |
|---|---|
| Guo et al., arXiv 1706.04599 (the canonical reference) | temperature scaling on a standard classifier: ECE **16.53 % → 1.26 %** on CIFAR-100/ResNet-110. This is an *end-to-end fine-tuned* model, not a frozen-encoder probe |
| PRISM, arXiv 2610.36517 (pathology FMs, **864 linear-probe runs**: 8 models × 6 datasets × 6 label fractions × 3 seeds) | miscalibration is **under-confident in 252 of 288 cells** — i.e. the fitted temperature is **T < 1** — and temperature scaling *fails to help* in 49 of 288 |
| SCALE, arXiv 2609.38705 (8 pathology FMs) | calibration error is systematically higher on low-agreement cases than high-agreement ones in aggregate ECE. Temperature scaling buys only **0.035 → 0.031** overall and **0.191 → 0.175** on low-agreement cases (−8.1 %), whereas **retraining the head with agreement-aware label smoothing reaches ECE 3.1 %** against a 9.1 % end-to-end baseline at equal AUC |

Three design consequences, and one honest limit:

- **Frozen-encoder heads tend to be under-confident, not over-confident.** Before applying the textbook
  temperature scaling, fit T and *look at its sign*; a T > 1 correction applied to an already-under-confident
  head makes things worse.
- **Fixing the head beats post-hoc scaling.** The largest calibration gain in that evidence comes from the head's
  training objective (label smoothing / agreement-aware targets), not from rescaling logits afterwards.
- **Report ECE per subgroup, never only in aggregate.** The same model showed ECE 0.022 overall and **0.310** on
  its hardest subgroup. For a decision system, the per-option-count and per-image-type buckets are the ones that
  decide whether the confidence number is trustworthy.
- **The limit:** all of this evidence is pathology / medical imaging, over frozen *encoder* probes. **No published
  calibration numbers exist for a multi-encoder fusion head — this exact architecture is uncovered**, and none
  of it is about image-conditioned masked-option scoring. Treat the above as the best available prior and
  measure your own.

---

## 11. Ranked shortlist for a latency-sensitive, non-generative decision system

Scoring is against the actual constraints: 33 ms T4 batch 1, one forward pass, no generation, calibrated
confidence, and a masked-option scoring head.

### 🥇 1. SigLIP 2 ViT-B/16 — `google/siglip2-base-patch16-224` (or `-naflex` with `max_num_patches=256`)

| | |
|---|---|
| Vision params | **92,884,224** (total 375,187,970 — you ship only the vision prefix) |
| Tokens | 196 @224; 256 @256; 576 @384; NaFlex ≤256 aspect-preserving. **Pick NaFlex if images are documents/screens/panoramas** (at seq 256: −0.6 IN-1k but +1.3/+3.1/+1.9 R@1 on HierText/SciCap/Screen2Words); pick fixed-224 for square natural images |
| Width / projection | 768 → MAP head already outputs 768 |
| Quality | IN-1k 78.2, COCO T→I 52.1 / I→T 68.9, Flickr 80.7 / 93.0, IN 10-shot 72.1 |
| Licence / gating | **Apache-2.0, ungated** |
| ONNX | **16 community exports** (all fixed resolutions) **+ a NaFlex export with a dynamic patch axis** (needs ORT contrib ops) |
| Latency | 0.62 ms bandwidth floor; 17.4 GMAC image pass ≈ **1.9 ms** *(est)*; 196 tokens → 52.5 GMAC ≈ 5.7 ms *(est)*. Whole system at 64 tokens ≈ **8 ms** *(est)* vs a 33 ms budget |
| Why first | best licence/evidence/export triple; MAP pooled head + patch tokens in one forward; the SigLIP family is the best-measured encoder family in both Cambrian-1 and the SigLIP2 dense probes; backward-compatible with SigLIP-1 weights so you can A/B the recipe at identical shapes |

**Why not So400m/L:** 2.9/2.1 ms of pure bandwidth plus 576–729 tokens (154–195 GMAC in the text encoder) eats
most of a 33 ms budget for ~+5 IN-1k. If the decision task is subtle enough to need it, move to
`so400m-patch14-224` (256 tokens) rather than 384.

### 🥈 2. PE-Core-B16-224 — `facebook/PE-Core-B16-224`

| | |
|---|---|
| Vision params | **93,673,984**, width 768, 12 layers, patch 16, native 224 → 196 patch tokens + CLS |
| Global read-outs | **two**: 8-head attention pool *and* a CLS token |
| Quality | IN-1k **78.4**, IN-v2 71.7, ObjectNet 71.9, COCO-T2I 50.9 |
| Licence / gating | **Apache-2.0, ungated** (no LICENSE file in the repo — the only wart) |
| ONNX | 2 community exports (+8 quantised variants each): **B16-224 and L14-336 only** — no T/S/Spatial/Lang. B16's batch is frozen to 2 |
| Resolution | variable/unbounded in code (RoPE2D grid update, no token cap) |
| Why second | same accuracy as SigLIP2-B at the same token count; a second independent global read-out; the family ships the only published **frozen-probe battery** in the literature (PE-Core-G: 86.8 k-NN / 89.5 linear / 89.8 attention, beating DINOv2-g) which is the closest published proxy for "frozen features → small head"; an explicit "probe the middle layers" result (−14.0 DAVIS J&F between best and last layer) that gives you a principled tuning knob; and a *free upgrade path* to `PE-Spatial-B16-512` (86.4 M) if the options turn out to be spatial |

**Caveat that keeps it second:** the HF repos ship `.pt`-only checkpoints with an empty `config.yaml`, so you
must either use the `timm` mirror (`vit_pe_core_base_patch16_224.fb`) or port the `core.vision_encoder` code
path — more integration work than SigLIP2's three lines of `transformers`.

### 🥉 3. C-RADIOv4-SO400M — `nvidia/C-RADIOv4-SO400M` (fallback: `C-RADIOv4-H`)

| | |
|---|---|
| Params | **431,237,240** (SO400M, width 1152, 27 layers) / 651,645,448 (H, width 1280, 32) |
| Outputs | `summary` (B,C) **and** a patch-feature tensor (documented as `spatial_features` in the card, `features` (B,T,D) in the code) — note **C ≠ D**: the global and spatial widths differ, so plan for two projections. That is the contract you actually want |
| Resolution | up to **2048×2028 in 16-px increments**, 16,384-entry position table |
| Provenance | distilled from **SigLIP2-g + DINOv3-7B + SAM3** (released 2026-01-27, arXiv 2601.17237) |
| Licence | **NVIDIA Open Model License — commercial use permitted** ("ready for commercial/non-commercial use") |
| Latency | tower alone: **9.4 ms @224 px / 54.7 ms @its native 512 px** for SO400M; **13.7 / 78.9 ms** for H *(est, 18.5 TFLOP/s)*. Bandwidth floor 2.87 / 4.34 ms. **Run it at 224, not 512, if you use it at all** |
| Why third | the only family that is *architecturally* global+patch, and the only 2026 model that fuses the two best teachers (SigLIP2 for semantics, DINOv3 for dense) under a commercial licence. Use it when the decision depends on resolution or dense detail (documents, small objects, spatial relations) — that is where B-size contrastive encoders measurably lose (SigLIP2-B ADE20K 41.6 vs DINOv3-B 51.8) |

**Why not first:** even at 224 px it is 4.6× the parameters of a B/16 and ~5× its image-pass cost, for a system
whose bottleneck is the text encoder — and it buys accuracy you cannot yet demonstrate on a decision task. At
its native 512 px it simply does not fit the budget (≈55 ms of tower alone). There is **no official ONNX
export** (community-only, 0-download provenance), the HF repos are `.pt`-only, and loading requires
`trust_remote_code=True` — three integration risks SigLIP2 does not have. Its real niche is the *document /
high-resolution* case, where you would run it at 512–1024 px and accept 30–80 ms.

### 🏅 4. Honourable mention: PE-Spatial-B16-512 — `facebook/PE-Spatial-B16-512`

**86,433,792 params** (smaller than PE-Core-B!), 1024+CLS tokens at 512 px, attention-pool **disabled**
(`pool_type="none"`), Apache-2.0, and the best frozen dense numbers in the PE family: ADE20K linear probe
**44.4**, zero-shot DAVIS J&F **58.9** — i.e. it recovers the spatial information the PE paper says PE-Core
loses at its output (−14.0 J&F, §7.2). Latency: 1025 tokens × 12 layers × 768 width ≈ 106 GMAC ≈ **11.5 ms**
*(est)* for the tower, plus **275 GMAC (≈30 ms)** to inject 1024 tokens into the text encoder — so **pool it
before fusing** (2×2 unshuffle → 256 tokens → 7.4 ms) or use it only as a global+pooled feature source.
Pick this instead of #2 if the options turn out to be spatial/relational rather than global-attribute.

### Deliberately not recommended
- **DINOv3 ViT-S/16** — technically attractive (cheapest dense features, ~21 M params, any resolution via
  RoPE), but `gated: manual` + custom licence with a "Built with DINOv3" obligation and no ungated ONNX
  workflow makes it a compliance/procurement decision, not an engineering one. Reconsider it *only* if the
  options turn out to be spatial/relational, in which case pair it via channel concat with SigLIP2 (Prismatic's
  recipe) rather than replacing.
- **Any So400m/Giant/7B tower, Qwen2.5-VL, InternViT-6B** — bandwidth floor plus token bill exceed the budget.
- **MetaCLIP 2** (CC-BY-NC), **DFN5B** (AML licence, no safetensors), **RADIO v1** (non-commercial NSCLv1).
- **EVA-02** — no `transformers` implementation, so everything (including ONNX export) is on you.

### Recommended wiring (concrete)

1. Vision tower at 224 (fixed) → 196 patch tokens + one MAP-pooled vector.
2. **2×2 pixel-unshuffle** the 196 tokens → 49–64 tokens (InternVL's deployed trick), or start with 196 and
   measure; then one `Linear(768 or 1152 → 1024)` + LayerNorm — a 2-layer GELU MLP is the LLaVA/Prismatic
   default and costs ~2 M params.
3. Prepend `[pooled_vector]` as one extra token, then `[64 visual tokens][text][option spans]` in the 28-layer
   bidirectional encoder. Budget: 64 + ~128 text + ~20 option ≈ 212 tokens ≈ 57 GMAC of the text encoder.
4. Fine-tune: stage 1 = projection only, text encoder frozen; stage 2 = projection + text encoder; stage 3 only
   if data allows. Keep the vision tower frozen (Prismatic's evidence) unless Cambrian-style gains appear.
5. **Head: attention over the patch tokens, not a projection of the pooled vector.** The published head-to-head
   (§6.6) is worth **+0.131 prec@1** (0.696 vs 0.565) for group-query cross-attention over 196 patches, while
   adding the CLS token to that context buys +0.005 MRR. Cheap version: 16–32 learned queries, one cross-attention
   layer over the projected patch tokens, then score options from the query outputs.
6. **Loss before backbone.** In the same study, BCE → Asymmetric Loss moved the flat head **+0.130 prec@1** —
   as large as the entire head redesign. With many options and few positives per decision, handle the imbalance
   in the loss first.
7. Calibration: temperature-scale on a held-out split after stage 2 and report ECE per option-count bucket —
   and **check the sign before trusting it**. The only published calibration evidence on frozen-encoder probes
   (medical imaging, §10.5) finds the standard story inverts: temperature scaling buys only ~8 % of the error
   and the probes are *under*-confident, i.e. fitted T < 1, in most cells. Fixing the head (agreement-aware label
   smoothing) beat temperature scaling there. Frozen vision features also make the head's logits more stable
   across domains, which is itself a calibration argument for freezing.

---

## 12. What the numbers do not tell us

**1. Almost nothing is measured on a T4, and nothing at all for the models you are actually choosing between.**
The two measured datapoints in §8.1 are for a DINOv2+DPT *pipeline* at 770² and a CLIP ViT-L/14 at batch 8. There
is **no** T4 measurement of SigLIP, SigLIP2, DINOv3, PE-Core, InternViT or Qwen, at any batch, anywhere I could
find — and the standard "efficient ViT" papers (MobileViT, EfficientViT, LeViT, FastViT, MobileCLIP) never
benchmark on a T4 at all (V100/A100/iPhone/Volta). My §8.4 numbers are arithmetic on the 16.5–18.5 TFLOP/s
constant derived from those two datapoints. **The first engineering task in the project should be a batch-1
fp16 T4 sweep of your two or three finalists, not more reading.**

**2. No benchmark measures this task.** Every quality number in §6 is zero-shot classification, retrieval,
linear probing or VQA. Nothing published scores masked options over (image + bidirectional text) with a small
head. The gap between "SigLIP2 wins Cambrian-1's VQA-shaped encoder ranking" and "SigLIP2 gives the best
masked-option decisions" is exactly one experiment you have to run yourself — and Cambrian-1's own framing
warns that its ranking measures readability by an LLM, not decodability by a linear head.

**3. Calibration evidence exists, but only for medical frozen-encoder probes — not for this architecture.**
§10.5 has the numbers (under-confident in 252/288 probe cells; temperature scaling buying ~8 %; head-retraining
beating it). But there is **no published calibration result for a multi-encoder fusion head, and none for
image-conditioned masked-option scoring**. The claim that frozen features calibrate better is a reasonable prior,
not a finding.

**4. Accuracy deltas at B-size are inside the noise of a downstream decision task.** 78.2 (SigLIP2-B/16),
78.4 (PE-Core-B/16) and 79.1 (SigLIP2-B/16@256) are different *token budgets and recipes*, not a reliable
ordering of decision quality. Choose on licence, export, token budget and available global read-out; treat the
0.2-point gaps as noise.

**5. Retrieval protocols are ambiguous in the primary sources.** Neither SigLIP paper states whether COCO is
1k or 5k, nor labels Flickr30k, nor specifies R@k beyond "recall@1". Numbers from these papers should not be
compared to 5k-split numbers from other papers.

**6. Dense evidence is thin exactly where you would want it.** SigLIP2's dense probes exist **only for
So400m/14 at 224/384** — there is no B/16 dense number anywhere. DINOv3's per-size table reports **IN-ReaL**,
not IN-1k val. PE's 78.4/83.5/85.4 are zero-shot with the text tower, not linear probes.

**7. ONNX "exists" ≠ "numerically matches".** 16 community SigLIP2 exports exist; none publishes a
PyTorch-vs-ONNX parity diff or an fp16 accuracy delta, and the two PE-Core exports ship **no README, no licence
tag and no config**. Half the "good" exports depend on `com.microsoft` fused ops, which pins you to ONNX Runtime
(the portable ones are DINOv2 opset 14 and PE-Core opset 14 — `ai.onnx` only). Verify parity yourself.

**8. Batch-32 behaviour is unmeasured.** Timm's numbers are at batch 384–1024. Whether a ViT-B at 224 saturates
the T4 at batch 32, and what a 16 GB card does with ViT-L fp16 at 384×32, is arithmetic in §8.6, not evidence.

**9. No licence here covers the training data.** All these encoders are trained on undisclosed web-scale
image–text data. "Apache-2.0" on the weights says nothing about the provenance of the images, and none of the
cards address it. This is a counsel question, not a table cell.

**10. The one thing the numbers *do* say clearly, and it is not about encoders.** For this architecture the
vision encoder choice is a second-order decision. The first-order decision is **how many visual tokens reach
the 28-layer bidirectional encoder** — 267.9 MMAC each, 3× the cost of a whole B/16 image pass at 196 tokens.
Any of the top three encoders, pooled to 64 tokens, will beat any encoder at 729 tokens on latency, and by a
wide margin.

---

## 13. Sources

Primary, all fetched 2026-11-02.

**Papers**
- SigLIP 2 — Tschannen et al., Google DeepMind, arXiv:2502.14786 (20 Feb 2025): <https://arxiv.org/abs/2502.14786>, full text <https://arxiv.org/html/2502.14786v1>
- SigLIP — Zhai et al., arXiv:2303.15343: <https://arxiv.org/abs/2303.15343>
- Perception Encoder — Bolya et al., arXiv:2504.13181: <https://arxiv.org/abs/2504.13181>, <https://arxiv.org/html/2504.13181v2>
- DINOv2 — Oquab et al., arXiv:2304.07193: <https://arxiv.org/abs/2304.07193>
- DINOv3 — Meta FAIR, arXiv:2508.10104: <https://arxiv.org/abs/2508.10104>, <https://arxiv.org/html/2508.10104v1>
- AM-RADIO — arXiv:2312.06709: <https://arxiv.org/html/2312.06709v2>
- RADIOv2.5 — arXiv:2412.07679: <https://arxiv.org/html/2412.07679v1>
- C-RADIOv4 tech report — arXiv:2601.17237
- Cambrian-1 — arXiv:2406.16860: <https://arxiv.org/html/2406.16860v1>
- Prismatic VLMs — arXiv:2402.07865: <https://arxiv.org/html/2402.07865v1>
- Honeybee — arXiv:2312.06742: <https://arxiv.org/html/2312.06742v1>
- InternVL 1.5 — arXiv:2404.16821; InternVL 2.5 — arXiv:2412.05271: <https://arxiv.org/html/2412.05271v1>
- Qwen2.5-VL — arXiv:2502.13923: <https://arxiv.org/html/2502.13923v1>
- EVA-02 — arXiv:2303.11331; EVA-CLIP — arXiv:2303.15389
- Vision Transformers need registers — arXiv:2309.16588
- Frozen-ViT pooling study (CLS vs patch-mean vs region-restricted; n≈493 k chest radiographs, 5 frozen ViTs) — arXiv:2606.11606
- C-RADIOv4 tech report — arXiv:2601.17237 (Table 1 = the cross-family frozen-probe table in §6.5; Table 2 = resolution scaling)
- Frozen-encoder + small-head decision study — arXiv:2610.38462 (Table 2 = the head/LoRA/loss ablation in §6.6; PDF parsed with `pdftotext`, no HTML available)
- Calibration on frozen-encoder probes: Guo et al. arXiv:1706.04599 (canonical temperature scaling); PRISM arXiv:2610.36517 (864 probe runs, under-confidence in 252/288 cells); SCALE arXiv:2609.38705 (agreement-aware label smoothing vs temperature scaling) — all three in §10.5
- MM1 — arXiv:2403.09611 (connector-type ablation; full text not fetchable as HTML in this session, cited via the secondary reading noted in §10.3)

**Checkpoints / licences / configs** (representative; every id in §2 was probed the same way)
- `https://huggingface.co/api/models/google/siglip2-base-patch16-224`, `.../facebook/PE-Core-B16-224`, `.../facebook/dinov3-vits16-pretrain-lvd1689m`, `.../nvidia/C-RADIOv4-H`, `.../OpenGVLab/InternViT-300M-448px-V2_5`, `.../Qwen/Qwen2.5-VL-7B-Instruct`
- `https://huggingface.co/google/siglip2-base-patch16-naflex/resolve/main/preprocessor_config.json` (`max_num_patches: 256`)
- `https://huggingface.co/answerdotai/ModernBERT-large/resolve/main/config.json`
- DINOv3 License (14 Aug 2025): <https://ai.meta.com/resources/models-and-libraries/dinov3-license/>
- NVIDIA Open Model License: <https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf>
- PE code config: `https://raw.githubusercontent.com/facebookresearch/perception_models/main/core/vision_encoder/config.py`; PE card table: `.../apps/pe/README.md`
- `transformers` DINOv3 modelling: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py`
- NAFlex processor: `https://github.com/google-research/big_vision/blob/main/big_vision/pp/proj/image_text/ops_naflex.py`

**Latency / hardware / ONNX**
- NVIDIA T4 datasheet (**300 GB/s**, 65 TFLOPS fp16 tensor, 130 TOPS INT8, 16 GB GDDR6, 8.1 TFLOPS fp32): <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf> — note NVIDIA's product page says "320+ GB/s"; the datasheet says 300. Use 300 for modelling and do not cite "320" as the datasheet figure.
- Measured T4 batch-1 fp16/TensorRT (DINOv2 ViT-B/S + DPT, 770² and 644²): <https://github.com/ultralytics/ultralytics/blob/main/docs/macros/yolo-depth-speed-comparison.md>
- Measured T4 batch-8 fp16/INT8 (CLIP ViT-L/14 @224, `trtexec`): <https://github.com/NVIDIA/Model-Optimizer/issues/167>
- timm measured AMP inference benchmarks (RTX 3090): <https://github.com/huggingface/pytorch-image-models/tree/main/results>
- ORT attention-fusion failure on CLIP-ViT: <https://github.com/microsoft/onnxruntime/issues/21208>; ViT optimizer support: <https://github.com/microsoft/onnxruntime/pull/15289>; SigLIP fusions: <https://github.com/microsoft/onnxruntime/pull/23528>; graph-optimization scope: <https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html>
- Artifacts inspected directly: `onnx-community/siglip2-base-patch16-naflex-ONNX`, `onnx-community/dinov2-base`, `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX` (+`-MHA`), `onnx-community/PE-Core-B16-224-ONNX`, `onnx-community/Qwen2.5-VL-3B-Instruct-ONNX`
- NaFlex's `actual_patches != max_num_patches` failure: <https://github.com/huggingface/transformers/issues/37907>

**Working data for this survey** (all reproducible, no network needed to re-derive the tables)
- `.survey/hdr/` — cached safetensors headers; `.survey/extract.json` — per-repo shape extraction
- `.survey/compute.py` — FLOP/token model (cross-checks against timm's measured GMACs); `.survey/hf_probe.py`, `.survey/dims.py` — the ranged-header probes
- `.survey/notes-siglip.md`, `.survey/notes-dino.md`, `.survey/notes-pe.md`, `.survey/notes-intern-qwen-radio-eva.md`, `.survey/notes-onnx-latency.md`, `.survey/notes-consensus.md` — per-family evidence with a URL on every row; the ONNX claims in §8.5 come from parsing the shipped `.onnx` graphs, not from READMEs
