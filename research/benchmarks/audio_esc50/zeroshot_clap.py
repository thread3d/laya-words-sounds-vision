"""A zero-shot comparator measured here: CLAP's text-audio similarity on ESC-50.

The published table in `docs/audio-benchmark-assessment.md` quotes CLAP at 82.6 % zero-shot on ESC-50.
That number is the cheapest strong comparator to reproduce, because nothing is trained: the model
embeds each clip and 50 text prompts, and the answer is the most similar prompt. Running it here makes
the comparison local rather than borrowed -- same clips, same protocol, same machine.

Reading the audio: ESC-50 clips are 44.1 kHz mono PCM, and the standard library reads them, which keeps
this script free of the `soundfile` the ROCm interpreter does not have. CLAP's processor wants the
sample rate it was trained on, so the raw 44.1 kHz samples are handed over as they are.

    source research/benchmarks/audio_esc50/gpuenv.sh
    python3 research/benchmarks/audio_esc50/zeroshot_clap.py [--limit N] [--templates 1]

Writes `$AUDIO_BENCH_DATA/esc50/results_clap_zeroshot.json`.
"""
import argparse
import glob
import json
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

DATA_ROOT = os.environ.get("AUDIO_BENCH_DATA", os.path.join(REPO, ".audio-bench"))
ROWS = os.path.join(DATA_ROOT, "esc50")
MODEL = os.environ.get("AUDIO_BENCH_CLAP", "laion/clap-htsat-unfused")

# The published CLAP protocol averages several prompts per class; `--templates` picks how many.
TEMPLATES = [
    "This is the sound of {}.",
    "This is a sound of {}.",
    "a photo of {}",
    "sound of {}",
]


def _feature(output):
    """The contrastive embedding out of `get_{audio,text}_features`.

    transformers 5.13 returns a `BaseModelOutputWithPooling` rather than a tensor; the projected,
    pooled vector is `.pooler_output`, with the head baked in.
    """
    import torch
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


def read_wav(path, sample_rate=48000):
    """`[samples]` float32 at CLAP's own rate (48 kHz), through the package's own loader.

    ESC-50 is 44.1 kHz and CLAP's feature extractor refuses anything else, so the clip is resampled
    rather than handed over at the wrong rate -- a model given 44.1 kHz samples labelled 48 kHz hears
    everything ~9 % sharp, which is a different clip.
    """
    from laya.audio import load_waveform
    return load_waveform(path, sample_rate).numpy(), sample_rate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="score at most N clips")
    parser.add_argument("--templates", type=int, default=len(TEMPLATES),
                        help="how many prompt templates to average per class (default: all 4)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    import torch
    from transformers import ClapModel, ClapProcessor

    with open(os.path.join(ROWS, "meta.json"), encoding="utf-8") as handle:
        classes = json.load(handle)["classes50"]

    # Every clip once: the zero-shot protocol trains on nothing, so all 2 000 are scored and the
    # accuracy is the mean over the five cross-validation folds by construction.
    audio_dir = os.path.join(DATA_ROOT, "data", "ESC-50-master", "audio")
    paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not paths:
        print("no clips under %s -- run prep.py first" % audio_dir, file=sys.stderr)
        return 2
    if args.limit:
        paths = paths[:args.limit]

    processor = ClapProcessor.from_pretrained(MODEL)
    model = ClapModel.from_pretrained(MODEL).eval()
    if args.device != "cpu":
        # CLAP's HTSAT encoder is convolutional, and MIOpen segfaults on every conv on this build;
        # the harness patch routes them through unfold + matmul (verified against `conv2d` on the
        # model's own weights before any of it is used).
        import gpu_patch
        print("gpu_patch: conv1d=%d conv2d=%d" % gpu_patch.patch_model(model), flush=True)
        model = model.to(args.device)

    prompts = [TEMPLATES[t].format(name) for name in classes
               for t in range(args.templates)]
    text = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeds = _feature(model.get_text_features(
            **{k: v.to(args.device) for k, v in text.items()}))
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds.view(len(classes), args.templates, -1).mean(dim=1)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    truth, preds = [], []
    started = time.time()
    for start in range(0, len(paths), args.batch):
        chunk = paths[start:start + args.batch]
        waves, rates = zip(*(read_wav(path) for path in chunk))
        audio = processor(audio=list(waves), sampling_rate=rates[0], return_tensors="pt",
                          padding=True)
        with torch.no_grad():
            audio_embeds = _feature(model.get_audio_features(
                **{k: v.to(args.device) for k, v in audio.items()}))
        audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
        similarity = audio_embeds @ text_embeds.t()
        preds.extend(similarity.argmax(dim=-1).cpu().tolist())
        # The filename's last field *is* the class index (`1-100032-A-0.wav` -> 0). Keep the truth as
        # an index, like the prediction: comparing an index with a class name scores zero and looks
        # exactly like a model that has learned nothing.
        truth.extend(int(os.path.splitext(os.path.basename(p))[0].split("-")[-1]) for p in chunk)
        done = min(start + args.batch, len(paths))
        if done % 200 < args.batch or done == len(paths):
            print("  %d/%d  (%d classes)  %.1fs" % (done, len(paths), len(classes),
                                                   time.time() - started), flush=True)

    correct = sum(1 for p, t in zip(preds, truth) if p == t)
    accuracy = correct / len(truth)
    report = {"model": MODEL, "clips": len(truth), "classes": len(classes),
              "templates": args.templates, "accuracy": accuracy,
              "correct": correct, "device": args.device,
              "seconds": round(time.time() - started, 1),
              "protocol": "zero-shot, no training; accuracy over all scored clips"}
    out = args.out or os.path.join(ROWS, "results_clap_zeroshot.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("CLAP zero-shot: %.4f on %d clips (%d classes, %d template(s))"
          % (accuracy, len(truth), len(classes), args.templates))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
