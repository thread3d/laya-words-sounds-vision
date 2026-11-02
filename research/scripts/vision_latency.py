"""Relative cost of candidate vision towers vs the Laya text encoder, on one machine.

Absolute CPU numbers are not GPU numbers; the ratio is the portable part. The text encoder run is
ModernBERT-large at the Laya checkpoint's default max_len=512 with a realistic head.
"""
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)   # `import laya` works without an installed package
MODELS_ROOT = os.environ.get("LAYA_MODELS", os.path.join(REPO, "models"))
CKPT = os.path.join(MODELS_ROOT, "laya")
import time, torch

torch.set_num_threads(8)
from transformers import SiglipVisionModel

def bench(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1000

import laya
from laya.common import build_sequence
agent = laya.load(CKPT)
enc = agent.model.encoder.eval()
q = {"k": {"type": "choice", "instructions": "Which team should handle this?",
           "criteria": {"billing": "a", "technical": "b", "sales": "c", "other": "d"}}}
internal = [agent._to_internal(v) for v in q.values()]
ids, _ = build_sequence(agent.tok, "The invoice charged me twice " * 40, internal[0], 512, 192)
inp = torch.tensor([ids[:512]])
att = torch.ones_like(inp)
pos = torch.arange(inp.size(1)).unsqueeze(0)
with torch.no_grad():
    t_text = bench(lambda: enc(input_ids=inp, attention_mask=att, position_ids=pos))
L = inp.size(1)
print(f"TEXT  ModernBERT-large@{L} tokens           {t_text:8.1f} ms   (baseline)")

for name, res in [("google/siglip2-base-patch16-224", 224), ("google/siglip2-so400m-patch14-384", 384)]:
    try:
        vm = SiglipVisionModel.from_pretrained(name).eval()
        px = torch.rand(1, 3, res, res)   # no torchvision needed; shape is all that matters
        with torch.no_grad():
            out = vm(pixel_values=px)
        ntok = out.last_hidden_state.size(1)
        params = sum(p.numel() for p in vm.parameters()) / 1e6
        with torch.no_grad():
            t = bench(lambda: vm(pixel_values=px))
        print(f"VIS   {name.split('/')[-1]:38s} {t:8.1f} ms   {ntok:4d} patch tokens, {params:5.1f}M params"
              f"   ratio {t/t_text:5.2f}x text")
    except Exception as e:
        print(f"VIS   {name}: {type(e).__name__}: {str(e)[:120]}")
