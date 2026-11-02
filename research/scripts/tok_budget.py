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
import laya
agent = laya.load(CKPT)
enc = agent.model.encoder.eval()
def bench(fn, n=5, warm=2):
    for _ in range(warm): fn()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    ts.sort(); return ts[len(ts)//2]*1000
base=None
for L in (192, 256, 512, 768, 1024, 2048):
    inp = torch.randint(0, 50000, (1, L))
    att = torch.ones_like(inp); pos = torch.arange(L).unsqueeze(0)
    with torch.no_grad():
        t = bench(lambda: enc(input_ids=inp, attention_mask=att, position_ids=pos))
    if base is None: base = t
    print(f"L={L:5d}  {t:9.1f} ms   relative to L=192: {t/base:5.2f}x")
