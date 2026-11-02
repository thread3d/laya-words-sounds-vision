"""Why did the Stage-1 spike collapse? Isolate where the colour signal dies."""
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)   # `import laya` works without an installed package
MODELS_ROOT = os.environ.get("LAYA_MODELS", os.path.join(REPO, "models"))
CKPT = os.path.join(MODELS_ROOT, "laya")
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8); torch.manual_seed(0)
import laya
from laya.common import build_head
from transformers import SiglipVisionModel

COLOURS = [(220,40,40),(40,60,220),(40,180,60),(225,200,50)]
def make_images(n_per=120, res=224, size=32):
    imgs, labels = [], []
    for ci, col in enumerate(COLOURS):
        for _ in range(n_per):
            base = torch.tensor(col).view(3,1,1).float().div(255).expand(3,size,size).clone()
            base += 0.06*torch.randn(3,size,size)
            other = torch.tensor(COLOURS[(ci+1)%4]).view(3,1,1).float().div(255)
            y = torch.randint(0, size-6, (1,)).item()
            base[:, y:y+3, :] = 0.8*other + 0.2*base[:, y:y+3, :]
            imgs.append(F.interpolate(base.unsqueeze(0), size=(res,res), mode="bilinear").squeeze(0))
            labels.append(ci)
    return torch.stack(imgs), torch.tensor(labels)

tower = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224").eval()
X, Y = make_images()
with torch.no_grad():
    P = tower(pixel_values=X).last_hidden_state          # [N,196,768]

def probe(feats, name, steps=300, dim=None):
    feats = feats.detach()
    d = dim or feats.size(-1)
    torch.manual_seed(0)
    W = nn.Linear(d, 4)
    opt = torch.optim.Adam(W.parameters(), lr=1e-2)
    n = int(len(Y)*0.8)
    for _ in range(steps):
        idx = torch.randint(0, n, (64,))
        loss = F.cross_entropy(W(feats[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (W(feats[n:]).argmax(-1) == Y[n:]).float().mean().item()
    print(f"  {name:52s} linear-probe acc = {acc:.3f}")
    return acc

print("1) is the class signal present in the tower output?")
probe(P.mean(1), "SigLIP2 patches, mean-pooled [768]")

agent = laya.load(CKPT)
model = agent.model; model.encoder.eval()
for p in model.encoder.parameters(): p.requires_grad_(False)
D = model.encoder.config.hidden_size
q = {"k": {"type":"choice","instructions":"What is the background colour?","criteria":{c:None for c in ["red","blue","green","yellow"]}}}
internal=[agent._to_internal(v) for v in q.values()]
head_ids, markers, _ = build_head(agent.tok, internal[0], 192)
splice = len(head_ids); NV = 8
ids = head_ids + [agent.tok.sep_token_id]

def run(patches, bridge, drop_vis=False):
    B = patches.size(0)
    with torch.no_grad():
        tok = model.encoder.embeddings.tok_embeddings(torch.tensor(ids).unsqueeze(0).expand(B,-1))
    vis = bridge(patches) if not drop_vis else torch.zeros(B,NV,D)
    emb = torch.cat([tok[:, :splice], vis, tok[:, splice:]], 1)
    att = torch.ones(B, emb.size(1), dtype=torch.long)
    pos = torch.cat([torch.arange(splice), torch.arange(NV)+splice,
                     torch.arange(len(ids)-splice)+splice+NV]).unsqueeze(0).expand(B,-1)
    h = model.encoder(inputs_embeds=emb, attention_mask=att, position_ids=pos).last_hidden_state
    return h

class Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(NV); self.proj = nn.Linear(768, D)
    def forward(self, patches):
        return self.proj(self.pool(patches.transpose(1, 2)).transpose(1, 2))
bridge = Bridge()
with torch.no_grad():
    h = run(P, bridge)
print("2) where does the signal survive inside the frozen encoder?")
probe(h[:, splice:splice+NV].reshape(len(X), -1), "visual slots after frozen encoder (mean-pool)")
probe(torch.cat([h[:, m] for m in markers], -1), "MARKER positions after frozen encoder (the head's input)")
with torch.no_grad():
    hd = run(P, bridge, drop_vis=True)
probe(torch.cat([hd[:, m] for m in markers], -1), "marker positions, image DROPPED (control)")
