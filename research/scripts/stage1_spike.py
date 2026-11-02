"""Stage-1 spike: does early fusion actually LEARN a vision-dependent decision?

Freeze ModernBERT (the Laya checkpoint encoder) and SigLIP2; train only a projector and the
decision head on a 4-way colour decision. Then ablate the visual block: if accuracy collapses,
the decision is genuinely grounded in the pixels rather than in the question text.

This is the smallest possible rehearsal of the M1 gate in VISION_SPEC.md §9.1.
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
import time, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
torch.manual_seed(0)

import laya
from laya.common import build_head, QTYPES
from transformers import SiglipVisionModel

CLASSES = ["red", "blue", "green", "yellow"]
COLOURS = [(220, 40, 40), (40, 60, 220), (40, 180, 60), (225, 200, 50)]
N_PER_CLASS, NV = 150, 8

def make_images(n_per=N_PER_CLASS, res=224, size=32):
    imgs, labels = [], []
    for ci, col in enumerate(COLOURS):
        for _ in range(n_per):
            base = torch.tensor(col).view(3, 1, 1).float() / 255.0
            base = base.expand(3, size, size).clone()
            base += 0.06 * torch.randn(3, size, size)                 # sensor noise
            # a distractor stripe whose colour is NOT the class colour
            other = torch.tensor(COLOURS[(ci + 1) % 4]).view(3, 1, 1).float() / 255.0
            y = torch.randint(0, size - 6, (1,)).item()
            base[:, y:y + 3, :] = 0.8 * other + 0.2 * base[:, y:y + 3, :]
            img = F.interpolate(base.unsqueeze(0), size=(res, res), mode="bilinear")
            imgs.append(img.squeeze(0)); labels.append(ci)
    return torch.stack(imgs), torch.tensor(labels)

tower = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224").eval()
for p in tower.parameters():
    p.requires_grad_(False)

print("building toy images + caching tower output ...")
X, Y = make_images()
with torch.no_grad():
    PATCH = tower(pixel_values=X).last_hidden_state          # [N, 196, 768]
print("cached patches:", tuple(PATCH.shape))

agent = laya.load(CKPT)
model = agent.model
model.encoder.eval()
for p in model.encoder.parameters():
    p.requires_grad_(False)

q = {"k": {"type": "choice", "instructions": "What is the background colour?",
           "criteria": {c: None for c in CLASSES}}}
internal = [agent._to_internal(v) for v in q.values()]
head_ids, markers, _ = build_head(agent.tok, internal[0], 192)
# no state text at all: the image is the state
ids = head_ids + [agent.tok.sep_token_id]                 # no state text: the image is the state
splice = len(head_ids)
qtype = torch.tensor([QTYPES["choice"]])

proj = nn.Sequential(nn.AdaptiveAvgPool1d(NV), nn.Flatten(2), nn.Linear(768, 1024))
# AdaptiveAvgPool1d on [N,196,768] needs [N,768,196]
class Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(NV)
        self.proj = nn.Linear(768, model.encoder.config.hidden_size)
    def forward(self, patches):                          # [N,196,768] -> [N,NV,1024]
        return self.proj(self.pool(patches.transpose(1, 2)).transpose(1, 2))

bridge = Bridge()
head_params = list(model.head.parameters()) + list(model.scorer.parameters()) + list(model.type_emb.parameters())
for p in head_params:
    p.requires_grad_(True)
opt = torch.optim.AdamW([{"params": bridge.parameters(), "lr": 3e-3},
                         {"params": head_params, "lr": 1e-3}], weight_decay=0.01)

def forward(batch_patches, ids=ids, drop_vis=False):
    B = batch_patches.size(0)
    with torch.no_grad():
        tok_emb = model.encoder.embeddings.tok_embeddings(torch.tensor(ids).unsqueeze(0).expand(B, -1))
    vis = bridge(batch_patches) if not drop_vis else torch.zeros(B, NV, tok_emb.size(-1))
    emb = torch.cat([tok_emb[:, :splice], vis, tok_emb[:, splice:]], dim=1)
    L = emb.size(1)
    att = torch.ones(B, L, dtype=torch.long)
    pos = torch.cat([torch.arange(splice), torch.arange(NV) + splice,
                     torch.arange(len(ids) - splice) + splice + NV]).unsqueeze(0).expand(B, -1)
    h = model.encoder(inputs_embeds=emb, attention_mask=att, position_ids=pos).last_hidden_state
    h = h + model.type_emb(qtype)[:, None, :]
    for layer in model.head.layers:
        h = layer(h, src_key_padding_mask=~att.bool())
    mpos = torch.tensor([[m if m < splice else m + NV for m in markers]])
    idx = mpos[:, :, None].expand(B, -1, h.size(-1))
    m = torch.gather(h, 1, idx)
    return model.scorer(m).squeeze(-1)

def accuracy(model_eval_batches, drop_vis=False):
    correct = 0
    with torch.no_grad():
        for pb, yb in model_eval_batches:
            pred = forward(pb, drop_vis=drop_vis).argmax(-1)
            correct += (pred == yb).sum().item()
    return correct / sum(len(y) for _, y in model_eval_batches)

perm = torch.randperm(len(X))
n_val = 90
val = perm[:n_val]; tr = perm[n_val:]
val_b = [(PATCH[val[i:i + 30]], Y[val[i:i + 30]]) for i in range(0, n_val, 30)]

t0 = time.time()
BS, STEPS = 16, 260
print(f"\n{'step':>5} {'loss':>7} {'train_acc':>10} {'val_acc':>8}")
for step in range(1, STEPS + 1):
    sel = tr[torch.randint(0, len(tr), (BS,))]
    logits = forward(PATCH[sel])
    loss = F.cross_entropy(logits, Y[sel])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0 or step == 1:
        with torch.no_grad():
            tr_acc = (forward(PATCH[tr[:240]]).argmax(-1) == Y[tr[:240]]).float().mean().item()
        print(f"{step:5d} {loss.item():7.3f} {tr_acc:10.3f} {accuracy(val_b):8.3f}")

print(f"\ntrained in {time.time()-t0:.1f}s on CPU")
a_img = accuracy(val_b, drop_vis=False)
a_noimg = accuracy(val_b, drop_vis=True)
print(f"  accuracy WITH image    : {a_img:.3f}")
print(f"  accuracy WITHOUT image : {a_noimg:.3f}   (chance = 0.250)")
print(f"  image-ablation gain    : {a_img - a_noimg:+.3f}")
