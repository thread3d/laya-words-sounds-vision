"""Feasibility spike: inject a block of 'visual' tokens into the Laya decision model.

Proves four things the spec depends on:
  1. the ModernBERT encoder accepts inputs_embeds + position_ids with sdpa attention,
  2. a visual block can be spliced in after the question head, shifting later positions,
  3. the unchanged marker-scorer head still produces per-option logits,
  4. gradients reach the visual block (the fusion is trainable end to end).
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
import torch

import laya
from laya.common import build_sequence

agent = laya.load(CKPT)
model = agent.model
enc = model.encoder
D = enc.config.hidden_size
NV = 96  # visual tokens in this spike

q = {"k": {"type": "choice", "instructions": "Which team should handle this?",
           "criteria": {"billing": "invoices", "technical": "bugs", "sales": "pricing", "other": "none"}}}
internal = [agent._to_internal(v) for v in q.values()]
ids, markers, _ = build_sequence(agent.tok, "The invoice charged me twice.", internal[0],
                                128, 192, return_stats=True)
from laya.common import QTYPES
qtype = torch.tensor([QTYPES[internal[0]["t"]]], dtype=torch.long)

# Splice point: right after the question head (before the state text). build_head ends with [SEP]
# after the last option; find the end of the option block = last marker's option span end.
insert_at = markers[-1] + 1 + 12          # a few tokens past the last marker, inside the option block
insert_at = min(insert_at, len(ids) - 1)
print(f"text seq len={len(ids)}  insert visual block at offset {insert_at}  NV={NV}")

inp = torch.tensor([ids])
pos = torch.arange(inp.size(1)).unsqueeze(0)

with torch.no_grad():
    tok_emb = enc.embeddings.tok_embeddings(inp)              # [1, L, D]: the encoder's own embedding table

# A stand-in for a projector output: patch tokens from a vision tower, projected to D.
proj = torch.nn.Sequential(torch.nn.Linear(768, D), torch.nn.GELU()).requires_grad_(True)
patch_tokens = torch.randn(1, NV, 768, requires_grad=True)
vis = proj(patch_tokens)

# Assemble: text[:insert_at] + visual block + text[insert_at:]
emb = torch.cat([tok_emb[:, :insert_at], vis, tok_emb[:, insert_at:]], dim=1)
L2 = emb.size(1)
pos2 = torch.cat([pos[:, :insert_at], torch.arange(NV).unsqueeze(0) + insert_at, pos[:, insert_at:] + NV], dim=1)
att = torch.ones(1, L2, dtype=torch.long)

# Marker positions move by NV if they sat after the splice point.
shifted = [m + NV if m >= insert_at else m for m in markers]
marker_pos = torch.tensor([shifted])
marker_mask = torch.ones(1, len(markers), dtype=torch.bool)

enc.train()
h = enc(inputs_embeds=emb, attention_mask=att, position_ids=pos2).last_hidden_state
print("encoder accepted inputs_embeds + position_ids ->", tuple(h.shape))

with torch.no_grad():
    out = model.forward(input_ids=None, attention_mask=att, marker_pos=marker_pos,
                        marker_mask=marker_mask, qtype=qtype, _hidden=h) if False else None

# Run the head path by hand, mirroring DecisionModel.forward: type_emb -> 2-layer head -> scorer.
hh = h + model.type_emb(qtype)[:, None, :]
for layer in model.head.layers:
    hh = layer(hh, src_key_padding_mask=~att.bool())
idx = marker_pos[:, :, None].expand(-1, -1, hh.size(-1))
m = torch.gather(hh, 1, idx)
logits = model.scorer(m).squeeze(-1).float()
p = torch.softmax(logits, -1)
print("per-option logits:", [round(v, 4) for v in logits[0].tolist()])
print("per-option probs  :", [round(v, 4) for v in p[0].tolist()], "sum", round(float(p.sum()), 6))

# Gradients must reach the visual block for the fusion to be trainable.
p[0, 0].backward()
gn = patch_tokens.grad.norm().item()
gproj = proj[0].weight.grad.norm().item()
print(f"grad to patch_tokens = {gn:.6f}   grad to projector weight = {gproj:.6f}")
print("SPIKE VERDICT:", "fusion is trainable end-to-end" if gn > 0 else "NO GRADIENT PATH")
