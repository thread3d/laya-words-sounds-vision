"""A/B: does a cross-attention readout learn where concat-through-frozen-encoder stalls?

Variant A: visual tokens in the sequence, markers read via the existing head+scorer (concat fusion).
Variant B: same sequence, PLUS a cross-attention block in the head where each option's marker
           hidden state attends directly over the visual tokens before scoring.
Same data, same step budget, same frozen towers. Only the readout differs.
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

torch.set_num_threads(8); torch.manual_seed(0)
import laya
from laya.common import build_head, QTYPES
from transformers import SiglipVisionModel

COLOURS=[(220,40,40),(40,60,220),(40,180,60),(225,200,50)]
CLASSES=["red","blue","green","yellow"]
def make_images(n_per=150,res=224,size=32):
    imgs,labels=[],[]
    for ci,col in enumerate(COLOURS):
        for _ in range(n_per):
            b=torch.tensor(col).view(3,1,1).float().div(255).expand(3,size,size).clone()
            b+=0.06*torch.randn(3,size,size)
            oth=torch.tensor(COLOURS[(ci+1)%4]).view(3,1,1).float().div(255)
            y=torch.randint(0,size-6,(1,)).item(); b[:,y:y+3,:]=0.8*oth+0.2*b[:,y:y+3,:]
            imgs.append(F.interpolate(b.unsqueeze(0),size=(res,res),mode="bilinear").squeeze(0))
            labels.append(ci)
    return torch.stack(imgs),torch.tensor(labels)

tower=SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224").eval()
for p in tower.parameters(): p.requires_grad_(False)
X,Y=make_images()
with torch.no_grad(): P=tower(pixel_values=X).last_hidden_state
print("patches",tuple(P.shape))

agent=laya.load(CKPT)
model=agent.model; model.encoder.eval()
for p in model.encoder.parameters(): p.requires_grad_(False)
D=model.encoder.config.hidden_size
q={"k":{"type":"choice","instructions":"What is the background colour?","criteria":{c:None for c in CLASSES}}}
internal=[agent._to_internal(v) for v in q.values()]
head_ids,markers,_=build_head(agent.tok,internal[0],192)
splice=len(head_ids); NV=8
ids=head_ids+[agent.tok.sep_token_id]
qtype=torch.tensor([QTYPES["choice"]])
mpos=torch.tensor([[m if m<splice else m+NV for m in markers]])
POS=torch.cat([torch.arange(splice),torch.arange(NV)+splice,
               torch.arange(len(ids)-splice)+splice+NV]).unsqueeze(0)

class Bridge(nn.Module):
    def __init__(s):
        super().__init__(); s.pool=nn.AdaptiveAvgPool1d(NV); s.proj=nn.Linear(768,D)
    def forward(s,p): return s.proj(s.pool(p.transpose(1,2)).transpose(1,2))

class XAttn(nn.Module):
    """Each marker hidden state (query) attends over the visual tokens (keys/values)."""
    def __init__(s):
        super().__init__(); s.q=nn.Linear(D,D); s.k=nn.Linear(D,D); s.v=nn.Linear(D,D)
        s.o=nn.Linear(D,D); s.norm=nn.LayerNorm(D); s.h=8
    def forward(s,m,vis):
        B,K,_=m.shape; V=vis.size(1); hd=D//s.h
        q=s.q(m).view(B,K,s.h,hd).transpose(1,2)
        k=s.k(vis).view(B,V,s.h,hd).transpose(1,2)
        v=s.v(vis).view(B,V,s.h,hd).transpose(1,2)
        a=(q@k.transpose(-1,-2))/hd**0.5
        out=(a.softmax(-1)@v).transpose(1,2).reshape(B,K,D)
        return s.norm(m+s.o(out))

def build(variant):
    bridge=Bridge(); xa=XAttn() if variant=="xattn" else None
    head=list(model.head.parameters())+list(model.scorer.parameters())+list(model.type_emb.parameters())
    params=[{"params":bridge.parameters(),"lr":3e-3}]
    if xa is not None: params.append({"params":xa.parameters(),"lr":3e-3})
    params.append({"params":head,"lr":3e-3})
    return bridge,xa,torch.optim.AdamW(params,weight_decay=0.01)

def forward(Pb,bridge,xa,drop_vis=False):
    B=Pb.size(0)
    with torch.no_grad():
        tok=model.encoder.embeddings.tok_embeddings(torch.tensor(ids).unsqueeze(0).expand(B,-1))
    vis=bridge(Pb) if not drop_vis else torch.zeros(B,NV,D)
    emb=torch.cat([tok[:,:splice],vis,tok[:,splice:]],1)
    att=torch.ones(B,emb.size(1),dtype=torch.long)
    h=model.encoder(inputs_embeds=emb,attention_mask=att,
                    position_ids=POS.expand(B,-1)).last_hidden_state
    h=h+model.type_emb(qtype)[:,None,:]
    for layer in model.head.layers: h=layer(h,src_key_padding_mask=~att.bool())
    m=torch.gather(h,1,mpos[:,:,None].expand(B,-1,h.size(-1)))
    if xa is not None: m=xa(m,vis)                      # <-- the only difference
    return model.scorer(m).squeeze(-1)

perm=torch.randperm(len(X)); n_val=120
val,tr=perm[:n_val],perm[n_val:]
def acc(bridge,xa,drop=False):
    c=0
    with torch.no_grad():
        for i in range(0,n_val,40):
            b=val[i:i+40]
            c+=(forward(P[b],bridge,xa,drop).argmax(-1)==Y[b]).sum().item()
    return c/n_val

STEPS=300
for variant in ("concat","xattn"):
    torch.manual_seed(1)
    bridge,xa,opt=build(variant)
    t0=time.time(); curve=[]
    for step in range(1,STEPS+1):
        sel=tr[torch.randint(0,len(tr),(32,))]
        loss=F.cross_entropy(forward(P[sel],bridge,xa),Y[sel])
        opt.zero_grad(); loss.backward(); opt.step()
        if step%100==0: curve.append((step,round(loss.item(),3),round(acc(bridge,xa),3)))
    a_img, a_no = acc(bridge,xa), acc(bridge,xa,drop=True)
    print(f"{variant:7s} {time.time()-t0:5.0f}s  " + "  ".join(f"@{s}:loss={l},val={a}" for s,l,a in curve)
          + f"   ||  WITH image {a_img:.3f}   WITHOUT {a_no:.3f}   gain {a_img-a_no:+.3f}")
