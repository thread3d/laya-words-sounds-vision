"""Work around a broken MIOpen on this machine: run the vision tower's patch-embed conv as GEMM.

`torch.nn.Conv2d` segfaults on **every** card here (2x gfx1030, 1x gfx906), including a trivial
3x3 conv, while `F.unfold`, matmul, LayerNorm, GELU and softmax are all healthy. MIOpen gets as far
as `miopenConvolutionForwardGetWorkspaceSize` and dumps core, no kernel database is shipped
(`share/miopen/db/*.kdb` is absent), and no MIOpen knob changes the outcome. That is an environment
defect, not a Laya one -- but SigLIP2's patch embedding is `Conv2d(3, 1152, kernel=16, stride=16)`,
so every conv-based tower dies with SIGSEGV and no vision work is possible on these GPUs.

A convolution whose kernel equals its stride has no receptive-field overlap: it is exactly a
per-patch linear map. So the conv can be evaluated as `F.unfold` -> matmul with its own reshaped
weight. Checked on CPU against the real conv (where it works), same weights:

    shapes (2, 16, 14, 14) == (2, 16, 14, 14);  max |diff| = 1.13e-06   (fp32 rounding)

The module is kept -- only `forward` is rebound -- because `transformers` reads
`self.patch_embedding.weight.dtype` in its own forward, and a stand-in object would break that.

Harness-only on purpose: this is an environment workaround, not evidence that the package needs a
conv-free path. Landing it in `laya` should be a deliberate decision made on a machine where
MIOpen works.

    import tower_patch; tower_patch.install()
"""
import types

import torch
import torch.nn.functional as F


def _padding_int(padding, kernel):
    """`Conv2d.padding` as (ph, pw) ints. HuggingFace builds SigLIP with padding="valid"."""
    if isinstance(padding, str):
        if padding == "valid":
            return 0, 0
        if padding == "same":
            return kernel[0] // 2, kernel[1] // 2
        raise ValueError("unknown conv padding %r" % (padding,))
    if isinstance(padding, tuple):
        return padding
    return padding, padding


def _unfold_forward(self, pixel_values):
    """`Conv2d` as im2col + matmul. Kernel must equal stride (checked at install time)."""
    weight = self.weight                                  # [C_out, C_in, kh, kw]
    kh, kw = weight.shape[-2], weight.shape[-1]
    sh, sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
    ph, pw = _padding_int(self.padding, (kh, kw))
    cols = F.unfold(pixel_values, kernel_size=(kh, kw), stride=(sh, sw), padding=(ph, pw))
    cols = cols.transpose(1, 2)                           # [B, n, C_in*kh*kw]
    out = cols @ weight.reshape(weight.shape[0], -1).t()   # [B, n, C_out]
    if self.bias is not None:
        out = out + self.bias
    out = out.transpose(1, 2)                             # [B, C_out, n]
    h = (pixel_values.shape[-2] + 2 * ph - kh) // sh + 1
    w = (pixel_values.shape[-1] + 2 * pw - kw) // sw + 1
    return out.reshape(out.shape[0], weight.shape[0], h, w)


def patch_tower(tower):
    """Rebind `forward` on every stride==kernel Conv2d in `tower`; returns how many changed."""
    replaced = 0
    for module in tower.modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        if tuple(module.kernel_size) != tuple(module.stride):
            continue
        module.forward = types.MethodType(_unfold_forward, module)
        replaced += 1
    return replaced


def install():
    """Wrap `laya.vision.load_tower` so every tower it builds avoids MIOpen convolutions."""
    import laya.vision as V

    if getattr(V.load_tower, "_conv_free", False):
        return 0
    original = V.load_tower

    def load_tower(*args, **kwargs):
        tower = original(*args, **kwargs)
        replaced = patch_tower(tower)
        print("tower_patch: rerouted %d patch-embed conv(s) through im2col+matmul" % replaced,
              flush=True)
        return tower

    load_tower._conv_free = True
    V.load_tower = load_tower
    return 1
