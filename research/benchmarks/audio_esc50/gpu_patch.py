"""Work around a broken MIOpen on this machine: run the audio tower's Conv1d front-end as GEMM.

`torch.nn.Conv1d` segfaults on **every** card here (3x gfx1030), including a trivial single-layer
conv on a 1-second input: `conv1d(...)` exits 139 before it can raise. `F.pad`, matmul, LayerNorm,
GELU and softmax are all healthy, and the same defect is why the vision tower's `Conv2d` needed
`tower_patch.py` next door — MIOpen gets as far as its workspace query and dumps core, no kernel
database ships with this build, and no MIOpen knob changes the outcome. It is an environment defect,
not a Laya one, but wav2vec2's feature extractor is seven `Conv1d` layers, so no audio work is
possible on these GPUs without routing the convolution around MIOpen.

A 1-D convolution is a strided unfold plus one matmul, which is all `im2col` ever was:

    cols = x.unfold(2, kernel, stride)            # [B, C_in, L_out, K]
    out  = cols.reshape(B, L_out, C_in*K) @ W.t() # [B, L_out, C_out]

`install()` verifies itself against the CPU convolution with the same weights before rebinding
anything, so a wrong reimplementation fails loudly instead of quietly scoring a broken tower.

Harness-only on purpose, like the vision patch: this is an environment workaround, not evidence that
the package needs a conv-free path. Landing it in `laya` should be a deliberate decision made on a
machine where MIOpen works.

    source gpuenv.sh                      # LD_LIBRARY_PATH, HSA override, PYTHON
    python3 -c "import gpu_patch; print(gpu_patch.install())"
"""
import types

import torch
import torch.nn.functional as F


def _ints(value, width):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,) * width


def _conv1d_gemm(self, x):
    """`Conv1d` as unfold + one batched matmul, groups included.

    wav2vec2's front-end mixes both cases: `Conv1d(1, 512, kernel=10, stride=5)` with `groups=1` and
    later depthwise layers, so a groups-1-only workaround fails on the model it was written for.
    The grouped matmul lives in :func:`_grouped_matmul`, which both conv paths share.

    Verified against `F.conv1d` on every layer of the real tower by :func:`patch_tower`.
    """
    if _ints(self.dilation, 1)[0] != 1:
        raise NotImplementedError("gpu_patch: dilated Conv1d is not supported")
    kernel, = _ints(self.kernel_size, 1)
    stride, = _ints(self.stride, 1)
    padding, = _ints(self.padding, 1)
    if padding:
        x = F.pad(x, (padding, padding))
    batch, in_channels, length = x.shape
    groups = self.groups
    out_channels = self.weight.shape[0]
    out_len = (length - kernel) // stride + 1
    cols = x.unfold(2, kernel, stride)                        # [B, C_in, L_out, K]
    cols = cols.permute(0, 2, 1, 3).contiguous()
    cols = cols.view(batch, out_len, groups, in_channels // groups * kernel).permute(2, 0, 1, 3)
    out = _grouped_matmul(cols, self.weight, groups, batch, out_len)
    if self.bias is not None:
        out = out + self.bias
    return out.transpose(1, 2).contiguous()


def _grouped_matmul(cols, weight, groups, batch, out_len):
    """`[B, L, g, C_in/g*K]`-style columns against a grouped weight, as one `bmm`.

    `torch.matmul` will not do this: a 4-D `[g, B, L, F]` against a 3-D `[g, F, C]` broadcast-aligns
    `g` with `B` and fails, so the group axis is folded into the batch axis by hand.
    """
    out_channels, per_group = weight.shape[0], weight.shape[0] // groups
    cols = cols.reshape(groups * batch, out_len, -1)          # `cols` arrives as [g, B, L, F]
    weight = weight.view(groups, per_group, -1).permute(0, 2, 1)
    weight = weight.unsqueeze(1).expand(groups, batch, weight.shape[-2], weight.shape[-1])
    out = torch.bmm(cols, weight.reshape(groups * batch, weight.shape[-2], weight.shape[-1]))
    return out.view(groups, batch, out_len, per_group).permute(1, 2, 0, 3).reshape(
        batch, out_len, out_channels)


def _conv2d_gemm(self, x):
    """`Conv2d` as `F.unfold` + grouped matmul, for models that are not a wav2vec tower.

    CLAP's HTSAT encoder is the second model in this harness that dies on MIOpen, and its convs are
    2-D -- the same defect the vision harness patches for SigLIP, with a kernel that does not always
    equal its stride, so the general unfold form is used here rather than the patch-embed identity.
    """
    if _ints(self.dilation, 2)[0] != 1 or _ints(self.dilation, 2)[1] != 1:
        raise NotImplementedError("gpu_patch: dilated Conv2d is not supported")
    kh, kw = _ints(self.kernel_size, 2)
    sh, sw = _ints(self.stride, 2)
    ph, pw = _ints(self.padding, 2)
    if ph or pw:
        x = F.pad(x, (pw, pw, ph, ph))
    batch, in_channels, height, width = x.shape
    out_h = (height - kh) // sh + 1
    out_w = (width - kw) // sw + 1
    cols = F.unfold(x, kernel_size=(kh, kw), stride=(sh, sw))      # [B, C_in*kh*kw, L]
    cols = cols.transpose(1, 2).contiguous()                        # [B, L, C_in*kh*kw]
    cols = cols.view(batch, out_h * out_w, self.groups,
                     in_channels // self.groups * kh * kw).permute(2, 0, 1, 3)
    out = _grouped_matmul(cols, self.weight, self.groups, batch, out_h * out_w)
    out = out.view(batch, out_h, out_w, -1).permute(0, 3, 1, 2).contiguous()
    if self.bias is not None:
        out = out + self.bias.view(1, -1, 1, 1)
    return out


def patch_model(model, verify: bool = True):
    """Patch every `Conv1d`/`Conv2d` in any model (a zero-shot comparator, say).

    Returns ``(conv1d, conv2d)`` counts. `verify` runs each conv on CPU with its own weights and
    refuses to install a layer whose GEMM form disagrees.
    """
    counts = [0, 0]
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            forward = _conv1d_gemm
            probe_shape = (2, module.in_channels, 400)
            reference_args = dict(stride=module.stride, padding=module.padding)
            index = 0
        elif isinstance(module, torch.nn.Conv2d):
            forward = _conv2d_gemm
            probe_shape = (2, module.in_channels, 64, 64)
            reference_args = dict(stride=module.stride, padding=module.padding)
            index = 1
        else:
            continue
        if verify:
            probe = torch.randn(*probe_shape, dtype=module.weight.dtype)
            reference = F.conv1d(probe, module.weight, module.bias, groups=module.groups,
                                 dilation=module.dilation, **reference_args) \
                if index == 0 else F.conv2d(probe, module.weight, module.bias,
                                            groups=module.groups, dilation=module.dilation,
                                            **reference_args)
            patched = forward(module, probe)
            if patched.shape != reference.shape:
                raise AssertionError("gpu_patch: shape %s != conv %s"
                                     % (tuple(patched.shape), tuple(reference.shape)))
            if (patched - reference).abs().max().item() > 1e-4:
                raise AssertionError("gpu_patch: GEMM conv differs from conv by more than 1e-4")
        module.forward = types.MethodType(forward, module)
        counts[index] += 1
    return tuple(counts)


def patch_tower(tower, verify: bool = True):
    """Rebind `forward` on every `Conv1d` in `tower`; returns how many changed.

    `verify` runs each layer's own weights through the CPU convolution and the GEMM path on the same
    input and refuses to install it when the two disagree: a workaround that changes the model is
    worse than the segfault it avoids.
    """
    replaced = 0
    for module in tower.modules():
        if not isinstance(module, torch.nn.Conv1d):
            continue
        if verify:
            probe = torch.randn(2, module.in_channels, 400, dtype=module.weight.dtype)
            reference = F.conv1d(probe, module.weight, module.bias, stride=module.stride,
                                 padding=module.padding, dilation=module.dilation,
                                 groups=module.groups)
            patched = _conv1d_gemm(module, probe)
            if patched.shape != reference.shape:
                raise AssertionError("gpu_patch: shape %s != conv %s"
                                     % (tuple(patched.shape), tuple(reference.shape)))
            delta = (patched - reference).abs().max().item()
            if delta > 1e-4:
                raise AssertionError("gpu_patch: GEMM conv differs from conv1d by %.2e" % delta)
        module.forward = types.MethodType(_conv1d_gemm, module)
        replaced += 1
    return replaced


def install(verify: bool = True):
    """Wrap `laya.audio.load_tower` so every tower it builds avoids MIOpen convolutions."""
    import laya.audio as A

    if getattr(A.load_tower, "_conv_free", False):
        return 0
    original = A.load_tower

    def load_tower(*args, **kwargs):
        tower = original(*args, **kwargs)
        count = patch_tower(tower, verify=verify)
        if count:
            print("gpu_patch: %d Conv1d layer(s) run as GEMM (MIOpen segfaults on this build)"
                  % count, flush=True)
        return tower

    load_tower._conv_free = True
    A.load_tower = load_tower
    return 1


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath("."))
    from laya.audio import build_audio_encoder, load_waveform, parse_audio

    config = parse_audio({"audio": {"n_audio_tokens": 32}})
    # Install before building: `install()` wraps `laya.audio.load_tower`, which is what
    # `build_audio_encoder` calls.
    install()
    encoder = build_audio_encoder(config, hidden_size=1024)
    waveform = load_waveform(
        ".audio-bench/data/ESC-50-master/audio/1-100032-A-0.wav", config.sample_rate)
    encoder.to("cuda")
    import time
    for run in range(3):
        torch.cuda.synchronize()
        started = time.time()
        with torch.no_grad():
            block = encoder.encode(waveform.unsqueeze(0).to("cuda"), summary=True)
        torch.cuda.synchronize()
        print("encode %d: %.3fs  %s" % (run, time.time() - started, tuple(block.shape)), flush=True)
