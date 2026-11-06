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
    later `Conv1d(512, 512, ...)` layers, so a groups-1-only workaround fails on the model it was
    written for. A grouped conv is `g` independent convs, which `bmm` expresses by folding the group
    axis into the batch axis -- `torch.matmul` will not do it, because a 4-D `[g, B, L, F]` against a
    3-D `[g, F, C]` broadcast-aligns `g` with `B` and fails.

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
    cols = cols.view(batch, out_len, groups, in_channels // groups * kernel)
    cols = cols.permute(2, 0, 1, 3).reshape(groups * batch, out_len, -1)
    weight = self.weight.view(groups, out_channels // groups, -1).permute(0, 2, 1)
    weight = weight.unsqueeze(1).expand(groups, batch, weight.shape[-2], weight.shape[-1])
    out = torch.bmm(cols, weight.reshape(groups * batch, weight.shape[-2], weight.shape[-1]))
    out = out.view(groups, batch, out_len, out_channels // groups)
    out = out.permute(1, 2, 0, 3).reshape(batch, out_len, out_channels)
    if self.bias is not None:
        out = out + self.bias
    return out.transpose(1, 2).contiguous()


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
