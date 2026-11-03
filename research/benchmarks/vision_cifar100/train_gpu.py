#!/usr/bin/env python3
"""Run `laya-train` on a GPU, with this machine's environment facts applied.

Three of them are not optional, and each one was found by watching something fail silently or
segfault:

  * `LD_LIBRARY_PATH` must contain torch/lib, or ROCm lazy init dies on `libcaffe2_nvrtc.so`
    (torch's ROCm build dlopens it by bare name).
  * `HSA_OVERRIDE_GFX_VERSION` must be unset: it is right for the two gfx1030 cards and wrong for
    the gfx906 one, which then reports "no kernel image is available".
  * MIOpen's Conv2d segfaults, so the frozen SigLIP tower's patch-embed conv is rewritten as the
    numerically identical unfold+linear (see `tower_patch.py`).

`--device cuda` is appended if the caller did not choose a device: a run that silently lands on CPU
is the exact failure this file exists to prevent.

Usage:  python3 train_gpu.py --data ... --out ... [usual laya-train flags]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import torch  # noqa: E402  (import order matters: torch first, then the shim)

import tower_patch  # noqa: E402


def main(argv):
    if not torch.cuda.is_available():
        print("train_gpu: no CUDA/HIP device visible -- refusing to run on CPU by accident.\n"
              "           bwrap's `--dev /dev` hides /dev/kfd; run this unsandboxed.",
              file=sys.stderr)
        return 3
    print("train_gpu: %d device(s), using %s (%s)"
          % (torch.cuda.device_count(), torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).gcnArchName), flush=True)
    tower_patch.install()

    args = list(argv)
    if "--device" not in args:
        args += ["--device", "cuda"]
    from laya.train_cli import main as train_main
    return train_main(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
