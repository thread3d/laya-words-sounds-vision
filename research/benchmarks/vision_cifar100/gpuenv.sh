#!/bin/bash
# Environment for any GPU command in this directory.
#
# Three facts about this machine, each found by watching something fail silently or segfault
# (see the repository's vision ledger and `tower_patch.py`):
#   * torch's ROCm build dlopens `libcaffe2_nvrtc.so` by bare name, so `torch/lib` must be on the
#     search path or lazy init dies even though the file is right there;
#   * `HSA_OVERRIDE_GFX_VERSION` is set in the ambient environment and is wrong for one of the
#     cards, which then reports "no kernel image is available";
#   * MIOpen's `Conv2d` segfaults, which `tower_patch.py` routes around.
#
# Source it, do not execute it:  source gpuenv.sh
export TORCHLIB="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib"
export LD_LIBRARY_PATH="$TORCHLIB:${LD_LIBRARY_PATH:-}"
unset HSA_OVERRIDE_GFX_VERSION
