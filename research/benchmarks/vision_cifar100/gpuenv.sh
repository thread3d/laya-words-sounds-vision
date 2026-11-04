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
#
# `PYTHON` names the interpreter that will run the model; `torch/lib` is taken from *that*
# interpreter, so pointing `LD_LIBRARY_PATH` at somebody else's build is impossible by
# construction. Sourcing this with an interpreter whose torch is a different build puts the wrong
# libtorch on the search path and torch then fails to import with a bare `AttributeError: module
# 'torch._C' has no attribute ...`, which reads like a broken install. The repository's `.venv` is
# the CPU interpreter the CI gates use; GPU work needs the HIP one (`PYTHON=python3`).
PYTHON="${PYTHON:-python3}"
export TORCHLIB="$("$PYTHON" -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib"
export LD_LIBRARY_PATH="$TORCHLIB:${LD_LIBRARY_PATH:-}"
unset HSA_OVERRIDE_GFX_VERSION

if ! "$PYTHON" -c 'import torch' >/dev/null 2>&1; then
  echo "gpuenv.sh: $PYTHON cannot import torch with $TORCHLIB on LD_LIBRARY_PATH." >&2
  echo "gpuenv.sh: interpreter and torch build do not match; GPU runs need PYTHON=<a HIP build>." >&2
fi
