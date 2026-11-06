#!/bin/bash
# Environment for any GPU command in this directory. Source it, do not execute it:
#
#   source research/benchmarks/audio_esc50/gpuenv.sh
#   python3 research/benchmarks/audio_esc50/gpu_patch.py
#
# This machine's GPUs are AMD (3x gfx1030, 34 GB each) and are reached with the ROCm torch build in
# the miniconda interpreter -- the repository's `.venv` is a CPU build for the CI gates. The three
# facts below were each found by watching something fail in its own way (see `gpuenv.sh` next door
# in `vision_cifar100/`, and `gpu_patch.py` here):
#
#   * torch's ROCm build dlopens `libcaffe2_nvrtc.so` by bare name, so its own `lib` must be on the
#     search path or lazy device init dies although the file is right there;
#   * `HSA_OVERRIDE_GFX_VERSION` is set in the ambient environment and is wrong for these cards;
#   * MIOpen **segfaults on every convolution** here (`conv1d` exits 139), which is why
#     `gpu_patch.py` routes the audio tower's conv front-end through unfold + matmul.
#
# `PYTHON` names the interpreter that will run the model, and `torch/lib` is taken from *that*
# interpreter. Sourcing this with a CPU build puts the wrong libtorch on the path and torch then
# fails to import with a bare `AttributeError: module 'torch._C' has no attribute ...`.
PYTHON="${PYTHON:-$HOME/miniconda3/bin/python3}"
export TORCHLIB="$("$PYTHON" -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib"
export LD_LIBRARY_PATH="$TORCHLIB:${LD_LIBRARY_PATH:-}"
unset HSA_OVERRIDE_GFX_VERSION
# The GPU interpreter is a separate environment; give it the benchmark data and a writable cache.
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
export AUDIO_BENCH_DATA="${AUDIO_BENCH_DATA:-$PWD/.audio-bench}"

if ! "$PYTHON" -c 'import torch' >/dev/null 2>&1; then
  echo "gpuenv.sh: $PYTHON cannot import torch with $TORCHLIB on LD_LIBRARY_PATH." >&2
  echo "gpuenv.sh: GPU runs need PYTHON=<a ROCm/HIP build>; the repository .venv is CPU-only." >&2
fi
