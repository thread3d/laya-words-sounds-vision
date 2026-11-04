# GPU environment for the CIFAR-100 harness (see VISION_PROGRESS.md, "five obstacles").
#
# Two of this machine's facts are not optional on a ROCm build:
#   * torch's ROCm wheel dlopens `libcaffe2_nvrtc.so` by bare name, so `torch/lib` has to be on
#     the search path or lazy init dies;
#   * `HSA_OVERRIDE_GFX_VERSION` is set in the ambient environment, is right for the gfx1030 cards
#     and wrong for the gfx906 one, whose kernels then fail with "no kernel image is available".
#
# Source it, or let `train.sh` source it for you. `PYTHON` selects the interpreter used to locate
# torch: it must be the one that will run the model.
PYTHON="${PYTHON:-python3}"
TORCHLIB="$("$PYTHON" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="$TORCHLIB:${LD_LIBRARY_PATH:-}"
unset HSA_OVERRIDE_GFX_VERSION
