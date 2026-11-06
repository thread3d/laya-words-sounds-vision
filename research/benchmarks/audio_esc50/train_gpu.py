"""`laya-train` with this machine's GPU facts applied: the MIOpen conv workaround installed.

`laya-train` builds the audio tower through `laya.audio.load_tower`, which on this ROCm build dies
with SIGSEGV the moment the tower's `Conv1d` front-end reaches MIOpen. `gpu_patch.install()` wraps
that loader to run the convolutions as unfold + matmul (verified against `conv1d` on the real
weights), and this script then hands over to the ordinary CLI -- same arguments, same behaviour,
plus the workaround:

    source research/benchmarks/audio_esc50/gpuenv.sh
    python3 research/benchmarks/audio_esc50/train_gpu.py --data ... --device cuda ...

Harness-only, exactly like the vision harness's `train_gpu.py`: the workaround belongs to this
machine's broken MIOpen, not to the package.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import gpu_patch  # noqa: E402

gpu_patch.install()

from laya.train_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
