#!/bin/bash
# Create the Python environment for run_all.py on the cluster (no GPU is used by this script).
# Usage (from sim/):  nohup bash cluster/setup_env.sh > cluster_setup.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."   # sim/

echo "== checks $(date)"
df -h "$HOME" | tail -1
PY=""
for cand in python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PY=$cand; break; fi
done
echo "python: $PY ($($PY --version 2>&1))"
$PY -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10"'

echo "== venv"
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "== torch (CUDA wheels)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "== dependencies"
pip install -r cluster/requirements-cluster.txt
# ultralytics pulls opencv-python, which needs GUI libraries; use the headless build instead.
pip uninstall -y opencv-python || true
pip install opencv-python-headless
pip install --no-deps git+https://github.com/facebookresearch/vggt

echo "== import check (no GPU allocation)"
python - <<'EOF'
import torch, torchvision, transformers, ultralytics, cv2, scipy, bitsandbytes, vggt
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
print("transformers", transformers.__version__, "ultralytics", ultralytics.__version__, "cv2", cv2.__version__)
assert hasattr(cv2, "SIFT_create")
from perception.pipeline import process_trial  # noqa: F401
from memory import memory2d, memory_vggt       # noqa: F401
from baselines.vlm import HFBackend            # noqa: F401
import run_all                                 # noqa: F401
print("project imports ok")
EOF
echo "== done $(date)"
