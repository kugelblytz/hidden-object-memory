"""Pre-download all model weights used by run_all.py (no GPU needed; run with CUDA_VISIBLE_DEVICES=).

Usage (from sim/):  CUDA_VISIBLE_DEVICES= nohup python cluster/download_models.py > cluster_models.log 2>&1 &
"""
import time

from huggingface_hub import hf_hub_download, snapshot_download

VLMS = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3-VL-8B-Instruct"]

t0 = time.time()
print("VGGT-1B:", hf_hub_download("facebook/VGGT-1B", "model.pt"), flush=True)

from perception.detector import Detector  # downloads YOLOE weights + text encoder (CPU)

Detector(device="cpu")
print("YOLOE ready", flush=True)

for model_id in VLMS:
    path = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"])
    print(f"{model_id}: {path} ({time.time() - t0:.0f}s)", flush=True)
print("== done", flush=True)
