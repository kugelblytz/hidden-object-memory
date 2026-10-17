# Running the evaluation on the cluster

Rendering happens on the laptop (MIG GPU slices do not support Vulkan). Everything after the
recording (perception, 2D memory, VGGT, VLM baselines, report) runs in one job with `run_all.py`.

## 1. Copy to the cluster

- the `sim/` code (git clone or copy; without `.venv/` and `out/`);
- the recorded trials, e.g. `sim/out/test/` (about 4 GB for 180 trials; `inputs/` and `gt/` are needed).

## 2. Environment (once)

```bash
cd sim
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA build
pip install -r requirements.txt
pip install --no-deps git+https://github.com/facebookresearch/vggt
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/VGGT-1B','model.pt')"
```

YOLOE weights and the VLMs download automatically on first use. With little home quota, set
`HF_HOME` to a scratch directory first.

## 3. Smoke test (a few minutes)

```bash
python run_all.py --data out/test --limit 2 --vlm Qwen/Qwen3.5-9B --load-in-8bit
```

Check that every stage prints results and that `out/test/results/report.html` exists. Then delete
`out/test/results/` so that the smoke-test answers do not mix with the full run.

## 4. Full run

```bash
sbatch cluster/run_all.sbatch      # fill in partition/account/GPU slice first
```

The job is resumable. If it stops, submit it again, and finished trials and stages are skipped.
Results: `out/test/results/report.html`, `out/test/results/summary.json`,
and per trial `out/test/results/trials/*.json`.

## Memory budget (20 GB H100 slice)

- VGGT-1B in bf16: about 3 GB of weights plus 4–8 GB of activations (11 frames per pass, 22 for the joint pass).
- VGGT is released before the VLM is loaded.
- Qwen3.5-9B and Qwen3-VL-8B run in 8-bit (`--load-in-8bit`): about 11 GB each, one after the other.
