# Code: simulation, perception, memories and evaluation

All code for the study. The pipeline has four parts:

1. **Recording** of trials in ManiSkill3 (laptop).
2. **Shared perception** that only sees RGB images.
3. **Five answering conditions:** two memories, VGGT in two variants, and the VLM baselines.
4. **Evaluation** against simulator ground truth.

Results and their interpretation are in [../RESULTS.md](../RESULTS.md).

## Layout

```
sim/
  run_pilot.py          record pilot or test trials                      (entry point)
  run_all.py            evaluate recorded trials: all stages, resumable  (entry point)
  plot_results.py       figures for RESULTS.md                           (entry point)
  hidden_object/        ManiSkill3 scene HiddenObjectShell-v0, trial sampling and recording, textures
  perception/           detector (YOLOE), tracker, pipeline: events, keyframes, numbered candidates
  memory/               memory2d.py (homography), memory_vggt.py (separate and joint VGGT pass)
  baselines/            vlm.py: current-only and keyframe conditions with a vision-language model
  evaluation/           the ONLY package that reads ground truth: gt.py, metrics.py, scoring.py
  tools/                development and diagnostic tools (run as modules, see below)
  cluster/              setup script, cluster requirements, model download, SLURM template, CLUSTER.md
  requirements.txt      laptop environment (Windows, simulation + perception)
```

**Separation of inputs and ground truth.** Perception, memories and baselines read only
`trial/inputs/` (RGB frames) and the perception output. Only `evaluation/` and the tools that score
results read `trial/gt/`.

## One trial

| Stream | Frames | Content |
|---|---|---|
| `episode1/` | 26 | The ball is visible; an invisible external actor lifts a cup, carries it over the ball and lowers it. |
| `sweep_before/` | 10 | Continuous arc (±10° azimuth, ±3° elevation) starting at the main camera pose. |
| *(change)* | – | Not recorded. One of: `fixed`; `camera` (azimuth change 10–150°, so the left/right order can flip); `cover` (one cup moves 10–35 cm and changes its left/right order, the other two stay). |
| `sweep_after/` | 10 | Arc around the main camera pose of episode 2. |
| `query.png` | 1 | Final image of episode 2; the ball is fully hidden. |

**Design:** appearance (`identical` / `distinct`: same mug mesh, one colour or three colours) ×
motion (`fixed`, `camera`, `cover`), across three scenes with their own room textures and distractor
objects. Every trial is determined by its seed.

**Validity checks**, resampled on failure: the ball is visible at the start and fully hidden at query
time, and all three cups are visible in the query image.

```
trial_XXXX/
  inputs/        RGB frames: the only thing methods may see
  gt/            meta.json, frames.npz (depth, segmentation, camera parameters), candidates.json
  perception/    perception.json, query_marked.png            (written by the perception stage)
  vggt/          cached VGGT reconstructions                   (written by the VGGT stage)
  baselines/     VLM query bundles                             (written by the baseline stage)
  preview.jpg, recording.gif   for visual inspection only
```

## Setup

**Laptop** (Windows, tested on a Quadro T1000 4 GB; used for recording and the light stages):

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m mani_skill.utils.download_asset ycb -y
```

- Create environments with `render_backend="cpu"`. With CPU-only PyTorch the default GPU render
  backend crashes; rendering still runs on the GPU via Vulkan.
- `tools/smoke_test_shellgame.py` additionally needs
  `uv pip install --no-deps mikasa-robo-suite==1.0.0`. The package pins contradictory dependencies,
  hence `--no-deps`.

**GPU cluster** (VGGT and vision-language models; Linux, tested on a 20 GB H100 slice): see
[cluster/CLUSTER.md](cluster/CLUSTER.md).

## Workflow

```powershell
# 1. record (laptop): pilot data for development, test data once the protocol is frozen
.venv\Scripts\python.exe run_pilot.py --split pilot --per-cell 5 --gif
.venv\Scripts\python.exe run_pilot.py --split test --per-cell 30 --protocol-frozen

# 2. evaluate (GPU): perception -> 2D memory -> VGGT -> VLM baselines -> report
python run_all.py --data out/test --vlm Qwen/Qwen3.5-9B Qwen/Qwen3-VL-8B-Instruct --load-in-8bit

#    light stages only (works on the laptop)
.venv\Scripts\python.exe run_all.py --data out/pilot --stages perception memory2d report

# 3. figures
.venv\Scripts\python.exe plot_results.py --summary ../results/test_summary.json --out ../results
```

Seeds:
- Pilot seeds start at 10,000 and test seeds at 1,000,000, so the two sets never overlap.
- Test recordings require `--protocol-frozen`.

`run_all.py`:
- stores per-trial results in `<data>/results/trials/` and skips finished work, so a restarted job
  resumes;
- frees GPU memory between VGGT and the language models;
- writes `results/summary.json` and `results/report.html`, with accuracy and bootstrap CIs per
  condition, cell and motion size, the RQ1/RQ2 contrasts and the H2 interaction.

## Components

**Perception** (`perception/`), shared by all conditions:
1. YOLOE-11L text-prompted detection with roles `cover` / `target` / `distractor`. Prompts were tuned
   on pilot data and then frozen.
2. Tracking within the two observed segments; each track's role is decided by voting over its frames.
3. Detection of the covering event, giving the containment relation "target → cup track".
4. Up to five event-triggered keyframes.
5. Cups in the query image numbered from left to right.

**2D memory** (`memory/memory2d.py`):
1. SIFT features on the masked static background.
2. A RANSAC homography for the best before/after frame pair.
3. Old cup footprints mapped into the new view.
4. Hungarian assignment by position plus colour.

**VGGT memory** (`memory/memory_vggt.py`). `mode="separate"`:
1. One VGGT reconstruction per segment.
2. Background matches lifted to 3D and aligned with a RANSAC similarity transform.
3. Assignment of old to new cups in 3D.

`mode="joint"`: one VGGT pass over the frames before and after the change, which gives a shared 3D
frame without explicit matching.

**Baselines** (`baselines/vlm.py`):
- current-only (numbered query image) and keyframes (≤5 keyframes + query image);
- greedy decoding, thinking off, optional 8-bit;
- free-text replies get one answer-extraction turn.

Memory conditions answer by direct lookup. "Unknown" counts as wrong.

## Tools

Run from `sim/` as modules:

| Command | Purpose |
|---|---|
| `python -m tools.run_perception --data out/pilot` | perception only, scored against ground truth |
| `python -m tools.select_detector --data out/pilot` | compare detector weights, prompts and thresholds (pilot data only) |
| `python -m tools.run_memory --data out/pilot` | 2D memory with its diagnostics |
| `python -m tools.viz_memory --data out/pilot` | visual failure report for the 2D memory (`memory_report.html`) |
| `python -m tools.run_baselines --data out/pilot --backend dry_run` | write VLM query bundles (or run a VLM with `--backend hf`) |
| `python -m tools.smoke_test_shellgame` | render the original MIKASA-Robo ShellGame for comparison |
