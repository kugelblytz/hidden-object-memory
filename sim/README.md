# Hidden-object simulation (ManiSkill3)

Simulation and data recording for the research plan *Actor-Agnostic Persistent Object
Memory for Hidden-Object Localization*. One trial is two task episodes with an
unobserved change in between:

| Stream | Frames | Content |
|---|---|---|
| `episode1/` | 26 | target ball visible; an invisible external actor lifts the target cover, carries it over the ball and lowers it (main camera) |
| `sweep_before/` | 10 | short arc (±10° azimuth, ±3° elevation) around the main camera pose |
| *(change)* | – | not recorded: `fixed`, `camera` (azimuth change 10–60°), or `cover` (one cover moved 5–15 cm, no crossing) |
| `sweep_after/` | 10 | arc around the main camera pose of episode 2 |
| `query.png` | 1 | final image of episode 2; the ball is fully hidden |

Design factors: appearance (`identical` = three covers of one colour, `distinct` = three
colours; same mug mesh) × motion (`fixed`, `camera`, `cover`), across three scenes
(room textures + distractor objects). The target cover, layout, colours and motion
size are sampled from the trial seed.

## Output layout

```
trial_XXXX/
  inputs/      RGB only: what the methods may see
  gt/          meta.json (layout, target, change, segmentation ids, validity checks)
               frames.npz (depth in mm, segmentation, intrinsics K, OpenCV extrinsics per frame)
  preview.jpg  contact sheet of input frames for quick inspection
```

`gt/` is for scoring and diagnostics only and must never be passed to a method.
A trial is valid if the ball is visible at the start, hidden at query time and all
three covers are visible in the query image; invalid trials are resampled.

## Setup (Windows, tested on a Quadro T1000 4 GB)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe --no-deps mikasa-robo-suite==1.0.0
.venv\Scripts\python.exe -m mani_skill.utils.download_asset ycb -y
```

`mikasa-robo-suite` pins contradictory dependencies, so it is installed with `--no-deps`;
it is only needed for `smoke_test_shellgame.py`.

Always create environments with `render_backend="cpu"`: with CPU-only PyTorch the
default GPU render backend crashes (rendering still runs on the GPU via Vulkan).

## Run

```powershell
.venv\Scripts\python.exe run_pilot.py --split pilot --per-cell 5 --gif
.venv\Scripts\python.exe run_pilot.py --split test --per-cell 30 --protocol-frozen
```

Pilot seeds start at 10,000 and test seeds at 1,000,000, so pilot recordings can
never end up in the test set; test recordings require `--protocol-frozen`. `--gif`
writes a labelled `recording.gif` per trial for visual inspection, and a per-cell
summary (resampled/invalid trials, smallest visible cover, motion range) is printed
and saved to `summary.json`. About 4–6 s and 21 MB per trial on the laptop above.

## Files

- `hidden_object/env.py`: ManiSkill3 scene `HiddenObjectShell-v0` (table, textured room, parked Panda, kinematic covers, ball, distractors) and recording helpers
- `hidden_object/trial.py`: trial sampling, phase-by-phase recording, ground-truth export
- `hidden_object/textures.py`: seeded, non-repetitive wall and floor textures
- `run_pilot.py`: records one trial per design cell
- `smoke_test_shellgame.py`: renders the original MIKASA-Robo ShellGame for comparison
