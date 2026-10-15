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

## Perception pipeline

The shared RGB perception used by all conditions reads only `inputs/` and writes
`trial_XXXX/perception/perception.json` and `query_marked.png`:

1. **Detection:** YOLOE-11L (Ultralytics, open vocabulary), text prompts mapped to roles
   (`cover`, `target`, `distractor`); global threshold 0.05, target threshold 0.3;
   class-agnostic NMS.
2. **Tracking** within the two continuously observed segments (A = episode 1 + sweep before,
   B = sweep after + query), by Hungarian matching on overlap and centre distance. Each
   track's role is the score-weighted majority over its frames; tracks seen in fewer than
   50 % of a segment's frames are dropped. Linking tracks across the unobserved change is
   left to the memory systems.
3. **Events:** target first/last seen, the carried cover's motion start and settling, and the
   containment relation (target → parent cover track) from where the target disappeared.
4. **Keyframes** (≤5) for the keyframe baseline: the four events plus the first frame after the change.
5. **Query candidates:** cover tracks of segment B numbered left to right in the query image.

```powershell
.venv\Scripts\python.exe run_perception.py --data out/pilot
.venv\Scripts\python.exe select_detector.py --data out/pilot   # compare weights/prompts/thresholds
```

`run_perception.py` also scores the pipeline against ground truth (`perception/gt.py`,
evaluation only) and writes the number of the correct candidate to `gt/candidates.json`
for later scoring. About 35–45 s per trial on CPU.

## Files

- `hidden_object/env.py`: ManiSkill3 scene `HiddenObjectShell-v0` (table, textured room, parked Panda, kinematic covers, ball, distractors) and recording helpers
- `hidden_object/trial.py`: trial sampling, phase-by-phase recording, ground-truth export
- `hidden_object/textures.py`: seeded, non-repetitive wall and floor textures
- `run_pilot.py`: records trials per design cell (pilot/test seed ranges)
- `perception/`: detector, tracker, pipeline (events, keyframes, candidates), GT boxes and metrics (evaluation only)
- `run_perception.py`, `select_detector.py`: run and evaluate perception, compare detector settings
- `smoke_test_shellgame.py`: renders the original MIKASA-Robo ShellGame for comparison
