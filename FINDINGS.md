# Findings log: hidden-object memory study

Running record of findings, design decisions and open issues from building and piloting the
experiment. All numbers are **pilot data** (30 trials, 5 per cell), so they are indications,
not results. Last updated: 2026-10-16.

## 1. Simulation

- **Simulator: ManiSkill3** (SAPIEN), chosen over robosuite/RoboCasa and Isaac Sim: pip-installable,
  runs on Windows and on A100/H100 clusters, poses can be set directly between episodes, free cameras
  with ground-truth depth/segmentation. Isaac Sim does not run on A100/H100 (RTX GPUs only).
- **MIKASA-Robo's ShellGame fakes the hiding:** the ball is teleported 1000 m up while the cups appear,
  so no covering is ever visible. It is used only as a code template; our scene
  (`HiddenObjectShell-v0`) shows the covering explicitly (an invisible external actor lifts, carries and
  lowers the cup).
- **Windows:** with CPU-only PyTorch the default render backend crashes; `render_backend="cpu"` works
  (still renders on the GPU via Vulkan). The T1000 has no ray-tracing support; per the literature,
  texture and lighting matter more than ray tracing for VGGT and feature matching.
- **Recording:** about 4 s and 21 MB per trial; 180 trials would take about 15 min and 4 GB.
  Every trial is checked automatically (ball visible at the start, fully hidden at query time, all
  cups visible); 30/30 pilot trials were valid without resampling.
- RGB inputs (`inputs/`) and ground truth (`gt/`) are stored separately; pilot and test seeds come
  from disjoint ranges, and test recordings require `--protocol-frozen`.

## 2. The first task version was too easy

With camera changes of 10–60° and small cup moves (5–15 cm, no crossing), a memory that simply
compared **raw pixel positions** (no registration) was correct in **29/30** pilot trials: a camera
orbiting the table keeps the left/right order of the cups. No geometric method could show an
advantage. Changes made:

- camera azimuth changes now span **10–150°** (the order of the cups flips beyond ~90–110°);
- a moved cup travels **10–35 cm** and must **change its left/right order** in the main view
  (the move is unobserved, so its path is irrelevant; the other two cups stay, so trials remain solvable).

After this change, raw positions fail in every trial above ~110° (see section 5).

## 3. Perception (shared by all conditions)

Pipeline: YOLOE-11L text-prompted detection → tracking within the two observed segments → event
detection (target hidden under cup X) → ≤5 event-triggered keyframes → numbered candidates in the
query image. Evaluated against simulator ground truth.

**Detector choice** (240 pilot frames; P/R = precision/recall):

| Configuration | Cups P/R | Ball recall | Query image has exactly the 3 cups |
|---|---|---|---|
| YOLOE-11S, first prompts | 0.89 / 0.91 | 1.00 | 60 % |
| YOLOE-11L, first prompts | 0.96 / 0.96 | 1.00 | 73 % |
| YOLOE-11L, generic prompts (cup/bowl/container) | 0.88 / 0.99 | **0.02** | 50 % |
| YOLOE-11L, final tuned prompts + track voting | 0.99 / 0.97 | 0.95 | **93 %** |

Findings:
- **The detector is very sensitive to wording:** "small yellow ball" finds the ball in 95 % of frames,
  "yellow ball" in 2 %.
- **Cans are detected as cups** with generic prompts; naming can types as distractors fixes most of it.
- **Single-frame decisions are fragile.** Three fixes on the pilot: (1) an object's role (cup, ball,
  distractor) is decided by voting over its whole track; (2) tracks seen in less than half of the
  frames are dropped; (3) the ball never shares a track with other objects (otherwise a cup lowered
  over the ball continued the ball's track and the hiding went unnoticed).
- **Final perception on the harder pilot:** hiding detected and the correct cup assigned in **30/30**
  trials; correct cup among the candidates in **30/30**; exactly three candidates in 28/30.

## 4. Methodological decisions (with the reasoning)

- **We evaluate the memory, not the perception.** Perception sees what happens inside each episode
  ("ball went under cup B"). It cannot link the cups across the unobserved change ("which of the new
  cups is B?"); that link is the memory, and that is where RQ1/RQ2 are answered.
- **Memory conditions answer by direct lookup, without a VLM.** A VLM would only read "ball under #2"
  back. The VLM is needed only for the two frame-based baselines (current-only, keyframes), which have
  no memory structure. This revises the earlier decision (Q5: same VLM for all conditions).
- **Perception is tuned once on pilot data and then frozen**, identical for all conditions, and its
  errors are reported separately. A diagnostic with ground-truth containment as input isolates the
  memory's re-identification accuracy.
- **Honest limits on generality:**
  - the prompts are fitted to the scene objects;
  - the event rule covers only one kind of hiding (a container lowered over an object);
  - the 2D memory assumes all containers stand on one plane (the table).

  The memory record and the re-identification are object-agnostic. A generalisation test with unseen
  containers, objects and container heights would be needed to claim more. Open decision:
  options A, B or C, below.

## 5. 2D memory (condition 3) on the harder pilot

Memory record per cup: footprint (bottom centre of its box), colour histogram, contents. After the
change it:
1. finds the before/after sweep pair with the most **background matches** (SIFT on the masked static
   scene);
2. estimates a table-plane homography with RANSAC;
3. maps the old footprints into the new image;
4. assigns old to new cups jointly by position plus colour (Hungarian);
5. answers with the candidate number.

Fewer than 15 matches gives "unknown", which counts as wrong.

| Cell | 2D memory | Raw positions (no registration) |
|---|---|---|
| identical / fixed | 1.00 | 1.00 |
| distinct / fixed | 1.00 | 1.00 |
| identical / cup moved | 1.00 | 1.00 |
| distinct / cup moved | 0.80 | 0.80 |
| distinct / camera | 0.80 | 0.80 |
| identical / camera | 0.60 | 0.40 |
| **all** | **0.87** | 0.83 (chance 0.33) |

Camera condition by angle:
- 2D memory: correct at 13–48° and at 120–143°; fails at 66–92° in 3 of 4 trials.
- Raw positions: fail in all trials above ~110°.

**Failure types** (visual report: `sim/out/pilot/memory_report.html`):
1. **False registration** (trial 7, 66°): the repetitive wood grain produces consistent but wrong
   matches. The "homography" collapses all cups onto one point, so the choice is random. The memory
   should detect implausible registrations and answer "unknown".
2. **Too few matches from the side** (trials 19 and 22, 85–92°): the table is strongly foreshortened;
   only ~13 matches are found. This is the genuine 2D limit, and the case where VGGT is expected to help (H2).
3. **A cup missing in the chosen frame** (trial 23): the frame pair was chosen by match count alone,
   and in that frame a cup was not detected, so the moved cup was assigned to the wrong new cup. This
   is an implementation flaw: frames with all cups visible should be preferred, or cups pooled over the
   segment.

Fixes 1 and 3 are deferred until all four conditions exist.

## 6. Status of the four conditions

| Condition | Status |
|---|---|
| 1. Current-only (VLM, final image) | query bundles and VLM interface being built; VLM run needs the cluster |
| 2. Keyframes (VLM, ≤5 event frames) | keyframe selection done; VLM run needs the cluster |
| 3. 2D memory | built, 87 % on the pilot, two fixes pending |
| 4. VGGT memory | being built (VGGT installed, weights downloading); full runs need the cluster |

## 7. Open decisions and to-dos

- **Project direction** (to discuss with Haonan and the mentor):
  - (A) narrow, honestly scoped study;
  - (B) object-agnostic perception plus a generalisation test on unseen objects and containers
    (recommended);
  - (C) a VLM-centred study.
- **Research plan (v4) needs updating:**
  - camera change 10–150° instead of 10–60°;
  - cup moves that change the order;
  - memory conditions answer by lookup (Q5 revised);
  - MIKASA-Robo "adapted from", plus its reference entry;
  - limitations on perception specificity.
- **Statistics:** 5 trials per cell is far too few for conclusions; the planned 30 per cell
  (180 trials) are needed.
