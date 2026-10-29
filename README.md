# Actor-Agnostic Persistent Object Memory for Hidden-Object Localization

KTH course project by Haonan Yin and Joris Köster. A small ball is hidden under one of three cups;
then, while nobody watches, the camera moves around the table or one cup is moved. Can a robot memory
still name the cup that hides the ball? We compare five systems on 180 simulated test trials:

| System | Accuracy (chance 0.33) |
|---|---|
| VGGT memory, joint pass (old and new frames in one 3D reconstruction) | **0.93** |
| VGGT memory, separate passes (aligned with feature matching) | 0.88 |
| 2D memory (table-plane homography) | 0.83 |
| Vision-language model with keyframes (Qwen3.5-9B / Qwen3-VL-8B) | 0.66 / 0.68 |
| Vision-language model with the final image only | 0.27 / 0.08 |

- **RQ1** (explicit memory vs frame-based baselines): memory wins by +0.18 to +0.28 (all p_holm < 0.001).
- **RQ2** (3D vs 2D grounding): the joint VGGT pass beats 2D by +0.30 [+0.18, +0.43] under camera changes;
  separate passes give +0.13 (not significant).
- **H2** (the advantage grows with the viewpoint change): not supported.

**In short:** a controlled benchmark-design study. Trials are seeded and fully reproducible; pilot and test
use disjoint seed ranges, and the test protocol was frozen before the test run. Ground truth is read only by
the evaluation code. All systems receive the same RGB observations; comparisons are paired, with
Holm-corrected tests and confidence intervals, plus failure analyses for every system
([RESULTS.md](RESULTS.md), [FINDINGS.md](FINDINGS.md)).

## Contributions

Joint project of Haonan Yin and Joris Köster. All commits in this repository were made by Joris Köster.

<!-- TODO(joris): confirm the split with Haonan before publishing, then replace this list. -->
- **Haonan Yin:** _to be confirmed_
- **Joris Köster:** _to be confirmed_

## Where to start

| File | What it is |
|---|---|
| [RESULTS.md](RESULTS.md) | Results report: setup, conditions, tables, figures, research questions, related work, limitations |
| [results/where_did_the_ball_go.html](results/where_did_the_ball_go.html) | Interactive explainer: how each system processes its inputs, a schematic model, an explorer of all 180 trials (open in a browser) |
| [FINDINGS.md](FINDINGS.md) | Lab notebook: design decisions, pilot findings, failure analyses, protocol deviation |
| [docs/research_plan_v4.md](docs/research_plan_v4.md) | Research plan (v4, out of date). Later design changes are listed below and in FINDINGS.md |
| [sim/README.md](sim/README.md) | Code documentation: simulation, perception, memories, evaluation |
| [sim/cluster/CLUSTER.md](sim/cluster/CLUSTER.md) | Running the evaluation on the GPU cluster |

## Repository layout

```
RESULTS.md, FINDINGS.md        results report and lab notebook
docs/                          research plan (v4, out of date)
results/                       figures, summary.json, HTML report, interactive explainer
sim/
  run_pilot.py                 record pilot or test trials (disjoint seed ranges)
  run_all.py                   one resumable job: perception -> memories -> VLMs -> report
  plot_results.py              figures for RESULTS.md
  hidden_object/               ManiSkill3 scene and trial recording (two episodes, unobserved change)
  perception/                  shared RGB perception: detection, tracking, events, keyframes, candidates
  memory/                      2D memory and VGGT memory (separate and joint pass)
  baselines/                   vision-language-model baselines (current-only, keyframes)
  evaluation/                  the only code that reads ground truth: GT access, metrics, scoring
  tools/                       development and diagnostic tools (python -m tools.<name>)
  cluster/                     setup script, requirements, model download, SLURM template
```

## Reproducing the results

1. **Record the test set** (laptop, Windows; ~15 min, ~4 GB). Setup is described in `sim/README.md`.
   ```
   cd sim
   .venv\Scripts\python.exe run_pilot.py --split test --per-cell 30 --protocol-frozen
   ```
2. **Evaluate on a GPU** (tested on a 20 GB slice of an H100; ~2 h).
   ```
   bash cluster/setup_env.sh
   python cluster/download_models.py
   python run_all.py --data out/test --vlm Qwen/Qwen3.5-9B Qwen/Qwen3-VL-8B-Instruct --load-in-8bit
   ```
3. **Figures:**
   ```
   python plot_results.py --summary ../results/test_summary.json --out ../results
   ```

Trials are fully determined by their seeds. The perception and the memories use only the RGB images
in `inputs/`; simulator ground truth (`gt/`) is used for scoring only.

## Changes relative to research plan v4

The plan document was not updated after the design was tightened. Main differences (details in FINDINGS.md):

- **Camera change:** 10–150° instead of 10–60°. With only 10–60° the task was too easy: raw pixel
  positions already solved it.
- **Cup moves:** one cup moves 10–35 cm and changes its left/right order.
- **Answering:** memory conditions answer by direct lookup; the vision-language model is only used for
  the frame-based baselines.
- **VGGT variants:** VGGT is evaluated with separate passes *and* with one joint pass over old and new
  frames.
- **Simulator:** ManiSkill3 (scene adapted from [MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo)'s ShellGame, MIT license).
- **Containers:** inverted mugs ("shell game").
- **Language models:** Qwen3.5-9B and Qwen3-VL-8B in 8-bit, because the cluster GPU is a 20 GB slice.

## Known limitations

- Simulation only, one task family.
- Perception prompts are tuned to the scene objects.
- The 2D memory does not yet reject implausible homographies, which makes it a weaker baseline than
  necessary.
- The answer extraction of the language models was fixed after a first test run truncated their
  answers (documented in RESULTS.md).
- About 8% of trials fail for every system because of perception errors.

See RESULTS.md, "Limitations and deviations".
