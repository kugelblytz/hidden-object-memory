"""Record pilot trials: one per (appearance x motion) cell, cycling through scenes.

Usage: .venv\\Scripts\\python.exe run_pilot.py [--out out/pilot] [--seed 1000] [--per-cell 1]
Invalid trials (ball not hidden, a cover not visible) are resampled with a new seed.
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym

import hidden_object  # noqa: F401
from hidden_object.trial import APPEARANCES, MOTIONS, SCENES, record_trial, sample_trial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/pilot")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--per-cell", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=5)
    args = ap.parse_args()

    env = gym.make(
        "HiddenObjectShell-v0",
        num_envs=1,
        obs_mode="rgb+depth+segmentation",
        sim_backend="cpu",
        render_backend="cpu",  # GPU render backend needs CUDA torch (see sim-setup notes)
        enable_shadow=True,
    )
    out_root = Path(args.out)
    summary, trial_id = [], 0
    for rep in range(args.per_cell):
        for appearance in APPEARANCES:
            for motion in MOTIONS:
                scene_id = trial_id % len(SCENES)
                for attempt in range(args.max_attempts):
                    seed = args.seed + 1000 * attempt + trial_id
                    t0 = time.time()
                    trial = sample_trial(trial_id, seed, scene_id, appearance, motion)
                    out_dir = out_root / f"trial_{trial_id:04d}"
                    checks = record_trial(env, trial, out_dir)
                    print(f"trial {trial_id:3d} {appearance:9s} {motion:6s} scene {scene_id} seed {seed}: "
                          f"valid={checks['valid']} {checks} ({time.time() - t0:.1f}s)")
                    if checks["valid"]:
                        break
                summary.append({"trial_id": trial_id, "seed": seed, "appearance": appearance, "motion": motion,
                                "scene_id": scene_id, "attempts": attempt + 1, **checks})
                trial_id += 1
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
