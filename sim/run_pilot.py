"""Record trials for every (appearance x motion) cell, cycling through scenes.

Usage:
  .venv\\Scripts\\python.exe run_pilot.py --split pilot --per-cell 5 --gif
  .venv\\Scripts\\python.exe run_pilot.py --split test --per-cell 30 --protocol-frozen

Pilot and test seeds come from disjoint ranges, so pilot recordings can never end up
in the test set. Test recordings additionally require --protocol-frozen.
Invalid trials (ball not hidden, a cover not visible) are resampled with a new seed.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
from PIL import Image, ImageDraw

import hidden_object  # noqa: F401
from hidden_object.trial import APPEARANCES, MOTIONS, SCENES, record_trial, sample_trial

SEED_BASE = {"pilot": 10_000, "test": 1_000_000}
ATTEMPT_STRIDE = 1_000  # seed = base + attempt * stride + trial_id, so trial_id < 1000


def write_gif(trial_dir: Path):
    """Animated overview of all input frames, labelled by phase (for humans only)."""
    frames = []
    for stream, label in [("episode1", "episode 1"), ("sweep_before", "sweep before"),
                          ("sweep_after", "sweep after (after unobserved change)")]:
        for p in sorted((trial_dir / "inputs" / stream).glob("*.png")):
            im = Image.open(p).convert("RGB")
            ImageDraw.Draw(im).text((10, 10), label, fill="white")
            frames.append(im)
    query = Image.open(trial_dir / "inputs" / "query.png").convert("RGB")
    ImageDraw.Draw(query).text((10, 10), "query (episode 2)", fill="white")
    frames += [query] * 8
    frames[0].save(trial_dir / "recording.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SEED_BASE), default="pilot")
    ap.add_argument("--out", default=None, help="default: out/<split>")
    ap.add_argument("--per-cell", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--gif", action="store_true", help="also write recording.gif per trial")
    ap.add_argument("--protocol-frozen", action="store_true", help="required for --split test")
    args = ap.parse_args()
    if args.split == "test" and not args.protocol_frozen:
        ap.error("test recordings require --protocol-frozen (fix all settings on pilot data first)")

    env = gym.make(
        "HiddenObjectShell-v0",
        num_envs=1,
        obs_mode="rgb+depth+segmentation",
        sim_backend="cpu",
        render_backend="cpu",  # GPU render backend needs CUDA torch (see README)
        enable_shadow=True,
    )
    out_root = Path(args.out or f"out/{args.split}")
    base = SEED_BASE[args.split]
    summary, trial_id = [], 0
    for _ in range(args.per_cell):
        for appearance in APPEARANCES:
            for motion in MOTIONS:
                scene_id = trial_id % len(SCENES)
                for attempt in range(args.max_attempts):
                    seed = base + attempt * ATTEMPT_STRIDE + trial_id
                    t0 = time.time()
                    trial = sample_trial(trial_id, seed, scene_id, appearance, motion)
                    out_dir = out_root / f"trial_{trial_id:04d}"
                    checks = record_trial(env, trial, out_dir)
                    print(f"trial {trial_id:3d} {appearance:9s} {motion:6s} scene {scene_id} seed {seed}: "
                          f"valid={checks['valid']} min_cover_px={min(checks['cover_px_query'])} "
                          f"({time.time() - t0:.1f}s)", flush=True)
                    if checks["valid"]:
                        break
                if args.gif:
                    write_gif(out_dir)
                summary.append({"trial_id": trial_id, "seed": seed, "appearance": appearance, "motion": motion,
                                "scene_id": scene_id, "attempts": attempt + 1,
                                "magnitude": trial["change"].get("magnitude"), **checks})
                trial_id += 1
    env.close()
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    cells = defaultdict(list)
    for s in summary:
        cells[(s["appearance"], s["motion"])].append(s)
    print("\ncell                 trials  resampled  invalid  min_cover_px  magnitude range")
    for (appearance, motion), rows in cells.items():
        mags = [r["magnitude"] for r in rows if r["magnitude"] is not None]
        mag = f"{min(mags):.2f}-{max(mags):.2f}" if mags else "-"
        print(f"{appearance:9s} {motion:6s}  {len(rows):6d}  {sum(r['attempts'] > 1 for r in rows):9d}  "
              f"{sum(not r['valid'] for r in rows):7d}  {min(min(r['cover_px_query']) for r in rows):12d}  {mag}")


if __name__ == "__main__":
    main()
