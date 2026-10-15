"""Evaluate the memory conditions by direct lookup (no VLM) on perceived trials.

Usage: .venv\\Scripts\\python.exe run_memory.py --data out/pilot

Variants
  2d                 perception containment + table-plane registration (condition 3)
  2d_noreg           diagnostic: raw image positions, no registration
  2d_gt_containment  diagnostic: true parent cover as input, isolates re-identification
Unknown answers count as wrong. Requires run_perception.py to have been run first.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from memory.memory2d import answer
from perception.gt import gt_boxes, gt_meta
from perception.tracker import iou


def gt_parent_track(trial: Path, perception: dict) -> int | None:
    """Evaluation-only: segment-A track of the true target cover at the end of episode 1."""
    meta = gt_meta(trial)
    seg_a = perception["segments"]["A"]
    n_ep1 = meta["n_frames"]["episode1"]
    gt = next((g for g in gt_boxes(trial, "episode1", n_ep1 - 1) if g["name"] == f"cover_{meta['target_cover']}"), None)
    if gt is None:
        return None
    best = max(((iou(d["box"], gt["box"]), d["track_id"]) for d in seg_a[n_ep1 - 1]["dets"] if d["role"] == "cover"),
               default=(0, None))
    return best[1] if best[0] >= 0.5 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/pilot")
    args = ap.parse_args()

    variants = ("2d", "2d_noreg", "2d_gt_containment")
    cells, rows = defaultdict(lambda: defaultdict(list)), []
    for trial in sorted(Path(args.data).glob("trial_*")):
        perception = json.loads((trial / "perception" / "perception.json").read_text())
        correct = json.loads((trial / "gt" / "candidates.json").read_text())["correct_candidate"]
        meta = gt_meta(trial)
        parent = (perception["containment"] or {}).get("parent_track")
        results = {
            "2d": answer(trial, perception, parent),
            "2d_noreg": answer(trial, perception, parent, register=False),
            "2d_gt_containment": answer(trial, perception, gt_parent_track(trial, perception)),
        }
        cell = f"{meta['appearance']}/{meta['motion']}"
        row = {"trial": trial.name, "cell": cell, "correct_candidate": correct, "magnitude": meta["change"].get("magnitude")}
        for v, r in results.items():
            ok = r["candidate"] is not None and r["candidate"] == correct
            cells[cell][v].append(ok)
            row[v] = {**r, "correct": ok}
        rows.append(row)
        print(f"{trial.name} {cell:18s} " + "  ".join(
            f"{v}={'ok' if row[v]['correct'] else 'X'}({row[v]['reason']}, inl={row[v].get('inliers', '-')})" for v in variants), flush=True)

    print(f"\n{'cell':18s} " + " ".join(f"{v:>18s}" for v in variants))
    totals = defaultdict(list)
    for cell in sorted(cells):
        for v in variants:
            totals[v] += cells[cell][v]
        print(f"{cell:18s} " + " ".join(f"{sum(cells[cell][v]) / len(cells[cell][v]):18.2f}" for v in variants))
    print(f"{'all':18s} " + " ".join(f"{sum(totals[v]) / len(totals[v]):18.2f}" for v in variants) + "   (chance 0.33)")
    (Path(args.data) / "memory_eval.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
