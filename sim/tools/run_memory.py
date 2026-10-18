"""Evaluate the memory conditions by direct lookup (no VLM) on perceived trials.

Usage: .venv\\Scripts\\python.exe -m tools.run_memory --data out/pilot

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

from evaluation.gt import gt_meta
from evaluation.scoring import gt_parent_track
from memory.memory2d import answer


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
