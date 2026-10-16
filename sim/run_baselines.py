"""Run the VLM baselines (current-only, keyframe) on perceived trials and score them.

Usage:
  .venv\\Scripts\\python.exe run_baselines.py --data out/pilot --backend dry_run
  .venv\\Scripts\\python.exe run_baselines.py --data out/pilot --backend hf --model Qwen/Qwen2.5-VL-7B-Instruct --device cuda

dry_run only writes trial/baselines/<condition>.json bundles. Results are saved to
<data>/baselines_<model>.json; unknown answers count as wrong.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from baselines.vlm import CONDITIONS, DryRunBackend, HFBackend, build_bundle, parse_answer
from perception.gt import gt_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/pilot")
    ap.add_argument("--backend", choices=["dry_run", "hf"], default="dry_run")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    backend = DryRunBackend() if args.backend == "dry_run" else HFBackend(args.model, args.device)
    cells, rows = defaultdict(lambda: defaultdict(list)), []
    for trial in sorted(Path(args.data).glob("trial_*")):
        perception = json.loads((trial / "perception" / "perception.json").read_text())
        correct = json.loads((trial / "gt" / "candidates.json").read_text())["correct_candidate"]
        meta = gt_meta(trial)
        cell = f"{meta['appearance']}/{meta['motion']}"
        row = {"trial": trial.name, "cell": cell, "correct_candidate": correct, "magnitude": meta["change"].get("magnitude")}
        for cond in CONDITIONS:
            bundle = build_bundle(trial, perception, cond)
            t0 = time.time()
            reply = backend.generate([trial / p for p in bundle["images"]], bundle["prompt"])
            ans = parse_answer(reply, bundle["candidates"])
            ok = ans is not None and ans == correct
            cells[cell][cond].append(ok)
            row[cond] = {"reply": reply, "answer": ans, "correct": ok, "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"{trial.name} {cell:18s} correct=#{correct} " + "  ".join(
            f"{c}: {row[c]['reply']!r}->{row[c]['answer']} ({'ok' if row[c]['correct'] else 'X'})" for c in CONDITIONS), flush=True)

    if args.backend == "dry_run":
        print("bundles written to trial/baselines/")
        return
    print(f"\n{'cell':18s} " + " ".join(f"{c:>14s}" for c in CONDITIONS))
    totals = defaultdict(list)
    for cell in sorted(cells):
        for c in CONDITIONS:
            totals[c] += cells[cell][c]
        print(f"{cell:18s} " + " ".join(f"{sum(cells[cell][c]) / len(cells[cell][c]):14.2f}" for c in CONDITIONS))
    print(f"{'all':18s} " + " ".join(f"{sum(totals[c]) / len(totals[c]):14.2f}" for c in CONDITIONS) + "   (chance 0.33)")
    name = args.model.split("/")[-1]
    (Path(args.data) / f"baselines_{name}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
