"""Run the shared perception pipeline on recorded trials and evaluate it against GT.

Usage: .venv\\Scripts\\python.exe -m tools.run_perception --data out/pilot

Per trial, perception output goes to trial/perception/ (derived from inputs only).
The ground-truth number of the correct candidate goes to trial/gt/candidates.json
(for scoring only). A per-cell summary is printed and saved to <data>/perception_eval.json.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from evaluation.gt import gt_meta
from evaluation.metrics import summarize
from evaluation.scoring import score_perception
from perception.detector import Detector
from perception.pipeline import process_trial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/pilot")
    ap.add_argument("--weights", default="yoloe-11l-seg.pt")
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()

    detector = Detector(args.weights, conf=args.conf)
    rows, cells = [], defaultdict(list)
    for trial in sorted(Path(args.data).glob("trial_*")):
        t0 = time.time()
        res = process_trial(trial, detector)
        ev = score_perception(trial, res)
        meta = gt_meta(trial)
        cell = (meta["appearance"], meta["motion"])
        cells[cell].append(ev)
        rows.append({"trial": trial.name, "cell": cell, **{k: v for k, v in ev.items() if k != "counts"}})
        print(f"{trial.name} {cell[0]:9s} {cell[1]:6s} hidden={ev['hidden_detected']} "
              f"containment={'ok' if ev['containment_correct'] else 'WRONG'} candidates={ev['n_candidates']} "
              f"exact={ev['candidates_exact']} last_seen_err={ev['last_seen_err']} ({time.time() - t0:.1f}s)", flush=True)

    all_ev = [e for v in cells.values() for e in v]
    det = summarize([c for e in all_ev for c in e["counts"]])
    print("\nDetection over all frames: " + "  ".join(f"{r}: P={v['precision']:.3f} R={v['recall']:.3f}" for r, v in det.items()))
    print("\ncell               n  hidden  containment  cand_exact  target_in_cand")
    summary = {"detection": det, "detector": detector.config(), "cells": {}, "trials": rows}
    for cell, evs in sorted(cells.items()):
        m = lambda k: sum(bool(e[k]) for e in evs) / len(evs)
        summary["cells"]["/".join(cell)] = {k: m(k) for k in ("hidden_detected", "containment_correct", "candidates_exact", "target_in_candidates")}
        print(f"{cell[0]:9s} {cell[1]:6s} {len(evs):3d}  {m('hidden_detected'):6.2f}  {m('containment_correct'):11.2f}  "
              f"{m('candidates_exact'):10.2f}  {m('target_in_candidates'):14.2f}")
    m = lambda k: sum(bool(e[k]) for e in all_ev) / len(all_ev)
    print(f"{'all':16s} {len(all_ev):3d}  {m('hidden_detected'):6.2f}  {m('containment_correct'):11.2f}  "
          f"{m('candidates_exact'):10.2f}  {m('target_in_candidates'):14.2f}")
    (Path(args.data) / "perception_eval.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
