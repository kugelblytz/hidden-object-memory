"""Run the shared perception pipeline on recorded trials and evaluate it against GT.

Usage: .venv\\Scripts\\python.exe run_perception.py --data out/pilot

Per trial, perception output goes to trial/perception/ (derived from inputs only).
The ground-truth number of the correct candidate goes to trial/gt/candidates.json
(for scoring only). A per-cell summary is printed and saved to <data>/perception_eval.json.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from perception.detector import Detector
from perception.evaluate import match_frame, summarize
from perception.gt import gt_boxes, gt_meta
from perception.pipeline import process_trial
from perception.tracker import iou


def evaluate_trial(trial: Path, res: dict) -> dict:
    meta = gt_meta(trial)
    target_name = f"cover_{meta['target_cover']}"
    counts = []
    for seg in res["segments"].values():
        for f in seg:
            counts.append(match_frame(f["dets"], gt_boxes(trial, f["stream"], f["idx"]))["counts"])

    n_ep1 = meta["n_frames"]["episode1"]
    ball_vis = [i for i in range(n_ep1) if any(g["role"] == "target" for g in gt_boxes(trial, "episode1", i))]
    ev = res["events"]

    # Containment: which GT cover does the inferred parent box correspond to?
    parent_name = None
    if res["containment"]:
        best = max(((iou(res["containment"]["parent_box_end_ep1"], g["box"]), g["name"])
                    for g in gt_boxes(trial, "episode1", n_ep1 - 1) if g["role"] == "cover"), default=(0, None))
        parent_name = best[1] if best[0] >= 0.5 else None

    # Query candidates vs GT covers.
    gt_q = [g for g in gt_boxes(trial, "query", 0) if g["role"] == "cover"]
    cand_to_gt = {}
    for c in res["query_candidates"]:
        best = max(((iou(c["box"], g["box"]), g["name"]) for g in gt_q), default=(0, None))
        cand_to_gt[c["number"]] = best[1] if best[0] >= 0.5 else None
    correct = [n for n, name in cand_to_gt.items() if name == target_name]
    (trial / "gt" / "candidates.json").write_text(json.dumps(
        {"candidate_to_cover": cand_to_gt, "correct_candidate": correct[0] if len(correct) == 1 else None}, indent=2))

    return {
        "counts": counts,
        "hidden_detected": ev["hidden"],
        "first_seen_err": None if ev["target_first_seen"] is None or not ball_vis else ev["target_first_seen"] - ball_vis[0],
        "last_seen_err": None if ev["target_last_seen"] is None or not ball_vis else ev["target_last_seen"] - ball_vis[-1],
        "containment_correct": parent_name == target_name,
        "containment_found": res["containment"] is not None,
        "candidates_exact": len(cand_to_gt) == 3 and sorted(v for v in cand_to_gt.values() if v) == sorted(g["name"] for g in gt_q),
        "target_in_candidates": len(correct) == 1,
        "n_candidates": len(cand_to_gt),
        "n_keyframes": len(res["keyframes"]),
    }


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
        ev = evaluate_trial(trial, res)
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
