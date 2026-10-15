"""Compare detector weights, prompt sets and confidence thresholds on pilot frames.

Pilot data only. Prints per-role precision/recall and the share of query images in
which exactly the three covers are detected (the candidate set used by all conditions).
Usage: .venv\\Scripts\\python.exe select_detector.py --data out/pilot
"""
import argparse
import json
from pathlib import Path

from perception.detector import Detector
from perception.evaluate import match_frame, query_candidates_exact, summarize
from perception.gt import gt_boxes

FRAMES = [("episode1", 0), ("episode1", 6), ("episode1", 13), ("episode1", 25),
          ("sweep_before", 3), ("sweep_after", 0), ("sweep_after", 6), ("query", 0)]

# Generic prompt sets only: the target is named as in the task instruction, containers
# and other objects by general categories, nothing tuned to individual scene objects.
CONFIGS = {
    "g_containers_object": ("yoloe-11l-seg.pt", {"cup": "cover", "bowl": "cover", "container": "cover",
                                                 "yellow ball": "target", "object": "distractor"}),
    "g_containers_only": ("yoloe-11l-seg.pt", {"cup": "cover", "bowl": "cover", "container": "cover",
                                               "yellow ball": "target"}),
    "g_household": ("yoloe-11l-seg.pt", {"cup": "cover", "bowl": "cover", "yellow ball": "target",
                                         "can": "distractor", "box": "distractor", "toy": "distractor"}),
}
THRESHOLDS = (0.05, 0.1, 0.2)


def frame_path(trial: Path, stream: str, idx: int) -> Path:
    return trial / "inputs" / ("query.png" if stream == "query" else f"{stream}/{idx:03d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/pilot")
    ap.add_argument("--out", default="out/detector_selection.json")
    args = ap.parse_args()
    trials = sorted(Path(args.data).glob("trial_*"))
    results = {}
    for name, (weights, prompts) in CONFIGS.items():
        det = Detector(weights, prompts, conf=min(THRESHOLDS))
        raw = {(t, s, i): [d.to_dict() for d in det.detect(str(frame_path(t, s, i)))] for t in trials for s, i in FRAMES}
        for thr in THRESHOLDS:
            counts, exact = [], []
            for (t, s, i), dets in raw.items():
                dets = [d for d in dets if d["score"] >= thr]
                counts.append(match_frame(dets, gt_boxes(t, s, i))["counts"])
                if s == "query":
                    exact.append(query_candidates_exact(dets, t))
            summ = summarize(counts)
            key = f"{name}@{thr}"
            results[key] = {**summ, "query_exact_candidates": round(sum(exact) / len(exact), 3)}
            print(f"{key:18s} " + "  ".join(f"{r}: P={summ[r]['precision']:.2f} R={summ[r]['recall']:.2f}" for r in summ)
                  + f"  query_exact={results[key]['query_exact_candidates']:.2f}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
