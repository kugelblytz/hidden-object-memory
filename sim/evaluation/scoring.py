"""Scoring of perception output and ground-truth diagnostics (evaluation only)."""
import json
from pathlib import Path

from perception.tracker import iou

from .gt import gt_boxes, gt_meta
from .metrics import match_frame


def score_perception(trial: Path, res: dict) -> dict:
    """Compare one trial's perception output with ground truth.

    Also writes trial/gt/candidates.json: which query candidate is the correct one (used
    later to score every condition)."""
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


def gt_parent_track(trial: Path, perception: dict) -> int | None:
    """Diagnostic input: the segment-A track of the true target cover at the end of episode 1."""
    meta = gt_meta(trial)
    seg_a = perception["segments"]["A"]
    n_ep1 = meta["n_frames"]["episode1"]
    gt = next((g for g in gt_boxes(trial, "episode1", n_ep1 - 1) if g["name"] == f"cover_{meta['target_cover']}"), None)
    if gt is None:
        return None
    best = max(((iou(d["box"], gt["box"]), d["track_id"]) for d in seg_a[n_ep1 - 1]["dets"] if d["role"] == "cover"),
               default=(0, None))
    return best[1] if best[0] >= 0.5 else None
