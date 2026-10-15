"""Detection metrics against simulator ground truth (evaluation only)."""
from collections import defaultdict

from .gt import gt_boxes
from .tracker import iou

ROLES = ("cover", "target", "distractor")


def match_frame(dets: list, gts: list, iou_thr: float = 0.5) -> dict:
    """Greedy one-to-one matching per role; returns per-role tp/fp/fn and matches."""
    out = {r: {"tp": 0, "fp": 0, "fn": 0} for r in ROLES}
    matches = []
    for role in ROLES:
        d = sorted([x for x in dets if x["role"] == role], key=lambda x: -x["score"])
        g = [x for x in gts if x["role"] == role]
        used = set()
        for det in d:
            best, best_j = 0.0, None
            for j, gt in enumerate(g):
                if j not in used and (v := iou(det["box"], gt["box"])) > best:
                    best, best_j = v, j
            if best >= iou_thr:
                used.add(best_j)
                out[role]["tp"] += 1
                matches.append((det, g[best_j]))
            else:
                out[role]["fp"] += 1
        out[role]["fn"] += len(g) - len(used)
    return {"counts": out, "matches": matches}


def summarize(counts_list: list) -> dict:
    tot = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for c in counts_list:
        for role, v in c.items():
            for k in v:
                tot[role][k] += v[k]
    res = {}
    for role, v in tot.items():
        p = v["tp"] / max(v["tp"] + v["fp"], 1)
        r = v["tp"] / max(v["tp"] + v["fn"], 1)
        res[role] = {"precision": round(p, 3), "recall": round(r, 3), **v}
    return res


def query_candidates_exact(dets: list, trial_dir) -> bool:
    """True if the cover detections in the query image match the three covers one-to-one."""
    gts = gt_boxes(trial_dir, "query", 0)
    c = match_frame([d for d in dets if d["role"] == "cover"], [g for g in gts if g["role"] == "cover"])["counts"]["cover"]
    return c["tp"] == 3 and c["fp"] == 0 and c["fn"] == 0
