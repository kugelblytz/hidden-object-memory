"""Frame-to-frame tracking within one continuously observed segment.

Detections are associated with existing tracks by Hungarian matching on a cost
that combines box overlap and centre distance. Association may cross the cover and
distractor roles, because single-frame labels flicker (e.g. a can labelled "cup"),
but never the target role: otherwise a cover lowered over the target would continue
the target's track. The role of a track is decided afterwards by score-weighted
voting over its frames. Tracks survive
short detection gaps (``max_age`` frames). Linking tracks *across* the unobserved
change between episodes is not done here: that is the job of the memory systems.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre(b) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


class Tracker:
    def __init__(self, max_age: int = 3, max_centre_dist: float = 80.0):
        self.max_age, self.max_dist = max_age, max_centre_dist
        self.tracks = {}  # id -> {"role", "box", "last_frame"}
        self.next_id = 0

    def update(self, frame_idx: int, detections: list) -> list:
        """Assigns ``track_id`` to each detection in place and returns them."""
        alive = {tid: t for tid, t in self.tracks.items() if frame_idx - t["last_frame"] <= self.max_age}
        tids = list(alive)
        cost = np.full((len(detections), len(tids)), 1e6)
        for i, d in enumerate(detections):
            for j, tid in enumerate(tids):
                t = alive[tid]
                dist = np.linalg.norm(centre(d.box) - centre(t["box"]))
                if dist <= self.max_dist and (t["role"] == "target") == (d.role == "target"):
                    cost[i, j] = (1.0 - iou(d.box, t["box"])) + dist / self.max_dist
        matched = set()
        if cost.size:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e6:
                    detections[i].track_id = tids[j]
                    matched.add(i)
        for i, d in enumerate(detections):
            if i not in matched:
                d.track_id = self.next_id
                self.next_id += 1
            self.tracks[d.track_id] = {"role": d.role, "box": d.box, "last_frame": frame_idx}
        return detections
