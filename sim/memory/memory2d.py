"""2D containment memory with automatic tabletop-plane registration.

Inputs: the RGB frames in ``inputs/`` and the shared perception output; never ``gt/``.

At the end of episode 1 the memory holds every cover track (footprint = bottom centre
of its box, where the inverted cover touches the table, and an HSV colour histogram)
and the containment relation target -> parent cover. After the unobserved change it
must decide which cover in episode 2 is the old parent:

1. SIFT features are extracted on the static background of every frame of the
   before-sweep (segment A) and the after-sweep (segment B); detected objects are masked.
2. For every (A frame, B frame) pair a homography is estimated with RANSAC, which
   locks onto the dominant plane, the table. The pair with most inliers is used.
3. Old footprints are mapped into the B frame and old covers are assigned to new
   covers (Hungarian) by registered position plus appearance distance.
4. The new track of the parent is translated to its number among the query candidates.
If registration is too weak, or the parent cannot be assigned, the answer is unknown.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

MIN_INLIERS = 15
RATIO = 0.75
RANSAC_PX = 4.0
APPEARANCE_WEIGHT = 1.0
MASK_PAD = 12


def frame_path(trial_dir: Path, stream: str, idx: int) -> Path:
    return trial_dir / "inputs" / ("query.png" if stream == "query" else f"{stream}/{idx:03d}.png")


def footprint(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, box[3]])


def colour_hist(img_bgr: np.ndarray, box) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    crop = img_bgr[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def appearance_distance(h1, h2) -> float:
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


class Frame:
    """One frame with its detections and background features."""

    _sift = None

    def __init__(self, trial_dir: Path, frame: dict):
        self.stream, self.idx, self.dets = frame["stream"], frame["idx"], frame["dets"]
        self.img = cv2.imread(str(frame_path(trial_dir, self.stream, self.idx)))
        if Frame._sift is None:
            Frame._sift = cv2.SIFT_create(nfeatures=4000)
        mask = np.full(self.img.shape[:2], 255, np.uint8)
        for d in self.dets:  # every detected object may move, so only background is matched
            x0, y0, x1, y1 = [int(round(v)) for v in d["box"]]
            mask[max(y0 - MASK_PAD, 0):y1 + MASK_PAD, max(x0 - MASK_PAD, 0):x1 + MASK_PAD] = 0
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.kp, self.desc = Frame._sift.detectAndCompute(gray, mask)

    def covers(self) -> dict:
        return {d["track_id"]: d for d in self.dets if d["role"] == "cover"}


def homography(fa: Frame, fb: Frame):
    if fa.desc is None or fb.desc is None or len(fa.kp) < 8 or len(fb.kp) < 8:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(fa.desc, fb.desc, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < 8:
        return None, 0
    src = np.float32([fa.kp[m.queryIdx].pt for m in good])
    dst = np.float32([fb.kp[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    return (H, int(inl.sum())) if H is not None else (None, 0)


def answer(trial_dir, perception: dict, parent_track: int | None, register: bool = True) -> dict:
    """Return {"candidate": number | None, ...diagnostics} for one trial."""
    trial_dir = Path(trial_dir)
    out = {"candidate": None, "reason": None}
    if parent_track is None:
        out["reason"] = "no containment"
        return out
    seg_a, seg_b = perception["segments"]["A"], perception["segments"]["B"]
    n_ep1 = sum(f["stream"] == "episode1" for f in seg_a)
    frames_a = [Frame(trial_dir, f) for f in seg_a[n_ep1 - 1:] if any(d["track_id"] == parent_track for d in f["dets"])]
    frames_b = [Frame(trial_dir, f) for f in seg_b]
    if not frames_a:
        out["reason"] = "parent not visible after episode 1"
        return out

    if register:
        best = (None, None, None, -1)
        for fa in frames_a:
            for fb in frames_b:
                H, n = homography(fa, fb)
                if n > best[3]:
                    best = (fa, fb, H, n)
        fa, fb, H, n_inl = best
        out.update({"inliers": n_inl, "pair": [f"{fa.stream}/{fa.idx}", f"{fb.stream}/{fb.idx}"]})
        if H is None or n_inl < MIN_INLIERS:
            out["reason"] = "registration failed"
            return out
    else:  # diagnostic: raw image positions, first after-change frame
        fa, fb, H = frames_a[0], frames_b[0], np.eye(3)

    old, new = fa.covers(), fb.covers()
    old_ids, new_ids = list(old), list(new)
    if not new_ids:
        out["reason"] = "no covers after change"
        return out
    pts = np.float32([footprint(old[t]["box"]) for t in old_ids]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    scale = np.median([new[t]["box"][3] - new[t]["box"][1] for t in new_ids])
    cost = np.zeros((len(old_ids), len(new_ids)))
    for i, to in enumerate(old_ids):
        h_old = colour_hist(fa.img, old[to]["box"])
        for j, tn in enumerate(new_ids):
            pos = np.linalg.norm(mapped[i] - footprint(new[tn]["box"])) / max(scale, 1.0)
            cost[i, j] = pos + APPEARANCE_WEIGHT * appearance_distance(h_old, colour_hist(fb.img, new[tn]["box"]))
    rows, cols = linear_sum_assignment(cost)
    assign = {old_ids[r]: new_ids[c] for r, c in zip(rows, cols)}
    if parent_track not in assign:
        out["reason"] = "parent not assigned"
        return out
    new_parent = assign[parent_track]
    i = old_ids.index(parent_track)
    row = sorted(cost[i])
    out["margin"] = round(float(row[1] - row[0]), 3) if len(row) > 1 else None
    numbers = {c["track_id"]: c["number"] for c in perception["query_candidates"]}
    out["candidate"] = numbers.get(new_parent)
    out["reason"] = "ok" if out["candidate"] is not None else "parent track not among query candidates"
    return out
