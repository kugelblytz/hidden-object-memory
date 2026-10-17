"""VGGT-grounded 3D containment memory (condition 4).

Same memory record, inputs and assignment rule as the 2D memory; only the geometry differs.
Inputs: RGB frames in ``inputs/`` and the shared perception output; never ``gt/``.

1. VGGT reconstructs segment A (end of episode 1 + before-sweep) and segment B
   (after-sweep + query) in two separate forward passes. Each pass yields a per-pixel 3D
   point map in its own frame and scale (first camera of the pass, up to scale).
2. Every cover track gets a 3D position: median of confident points inside its box,
   pooled over all frames of the segment in which it was detected.
3. The same SIFT background matches as in the 2D memory are lifted to 3D with the point
   maps and pooled over all (A, B) frame pairs. A RANSAC similarity transform
   (rotation, translation, scale) aligns A to B. No planar assumption, no metric scale and
   no prior knowledge of object sizes are needed.
4. Old covers are mapped into B and assigned to new covers by 3D distance (relative to the
   cover spacing in B) plus colour distance (Hungarian), then the parent's new track is
   translated to its query candidate number. Too few inliers -> unknown.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .memory2d import APPEARANCE_WEIGHT, RATIO, Frame, appearance_distance, colour_hist, frame_path

IMG_W, IMG_H = 640, 480          # recorded resolution
VGGT_W, VGGT_H = 518, 392        # VGGT "crop" preprocessing of a 640x480 image (no crop needed)
MIN_INLIERS = 15
RANSAC_ITERS = 2000
INLIER_FRAC = 0.03               # inlier threshold as a fraction of the scene scale
SCALE_RANGE = (0.2, 5.0)         # plausible relative scale between the two reconstructions
MAX_CORR = 6000
BOX_SHRINK = 0.2

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        from huggingface_hub import hf_hub_download
        from vggt.models.vggt import VGGT

        _MODEL = VGGT()
        _MODEL.load_state_dict(torch.load(hf_hub_download("facebook/VGGT-1B", "model.pt"), map_location="cpu"))
        _MODEL.to(_device()).eval()
    return _MODEL


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def reconstruct(paths: list, cache: Path) -> dict:
    """Per-pixel 3D points (S,H,W,3) and confidence (S,H,W) at VGGT resolution; cached."""
    if cache.exists():
        z = np.load(cache)
        return {"points": z["points"].astype(np.float32), "conf": z["conf"].astype(np.float32)}
    import torch
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = load_and_preprocess_images([str(p) for p in paths]).to(_device())
    assert images.shape[-2:] == (VGGT_H, VGGT_W), images.shape
    # bf16 on GPU (H100) as recommended by the VGGT authors; fp32 on CPU.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=_device() == "cuda"):
        pred = _model()(images)
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = pred["depth"][0].float().cpu().numpy()
    points = unproject_depth_map_to_point_map(depth, extri[0].float().cpu().numpy(), intri[0].float().cpu().numpy())
    conf = pred["depth_conf"][0].float().cpu().numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, points=points.astype(np.float16), conf=conf.astype(np.float16))
    return {"points": points.astype(np.float32), "conf": conf.astype(np.float32)}


def _to_vggt(x, y):
    return (np.clip((np.asarray(y) * VGGT_H / IMG_H).astype(int), 0, VGGT_H - 1),
            np.clip((np.asarray(x) * VGGT_W / IMG_W).astype(int), 0, VGGT_W - 1))


def _box_point(rec: dict, s: int, box) -> np.ndarray | None:
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * BOX_SHRINK, (y1 - y0) * BOX_SHRINK
    r0, c0 = _to_vggt(x0 + dx, y0 + dy)
    r1, c1 = _to_vggt(x1 - dx, y1 - dy)
    pts = rec["points"][s, r0:r1 + 1, c0:c1 + 1].reshape(-1, 3)
    conf = rec["conf"][s, r0:r1 + 1, c0:c1 + 1].reshape(-1)
    if len(conf) < 5:
        return None
    keep = conf >= np.median(conf)
    return np.median(pts[keep], axis=0)


def cover_positions(rec: dict, frames: list) -> dict:
    """track_id -> {"xyz": pooled 3D position, "frame": Frame, "box": box (for appearance)}"""
    per_track = {}
    for s, f in enumerate(frames):
        for d in f.dets:
            if d["role"] != "cover":
                continue
            p = _box_point(rec, s, d["box"])
            if p is not None:
                per_track.setdefault(d["track_id"], {"pts": [], "frame": f, "box": d["box"]})["pts"].append(p)
    return {t: {"xyz": np.median(np.array(v["pts"]), axis=0), "frame": v["frame"], "box": v["box"]}
            for t, v in per_track.items()}


def _matches(fa: Frame, fb: Frame):
    if fa.desc is None or fb.desc is None or len(fa.kp) < 2 or len(fb.kp) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(fa.desc, fb.desc, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    return (np.float32([fa.kp[m.queryIdx].pt for m in good]).reshape(-1, 2),
            np.float32([fb.kp[m.trainIdx].pt for m in good]).reshape(-1, 2))


def correspondences(rec_a, frames_a, rec_b, frames_b):
    """3D-3D background correspondences pooled over all (A, B) frame pairs."""
    X, Y = [], []
    for ia, fa in enumerate(frames_a):
        conf_a = np.median(rec_a["conf"][ia])
        for ib, fb in enumerate(frames_b):
            src, dst = _matches(fa, fb)
            if not len(src):
                continue
            ra, ca = _to_vggt(src[:, 0], src[:, 1])
            rb, cb = _to_vggt(dst[:, 0], dst[:, 1])
            ok = (rec_a["conf"][ia, ra, ca] >= conf_a) & (rec_b["conf"][ib, rb, cb] >= np.median(rec_b["conf"][ib]))
            X.append(rec_a["points"][ia, ra[ok], ca[ok]])
            Y.append(rec_b["points"][ib, rb[ok], cb[ok]])
    if not X:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(X), np.concatenate(Y)


def umeyama(X: np.ndarray, Y: np.ndarray):
    """s, R, t minimising ||Y - (s R X + t)||."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_x = (Xc ** 2).sum() / len(X)
    s = np.trace(np.diag(D) @ S) / max(var_x, 1e-12)
    return s, R, my - s * R @ mx


def ransac_sim3(X, Y, rng):
    scale_b = np.median(np.linalg.norm(Y - np.median(Y, axis=0), axis=1))
    thr = INLIER_FRAC * scale_b
    best, best_inl = None, np.zeros(len(X), bool)
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(len(X), 3, replace=False)
        s, R, t = umeyama(X[idx], Y[idx])
        if not SCALE_RANGE[0] <= s <= SCALE_RANGE[1]:
            continue
        inl = np.linalg.norm(Y - (s * X @ R.T + t), axis=1) < thr
        if inl.sum() > best_inl.sum():
            best, best_inl = (s, R, t), inl
    if best is None or best_inl.sum() < 3:
        return None, best_inl
    return umeyama(X[best_inl], Y[best_inl]), best_inl


MODES = ("separate", "joint")


def answer(trial_dir, perception: dict, parent_track: int | None, seed: int = 0, mode: str = "separate") -> dict:
    """mode="separate": one VGGT pass per segment, aligned by lifted background matches + Sim(3).
    mode="joint": one VGGT pass over the frames of both segments, so before and after share a
    3D frame and no explicit matching is needed. VGGT assumes a static scene; when a cover
    moved, that cover violates the assumption (the rest of the scene is static)."""
    assert mode in MODES
    trial_dir = Path(trial_dir)
    out = {"candidate": None, "reason": None, "mode": mode}
    if parent_track is None:
        out["reason"] = "no containment"
        return out
    seg_a, seg_b = perception["segments"]["A"], perception["segments"]["B"]
    n_ep1 = sum(f["stream"] == "episode1" for f in seg_a)
    fa_dicts, fb_dicts = seg_a[n_ep1 - 1:], seg_b
    frames_a = [Frame(trial_dir, f) for f in fa_dicts]
    frames_b = [Frame(trial_dir, f) for f in fb_dicts]
    paths_a = [frame_path(trial_dir, f["stream"], f["idx"]) for f in fa_dicts]
    paths_b = [frame_path(trial_dir, f["stream"], f["idx"]) for f in fb_dicts]
    if mode == "joint":
        rec = reconstruct(paths_a + paths_b, trial_dir / "vggt" / "joint.npz")
        n_a = len(paths_a)
        rec_a = {k: v[:n_a] for k, v in rec.items()}
        rec_b = {k: v[n_a:] for k, v in rec.items()}
    else:
        rec_a = reconstruct(paths_a, trial_dir / "vggt" / "segment_A.npz")
        rec_b = reconstruct(paths_b, trial_dir / "vggt" / "segment_B.npz")

    old, new = cover_positions(rec_a, frames_a), cover_positions(rec_b, frames_b)
    if parent_track not in old:
        out["reason"] = "parent not reconstructed"
        return out
    if not new:
        out["reason"] = "no covers after change"
        return out

    if mode == "joint":
        s, R, t = 1.0, np.eye(3), np.zeros(3)  # already in one frame
    else:
        X, Y = correspondences(rec_a, frames_a, rec_b, frames_b)
        rng = np.random.default_rng(seed)
        if len(X) > MAX_CORR:
            keep = rng.choice(len(X), MAX_CORR, replace=False)
            X, Y = X[keep], Y[keep]
        out["correspondences"] = int(len(X))
        if len(X) < MIN_INLIERS:
            out["reason"] = "registration failed"
            return out
        sim3, inl = ransac_sim3(X, Y, rng)
        out["inliers"] = int(inl.sum())
        if sim3 is None or inl.sum() < MIN_INLIERS:
            out["reason"] = "registration failed"
            return out
        s, R, t = sim3
        out["scale"] = round(float(s), 3)

    old_ids, new_ids = list(old), list(new)
    new_xyz = np.array([new[t_]["xyz"] for t_ in new_ids])
    spacing = np.median([np.linalg.norm(a - b) for i, a in enumerate(new_xyz) for b in new_xyz[i + 1:]]) if len(new_ids) > 1 else 1.0
    cost = np.zeros((len(old_ids), len(new_ids)))
    for i, to in enumerate(old_ids):
        mapped = s * R @ old[to]["xyz"] + t
        h_old = colour_hist(old[to]["frame"].img, old[to]["box"])
        for j, tn in enumerate(new_ids):
            pos = np.linalg.norm(mapped - new[tn]["xyz"]) / max(spacing, 1e-9)
            cost[i, j] = pos + APPEARANCE_WEIGHT * appearance_distance(h_old, colour_hist(new[tn]["frame"].img, new[tn]["box"]))
    rows, cols = linear_sum_assignment(cost)
    assign = {old_ids[r]: new_ids[c] for r, c in zip(rows, cols)}
    if parent_track not in assign:
        out["reason"] = "parent not assigned"
        return out
    row = sorted(cost[old_ids.index(parent_track)])
    out["margin"] = round(float(row[1] - row[0]), 3) if len(row) > 1 else None
    numbers = {c["track_id"]: c["number"] for c in perception["query_candidates"]}
    out["candidate"] = numbers.get(assign[parent_track])
    out["reason"] = "ok" if out["candidate"] is not None else "parent track not among query candidates"
    return out
