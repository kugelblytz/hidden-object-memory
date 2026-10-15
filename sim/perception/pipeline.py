"""Shared RGB perception pipeline used by all experimental conditions.

Input: ``trial_dir/inputs`` only (RGB frames and the stream/episode structure).
Output: ``trial_dir/perception/perception.json`` and ``query_marked.png``.

Steps
1. Detect covers, the target and distractors in every frame (text-prompted detector).
2. Track objects within the two continuously observed segments:
   A = episode1 + sweep_before, B = sweep_after + query. Linking A and B across
   the unobserved change is left to the memory systems.
3. Detect events in episode 1: target first/last seen, start of the carried
   cover's motion and its settling; infer the containment relation
   (target -> parent cover track) from where the target disappeared.
4. Select at most five keyframes from these events plus the first frame after
   the change (for the keyframe baseline).
5. Number the cover detections in the query image left to right; these are the
   candidates every condition chooses from.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tracker import Tracker, centre

STREAMS_A, STREAMS_B = ("episode1", "sweep_before"), ("sweep_after", "query")
HIDE_CONFIRM_FRAMES = 3   # target must stay undetected this long to count as hidden
MOVE_PX, SETTLE_PX = 6.0, 2.0
MAX_KEYFRAMES = 5
MIN_TRACK_SHARE, MIN_TRACK_FRAMES = 0.5, 3


def list_frames(trial_dir: Path, stream: str) -> list:
    if stream == "query":
        return [trial_dir / "inputs" / "query.png"]
    return sorted((trial_dir / "inputs" / stream).glob("*.png"))


def _track_segment(detector, trial_dir: Path, streams) -> list:
    tracker, frames, k = Tracker(), [], 0
    for stream in streams:
        for idx, path in enumerate(list_frames(trial_dir, stream)):
            dets = tracker.update(k, detector.detect(str(path)))
            frames.append({"stream": stream, "idx": idx, "dets": [d.to_dict() for d in dets]})
            k += 1
    _resolve_tracks(frames)
    return frames


def _resolve_tracks(frames: list):
    """Give every detection the score-weighted majority role of its track and drop
    tracks seen in fewer than MIN_TRACK_SHARE of the segment's frames (one-off false
    positives). The single-frame label is kept as ``frame_role``."""
    votes, seen = {}, {}
    for f in frames:
        for d in f["dets"]:
            v = votes.setdefault(d["track_id"], {})
            v[d["role"]] = v.get(d["role"], 0.0) + d["score"]
            seen[d["track_id"]] = seen.get(d["track_id"], 0) + 1
    min_frames = max(MIN_TRACK_FRAMES, MIN_TRACK_SHARE * len(frames))
    role = {tid: max(v, key=v.get) for tid, v in votes.items()}
    # A target that disappears under a cover is legitimately short-lived: keep it.
    keep = {tid for tid in votes if seen[tid] >= min_frames or role[tid] == "target"}
    for f in frames:
        f["dets"] = [{**d, "frame_role": d["role"], "role": role[d["track_id"]]} for d in f["dets"] if d["track_id"] in keep]


def _track_centres(frames: list, role: str) -> dict:
    """track_id -> {frame position in segment: centre}"""
    out = {}
    for k, f in enumerate(frames):
        for d in f["dets"]:
            if d["role"] == role:
                out.setdefault(d["track_id"], {})[k] = centre(d["box"])
    return out


def _events(seg_a: list) -> dict:
    ep1 = [k for k, f in enumerate(seg_a) if f["stream"] == "episode1"]
    n_ep1 = len(ep1)
    target_frames = [k for k in ep1 if any(d["role"] == "target" for d in seg_a[k]["dets"])]
    ev = {"target_first_seen": None, "target_last_seen": None, "carry_start": None, "covered": None,
          "hidden": False, "moved_cover_track": None}
    if not target_frames:
        return ev
    first, last = target_frames[0], target_frames[-1]
    ev["target_first_seen"], ev["target_last_seen"] = first, last
    ev["hidden"] = last <= n_ep1 - 1 - HIDE_CONFIRM_FRAMES and not any(
        d["role"] == "target" for f in seg_a[last + 1:] for d in f["dets"])

    # The carried cover is the cover track with the largest displacement during episode 1.
    covers = _track_centres(seg_a[:n_ep1], "cover")
    disp = {tid: np.linalg.norm(c[max(c)] - c[min(c)]) for tid, c in covers.items() if len(c) >= 2}
    if disp:
        moved = max(disp, key=disp.get)
        if disp[moved] >= MOVE_PX:
            c = covers[moved]
            ev["moved_cover_track"] = int(moved)
            ks = sorted(c)
            ev["carry_start"] = next((k for k in ks if np.linalg.norm(c[k] - c[ks[0]]) > MOVE_PX), None)
            ev["covered"] = next((k for k in ks if k > last and np.linalg.norm(c[k] - c[ks[-1]]) < SETTLE_PX), ks[-1])
    return ev


def _containment(seg_a: list, ev: dict) -> dict | None:
    """Parent = cover track at the end of episode 1 closest to where the target disappeared."""
    if not ev["hidden"]:
        return None
    last = ev["target_last_seen"]
    target = next(d for d in seg_a[last]["dets"] if d["role"] == "target")
    p = centre(target["box"])
    n_ep1 = sum(f["stream"] == "episode1" for f in seg_a)
    end = seg_a[n_ep1 - 1]["dets"]
    scored = []
    for d in end:
        if d["role"] != "cover":
            continue
        h = d["box"][3] - d["box"][1]
        scored.append((np.linalg.norm(centre(d["box"]) - p) / max(h, 1.0), d))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    best_dist, best = scored[0]
    margin = scored[1][0] - best_dist if len(scored) > 1 else float("inf")
    return {
        "object_track": int(target["track_id"]),
        "parent_track": int(best["track_id"]),
        "parent_box_end_ep1": best["box"],
        "normalised_distance": round(float(best_dist), 3),
        "margin": None if margin == float("inf") else round(float(margin), 3),
        "agrees_with_moved_cover": best["track_id"] == ev["moved_cover_track"],
    }


def _keyframes(seg_a: list, ev: dict, n_after: int) -> list:
    n_ep1 = sum(f["stream"] == "episode1" for f in seg_a)
    picks = [ev[k] for k in ("target_first_seen", "carry_start", "target_last_seen", "covered") if ev.get(k) is not None]
    if not picks:  # no events found: spread over episode 1, still without simulator information
        picks = list(np.linspace(0, n_ep1 - 1, MAX_KEYFRAMES - 1).round().astype(int))
    keys = [{"stream": "episode1", "idx": int(k)} for k in dict.fromkeys(picks)]
    if n_after:
        keys.append({"stream": "sweep_after", "idx": 0})
    return keys[:MAX_KEYFRAMES]


def _query_candidates(seg_b: list) -> list:
    """Cover tracks of segment B, with their box in the query image. A confirmed cover
    track that the detector misses in the query image takes its box from the closest
    preceding frame (the end of the after-sweep, near the query viewpoint)."""
    boxes = {}
    for f in seg_b:  # later frames overwrite earlier ones -> latest box per track
        for d in f["dets"]:
            if d["role"] == "cover":
                boxes[d["track_id"]] = {"box": d["box"], "score": d["score"],
                                        "from_query": f["stream"] == "query"}
    covers = sorted(boxes.items(), key=lambda kv: centre(kv[1]["box"])[0])
    return [{"number": i + 1, "track_id": int(tid), **v} for i, (tid, v) in enumerate(covers)]


def _draw_marks(query_path: Path, candidates: list, out_path: Path):
    im = Image.open(query_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for c in candidates:
        x0, y0, x1, y1 = c["box"]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=2)
        tx, ty = (x0 + x1) / 2 - 8, y0 - 28
        draw.rectangle([tx - 4, ty, tx + 20, ty + 26], fill=(0, 0, 0))
        draw.text((tx, ty + 1), str(c["number"]), fill=(255, 255, 255), font=font)
    im.save(out_path)


def process_trial(trial_dir, detector) -> dict:
    trial_dir = Path(trial_dir)
    seg_a = _track_segment(detector, trial_dir, STREAMS_A)
    seg_b = _track_segment(detector, trial_dir, STREAMS_B)
    ev = _events(seg_a)
    result = {
        "detector": detector.config(),
        "segments": {"A": seg_a, "B": seg_b},
        "events": ev,
        "containment": _containment(seg_a, ev),
        "keyframes": _keyframes(seg_a, ev, n_after=sum(f["stream"] == "sweep_after" for f in seg_b)),
        "query_candidates": _query_candidates(seg_b),
    }
    out = trial_dir / "perception"
    out.mkdir(exist_ok=True)
    (out / "perception.json").write_text(json.dumps(result, indent=1, default=float))
    _draw_marks(trial_dir / "inputs" / "query.png", result["query_candidates"], out / "query_marked.png")
    return result
