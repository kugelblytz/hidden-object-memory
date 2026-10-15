"""Trial sampling and recording for the hidden-object experiment.

One trial = two task episodes with an unobserved change in between:

  episode1     the target ball is visible, an invisible external actor lifts the
               target cover, carries it over the ball and lowers it (main camera)
  sweep_before ~10 frames on a short arc around the main camera pose
  (change)     not recorded: fixed / camera viewpoint change / one cover moved
  sweep_after  ~10 frames on a short arc around the (possibly new) main camera pose
  query        final image of episode 2 from the main camera

Only RGB frames go to ``inputs/``; layout, poses, camera parameters, depth and
segmentation go to ``gt/`` and are meant for scoring and diagnostics only.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

PALETTE = {  # matte, mutually distinct, none close to the yellow target ball
    "red": [0.62, 0.16, 0.13],
    "blue": [0.16, 0.30, 0.62],
    "green": [0.20, 0.50, 0.26],
    "grey": [0.50, 0.50, 0.52],
    "brown": [0.55, 0.36, 0.18],
    "purple": [0.42, 0.26, 0.52],
}
SCENES = [  # two or three tabletop scenarios: room textures + distractor pool
    {"scene_id": 0, "texture_seed": 100, "distractors": ["077_rubiks_cube", "005_tomato_soup_can"]},
    {"scene_id": 1, "texture_seed": 200, "distractors": ["061_foam_brick", "010_potted_meat_can"]},
    {"scene_id": 2, "texture_seed": 300, "distractors": ["009_gelatin_box", "036_wood_block"]},
]
APPEARANCES = ("identical", "distinct")
MOTIONS = ("fixed", "camera", "cover")

# Geometry (metres / degrees); tabletop surface is z = 0.
SLOT_X, SLOT_Y = (-0.12, 0.12), (-0.25, 0.25)
TABLE_X, TABLE_Y = (-0.30, 0.30), (-0.42, 0.42)
MIN_COVER_SPACING = 0.17
MIN_CLEARANCE = 0.15
APPROACH_DIST = 0.16
LOOK_AT = np.array([0.0, 0.0, 0.03])
CAM_AZ0, CAM_EL, CAM_R = (-20.0, 20.0), (30.0, 45.0), (0.95, 1.10)
CAM_DELTA = (10.0, 60.0)
COVER_SHIFT = (0.05, 0.15)
SWEEP_FRAMES, SWEEP_HALF_ARC, SWEEP_EL_WOBBLE = 10, 10.0, 3.0
LIFT = 0.13
EP1_PHASES = {"hold_start": 4, "lift": 5, "carry": 8, "lower": 5, "hold_end": 4}
MIN_COVER_PIXELS, MIN_BALL_PIXELS_START = 400, 20


# ---------------------------------------------------------------- sampling
def _dist_point_segment(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _in_box(p, xr, yr) -> bool:
    return xr[0] <= p[0] <= xr[1] and yr[0] <= p[1] <= yr[1]


def camera_eye(az: float, el: float, r: float) -> np.ndarray:
    """Camera position on a sphere around LOOK_AT; azimuth/elevation in degrees, radius in metres."""
    a, e = np.deg2rad(az), np.deg2rad(el)
    return LOOK_AT + r * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def sample_trial(trial_id: int, seed: int, scene_id: int, appearance: str, motion: str, max_tries: int = 500) -> dict:
    """Sample one trial layout deterministically from ``seed``."""
    assert appearance in APPEARANCES and motion in MOTIONS
    rng = np.random.default_rng(seed)
    scene = SCENES[scene_id]

    names = list(PALETTE)
    if appearance == "identical":
        colour_names = [names[rng.integers(len(names))]] * 3
    else:
        colour_names = [names[i] for i in rng.choice(len(names), 3, replace=False)]

    target = int(rng.integers(3))
    cam0 = {"az": float(rng.uniform(*CAM_AZ0)), "el": float(rng.uniform(*CAM_EL)), "r": float(rng.uniform(*CAM_R))}
    # The target cover approaches the ball from the side facing away from the camera.
    away = -camera_eye(cam0["az"], cam0["el"], cam0["r"])[:2]
    away /= np.linalg.norm(away)

    # Three cover slots with a minimum spacing and a free approach path for the target
    # cover; layouts without such a path are redrawn.
    for _ in range(max_tries):
        slots = np.column_stack([rng.uniform(*SLOT_X, 3), rng.uniform(*SLOT_Y, 3)])
        if min(np.linalg.norm(slots[i] - slots[j]) for i in range(3) for j in range(i + 1, 3)) < MIN_COVER_SPACING:
            continue
        others = [slots[i] for i in range(3) if i != target]
        for ang in np.deg2rad(rng.permutation(np.arange(-90, 91, 15))):
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            start = slots[target] + APPROACH_DIST * (rot @ away)
            if _in_box(start, TABLE_X, TABLE_Y) and all(
                _dist_point_segment(o, start, slots[target]) >= MIN_CLEARANCE for o in others
            ):
                break
        else:
            continue
        break
    else:
        raise RuntimeError("could not sample cover slots with a free approach path")

    distractors = []
    n_distr = int(rng.integers(1, 3))
    for model_id in list(rng.permutation(scene["distractors"]))[:n_distr]:
        for _ in range(max_tries):
            p = np.array([rng.uniform(*TABLE_X), rng.uniform(*TABLE_Y)]) * 0.85
            blockers = list(slots) + [start] + [np.array(d["xy"]) for d in distractors]
            if all(np.linalg.norm(p - b) >= MIN_CLEARANCE for b in blockers):
                distractors.append({"model_id": str(model_id), "xy": p.tolist(), "yaw": float(rng.uniform(0, 2 * np.pi))})
                break

    final = slots.copy()  # cover positions after the change (episode 2)
    cam1 = dict(cam0)
    change = {"type": motion}
    if motion == "camera":
        delta = float(rng.uniform(*CAM_DELTA) * rng.choice([-1.0, 1.0]))
        cam1["az"] = cam0["az"] + delta
        change.update({"delta_az_deg": delta, "magnitude": abs(delta)})
    elif motion == "cover":
        blockers = [np.array(d["xy"]) for d in distractors]
        for _ in range(max_tries):
            k = int(rng.integers(3))
            dist, ang = rng.uniform(*COVER_SHIFT), rng.uniform(0, 2 * np.pi)
            new = slots[k] + dist * np.array([np.cos(ang), np.sin(ang)])
            rest = [slots[i] for i in range(3) if i != k]
            # bounded move, stays on the table, keeps spacing and never passes another cover
            if (
                _in_box(new, TABLE_X, TABLE_Y)
                and all(np.linalg.norm(new - o) >= MIN_COVER_SPACING for o in rest)
                and all(_dist_point_segment(o, slots[k], new) >= MIN_CLEARANCE for o in rest + blockers)
            ):
                final[k] = new
                change.update({"cover": k, "shift": (new - slots[k]).tolist(), "magnitude": float(dist)})
                break
        else:
            raise RuntimeError("could not sample a cover move")

    return {
        "trial_id": trial_id,
        "seed": seed,
        "scene": scene,
        "appearance": appearance,
        "motion": motion,
        "cover_color_names": colour_names,
        "cover_colors": [PALETTE[c] for c in colour_names],
        "cover_yaw": float(rng.uniform(0, 2 * np.pi)),  # shared yaw: the handle is no identity cue
        "target_cover": target,
        "slots_ep1": slots.tolist(),
        "target_start": start.tolist(),
        "ball_xy": slots[target].tolist(),
        "slots_ep2": final.tolist(),
        "ball_xy_ep2": final[target].tolist(),
        "distractors": distractors,
        "camera_ep1": cam0,
        "camera_ep2": cam1,
        "change": change,
    }


# --------------------------------------------------------------- recording
def _smooth(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return t * t * (3.0 - 2.0 * t)


def _sweep_poses(cam: dict) -> list:
    offsets = np.linspace(-SWEEP_HALF_ARC, SWEEP_HALF_ARC, SWEEP_FRAMES)
    wobble = SWEEP_EL_WOBBLE * np.sin(np.linspace(0, 2 * np.pi, SWEEP_FRAMES))
    return [camera_eye(cam["az"] + o, cam["el"] + w, cam["r"]) for o, w in zip(offsets, wobble)]


def record_trial(env, trial: dict, out_dir: Path) -> dict:
    """Render all phases of ``trial`` into ``out_dir``; returns validity checks."""
    env.reset(seed=trial["seed"], options={"reconfigure": True, "trial": trial})
    u = env.unwrapped
    seg_ids = u.segmentation_ids()
    target = trial["target_cover"]
    slots1, slots2 = np.array(trial["slots_ep1"]), np.array(trial["slots_ep2"])
    frames = {"episode1": [], "sweep_before": [], "sweep_after": [], "query": []}

    def shoot(stream: str, eye):
        u.set_camera(eye, LOOK_AT)
        frames[stream].append(u.capture())

    # Episode 1: the ball is visible, then the target cover is carried over it.
    u.place_distractors()
    u.place_ball(trial["ball_xy"])
    for i in range(3):
        u.place_cover(i, slots1[i])
    start, goal = np.array(trial["target_start"]), slots1[target]
    eye0 = camera_eye(**trial["camera_ep1"])
    path = (
        [(start, 0.0)] * EP1_PHASES["hold_start"]
        + [(start, LIFT * s) for s in _smooth(EP1_PHASES["lift"])]
        + [(start + (goal - start) * s, LIFT) for s in _smooth(EP1_PHASES["carry"])]
        + [(goal, LIFT * (1 - s)) for s in _smooth(EP1_PHASES["lower"])]
        + [(goal, 0.0)] * EP1_PHASES["hold_end"]
    )
    for xy, lift in path:
        u.place_cover(target, xy, lift)
        shoot("episode1", eye0)

    for eye in _sweep_poses(trial["camera_ep1"]):
        shoot("sweep_before", eye)

    # Unobserved change between the episodes; a hidden ball moves with its cover.
    for i in range(3):
        u.place_cover(i, slots2[i])
    u.place_ball(trial["ball_xy_ep2"])

    for eye in _sweep_poses(trial["camera_ep2"]):
        shoot("sweep_after", eye)
    shoot("query", camera_eye(**trial["camera_ep2"]))

    # Validity: ball visible at the start, hidden at query time, every cover visible at query time.
    q_seg, s_seg = frames["query"][0]["seg"], frames["episode1"][0]["seg"]
    cover_px = [int((q_seg == seg_ids[f"cover_{i}"]).sum()) for i in range(3)]
    checks = {
        "ball_px_start": int((s_seg == seg_ids["target_ball"]).sum()),
        "ball_px_query": int((q_seg == seg_ids["target_ball"]).sum()),
        "cover_px_query": cover_px,
    }
    checks["valid"] = (
        checks["ball_px_start"] >= MIN_BALL_PIXELS_START and checks["ball_px_query"] == 0 and min(cover_px) >= MIN_COVER_PIXELS
    )

    _write(out_dir, trial, frames, seg_ids, checks)
    return checks


def _write(out_dir: Path, trial: dict, frames: dict, seg_ids: dict, checks: dict):
    inputs, gt = out_dir / "inputs", out_dir / "gt"
    for stream, items in frames.items():
        if stream == "query":
            inputs.mkdir(parents=True, exist_ok=True)
            Image.fromarray(items[0]["rgb"]).save(inputs / "query.png")
        else:
            (inputs / stream).mkdir(parents=True, exist_ok=True)
            for k, f in enumerate(items):
                Image.fromarray(f["rgb"]).save(inputs / stream / f"{k:03d}.png")
    gt.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gt / "frames.npz",
        **{f"{s}_depth_mm": np.stack([np.round(f["depth"] * 1000).astype(np.uint16) for f in v]) for s, v in frames.items()},
        **{f"{s}_seg": np.stack([f["seg"].astype(np.uint16) for f in v]) for s, v in frames.items()},
        **{f"{s}_K": np.stack([f["K"] for f in v]) for s, v in frames.items()},
        **{f"{s}_extrinsic_cv": np.stack([f["extrinsic_cv"] for f in v]) for s, v in frames.items()},
    )
    meta = {**trial, "segmentation_ids": seg_ids, "checks": checks, "n_frames": {s: len(v) for s, v in frames.items()}}
    (gt / "meta.json").write_text(json.dumps(meta, indent=2))
    _preview(out_dir, frames)


def _preview(out_dir: Path, frames: dict):
    """Contact sheet of input frames for quick visual inspection (contains no ground truth)."""
    ep1 = frames["episode1"]
    picks = [ep1[0], ep1[len(ep1) // 3], ep1[len(ep1) // 2], ep1[-1],
             frames["sweep_before"][0], frames["sweep_after"][0], frames["query"][0]]
    h, w = picks[0]["rgb"].shape[:2]
    sheet = Image.new("RGB", (4 * w // 2, 2 * h // 2), "white")
    for k, f in enumerate(picks):
        tile = Image.fromarray(f["rgb"]).resize((w // 2, h // 2))
        sheet.paste(tile, ((k % 4) * w // 2, (k // 4) * h // 2))
    sheet.save(out_dir / "preview.jpg", quality=90)
