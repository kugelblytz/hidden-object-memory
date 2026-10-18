"""Ground-truth boxes from simulator segmentation, FOR EVALUATION ONLY.

Nothing in this module may be used by the perception pipeline or the memory
systems; it reads ``gt/`` which holds privileged simulator information.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

MIN_PIXELS = 30  # objects with fewer visible pixels count as not visible


@lru_cache(maxsize=4)
def _load(trial_dir: str):
    gt = Path(trial_dir) / "gt"
    return json.loads((gt / "meta.json").read_text()), dict(np.load(gt / "frames.npz"))


def gt_meta(trial_dir) -> dict:
    return _load(str(trial_dir))[0]


def gt_boxes(trial_dir, stream: str, idx: int) -> list:
    """Visible objects in one frame: [{"name", "role", "box", "pixels"}]."""
    meta, frames = _load(str(trial_dir))
    seg = frames[f"{stream}_seg"][idx]
    out = []
    for name, sid in meta["segmentation_ids"].items():
        mask = seg == sid
        n = int(mask.sum())
        if n < MIN_PIXELS:
            continue
        ys, xs = np.nonzero(mask)
        role = "cover" if name.startswith("cover") else "target" if name == "target_ball" else "distractor"
        out.append({"name": name, "role": role, "box": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                    "pixels": n})
    return out
