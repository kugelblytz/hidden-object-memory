"""Open-vocabulary object detection with text prompts (Ultralytics YOLOE).

Every prompt is mapped to a role used by the memory systems:
``cover`` (candidate containers), ``target`` (the object to be hidden) or
``distractor``. Non-maximum suppression runs across classes, so one object
yields one detection even if several prompts fire on it.
"""
from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLOE

# Chosen on pilot data (select_detector.py): "cup" alone also fires on cylindrical
# cans, so covers are prompted as mugs and several can types are named as distractors.
DEFAULT_PROMPTS = {
    "upside-down mug": "cover",
    "mug": "cover",
    "small yellow ball": "target",
    "soda can": "distractor",
    "food can": "distractor",
    "tin can": "distractor",
    "box": "distractor",
    "cube": "distractor",
}
# Per-role minimum scores on top of the global threshold: the target is detected with
# high confidence, and low-score target detections after covering would mask the event.
DEFAULT_ROLE_CONF = {"target": 0.3}


@dataclass
class Detection:
    role: str
    label: str
    score: float
    box: list  # xyxy in pixels
    track_id: int = -1
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"role": self.role, "label": self.label, "score": round(self.score, 4),
                "box": [round(v, 1) for v in self.box], "track_id": self.track_id}


class Detector:
    def __init__(self, weights: str = "yoloe-11l-seg.pt", prompts: dict | None = None,
                 conf: float = 0.05, device: str = "cpu", imgsz: int = 640, role_conf: dict | None = None):
        self.prompts = dict(prompts or DEFAULT_PROMPTS)
        self.role_conf = dict(DEFAULT_ROLE_CONF if role_conf is None else role_conf)
        self.names = list(self.prompts)
        self.model = YOLOE(weights)
        self.model.set_classes(self.names, self.model.get_text_pe(self.names))
        self.conf, self.device, self.imgsz, self.weights = conf, device, imgsz, weights

    def config(self) -> dict:
        return {"weights": self.weights, "prompts": self.prompts, "conf": self.conf,
                "role_conf": self.role_conf, "imgsz": self.imgsz}

    def detect(self, image) -> list[Detection]:
        """image: path or HxWx3 RGB array."""
        if isinstance(image, np.ndarray):
            image = image[..., ::-1]  # Ultralytics expects BGR arrays
        r = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, agnostic_nms=True,
                               verbose=False, device=self.device)[0]
        dets = []
        for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            label = self.names[int(c)]
            if s < self.role_conf.get(self.prompts[label], 0.0):
                continue
            dets.append(Detection(role=self.prompts[label], label=label, score=float(s), box=[float(v) for v in b]))
        return dets
