"""Frame-based baselines answered by a vision-language model (conditions 1 and 2).

current_only  the numbered query image only
keyframe      up to five event-triggered keyframes from the shared perception
              (episode 1 events plus the first frame after the change) + the numbered query image

Both conditions use the same candidate numbering, prompt structure and deterministic
decoding. Bundles (image list + prompt) are written to trial/baselines/<condition>.json so
that a cluster job can run them without re-running perception.
"""
import json
import re
from pathlib import Path

CONDITIONS = ("current_only", "keyframe")

TASK = ("A small yellow ball was hidden under one of the upside-down cups on the table. "
        "In the final image the cups are marked with numbers. Which numbered cup hides the yellow ball? "
        "Answer with the number only. If you cannot tell, answer 'unknown'.")


def frame_path(trial_dir: Path, stream: str, idx: int) -> Path:
    return trial_dir / "inputs" / ("query.png" if stream == "query" else f"{stream}/{idx:03d}.png")


def build_bundle(trial_dir, perception: dict, condition: str) -> dict:
    trial_dir = Path(trial_dir)
    query = trial_dir / "perception" / "query_marked.png"
    numbers = [c["number"] for c in perception["query_candidates"]]
    if condition == "current_only":
        images = [query]
        intro = "The image shows the current scene."
    elif condition == "keyframe":
        before = [k for k in perception["keyframes"] if k["stream"] == "episode1"]
        after = [k for k in perception["keyframes"] if k["stream"] != "episode1"]
        images = [frame_path(trial_dir, k["stream"], k["idx"]) for k in before + after] + [query]
        n_b, n_a = len(before), len(after)
        intro = (f"Images 1 to {n_b} are earlier frames in chronological order from a first episode. "
                 + (f"Image {n_b + 1} was taken after a pause during which the scene was not observed; the camera "
                    f"or the cups may have moved in between. " if n_a else "")
                 + f"The final image (image {n_b + n_a + 1}) shows the current scene.")
    else:
        raise ValueError(condition)
    # Paths relative to the trial folder, so bundles stay valid when trials are copied to a cluster.
    bundle = {"condition": condition, "images": [p.relative_to(trial_dir).as_posix() for p in images], "candidates": numbers,
              "prompt": f"{intro} {TASK} Possible answers: {', '.join(map(str, numbers))} or unknown."}
    out = trial_dir / "baselines"
    out.mkdir(exist_ok=True)
    (out / f"{condition}.json").write_text(json.dumps(bundle, indent=1))
    return bundle


def parse_answer(text: str, numbers: list) -> int | None:
    """First candidate number mentioned in the reply; None for 'unknown' or anything else."""
    if text is None:
        return None
    for tok in re.findall(r"\d+", text):
        if int(tok) in numbers:
            return int(tok)
    return None


class DryRunBackend:
    """Writes bundles only; used to prepare cluster runs."""
    name = "dry_run"

    def generate(self, images: list, prompt: str) -> str | None:
        return None


class HFBackend:
    """Any chat VLM loadable with transformers' AutoModelForImageTextToText
    (e.g. Qwen/Qwen2.5-VL-7B-Instruct on the cluster, HuggingFaceTB/SmolVLM-256M-Instruct for
    a local plumbing test). Greedy decoding."""

    def __init__(self, model_id: str, device: str = "cpu", max_new_tokens: int = 8):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name, self.device, self.max_new_tokens = model_id, device, max_new_tokens
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    def generate(self, images: list, prompt: str) -> str:
        import torch
        from PIL import Image

        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        pil = [Image.open(p).convert("RGB") for p in images]
        inputs = self.processor(text=text, images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
