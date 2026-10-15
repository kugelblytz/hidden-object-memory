"""Procedural, non-repetitive textures for the room walls and floor.

Feature matching on the static background (2D homography, VGGT Sim(3) alignment)
needs texture that is rich and does not repeat, so every wall gets its own
seeded mix of fractal noise and random shapes. Textures are cached as PNG files.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CACHE_DIR = Path(__file__).resolve().parent.parent / "out" / "textures"


def _fractal_noise(rng: np.random.Generator, size: int, octaves: int = 6) -> np.ndarray:
    out = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = 2 ** (o + 2)
        small = (rng.random((n, n)) * 255).astype(np.uint8)
        layer = Image.fromarray(small).resize((size, size), Image.BICUBIC)
        out += amp * np.asarray(layer, np.float32) / 255.0
        total += amp
        amp *= 0.55
    return out / total


def make_texture(seed: int, style: str = "wall", size: int = 1024) -> np.ndarray:
    """Return an RGB uint8 texture. style is "wall" (large shapes) or "floor" (fine speckle)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.3, 0.7, 3)
    tint = rng.uniform(-0.3, 0.3, 3)
    noise = _fractal_noise(rng, size)[..., None]
    img = np.clip(base + tint * (noise - 0.5) * 2.0, 0.0, 1.0)
    im = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(im, "RGBA")

    n_shapes, max_extent = (260, 0.10) if style == "wall" else (900, 0.025)
    for _ in range(n_shapes):
        colour = tuple(int(v) for v in rng.integers(0, 256, 3)) + (int(rng.integers(90, 230)),)
        x, y = rng.uniform(0, size, 2)
        w, h = rng.uniform(size * 0.004, size * max_extent, 2)
        kind = rng.integers(4)
        if kind == 0:
            draw.rectangle([x, y, x + w, y + h], fill=colour)
        elif kind == 1:
            draw.ellipse([x, y, x + w, y + h], fill=colour)
        elif kind == 2:
            end = (x + rng.uniform(-2, 2) * w, y + rng.uniform(-2, 2) * h)
            draw.line([(x, y), end], fill=colour, width=int(rng.integers(1, 7)))
        else:
            pts = [(x + rng.uniform(-w, w), y + rng.uniform(-h, h)) for _ in range(int(rng.integers(3, 7)))]
            draw.polygon(pts, fill=colour)
    return np.asarray(im.convert("RGB"))


def texture_file(seed: int, style: str = "wall") -> str:
    """Create (once) and return the path of a cached texture PNG."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{style}_{seed}.png"
    if not path.exists():
        Image.fromarray(make_texture(seed, style)).save(path)
    return str(path)
