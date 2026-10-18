"""Smoke test: can ManiSkill3 + MIKASA ShellGame run and render on this machine?

Renders the 512 px human-render camera at several steps of one episode and
saves PNGs to out/smoke/. Run: .venv\\Scripts\\python.exe -m tools.smoke_test_shellgame
"""
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import mikasa_robo_suite.vla.memory_envs.shell_game_shuffle_touch_vla  # noqa: F401 (registers env)

OUT = Path(__file__).resolve().parent.parent / "out" / "smoke"  # sim/out/smoke
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
env = gym.make(
    "ShellGameShuffleTouch-VLA-v0",
    num_envs=1,
    obs_mode="rgb",
    render_mode="rgb_array",
    sim_backend="cpu",
    # The default "gpu" render backend uses CUDA interop and crashes with CPU-only torch;
    # "cpu" still renders on the GPU via Vulkan and copies images through host memory.
    render_backend="cpu",
)
obs, info = env.reset(seed=0)
print(f"env ready in {time.time() - t0:.1f}s")

save_at = {0, 2, 6, 12, 20, 30, 45}
for step in range(46):
    if step in save_at:
        t1 = time.time()
        frame = env.render()
        frame = frame[0] if frame.ndim == 4 else frame
        frame = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
        Image.fromarray(frame.astype(np.uint8)).save(OUT / f"step_{step:02d}.png")
        print(f"step {step:2d}: saved {frame.shape}, render {time.time() - t1:.2f}s")
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

print("ball under cup (oracle):", int(env.unwrapped.oracle_info.flatten()[0]))
env.close()
print(f"total {time.time() - t0:.1f}s")
