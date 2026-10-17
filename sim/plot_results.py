"""Figures for RESULTS.md from run_all.py's summary.json.

Usage: .venv\\Scripts\\python.exe plot_results.py --summary ../results/test_summary.json --out ../results
Palette: reference categorical slots 1-5 (validated: CVD/normal-vision pass; contrast relief
via direct labels, distinct markers and the tables in RESULTS.md).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [  # fixed order: entity -> slot
    ("vggt_joint", "VGGT memory (joint pass)", "#2a78d6", "o"),
    ("vggt_separate", "VGGT memory (separate passes)", "#eb6834", "s"),
    ("2d", "2D memory (homography)", "#1baf7a", "D"),
    ("vlm:Qwen3.5-9B:keyframe", "Keyframes, Qwen3.5-9B", "#eda100", "^"),
    ("vlm:Qwen3-VL-8B-Instruct:keyframe", "Keyframes, Qwen3-VL-8B", "#e87ba4", "v"),
]
OVERALL = [
    ("vggt_joint", "VGGT memory, joint pass", True),
    ("vggt_separate", "VGGT memory, separate passes", True),
    ("2d", "2D memory (homography)", True),
    ("2d_noreg", "2D, raw positions (diagnostic)", False),
    ("vlm:Qwen3-VL-8B-Instruct:keyframe", "Keyframes, Qwen3-VL-8B", True),
    ("vlm:Qwen3.5-9B:keyframe", "Keyframes, Qwen3.5-9B", True),
    ("vlm:Qwen3.5-9B:current_only", "Current-only, Qwen3.5-9B", True),
    ("vlm:Qwen3-VL-8B-Instruct:current_only", "Current-only, Qwen3-VL-8B", True),
]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_angle(s, out: Path):
    bins = [k for k in s["conditions"]["2d"]["bins"] if k.startswith("camera")]
    labels = [f"{b.split()[1].replace('-151', '-150').replace('-', '–')}°\n(n={s['conditions']['2d']['bins'][b]['n']})"
              for b in bins]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=SURFACE)
    style(ax)
    x = list(range(len(bins)))
    ax.axhline(1 / 3, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(len(bins) - 1 + 0.15, 1 / 3 + 0.015, "chance (1/3)", color=MUTED, fontsize=8, va="bottom")
    for key, name, colour, marker in SERIES:
        y = [s["conditions"][key]["bins"][b]["acc"] for b in bins]
        ax.plot(x, y, color=colour, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.3, len(bins) - 1 + 0.9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (first choice)", color=INK2, fontsize=9)
    ax.set_xlabel("Camera viewpoint change between episodes", color=INK2, fontsize=9)
    ax.set_title("Camera-change trials: accuracy by size of the viewpoint change", color=INK, fontsize=11,
                 loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_accuracy_by_angle.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fig_overall(s, out: Path):
    rows = [(n, s["conditions"][k]["all"], main) for k, n, main in OVERALL if k in s["conditions"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axvline(1 / 3, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(1 / 3 + 0.01, -0.75, "chance (1/3)", color=MUTED, fontsize=8, va="center")
    ax.set_ylim(-1.0, len(rows) - 0.5)
    for i, (name, b, main) in enumerate(reversed(rows)):
        colour = "#2a78d6" if main else MUTED
        ax.plot(b["ci"], [i, i], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(b["acc"], i, "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.text(b["ci"][1] + 0.015, i, f"{b['acc']:.2f}", color=INK2, fontsize=8, va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], color=INK2)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Accuracy with 95% bootstrap CI (180 test trials)", color=INK2, fontsize=9)
    ax.set_title("Overall accuracy by condition", color=INK, fontsize=11, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out / "fig_overall_accuracy.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    s = json.loads(args.summary.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    fig_angle(s, args.out)
    fig_overall(s, args.out)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
