"""Visual failure analysis for the 2D memory.

For every trial writes trial/memory_debug.jpg and an overview <data>/memory_report.html
(failures first). Each image shows
  top left   before the change: covers A/B/C, the one hiding the target in yellow
  top right  after the change: old footprints mapped by the homography (crosses),
             assignment lines, the chosen cover (red) and the correct cover (green)
  bottom     background feature matches between the two views (green = RANSAC
             inliers used for the homography, red = rejected)
Ground truth is used here only to mark the correct cover.
Usage: .venv\\Scripts\\python.exe viz_memory.py --data out/pilot
"""
import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np

from memory.memory2d import answer, footprint
from perception.gt import gt_meta

OLD_COLOURS = [(0, 140, 255), (255, 200, 0), (255, 0, 255), (0, 255, 255)]  # BGR per old cover
RED, GREEN, YELLOW, WHITE = (0, 0, 255), (0, 200, 0), (0, 230, 255), (255, 255, 255)


def _box(img, box, colour, thickness=2, label=None):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x0, y0), (x1, y1), colour, thickness)
    if label:
        cv2.putText(img, label, (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


def _header(width, lines):
    h = np.full((22 * len(lines) + 10, width, 3), 30, np.uint8)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(h, text, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
    return h


def render(trial: Path, perception: dict, res: dict, correct_number, title_lines) -> np.ndarray:
    dbg = res.get("_debug")
    if not dbg or dbg.get("fa") is None:
        return _header(1280, title_lines + [("no registration frames available", RED)])
    fa, fb = dbg["fa"], dbg["fb"]
    left, right = fa.img.copy(), fb.img.copy()
    parent = (perception["containment"] or {}).get("parent_track")
    old = dbg.get("old") or fa.covers()
    letters = {}
    for k, (tid, d) in enumerate(sorted(old.items(), key=lambda kv: footprint(kv[1]["box"])[0])):
        letters[tid] = (chr(65 + k), OLD_COLOURS[k % len(OLD_COLOURS)])
        is_parent = tid == parent
        _box(left, d["box"], YELLOW if is_parent else letters[tid][1], 4 if is_parent else 2,
             letters[tid][0] + (" (has target)" if is_parent else ""))

    new = dbg.get("new") or fb.covers()
    numbers = {c["track_id"]: c["number"] for c in perception["query_candidates"]}
    correct_tid = next((c["track_id"] for c in perception["query_candidates"] if c["number"] == correct_number), None)
    for tid, d in new.items():
        label = f"#{numbers.get(tid, '?')}"
        if tid == correct_tid:
            _box(right, d["box"], GREEN, 4, label + " correct")
        if tid == dbg.get("new_parent"):
            _box(right, [v + (3 if i < 2 else -3) for i, v in enumerate(d["box"])], RED, 3, label + " chosen")
        if tid not in (correct_tid, dbg.get("new_parent")):
            _box(right, d["box"], WHITE, 1, label)
    for tid, p in (dbg.get("mapped") or {}).items():
        letter, colour = letters.get(tid, ("?", WHITE))
        x, y = int(p[0]), int(p[1])
        cv2.drawMarker(right, (x, y), colour, cv2.MARKER_CROSS, 22, 3)
        cv2.putText(right, letter, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
        tn = dbg["assign"].get(tid)
        if tn in new:
            q = footprint(new[tn]["box"])
            cv2.line(right, (x, y), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)
    top = np.hstack([left, right])

    w = fa.img.shape[1]
    bottom = np.hstack([fa.img, fb.img]) // 2 + 60
    rng = np.random.default_rng(0)
    matches = dbg.get("matches") or []
    inl = [m for m in matches if m[2]]
    out = [m for m in matches if not m[2]]
    for group, colour in ((out, (60, 60, 255)), (inl, (0, 255, 0))):
        if len(group) > 120:
            group = [group[i] for i in rng.choice(len(group), 120, replace=False)]
        for s, d, _ in group:
            cv2.line(bottom, (int(s[0]), int(s[1])), (int(d[0]) + w, int(d[1])), colour, 1, cv2.LINE_AA)
    cv2.putText(bottom, f"background matches: {len(inl)} inliers (green) / {len(out)} rejected (red)",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
    return np.vstack([_header(top.shape[1], title_lines), top, bottom])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/pilot")
    args = ap.parse_args()
    data = Path(args.data)
    entries = []
    for trial in sorted(data.glob("trial_*")):
        perception = json.loads((trial / "perception" / "perception.json").read_text())
        correct = json.loads((trial / "gt" / "candidates.json").read_text())["correct_candidate"]
        meta = gt_meta(trial)
        parent = (perception["containment"] or {}).get("parent_track")
        res = answer(trial, perception, parent, debug=True)
        res_noreg = answer(trial, perception, parent, register=False)
        ok = res["candidate"] is not None and res["candidate"] == correct
        ok_noreg = res_noreg["candidate"] is not None and res_noreg["candidate"] == correct
        mag = meta["change"].get("magnitude")
        mag_txt = "-" if mag is None else (f"{mag:.0f} deg" if meta["motion"] == "camera" else f"{mag * 100:.0f} cm")
        lines = [
            (f"{trial.name}  {meta['appearance']} / {meta['motion']}  change: {mag_txt}", WHITE),
            (f"2D memory: {'CORRECT' if ok else 'WRONG'} (answer #{res['candidate']}, correct #{correct}, "
             f"{res['reason']}, inliers {res.get('inliers', '-')}, pair {res.get('pair', '-')})", GREEN if ok else RED),
            (f"raw positions (no registration): {'correct' if ok_noreg else 'wrong'} (answer #{res_noreg['candidate']})",
             GREEN if ok_noreg else RED),
        ]
        img = render(trial, perception, res, correct, lines)
        cv2.imwrite(str(trial / "memory_debug.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        entries.append({"trial": trial.name, "ok": ok, "ok_noreg": ok_noreg, "cell": f"{meta['appearance']}/{meta['motion']}",
                        "mag": mag_txt, "reason": res["reason"], "inliers": res.get("inliers", "-")})
        print(f"{trial.name}: {'ok' if ok else 'WRONG'} ({res['reason']})", flush=True)

    entries.sort(key=lambda e: (e["ok"], e["cell"], e["trial"]))
    n_ok = sum(e["ok"] for e in entries)
    parts = [f"<html><head><meta charset='utf-8'><title>2D memory report</title><style>"
             "body{font-family:sans-serif;background:#111;color:#eee;margin:16px}img{max-width:100%}"
             "table{border-collapse:collapse}td,th{border:1px solid #444;padding:4px 8px}"
             ".bad{color:#f66}.good{color:#6c6}</style></head><body>",
             f"<h1>2D memory: {n_ok}/{len(entries)} correct</h1>",
             "<p>Top left: before the change (yellow = cover hiding the target). Top right: after the change "
             "(crosses = old covers mapped by the homography, red = chosen, green = correct). "
             "Bottom: background matches used for the homography.</p><table><tr><th>trial</th><th>cell</th>"
             "<th>change</th><th>2D</th><th>raw positions</th><th>reason</th><th>inliers</th></tr>"]
    for e in entries:
        parts.append(f"<tr><td><a href='#{e['trial']}'>{e['trial']}</a></td><td>{e['cell']}</td><td>{e['mag']}</td>"
                     f"<td class={'good' if e['ok'] else 'bad'}>{'ok' if e['ok'] else 'WRONG'}</td>"
                     f"<td class={'good' if e['ok_noreg'] else 'bad'}>{'ok' if e['ok_noreg'] else 'wrong'}</td>"
                     f"<td>{html.escape(str(e['reason']))}</td><td>{e['inliers']}</td></tr>")
    parts.append("</table>")
    for e in entries:
        t = e["trial"]
        parts.append(f"<h2 id='{t}' class={'good' if e['ok'] else 'bad'}>{t} - {e['cell']} - {e['mag']} - "
                     f"{'ok' if e['ok'] else 'WRONG'}</h2><p><a href='{t}/recording.gif'>recording</a> | "
                     f"<a href='{t}/preview.jpg'>preview</a> | <a href='{t}/perception/query_marked.png'>query candidates</a></p>"
                     f"<img src='{t}/memory_debug.jpg'>")
    parts.append("</body></html>")
    (data / "memory_report.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"report: {data / 'memory_report.html'}")


if __name__ == "__main__":
    main()
