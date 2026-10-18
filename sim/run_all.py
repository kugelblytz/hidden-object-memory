"""Run the whole evaluation on recorded trials in one resumable job.

Stages (in order; each skips work that is already done unless --force):
  perception  shared RGB perception per trial + perception scoring (writes gt/candidates.json)
  memory2d    2D memory (+ diagnostics: no registration, GT containment)
  vggt        VGGT memory, separate passes and joint pass (+ GT-containment diagnostics);
              afterwards the VGGT model is released from GPU memory
  baselines   VLM baselines (current-only, keyframe) for every --vlm model, one model at a time
  report      unified evaluation: accuracy with bootstrap CIs, per cell, per motion size,
              primary contrasts (RQ1, RQ2), H2 interaction; writes results/report.html

Usage (cluster, 20 GB H100 slice):
  python run_all.py --data out/test --vlm Qwen/Qwen3.5-9B Qwen/Qwen3-VL-8B-Instruct --load-in-8bit
Local (light stages only):
  python run_all.py --data out/pilot --stages perception memory2d report

Results per trial: <data>/results/trials/<trial>.json; summary: <data>/results/summary.json.
Rendering is not part of this job (MIG GPU slices do not support Vulkan); record trials first.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

STAGES = ("perception", "memory2d", "vggt", "baselines", "report")
CAMERA_BINS = [(10, 40), (40, 70), (70, 100), (100, 151)]
COVER_BINS = [(0.10, 0.20), (0.20, 0.30), (0.30, 0.36)]
N_BOOT = 2000


# ------------------------------------------------------------------ helpers
def device_default() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def free_gpu():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class TrialStore:
    """Per-trial result file: {"perception": {...}, "conditions": {name: {...}}}."""

    def __init__(self, data: Path):
        self.dir = data / "results" / "trials"
        self.dir.mkdir(parents=True, exist_ok=True)

    def load(self, trial: Path) -> dict:
        p = self.dir / f"{trial.name}.json"
        return json.loads(p.read_text()) if p.exists() else {"perception": None, "conditions": {}}

    def save(self, trial: Path, rec: dict):
        (self.dir / f"{trial.name}.json").write_text(json.dumps(rec, indent=1, default=float))


def load_perception(trial: Path) -> dict:
    return json.loads((trial / "perception" / "perception.json").read_text())


def correct_candidate(trial: Path):
    return json.loads((trial / "gt" / "candidates.json").read_text())["correct_candidate"]


def scored(res: dict, correct) -> dict:
    res = {k: v for k, v in res.items() if not k.startswith("_")}
    res["correct"] = res.get("candidate") is not None and res["candidate"] == correct
    return res


# ------------------------------------------------------------------- stages
def stage_perception(trials, store, args):
    from perception.detector import Detector
    from perception.pipeline import process_trial
    from evaluation.scoring import score_perception

    detector = None
    for trial in trials:
        rec = store.load(trial)
        done = (trial / "perception" / "perception.json").exists()
        if done and rec["perception"] and not args.force:
            continue
        t0 = time.time()
        if done and not args.force:  # perception output exists: only score it
            res = load_perception(trial)
        else:
            detector = detector or Detector(device=args.device)
            res = process_trial(trial, detector)
        ev = score_perception(trial, res)  # perception scoring against GT; writes gt/candidates.json
        rec["perception"] = {k: v for k, v in ev.items() if k != "counts"}
        store.save(trial, rec)
        print(f"[perception] {trial.name} containment={'ok' if ev['containment_correct'] else 'WRONG'} "
              f"candidates={ev['n_candidates']} ({time.time() - t0:.1f}s)", flush=True)
    del detector
    free_gpu()


def _run_conditions(trials, store, args, variants: dict, tag: str):
    """variants: name -> callable(trial, perception, parent_track, gt_parent) -> result dict."""
    from evaluation.scoring import gt_parent_track

    for trial in trials:
        rec = store.load(trial)
        todo = [v for v in variants if args.force or v not in rec["conditions"]]
        if not todo:
            continue
        perception, correct = load_perception(trial), correct_candidate(trial)
        parent = (perception["containment"] or {}).get("parent_track")
        gt_parent = gt_parent_track(trial, perception)  # diagnostics only
        t0 = time.time()
        for v in todo:
            try:
                rec["conditions"][v] = scored(variants[v](trial, perception, parent, gt_parent), correct)
            except Exception as exc:  # keep the job alive; the failure is recorded and counts as wrong
                rec["conditions"][v] = {"candidate": None, "reason": f"error: {exc!r}", "correct": False}
            store.save(trial, rec)
        print(f"[{tag}] {trial.name} " + " ".join(
            f"{v}={'ok' if rec['conditions'][v]['correct'] else 'X'}" for v in todo) + f" ({time.time() - t0:.1f}s)", flush=True)


def stage_memory2d(trials, store, args):
    from memory import memory2d

    _run_conditions(trials, store, args, {
        "2d": lambda t, p, par, gp: memory2d.answer(t, p, par),
        "2d_noreg": lambda t, p, par, gp: memory2d.answer(t, p, par, register=False),
        "2d_gt_containment": lambda t, p, par, gp: memory2d.answer(t, p, gp),
    }, "memory2d")


def stage_vggt(trials, store, args):
    from memory import memory_vggt

    _run_conditions(trials, store, args, {
        "vggt_separate": lambda t, p, par, gp: memory_vggt.answer(t, p, par, mode="separate"),
        "vggt_joint": lambda t, p, par, gp: memory_vggt.answer(t, p, par, mode="joint"),
        "vggt_separate_gt_containment": lambda t, p, par, gp: memory_vggt.answer(t, p, gp, mode="separate"),
        "vggt_joint_gt_containment": lambda t, p, par, gp: memory_vggt.answer(t, p, gp, mode="joint"),
    }, "vggt")
    memory_vggt._MODEL = None  # release VGGT before any VLM is loaded
    free_gpu()


def stage_baselines(trials, store, args):
    import warnings

    from baselines.vlm import CONDITIONS, HFBackend, build_bundle, is_unknown, parse_answer

    # bitsandbytes warns on every 8-bit matmul; this flooded the log with ~800k lines.
    warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
    for model_id in args.vlm:
        short = model_id.split("/")[-1]
        names = {c: f"vlm:{short}:{c}" for c in CONDITIONS}
        pending = [t for t in trials if args.force or any(n not in store.load(t)["conditions"] for n in names.values())]
        if not pending:
            continue
        backend = HFBackend(model_id, args.device, load_in_8bit=args.load_in_8bit)
        for trial in pending:
            rec = store.load(trial)
            perception, correct = load_perception(trial), correct_candidate(trial)
            for cond, name in names.items():
                bundle = build_bundle(trial, perception, cond)
                images = [trial / p for p in bundle["images"]]
                reply = backend.generate(images, bundle["prompt"])
                ans, extracted = parse_answer(reply, bundle["candidates"]), None
                if ans is None and not is_unknown(reply):  # free text: ask for the bare answer
                    extracted = backend.extract(images, bundle["prompt"], reply, bundle["candidates"])
                    ans = parse_answer(extracted, bundle["candidates"])
                rec["conditions"][name] = scored({"candidate": ans, "reply": reply, "extracted": extracted,
                                                  "reason": "ok" if ans else "unknown"}, correct)
            store.save(trial, rec)
            print(f"[baselines] {short} {trial.name} " + " ".join(
                f"{c}={rec['conditions'][n]['reply']!r}" for c, n in names.items()), flush=True)
        del backend
        free_gpu()


# ------------------------------------------------------------------- report
def _bootstrap_ci(values: np.ndarray, scenes: np.ndarray, rng) -> tuple:
    """Mean and 95% percentile CI, resampling trials within each scene."""
    groups = [np.flatnonzero(scenes == s) for s in np.unique(scenes)]
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        boots[b] = values[idx].mean()
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), boots


def _paired(a: np.ndarray, b: np.ndarray, scenes, rng) -> dict:
    mean, lo, hi, boots = _bootstrap_ci(a.astype(float) - b.astype(float), scenes, rng)
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"diff": mean, "ci": [lo, hi], "p_boot": float(p)}


def _holm(ps: list) -> list:
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def _logit_interaction(success, is_b, angle, rng) -> dict:
    """Logistic regression success ~ 1 + is_b + angle + is_b*angle (angle in 100 deg);
    returns the interaction coefficient with a bootstrap CI (H2)."""
    from scipy.optimize import minimize

    def fit(y, g, x):
        X = np.column_stack([np.ones_like(x), g, x, g * x])

        def nll(w):
            z = X @ w
            return np.sum(np.logaddexp(0, z) - y * z) + 1e-3 * np.sum(w ** 2)  # tiny ridge for separation

        return minimize(nll, np.zeros(4), method="BFGS").x

    y, g, x = success.astype(float), is_b.astype(float), angle / 100.0
    w = fit(y, g, x)
    boots = []
    for _ in range(500):
        idx = rng.choice(len(y), len(y))
        boots.append(fit(y[idx], g[idx], x[idx])[3])
    return {"interaction": float(w[3]), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "coefficients": [float(v) for v in w]}


def stage_report(trials, store, args):
    from evaluation.gt import gt_meta

    rng = np.random.default_rng(0)
    rows = []
    for trial in trials:
        rec, meta = store.load(trial), gt_meta(trial)
        rows.append({"trial": trial.name, "scene": meta["scene"]["scene_id"], "appearance": meta["appearance"],
                     "motion": meta["motion"], "magnitude": meta["change"].get("magnitude") or 0.0,
                     "perception": rec["perception"] or {},
                     "correct": {k: v["correct"] for k, v in rec["conditions"].items()}})
    conditions = sorted({k for r in rows for k in r["correct"]})
    complete = {c: [r for r in rows if c in r["correct"]] for c in conditions}
    summary = {"n_trials": len(rows), "conditions": {}, "contrasts": {}, "h2": {}}

    def acc_block(rs, c):
        v = np.array([r["correct"][c] for r in rs], float)
        if not len(v):
            return None
        m, lo, hi, _ = _bootstrap_ci(v, np.array([r["scene"] for r in rs]), rng)
        return {"n": len(v), "acc": m, "ci": [lo, hi]}

    for c in conditions:
        rs = complete[c]
        block = {"all": acc_block(rs, c), "cells": {}, "bins": {}}
        for app in ("identical", "distinct"):
            for mot in ("fixed", "camera", "cover"):
                sub = [r for r in rs if r["appearance"] == app and r["motion"] == mot]
                if sub:
                    block["cells"][f"{app}/{mot}"] = acc_block(sub, c)
        for mot, bins in (("camera", CAMERA_BINS), ("cover", COVER_BINS)):
            for lo, hi in bins:
                sub = [r for r in rs if r["motion"] == mot and lo <= r["magnitude"] < hi]
                if sub:
                    block["bins"][f"{mot} {lo}-{hi}"] = acc_block(sub, c)
        summary["conditions"][c] = block

    def paired(c1, c2, filt=lambda r: True):
        rs = [r for r in rows if c1 in r["correct"] and c2 in r["correct"] and filt(r)]
        if len(rs) < 3:
            return None
        a = np.array([r["correct"][c1] for r in rs])
        b = np.array([r["correct"][c2] for r in rs])
        return {"n": len(rs), **_paired(a, b, np.array([r["scene"] for r in rs]), rng)}

    memories = [c for c in ("2d", "vggt_separate", "vggt_joint") if c in conditions]
    keyframes = [c for c in conditions if c.endswith(":keyframe")]
    if args.primary_vlm:
        keyframes = [c for c in keyframes if args.primary_vlm.split("/")[-1] in c] or keyframes
    rq1 = {}
    for kf in keyframes[:1]:
        for mem in memories:
            res = paired(mem, kf)
            if res:
                rq1[f"{mem} - {kf}"] = res
    for k, p in zip(rq1, _holm([v["p_boot"] for v in rq1.values()])):
        rq1[k]["p_holm"] = p
    summary["contrasts"]["RQ1"] = rq1
    rq2 = {}
    for v3d in [c for c in ("vggt_separate", "vggt_joint") if c in conditions]:
        if "2d" not in conditions:
            continue
        for mot in ("fixed", "camera", "cover"):
            res = paired(v3d, "2d", lambda r, m=mot: r["motion"] == m)
            if res:
                rq2[f"{v3d} - 2d ({mot})"] = res
    summary["contrasts"]["RQ2"] = rq2
    for v3d in [c for c in ("vggt_separate", "vggt_joint") if c in conditions]:
        rs = [r for r in rows if r["motion"] in ("camera", "fixed") and v3d in r["correct"] and "2d" in r["correct"]]
        if len(rs) >= 10:
            y = np.array([r["correct"][m] for r in rs for m in (v3d, "2d")])
            g = np.array([1 if m == v3d else 0 for r in rs for m in (v3d, "2d")])
            x = np.array([r["magnitude"] if r["motion"] == "camera" else 0.0 for r in rs for _ in (0, 1)])
            summary["h2"][f"{v3d} vs 2d"] = _logit_interaction(y, g, x, rng)

    pr = [r["perception"] for r in rows if r["perception"]]
    if pr:
        summary["perception"] = {k: float(np.mean([bool(p.get(k)) for p in pr]))
                                 for k in ("hidden_detected", "containment_correct", "candidates_exact", "target_in_candidates")}
    out = args.data / "results"
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "report.html").write_text(_html(summary), encoding="utf-8")
    print(_text(summary))
    print(f"report: {out / 'report.html'}")


def _fmt(b):
    return "-" if not b else f"{b['acc']:.2f} [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}] (n={b['n']})"


def _text(s: dict) -> str:
    lines = [f"\n{s['n_trials']} trials (chance 0.33)", f"{'condition':40s} accuracy [95% CI]"]
    for c, b in s["conditions"].items():
        lines.append(f"{c:40s} {_fmt(b['all'])}")
    for name, block in s["contrasts"].items():
        for k, v in block.items():
            lines.append(f"{name} {k}: diff {v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] p={v['p_boot']:.3f}"
                         + (f" p_holm={v['p_holm']:.3f}" if "p_holm" in v else ""))
    for k, v in s["h2"].items():
        lines.append(f"H2 {k}: interaction {v['interaction']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] per 100 deg")
    return "\n".join(lines)


def _html(s: dict) -> str:
    conds = list(s["conditions"])
    cells = sorted({k for b in s["conditions"].values() for k in b["cells"]})
    bins = list(dict.fromkeys(k for b in s["conditions"].values() for k in b["bins"]))

    def table(keys, getter):
        head = "<tr><th>condition</th>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>"
        body = "".join("<tr><td>" + c + "</td>" + "".join(f"<td>{_fmt(getter(c, k))}</td>" for k in keys) + "</tr>"
                       for c in conds)
        return f"<table>{head}{body}</table>"

    contrasts = "".join(
        f"<tr><td>{n}</td><td>{k}</td><td>{v['diff']:+.2f}</td><td>[{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]</td>"
        f"<td>{v['p_boot']:.3f}</td><td>{v.get('p_holm', '')}</td><td>{v['n']}</td></tr>"
        for n, block in s["contrasts"].items() for k, v in block.items())
    h2 = "".join(f"<li>{k}: interaction {v['interaction']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] "
                 f"(log-odds per 100 deg)</li>" for k, v in s["h2"].items())
    perc = "".join(f"<li>{k}: {v:.2f}</li>" for k, v in s.get("perception", {}).items())
    return ("<html><head><meta charset='utf-8'><title>Evaluation report</title><style>"
            "body{font-family:sans-serif;margin:16px}table{border-collapse:collapse;margin-bottom:20px}"
            "td,th{border:1px solid #bbb;padding:4px 8px;font-size:13px}</style></head><body>"
            f"<h1>Evaluation report ({s['n_trials']} trials, chance 0.33)</h1>"
            "<p>Accuracy with 95% bootstrap CI (trials resampled within scenes); unknown answers count as wrong.</p>"
            "<h2>Overall</h2>" + table(["all"], lambda c, k: s["conditions"][c]["all"])
            + "<h2>By cell</h2>" + table(cells, lambda c, k: s["conditions"][c]["cells"].get(k))
            + "<h2>By motion size</h2>" + table(bins, lambda c, k: s["conditions"][c]["bins"].get(k))
            + "<h2>Primary contrasts (paired, bootstrap)</h2><table><tr><th>RQ</th><th>contrast</th><th>diff</th>"
              "<th>95% CI</th><th>p</th><th>p (Holm)</th><th>n</th></tr>" + contrasts + "</table>"
            + "<h2>H2: memory type x viewpoint change</h2><ul>" + h2 + "</ul>"
            + "<h2>Perception</h2><ul>" + perc + "</ul></body></html>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("out/test"))
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    ap.add_argument("--vlm", nargs="*", default=["Qwen/Qwen3.5-9B", "Qwen/Qwen3-VL-8B-Instruct"])
    ap.add_argument("--primary-vlm", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N trials (smoke tests)")
    ap.add_argument("--force", action="store_true", help="recompute stages even if results exist")
    args = ap.parse_args()
    args.device = args.device or device_default()

    trials = sorted(args.data.glob("trial_*"))[: args.limit]
    store = TrialStore(args.data)
    print(f"{len(trials)} trials in {args.data}, device {args.device}, stages {args.stages}", flush=True)
    for stage in STAGES:
        if stage in args.stages:
            t0 = time.time()
            globals()[f"stage_{stage}"](trials, store, args)
            print(f"== stage {stage} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
