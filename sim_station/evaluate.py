"""Monte Carlo evaluation: N randomized headless trials per level, in parallel, with a failure-reason histogram.

    python -m sim_station.evaluate --levels 1 2 3 4 5 --trials 50 --jobs 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sim_station.jsonsafe import json_ready

TARGETS = {1: 0.98, 2: 0.98, 3: 0.95, 4: 0.95, 5: 0.90}

_SESSION = None


def _run_trial(args) -> dict:
    global _SESSION
    level, seed, material = args
    from sim_station.runner import Session

    if _SESSION is None:
        _SESSION = Session()
    S = _SESSION
    t0 = time.time()
    try:
        S.reset(level, material, seed=seed, draw=True)
        S.run(max_sim_s=420.0, wall_s=400.0)
        r = S.check_pass()
    except Exception as exc:                       # a crash is a failure with a reason, never a silent skip
        r = dict(ok=False, level=level, seed=seed, material=material, reasons=[f"crash: {type(exc).__name__}: {exc}"], sim_time_s=0.0)
    r["wall_s"] = time.time() - t0
    r.pop("checks", None)
    return r


def wilson_interval(passed: int, trials: int, z: float = 1.96) -> tuple:
    """95% Wilson interval for a binomial rate. Returns (low, high) in [0, 1]."""
    n = int(trials)
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = float(passed) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2.0 * n)) / denom
    margin = z * float(np.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)) / denom
    return (float(max(0.0, centre - margin)), float(min(1.0, centre + margin)))


def summarize(results: List[dict]) -> dict:
    ok = [r for r in results if r["ok"]]
    first = Counter()
    every = Counter()
    for r in results:
        if not r["ok"]:
            reasons = r.get("reasons") or ["unknown"]
            first[str(reasons[0])[:90]] += 1
            for reason in reasons:
                every[str(reason)[:90]] += 1
    def col(rows, key, fn=np.mean):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(fn(vals)) if vals else float("nan")
    lo, hi = wilson_interval(len(ok), len(results))
    return dict(
        trials=len(results), passed=len(ok), success_rate=len(ok) / max(len(results), 1),
        wilson95=[lo, hi],
        sim_time_mean_s=col(ok, "sim_time_s"),
        sim_time_mean_s_note="mean over passing trials only",
        sim_time_all_mean_s=col(results, "sim_time_s"),
        sim_time_p95_s=col(ok, "sim_time_s", lambda x: np.percentile(x, 95)),
        peak_grip_n_max=float(max((r.get("peak_grip_n", 0.0) for r in results), default=0.0)),
        max_tilt_deg_max=float(max((r.get("max_tilt_deg", 0.0) for r in results), default=0.0)),
        coaster_offset_mm_mean=col(ok, "coaster_offset_mm"),
        fit_err_mm=[round(r["fit_err_mm"], 1) for r in results if r.get("fit_err_mm") is not None],
        failure_reasons=first.most_common(8),
        failure_reasons_all=every.most_common(12),
    )


def write_report(path: str, allres: Dict[int, dict]) -> None:
    """Strict JSON summary plus one JSONL row per trial. NaN becomes null."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    clean = json_ready(allres)
    dest.write_text(json.dumps(clean, allow_nan=False, indent=1))
    rows = dest.with_suffix(".jsonl")
    with rows.open("w") as fh:
        for block in clean.values():
            for row in block.get("results") or []:
                fh.write(json.dumps(row, allow_nan=False) + "\n")


def run_level(level: int, trials: int, jobs: int, material: str = "paper", seed0: int = 0, progress=None) -> dict:
    tasks = [(level, seed0 + i, ("plastic" if (i % 4 == 3) else material)) for i in range(trials)]
    results: List[dict] = []
    from sim_station import scene
    scene.build_xml()                      # write the mesh and texture assets once, before the workers start
    os.environ["SIM_ASSETS_READY"] = "1"
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, min(jobs, trials))) as pool:
        for r in pool.imap_unordered(_run_trial, tasks):
            results.append(r)
            if progress is not None:
                progress(level, len(results), trials, r)
    out = summarize(results)
    out["level"] = level
    out["target"] = TARGETS[level]
    out["meets_target"] = out["success_rate"] >= TARGETS[level]
    out["results"] = sorted(results, key=lambda r: r.get("seed", 0))
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args(argv)
    allres: Dict[int, dict] = {}
    for lv in a.levels:
        def prog(level, i, n, r):
            print(f"  L{level} {i}/{n} seed {r.get('seed')} {'ok' if r['ok'] else 'FAIL ' + (r['reasons'][0][:80] if r['reasons'] else '')}", flush=True)
        t0 = time.time()
        out = run_level(lv, a.trials, a.jobs, seed0=a.seed0, progress=prog)
        allres[lv] = out
        lo, hi = out["wilson95"]
        print(f"Level {lv}: {out['passed']}/{out['trials']} = {100 * out['success_rate']:.0f}% "
              f"(95% Wilson {100 * lo:.0f}–{100 * hi:.0f}%, target {100 * out['target']:.0f}%) "
              f"pass-only mean {out['sim_time_mean_s']:.0f}s sim, all-trial mean {out['sim_time_all_mean_s']:.0f}s, "
              f"peak grip {out['peak_grip_n_max']:.0f} N, max tilt {out['max_tilt_deg_max']:.0f} deg, wall {time.time() - t0:.0f}s")
        for reason, n in out["failure_reasons"]:
            print(f"    {n:3d} x {reason}")
    if a.json:
        write_report(a.json, allres)


if __name__ == "__main__":
    main()
