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
from typing import Dict, List, Optional

import numpy as np

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


def summarize(results: List[dict]) -> dict:
    ok = [r for r in results if r["ok"]]
    reasons = Counter()
    for r in results:
        if not r["ok"]:
            reasons[(r["reasons"][0] if r["reasons"] else "unknown")[:90]] += 1
    f = lambda key, fn=np.mean: float(fn([r[key] for r in ok])) if ok else float("nan")
    return dict(
        trials=len(results), passed=len(ok), success_rate=len(ok) / max(len(results), 1),
        sim_time_mean_s=f("sim_time_s"), sim_time_p95_s=f("sim_time_s", lambda x: np.percentile(x, 95)),
        peak_grip_n_max=float(max((r.get("peak_grip_n", 0.0) for r in results), default=0.0)),
        max_tilt_deg_max=float(max((r.get("max_tilt_deg", 0.0) for r in results), default=0.0)),
        coaster_offset_mm_mean=f("coaster_offset_mm"),
        fit_err_mm=[round(r["fit_err_mm"], 1) for r in results if r.get("fit_err_mm") is not None],
        failure_reasons=reasons.most_common(8),
    )


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
        print(f"Level {lv}: {out['passed']}/{out['trials']} = {100 * out['success_rate']:.0f}% (target {100 * out['target']:.0f}%) "
              f"mean {out['sim_time_mean_s']:.0f}s sim, peak grip {out['peak_grip_n_max']:.0f} N, max tilt {out['max_tilt_deg_max']:.0f} deg, wall {time.time() - t0:.0f}s")
        for reason, n in out["failure_reasons"]:
            print(f"    {n:3d} x {reason}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(allres, fh, default=float, indent=1)


if __name__ == "__main__":
    main()
