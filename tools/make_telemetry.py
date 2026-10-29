"""Record the sensor traces of each nominal level run for the project page.

    python tools/make_telemetry.py 1 2 3 4 5

Re-runs the same seed-0 paper-cup run that `make_videos.py` renders (video time is simulation time) and adds a
`telemetry` block, sampled at 10 Hz, to site/videos/levelN.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

from sim_station.plant import ARMS
from sim_station.runner import Session

HZ = 10.0
OUT = Path(__file__).resolve().parents[1] / "site" / "videos"


def record(lv: int) -> dict:
    S = Session()
    S.reset(lv, "paper", seed=0)
    P, ctx = S.plant, S.ctx
    tel = {k: [] for k in ("t", "cup_w", "tap_w", "cup_f", "tap_f", "contact", "lever", "flow")}
    nxt, last_flow = 0.0, P.flow_time
    while not S.ex.finished and not S.ex.fault and P.time - S.t_zero < 400:
        S.tick()
        t = P.time - S.t_zero
        if t < nxt:
            continue
        nxt += 1.0 / HZ
        row = P._history_sample()
        gs = ctx.guard_state
        if gs.get("label"):
            contact = float(gs["value"])
        else:
            mon = [ctx.mon[a].force for a in ARMS if ctx.mon_on[a]]
            contact = float(np.linalg.norm(mon[0])) if mon else 0.0
        tel["t"].append(round(t, 2))
        for a in ARMS:
            tel[f"{a}_w"].append(round(row[f"{a}_width_mm"], 1))
            tel[f"{a}_f"].append(round(row[f"{a}_grip_f"], 2))
        tel["contact"].append(round(contact, 2))
        tel["lever"].append(round(float(np.degrees(row["lever"])), 1))
        tel["flow"].append(int(P.flow_time > last_flow))
        last_flow = P.flow_time
    res = S.check_pass()
    print(f"level {lv}: pass={res['ok']} sim {res['sim_time_s']:.1f} s, {len(tel['t'])} samples")
    return tel, res


if __name__ == "__main__":
    for lv in map(int, sys.argv[1:] or "12345"):
        tel, res = record(lv)
        p = OUT / f"level{lv}.json"
        d = json.loads(p.read_text())
        assert abs(d["sim_time_s"] - res["sim_time_s"]) < 0.5, "run differs from the rendered video"
        d["telemetry"] = tel
        p.write_text(json.dumps(d, separators=(",", ":")))
