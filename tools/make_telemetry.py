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
    scalar = ("t", "cup_w", "tap_w", "cup_f", "tap_f", "cup_cap", "tap_cap", "cup_speed", "tap_speed", "contact", "contact_thr", "lever", "flow")
    tel = {k: [] for k in scalar}
    joints = {f"{a}_{k}": [] for a in ARMS for k in ("q", "qd", "tau")}
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
            obs = P.read_arm(a)
            tel[f"{a}_w"].append(round(row[f"{a}_width_mm"], 1))
            tel[f"{a}_f"].append(round(row[f"{a}_grip_f"], 2))
            tel[f"{a}_cap"].append(round(float(P.ctl[a]["grip_cap"]), 2))
            tel[f"{a}_speed"].append(round(float(row[f"{a}_speed"]) * 1000.0, 1))
            joints[f"{a}_q"].append([round(float(np.degrees(x)), 2) for x in obs.q])
            joints[f"{a}_qd"].append([round(float(np.degrees(x)), 2) for x in obs.qd])
            joints[f"{a}_tau"].append([round(float(x), 3) for x in obs.tau])
        tel["contact"].append(round(contact, 2))
        tel["contact_thr"].append(round(float(gs.get("threshold") or 0.0), 2))
        tel["lever"].append(round(float(np.degrees(row["lever"])), 1))
        tel["flow"].append(int(P.flow_time > last_flow))
        last_flow = P.flow_time
    n = len(tel["t"])
    for key, rows in joints.items():
        tel[key] = [[rows[s][j] for s in range(n)] for j in range(6)]
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
