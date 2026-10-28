"""Proof that the twin is 1:1 with the real station (items 33 and 34).

replay_l1       plays the real Level 1 joint waypoints from the Pi log through the same control loop and lets physics decide whether
                the cup comes out of the coaster, goes under the outlet and goes back. A failure here is a calibration or geometry
                error, never something to tune away in the controller.
equivalence_l1  asks, for every real L1 waypoint, whether our strict Cartesian IK reproduces the real hand pose, and how far the
                poses our skills command (grasp, seat) are from the real ones.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from sim_station import layout as L
from sim_station import motion as M
from sim_station.kin import rot_error
from sim_station.plant import ARMS
from sim_station.runner import Session
from sim_station.runtime import CTRL_STEPS

GRIP_SPAN = 0.0776               # calibrated jaw gap at command 1.0


def _play(S: Session, q_to: np.ndarray, T: float, width: float, cap: float, hold: float = 0.35) -> None:
    ctx, P = S.ctx, S.plant
    traj = M.joint_move(ctx.q_cmd["cup"], q_to, min_T=T)
    P.command_gripper("cup", width, cap_n=cap, rate=0.05)
    for k in range(len(traj)):
        ctx.sense()
        ctx.q_cmd["cup"] = traj.q[k]
        P.command_arm("cup", traj.q[k], traj.qd[k])
        ctx.hold("tap")
        P.step(CTRL_STEPS)
        S._observe()
    for _ in range(int(hold / (CTRL_STEPS * P.dt))):
        ctx.sense()
        ctx.hold("cup")
        ctx.hold("tap")
        P.step(CTRL_STEPS)
        S._observe()


def replay_l1(material: str = "paper") -> dict:
    S = Session()
    S.reset(1, material, seed=0)
    P = S.plant
    W = L.REAL_L1_CUP_WAYPOINTS
    cap = 20.0 if material == "paper" else 10.0
    # the real cup started where the real hand closed on it: 2 cm beside the recess centre (the grasp and in_coaster poses are 2.0 cm apart)
    kin = S.ctx.kin["cup"]
    g = kin.fk(W["grasp"][:6])[:3, 3]
    qa = P.free_qadr[P.active_cup]
    P.data.qpos[qa: qa + 2] = g[:2]
    mujoco_forward = __import__("mujoco").mj_forward
    mujoco_forward(P.model, P.data)
    for _ in range(200):
        P.step(1)
    c0 = P.cup_state()["xyz"].copy()
    log: List[dict] = []

    def snap(name: str) -> None:
        cs = P.cup_state()
        outlet = P.outlet_xy_true()
        log.append(dict(step=name, cup_xyz=[round(float(x), 4) for x in cs["xyz"]], tilt_deg=round(float(np.degrees(np.arccos(np.clip(cs["up"], -1, 1)))), 1),
                        to_outlet_mm=round(float(np.linalg.norm(cs["xyz"][:2] - outlet)) * 1000, 1),
                        to_coaster_mm=round(float(np.linalg.norm(cs["xyz"][:2] - P.model.body_pos[P.body["coaster"]][:2])) * 1000, 1)))

    seq = [("approach", 2.6, 1.0), ("grasp", 1.6, 1.0), ("close", 0, 0.0), ("paper_hold", 1.2, None), ("travel_clear", 1.8, None),
           ("lower_for_spout", 1.8, None), ("under_spout", 2.0, None), ("spout_exit", 2.0, None), ("lift_out", 1.6, None),
           ("above_coaster", 2.2, None), ("in_coaster", 1.4, None), ("release", 1.0, 1.0), ("rest", 2.8, 1.0)]
    held_cmd = None
    for name, T, g in seq:
        if name == "close":
            _play(S, W["grasp"][:6], 0.3, 0.0, cap, hold=0.9)
            held_cmd = W["paper_hold"][6]
            snap("closed on cup")
            continue
        q = W[name][:6]
        cmd = g if g is not None else W[name][6]
        width = GRIP_SPAN * (cmd if g is not None else held_cmd)
        _play(S, q, T, width if g is not None else 0.0, cap)        # a held cup is never squeezed past what contact allows
        snap(name)
    cs = P.cup_state()
    outlet_err = [r["to_outlet_mm"] for r in log if r["step"] == "under_spout"]
    end = log[-1]
    res = dict(
        kind="replay_l1", material=material,
        lifted_out_of_coaster=any(r["cup_xyz"][2] > c0[2] + 0.02 for r in log),
        under_outlet_mm=outlet_err[0] if outlet_err else None,
        under_outlet_ok=bool(outlet_err and outlet_err[0] <= 5.0),
        returned_to_coaster_mm=end["to_coaster_mm"], returned_ok=bool(end["to_coaster_mm"] <= 15.0 and end["tilt_deg"] <= 10.0),
        max_tilt_deg=max(r["tilt_deg"] for r in log), peak_grip_n=round(S.metrics.peak_grip_n, 1), log=log,
    )
    res["ok"] = bool(res["lifted_out_of_coaster"] and res["under_outlet_ok"] and res["returned_ok"])
    return res


def _pose_err(Ta: np.ndarray, Tb: np.ndarray) -> Dict[str, float]:
    return dict(pos_mm=float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1000.0), ori_deg=float(np.degrees(np.linalg.norm(rot_error(Ta[:3, :3], Tb[:3, :3])))))


def equivalence_l1() -> dict:
    S = Session()
    S.reset(1, "paper", seed=0)
    ctx = S.ctx
    kin = ctx.kin["cup"]
    W = L.REAL_L1_CUP_WAYPOINTS
    rows = []
    for name, v in W.items():
        q_real = v[:6]
        T_real = kin.fk(q_real)
        seeds = [L.REST_Q, np.array([0.0, 1.6, 1.1, 0.7, 0.0, 0.0]), np.array([0.3, 1.2, 0.9, 0.5, 0.0, 0.0])]       # never the real q itself
        r = kin.ik_multi(T_real, seeds)
        if r.ok:
            e = _pose_err(kin.fk(r.q), T_real)
            rows.append(dict(waypoint=name, ik_ok=True, pos_mm=round(e["pos_mm"], 2), ori_deg=round(e["ori_deg"], 2),
                             max_joint_diff_rad=round(float(np.max(np.abs(r.q - q_real))), 3)))
        else:
            rows.append(dict(waypoint=name, ik_ok=False, reason=r.reason))
    # poses the skills command, against the real ones at the same moments
    cmp_rows = []
    fr = ctx.side_frames("cup", ctx.bel.cup_xy, L.COASTER_FLOOR_Z + 0.07, depth=0.004, standoff=0.05)
    cmp_rows.append(dict(moment="grasp (cup in the coaster)", **{k: round(v, 1) for k, v in _pose_err(fr["grasp"], kin.fk(W["grasp"][:6])).items()}))
    seat = L.COASTER_FLOOR_Z + 0.07
    fr2 = ctx.side_frames("cup", ctx.bel.coaster_xy, seat, depth=0.004, standoff=0.05)
    cmp_rows.append(dict(moment="in_coaster (hand pose at the seat)", **{k: round(v, 1) for k, v in _pose_err(fr2["grasp"], kin.fk(W["in_coaster"][:6])).items()}))
    out = dict(kind="equivalence_l1", ik_rows=rows, skill_rows=cmp_rows)
    out["ik_ok"] = all(r["ik_ok"] and r["pos_mm"] <= 2.0 and r["ori_deg"] <= 2.0 for r in rows)
    out["skills_within"] = all(r["pos_mm"] <= 20.0 and r["ori_deg"] <= 25.0 for r in cmp_rows)
    out["ok"] = bool(out["ik_ok"])
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(replay_l1(), indent=1, default=float))
    print(json.dumps(equivalence_l1(), indent=1, default=float))
