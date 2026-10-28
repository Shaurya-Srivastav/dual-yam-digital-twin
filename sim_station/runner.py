"""A headless run of one level: reset, execute the step graph, referee the result.

The referee (`check_pass`) is the only code that reads the true object poses. Skills never do.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from sim_station import layout as L
from sim_station import levels
from sim_station.kin import ArmKin
from sim_station.plant import ARMS, Plant, Randomization
from sim_station import motion as M
from sim_station.runtime import CTRL_STEPS, Ctx, Executor

_KIN: Optional[Dict[str, ArmKin]] = None


def shared_kin() -> Dict[str, ArmKin]:
    global _KIN
    if _KIN is None:
        _KIN = {a: ArmKin(a) for a in ARMS}
    return _KIN


@dataclass
class Metrics:
    peak_grip_n: float = 0.0
    max_tilt_deg: float = 0.0
    min_spout_clearance_mm: float = 1e9
    hand_path_m: float = 0.0


class Session:
    def __init__(self, plant: Optional[Plant] = None) -> None:
        self.plant = plant or Plant()
        self.level = 1
        self.material = "paper"
        self.seed = 0
        self.ctx: Optional[Ctx] = None
        self.ex: Optional[Executor] = None
        self.metrics = Metrics()
        self._last_tcp: Dict[str, np.ndarray] = {}
        self.reset(1)

    # ------------------------------------------------------------------ setup
    def reset(self, level: int, material: str = "paper", seed: int = 0, rand: Optional[Randomization] = None, draw: bool = False, calibrate: bool = True) -> None:
        self.level, self.material, self.seed = int(level), material, int(seed)
        if rand is None:
            rand = Randomization.draw(np.random.default_rng(self.seed), self.level) if draw else Randomization()
        self.rand = rand
        self.plant.reset(self.level, material, rand=rand, seed=self.seed)
        self.ctx = Ctx(self.plant, kin=shared_kin())
        self.ctx.reset()
        self.steps = levels.build_level(self.level, self.ctx)
        self.ex = Executor(self.ctx, self.steps)
        self.metrics = Metrics()
        if calibrate:
            self.calibrate_gravity()
        self.cup0 = self.plant.cup_state()["xyz"][:2].copy()
        self.t_zero = self.plant.time
        self._last_tcp = {a: self.plant.tcp_true(a)[0].copy() for a in ARMS}

    # ------------------------------------------------- power-up self-calibration
    def _drive(self, trajs: Dict[str, M.Traj], hold_s: float = 0.0) -> None:
        """Blocking helper for the power-up routine: play joint trajectories on both arms, then hold."""
        ctx, P = self.ctx, self.plant
        n = max(len(t) for t in trajs.values())
        for k in range(n):
            ctx.sense()
            for a, t in trajs.items():
                i = min(k, len(t) - 1)
                ctx.q_cmd[a] = t.q[i]
                P.command_arm(a, t.q[i], t.qd[i] if k < len(t) else np.zeros(6))
            P.step(CTRL_STEPS)
        for _ in range(int(hold_s / (CTRL_STEPS * P.dt))):
            ctx.sense()
            for a in trajs:
                ctx.hold(a)
            P.step(CTRL_STEPS)

    def calibrate_gravity(self) -> None:
        """What a robot does at power-up: hold a loaded pose, and from the position sag at the known commanded torque work out
        how far the real arm's effective gravity feed-forward is from the model. a_i = 1 + kp_i * e_i / g_i per joint."""
        ctx, P = self.ctx, self.plant
        # two loaded poses held high above the bench, clear of every object and of the other arm
        pose = np.array([0.0, 1.2, 1.1, 0.3, 0.0, 0.0])
        pose2 = np.array([0.0, 1.5, 1.5, 0.0, 0.0, 0.0])
        num = {a: np.zeros(6) for a in ARMS}
        den = {a: np.zeros(6) for a in ARMS}
        for target in (pose, pose2):
            trajs = {a: M.joint_move(ctx.q_cmd[a], target, min_T=1.6) for a in ARMS}
            self._drive(trajs, hold_s=0.7)
            es = {a: [] for a in ARMS}
            for _ in range(int(0.5 / (CTRL_STEPS * P.dt))):
                ctx.sense()
                for a in ARMS:
                    ctx.hold(a)
                    es[a].append(ctx.q_cmd[a] - ctx.mon[a].q)
                P.step(CTRL_STEPS)
            for a in ARMS:
                e = np.mean(es[a], axis=0)
                g = P.gravity_model(a, ctx.mon[a].q, adapted=False)
                w = g * g
                num[a] += w * (1.0 + P.kp[a] * e / np.where(np.abs(g) > 1e-6, g, 1.0))
                den[a] += w
        for a in ARMS:
            use = den[a] > 0.5
            adapt = np.ones(6)
            adapt[use] = num[a][use] / den[a][use]
            P.grav_adapt[a] = np.clip(adapt, 0.8, 1.25)
        # back to rest, unhurried, then forget everything the power-up moves touched
        self._drive({a: M.joint_move(ctx.q_cmd[a], L.REST_Q, min_T=2.0) for a in ARMS}, hold_s=0.5)
        P.violations = {}
        P.flow_time = 0.0
        P.flow_time_with_cup = 0.0
        P.flow_events = []
        ctx.sense(force=True)

    # -------------------------------------------------------------- execution
    def tick(self) -> None:
        self.ex.tick()
        self._observe()
        # any arm-to-arm contact aborts the run (a twin of the real e-stop: two arms touching is never part of a plan)
        if not self.ex.fault and any(v["kind"] == "arm_arm" for v in self.plant.violations.values()):
            self.ex.abort("arms touched each other: stopped")

    def _observe(self) -> None:
        P, m = self.plant, self.metrics
        for a in ARMS:
            m.peak_grip_n = max(m.peak_grip_n, float(P.pad_force(a)))
            p = P.tcp_true(a)[0]
            m.hand_path_m += float(np.linalg.norm(p - self._last_tcp[a]))
            self._last_tcp[a] = p.copy()
        cs = P.cup_state()
        m.max_tilt_deg = max(m.max_tilt_deg, float(np.degrees(np.arccos(np.clip(cs["up"], -1, 1)))))

    def run(self, max_sim_s: float = 400.0, wall_s: float = 900.0) -> None:
        t0 = time.time()
        tmax = self.plant.time + max_sim_s
        while not self.ex.finished and not self.ex.fault and self.plant.time < tmax and time.time() - t0 < wall_s:
            self.tick()

    # ---------------------------------------------------------------- referee
    def check_pass(self) -> dict:
        P, ex = self.plant, self.ex
        cs = P.cup_state()
        xyz = cs["xyz"]
        coaster = P.model.body_pos[P.body["coaster"]][:2]
        tilt = float(np.degrees(np.arccos(np.clip(cs["up"], -1, 1))))
        off = float(np.linalg.norm(xyz[:2] - coaster))
        disp = float(np.linalg.norm(P.data.xpos[P.body["dispenser"]][:2] - P.dispenser_rest)) * 1000.0
        rest_err = {a: float(np.linalg.norm(P.arm_q(a)[:5] - L.REST_Q[:5])) for a in ARMS}
        stack_ok = True
        if P.level in (2, 3) and P.model.body_gravcomp[P.body[P.active_rest]] == 0.0:
            ps, Rs = P.body_pose(P.active_rest)
            stack_ok = bool(Rs[2, 2] > np.cos(np.radians(20.0)) and ps[2] > 0.0)       # the remaining stack still stands on the mat
        checks = {
            "stack_still_standing": stack_ok,
            "stack_never_touched": not any(v["kind"] == "arm_stack" for v in P.violations.values()),
            "completed": bool(ex.finished and not ex.fault),
            "cup_upright_10deg": tilt <= 10.0,
            "cup_in_coaster_15mm": off <= 0.015,
            "flow_7s_with_cup": abs(P.flow_time_with_cup - 7.0) <= 0.2 and abs(P.flow_time - P.flow_time_with_cup) <= 0.2,
            "lever_closed": P.lever_angle() < 0.1,
            "dispenser_still_2mm": disp < 2.0,
            "no_collisions": len(P.fatal_violations()) == 0,
            "arms_at_rest": all(e < 0.12 for e in rest_err.values()),
        }
        reasons = [k for k, v in checks.items() if not v]
        if ex.fault:
            reasons.insert(0, f"fault: {ex.fault}")
        fit = self.ctx.scratch.get("fit_xy")
        return dict(
            fit_err_mm=(float(np.linalg.norm(fit - self.cup0)) * 1000.0 if fit is not None else None),
            ok=not reasons, checks=checks, reasons=reasons, level=self.level, material=self.material, seed=self.seed,
            sim_time_s=float(P.time - self.t_zero), tilt_deg=tilt, coaster_offset_mm=off * 1000.0, flow_s=float(P.flow_time_with_cup),
            dispenser_moved_mm=disp, peak_grip_n=self.metrics.peak_grip_n, max_tilt_deg=self.metrics.max_tilt_deg,
            hand_path_m=self.metrics.hand_path_m, violations=[(v["kind"], v["a"], v["b"], round(v["t"], 1)) for v in P.violations.values()],
        )
